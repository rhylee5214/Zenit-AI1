"""GPU trainer for the PyTorch ZENIT model.

Same run layout, metrics and registry as the NumPy trainer, so the dashboard
shows both. Supports bf16/fp16 autocast, gradient accumulation, gradient
checkpointing, torch.compile, resume and the STOP file.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ..data.loader import open_dataset
from ..model.torch_model import ZenitModel, load_zenit, save_zenit
from ..tokenizer.bpe import ZenitTokenizer
from .common import export_and_register, plot_losses
from .config import TrainConfig
from .optim_np import lr_at
from .runlog import RunLogger, system_stats


def pick_device(requested: str) -> str:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    if requested in ("cuda", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_ctx(device: str, precision: str):
    if device == "cpu" or precision == "fp32":
        return nullcontext(), None
    if precision == "bf16" and (device != "cuda" or torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16), None
    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.float16), scaler


@torch.no_grad()
def evaluate(model, data, tc, device, ctx) -> float:
    model.eval()
    rng = np.random.default_rng(12345)
    losses = []
    for _ in range(tc.eval_batches):
        x, y = data.batch(tc.batch_size, tc.seq_len, rng)
        with ctx:
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, model, opt, scaler, step: int, extra: dict) -> None:
    raw = getattr(model, "_orig_mod", model)
    save_zenit(raw, path, {"step": step, **extra})
    state = {"optimizer": opt.state_dict(), "step": step}
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path / "optimizer.pt")


def train(tc: TrainConfig, resume_dir: str | None = None, log: RunLogger | None = None) -> dict:
    device = pick_device(tc.device)
    torch.manual_seed(tc.seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tok = ZenitTokenizer.load(tc.tokenizer)
    train_data, meta = open_dataset(tc.data_dir, "train", tc.mode, pad_id=tok.pad_id)
    val_data, _ = open_dataset(tc.data_dir, "val", tc.mode, pad_id=tok.pad_id)
    log = log or RunLogger(tc.runs_dir, tc.run_name, resume_dir)

    ckpt = log.latest_checkpoint() if resume_dir else None
    start_step = 0
    if ckpt:
        model = load_zenit(ckpt, device=device)
    elif tc.init_from:
        model = load_zenit(tc.init_from, device=device)
    else:
        model = ZenitModel(tc.model_config(tok.vocab_size)).to(device)
    model.train()
    model.grad_checkpointing = tc.grad_checkpointing
    if model.cfg.vocab_size != tok.vocab_size:
        raise ValueError("tokenizer and model vocab sizes differ")

    opt = model.optimizer(tc.lr, tc.weight_decay, (tc.beta1, tc.beta2))
    ctx, scaler = autocast_ctx(device, tc.precision)
    if ckpt:
        state = torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=False)
        opt.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
    run_model = torch.compile(model) if tc.compile else model

    n_params = model.num_parameters()
    tokens_train = len(train_data) if tc.mode == "pretrain" else train_data.num_tokens()
    hw = system_stats()
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        hw.update(gpu=props.name, gpu_total_gb=round(props.total_memory / 1e9, 1))
    log.update(status="running", backend="torch", device=device, version=tc.version,
               config=tc.to_dict(), model_config=model.cfg.to_dict(), parameters=n_params,
               max_steps=tc.max_steps, hardware=hw,
               dataset={"dir": tc.data_dir, "tokens_train": tokens_train,
                        "tokens_val": meta["tokens"]["val"],
                        "manifest_fingerprint": meta.get("manifest_fingerprint")})
    print(f"[zenit] run {log.run_id} on {device}: {n_params:,} params")

    rng = np.random.default_rng(tc.seed + start_step)
    best_val = log.state.get("best_val_loss")
    tokens_seen = start_step * tc.batch_size * tc.seq_len * tc.grad_accum
    last_loss, status, step = None, "completed", start_step
    t_last = time.time()
    try:
        for step in range(start_step, tc.max_steps):
            lr = lr_at(step, tc.max_steps, tc.lr, tc.min_lr, tc.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for _ in range(tc.grad_accum):
                x, y = train_data.batch(tc.batch_size, tc.seq_len, rng)
                x = torch.from_numpy(x).pin_memory().to(device, non_blocking=True) if device.startswith("cuda") \
                    else torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)
                with ctx:
                    _, loss = run_model(x, y)
                loss = loss / tc.grad_accum
                (scaler.scale(loss) if scaler else loss).backward()
                loss_sum += loss.item()
            if scaler:
                scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            last_loss = loss_sum
            tokens_seen += tc.batch_size * tc.seq_len * tc.grad_accum
            done = step + 1

            if done % tc.log_interval == 0 or done == 1:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                dt = time.time() - t_last
                t_last = time.time()
                tps = tc.log_interval * tc.batch_size * tc.seq_len * tc.grad_accum / max(dt, 1e-9)
                log.log(step=done, loss=round(last_loss, 5), lr=lr, grad_norm=round(gnorm, 4),
                        tokens_seen=tokens_seen, tok_per_sec=round(tps), **system_stats())
                log.update(step=done, last_loss=round(last_loss, 5), tokens_seen=tokens_seen)
                print(f"step {done:6d}/{tc.max_steps} loss {last_loss:.4f} lr {lr:.2e} "
                      f"gnorm {gnorm:.2f} {tps:,.0f} tok/s")

            if done % tc.eval_interval == 0 or done == tc.max_steps:
                val = evaluate(run_model, val_data, tc, device, ctx)
                log.log(step=done, val_loss=round(val, 5), val_ppl=round(float(np.exp(val)), 3))
                if best_val is None or val < best_val:
                    best_val = val
                    save_checkpoint(log.dir / "checkpoints" / "best", model, opt, scaler, done, {"val_loss": val})
                log.update(best_val_loss=round(best_val, 5), last_val_loss=round(val, 5))

            if done % tc.ckpt_interval == 0:
                path = log.checkpoint_dir(done)
                save_checkpoint(path, model, opt, scaler, done, {"loss": last_loss})
                log.mark_latest(path)
                log.prune(tc.keep_checkpoints)

            if log.stop_requested():
                status = "stopped"
                break
    except KeyboardInterrupt:
        status = "stopped"
    except Exception as err:
        log.update(status="failed", error=repr(err))
        raise

    final = step + 1
    path = log.checkpoint_dir(final)
    save_checkpoint(path, model, opt, scaler, final, {"loss": last_loss})
    log.mark_latest(path)
    log.prune(tc.keep_checkpoints)
    plot_losses(log.dir)
    best_dir = log.dir / "checkpoints" / "best"
    src = best_dir if best_dir.exists() else path
    fmt = "zenit-safetensors" if (src / "model.safetensors").exists() else "zenit-torch-pt"
    result = export_and_register(log, tc, src, {
        "parameters": n_params, "steps": final,
        "final_train_loss": round(last_loss, 5) if last_loss is not None else None,
        "best_val_loss": round(best_val, 5) if best_val is not None else None,
        "exported_checkpoint": src.name,
    }, weights_format=fmt)
    log.update(status=status, ended=True, export=result["export_dir"])
    return {"run_dir": str(log.dir), **result}
