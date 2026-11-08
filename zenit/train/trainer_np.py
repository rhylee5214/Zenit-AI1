"""CPU trainer for the NumPy ZENIT model."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..data.loader import open_dataset
from ..model.numpy_model import NumpyZenit
from ..tokenizer.bpe import ZenitTokenizer
from .common import export_and_register, plot_losses
from .config import TrainConfig
from .optim_np import AdamW, clip_grads, lr_at
from .runlog import RunLogger, system_stats


def evaluate(model: NumpyZenit, data, tc: TrainConfig) -> float:
    rng = np.random.default_rng(12345)          # same batches every eval
    losses = []
    for _ in range(tc.eval_batches):
        x, y = data.batch(tc.batch_size, tc.seq_len, rng)
        losses.append(model.forward(x, y)[1])
    return float(np.mean(losses))


def save_checkpoint(path: Path, model: NumpyZenit, opt: AdamW, step: int, extra: dict) -> None:
    model.save(path, extra_meta={"step": step, **extra})
    np.savez(path / "optimizer.npz", **opt.state_dict())


def train(tc: TrainConfig, resume_dir: str | None = None, log: RunLogger | None = None) -> dict:
    tok = ZenitTokenizer.load(tc.tokenizer)
    train_data, meta = open_dataset(tc.data_dir, "train", tc.mode, pad_id=tok.pad_id)
    val_data, _ = open_dataset(tc.data_dir, "val", tc.mode, pad_id=tok.pad_id)
    log = log or RunLogger(tc.runs_dir, tc.run_name, resume_dir)

    start_step = 0
    ckpt = log.latest_checkpoint() if resume_dir else None
    if ckpt:
        model = NumpyZenit.load(ckpt)
        opt = AdamW(model.params, (tc.beta1, tc.beta2), weight_decay=tc.weight_decay)
        with np.load(ckpt / "optimizer.npz") as st:
            opt.load_state_dict(st)
        start_step = opt.t
    elif tc.init_from:
        model = NumpyZenit.load(tc.init_from)
        opt = AdamW(model.params, (tc.beta1, tc.beta2), weight_decay=tc.weight_decay)
    else:
        mc = tc.model_config(tok.vocab_size)
        model = NumpyZenit(mc, seed=tc.seed)
        opt = AdamW(model.params, (tc.beta1, tc.beta2), weight_decay=tc.weight_decay)
    if model.cfg.vocab_size != tok.vocab_size:
        raise ValueError("tokenizer and model vocab sizes differ")

    n_params = model.num_parameters()
    tokens_train = len(train_data) if tc.mode == "pretrain" else train_data.num_tokens()
    log.update(status="running", backend="numpy", device="cpu", version=tc.version,
               config=tc.to_dict(), model_config=model.cfg.to_dict(), parameters=n_params,
               max_steps=tc.max_steps, dataset={"dir": tc.data_dir, "tokens_train": tokens_train,
                                                "tokens_val": meta["tokens"]["val"],
                                                "manifest_fingerprint": meta.get("manifest_fingerprint")},
               hardware=system_stats())
    print(f"[zenit] run {log.run_id}: {n_params:,} params, {tokens_train:,} train tokens")

    rng = np.random.default_rng(tc.seed + start_step)
    best_val = log.state.get("best_val_loss")
    last_loss = None
    tokens_seen = start_step * tc.batch_size * tc.seq_len * tc.grad_accum
    t_last = time.time()
    status = "completed"
    step = start_step
    try:
        for step in range(start_step, tc.max_steps):
            lr = lr_at(step, tc.max_steps, tc.lr, tc.min_lr, tc.warmup_steps)
            grads, loss_sum = None, 0.0
            for _ in range(tc.grad_accum):
                x, y = train_data.batch(tc.batch_size, tc.seq_len, rng)
                _, loss, cache = model.forward(x, y)
                g = model.backward(cache)
                loss_sum += loss
                if grads is None:
                    grads = g
                else:
                    for k in grads:
                        grads[k] += g[k]
            if tc.grad_accum > 1:
                for k in grads:
                    grads[k] /= tc.grad_accum
            gnorm = clip_grads(grads, tc.grad_clip)
            opt.step(grads, lr)
            last_loss = loss_sum / tc.grad_accum
            tokens_seen += tc.batch_size * tc.seq_len * tc.grad_accum
            done = step + 1

            if done % tc.log_interval == 0 or done == 1:
                dt = time.time() - t_last
                t_last = time.time()
                tps = tc.log_interval * tc.batch_size * tc.seq_len * tc.grad_accum / max(dt, 1e-9)
                log.log(step=done, loss=round(last_loss, 5), lr=lr, grad_norm=round(gnorm, 4),
                        tokens_seen=tokens_seen, tok_per_sec=round(tps), **system_stats())
                log.update(step=done, last_loss=round(last_loss, 5), tokens_seen=tokens_seen)
                print(f"step {done:5d}/{tc.max_steps} loss {last_loss:.4f} lr {lr:.2e} "
                      f"gnorm {gnorm:.2f} {tps:,.0f} tok/s")

            if done % tc.eval_interval == 0 or done == tc.max_steps:
                val = evaluate(model, val_data, tc)
                log.log(step=done, val_loss=round(val, 5), val_ppl=round(float(np.exp(val)), 3))
                print(f"           val_loss {val:.4f}")
                if best_val is None or val < best_val:
                    best_val = val
                    save_checkpoint(log.dir / "checkpoints" / "best", model, opt, done, {"val_loss": val})
                log.update(best_val_loss=round(best_val, 5), last_val_loss=round(val, 5))

            if done % tc.ckpt_interval == 0:
                path = log.checkpoint_dir(done)
                save_checkpoint(path, model, opt, done, {"loss": last_loss})
                log.mark_latest(path)
                log.prune(tc.keep_checkpoints)

            if log.stop_requested():
                status = "stopped"
                print("[zenit] STOP file found, stopping")
                break
    except KeyboardInterrupt:
        status = "stopped"
    except Exception as err:
        log.update(status="failed", error=repr(err))
        raise

    final_step = step + 1
    path = log.checkpoint_dir(final_step)
    save_checkpoint(path, model, opt, final_step, {"loss": last_loss})
    log.mark_latest(path)
    log.prune(tc.keep_checkpoints)
    plot_losses(log.dir)

    best_dir = log.dir / "checkpoints" / "best"
    src = best_dir if best_dir.exists() else path
    result = export_and_register(log, tc, src, {
        "parameters": n_params, "steps": final_step,
        "final_train_loss": round(last_loss, 5) if last_loss is not None else None,
        "best_val_loss": round(best_val, 5) if best_val is not None else None,
        "exported_checkpoint": src.name,
    }, weights_format="zenit-npz")
    log.update(status=status, ended=True, export=result["export_dir"])
    print(f"[zenit] {status}. exported {result['export_dir']}")
    return {"run_dir": str(log.dir), **result}
