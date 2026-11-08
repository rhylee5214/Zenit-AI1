"""Direct Preference Optimisation (DPO) on top of a ZENIT SFT adapter.

    python -m zenit.finetune.dpo --config configs/zenit-4.0-dpo.json

Preference data rows: {"prompt": "..." or [messages], "chosen": "...", "rejected": "..."}.
The reference policy is the SFT adapter as it was before DPO starts: its log
probabilities get computed once up front, so only one copy of the model sits
in GPU memory.

Loss: -log sigmoid(beta * ((logp_c - ref_c) - (logp_r - ref_r)))
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from ..data.loading import iter_records
from ..data.manifest import load_manifest
from ..persona.identity import ZENIT_SYSTEM_PROMPT
from ..train.registry import Registry
from ..train.runlog import RunLogger, system_stats
from .common import FALLBACK_TEMPLATE, FinetuneConfig, decide_4bit, write_card


def load_pairs(cfg: FinetuneConfig) -> list[dict]:
    rows = []
    if cfg.pref_data and Path(cfg.pref_data).exists():
        for line in Path(cfg.pref_data).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if cfg.pref_manifest:
        m = load_manifest(cfg.pref_manifest)
        for src in m.sources:
            for i, rec in enumerate(iter_records(src, Path.cwd())):
                if cfg.max_examples and i >= cfg.max_examples:
                    break
                rows.append(rec)
    out = []
    for r in rows:
        prompt = r["prompt"]
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        chosen, rejected = r["chosen"], r["rejected"]
        if isinstance(chosen, list):        # some datasets store full conversations
            chosen = chosen[-1]["content"]
        if isinstance(rejected, list):
            rejected = rejected[-1]["content"]
        if chosen and rejected and chosen != rejected:
            out.append({"messages": msgs, "chosen": chosen, "rejected": rejected})
    return out


def encode_pair(tok, msgs, answer, max_len, kw):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    ids = (p_ids + a_ids)[-max_len:]
    n_answer = min(len(a_ids), len(ids))
    return ids, n_answer


def seq_logprob(model, ids_list, n_answer_list, device, pad_id):
    import torch
    width = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x, device=device)
        mask[i, :len(x)] = 1
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    targets = ids[:, 1:]
    lp = torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    out = []
    for i, x in enumerate(ids_list):
        end = len(x) - 1
        out.append(lp[i, end - n_answer_list[i]:end].sum())
    return torch.stack(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = FinetuneConfig.load(a.config, a.set)
    if not cfg.init_adapter:
        print("[zenit] DPO needs init_adapter (the SFT adapter to improve)")
        return 2
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = load_pairs(cfg)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    for p in pairs:
        if p["messages"][0]["role"] != "system" and rng.random() < cfg.system_prompt_rate:
            p["messages"] = [{"role": "system", "content": ZENIT_SYSTEM_PROMPT}] + p["messages"]
    print(f"[zenit] {len(pairs)} preference pairs")
    tok = AutoTokenizer.from_pretrained(cfg.init_adapter)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = []
    for p in pairs:
        c, nc = encode_pair(tok, p["messages"], p["chosen"], cfg.max_seq_len, cfg.chat_template_kwargs)
        r, nr = encode_pair(tok, p["messages"], p["rejected"], cfg.max_seq_len, cfg.chat_template_kwargs)
        enc.append((c, nc, r, nr))

    use_4bit = decide_4bit(cfg, 8.0)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, cfg.init_adapter, is_trainable=True)
    device = next(model.parameters()).device

    print("[zenit] computing reference log-probs ...")
    model.eval()
    ref = []
    with torch.no_grad():
        for c, nc, r, nr in enc:
            lp = seq_logprob(model, [c, r], [nc, nr], device, tok.pad_token_id)
            ref.append((lp[0].item(), lp[1].item()))

    log = RunLogger(cfg.runs_dir, cfg.run_name)
    steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(len(enc) * cfg.epochs / (cfg.batch_size * cfg.grad_accum))
    log.update(status="running", backend="torch+peft", device=torch.cuda.get_device_name(0), version=cfg.version,
               config=cfg.to_dict(), max_steps=steps, hardware=system_stats(),
               dataset={"pairs": len(enc), "tokens_train": sum(len(e[0]) + len(e[2]) for e in enc)})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(steps * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    order = list(range(len(enc)))
    cursor, status = 0, "completed"
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "acc": 0.0, "margin": 0.0}
        for _ in range(cfg.grad_accum):
            batch = []
            for _ in range(cfg.batch_size):
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                batch.append(order[cursor])
                cursor += 1
            ids = [enc[i][0] for i in batch] + [enc[i][2] for i in batch]
            ns = [enc[i][1] for i in batch] + [enc[i][3] for i in batch]
            lp = seq_logprob(model, ids, ns, device, tok.pad_token_id)
            pc, pr = lp[:len(batch)], lp[len(batch):]
            rc = torch.tensor([ref[i][0] for i in batch], device=device)
            rr = torch.tensor([ref[i][1] for i in batch], device=device)
            logits = cfg.beta * ((pc - rc) - (pr - rr))
            loss = -torch.nn.functional.logsigmoid(logits).mean() / cfg.grad_accum
            loss.backward()
            tot["loss"] += loss.item()
            tot["acc"] += (logits > 0).float().mean().item() / cfg.grad_accum
            tot["margin"] += logits.mean().item() / cfg.beta / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % cfg.logging_steps == 0 or step == 0:
            log.log(step=step + 1, loss=round(tot["loss"], 5), reward_accuracy=round(tot["acc"], 4),
                    reward_margin=round(tot["margin"], 4), lr=sched.get_last_lr()[0], **system_stats())
            log.update(step=step + 1, last_loss=round(tot["loss"], 5))
            print(f"step {step + 1}/{steps} loss {tot['loss']:.4f} pref-acc {tot['acc']:.2f}")
        if log.stop_requested():
            status = "stopped"
            break

    out = Path(cfg.models_dir) / cfg.run_name / "adapter"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    card = write_card(out, cfg, {"run_id": log.run_id, "load_in_4bit": use_4bit, "dpo_beta": cfg.beta,
                                 "preference_pairs": len(enc), "init_adapter": cfg.init_adapter})
    Registry(cfg.models_dir).register({"id": cfg.run_name, "name": "ZENIT", "version": cfg.version, "stage": "dpo",
                                       "engine": "hf", "path": cfg.base_model, "adapter": str(out),
                                       "base_model": cfg.base_model, "run_id": log.run_id,
                                       "metrics": {"final_train_loss": log.state.get("last_loss")}})
    log.update(status=status, ended=True, export=str(out))
    print(f"[zenit] {status}. adapter saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
