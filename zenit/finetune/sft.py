"""Supervised fine-tuning (LoRA / QLoRA) of an open-weight base model into ZENIT.

    python -m zenit.finetune.sft --config configs/zenit-2.0-sft.json
    python -m zenit.finetune.sft --config configs/zenit-2.0-sft.json --set max_examples=500 --dry-run

The data manifest is enforced: every source needs a license and
permission: true. Persona data teaches ZENIT's identity and honesty rules.
Training logs go to runs/<run_id>/ so the dashboard shows progress, and a
STOP file in the run folder ends training cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path

from ..data.cleaning import CleanOptions, clean_text
from ..data.loading import iter_records
from ..data.manifest import load_manifest
from ..inference.hf_engine import HFEngine
from ..persona.identity import ZENIT_SYSTEM_PROMPT
from ..train.registry import Registry
from ..train.runlog import RunLogger, system_stats
from .common import FALLBACK_TEMPLATE, FinetuneConfig, decide_4bit, render_persona, tokenize_chat, write_card


def collect_examples(cfg: FinetuneConfig) -> tuple[list[dict], dict]:
    rng = random.Random(cfg.seed)
    opt = CleanOptions(scrub_pii=True, min_chars=2, min_alpha_ratio=0.0)
    seen: set[str] = set()
    examples: list[dict] = []
    stats: dict = {"sources": {}}
    if cfg.manifest and Path(cfg.manifest).exists():
        manifest = load_manifest(cfg.manifest)
        stats["manifest_fingerprint"] = manifest.fingerprint()
        for src in manifest.sources:
            n = 0
            for rec in iter_records(src, Path.cwd()):
                msgs = rec.get("messages")
                if not msgs:
                    continue
                msgs = [{"role": m["role"], "content": clean_text(m["content"], opt)} for m in msgs
                        if m.get("role") in ("system", "user", "assistant") and m.get("content")]
                if not any(m["role"] == "assistant" for m in msgs):
                    continue
                h = hashlib.sha1(json.dumps(msgs).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                examples.append({"messages": msgs, "source": src.name})
                n += 1
                if cfg.max_examples and n >= cfg.max_examples:
                    break
            stats["sources"][src.name] = {"examples": n, "license": src.license}
    persona = render_persona(cfg.persona_data, cfg.base_model, cfg.version)
    for _ in range(cfg.persona_weight):
        examples.extend({"messages": json.loads(json.dumps(p["messages"])), "source": "zenit-persona"} for p in persona)
    stats["sources"]["zenit-persona"] = {"examples": len(persona) * cfg.persona_weight, "license": "CC0-1.0"}
    for ex in examples:
        if ex["messages"][0]["role"] != "system" and rng.random() < cfg.system_prompt_rate:
            ex["messages"].insert(0, {"role": "system", "content": ZENIT_SYSTEM_PROMPT})
        # Same folding of memory/context roles as the HF inference engine, so
        # training and serving see identical prompt layouts.
        ex["messages"] = HFEngine._fold_roles(ex["messages"])
    rng.shuffle(examples)
    stats["examples"] = len(examples)
    return examples, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true", help="build and tokenize the data, print stats, stop")
    ap.add_argument("--yes", action="store_true", help="confirm long runs")
    a = ap.parse_args(argv)
    cfg = FinetuneConfig.load(a.config, a.set)

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainerCallback, TrainingArguments)

    print(f"[zenit] collecting data for {cfg.run_name} ...")
    examples, stats = collect_examples(cfg)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = []
    for ex in examples:
        r = tokenize_chat(tok, ex["messages"], cfg.max_seq_len, cfg.chat_template_kwargs)
        if r:
            rows.append(r)
    n_tokens = sum(len(r["input_ids"]) for r in rows)
    n_sup = sum(sum(1 for x in r["labels"] if x != -100) for r in rows)
    n_eval = max(1, int(len(rows) * cfg.eval_fraction))
    train_rows, eval_rows = rows[n_eval:], rows[:n_eval]
    stats.update(tokenized=len(rows), tokens=n_tokens, supervised_tokens=n_sup)
    print(json.dumps(stats, indent=2))
    if a.dry_run:
        return 0
    if not torch.cuda.is_available():
        print("[zenit] no CUDA GPU found. Fine-tuning a multi-billion parameter model on CPU is not practical.")
        return 2

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from ..train.estimate import measure_torch_tflops
    from huggingface_hub import model_info
    try:
        n_params_b = (model_info(cfg.base_model).safetensors.total or 0) / 1e9
    except Exception:
        n_params_b = 0.0
    use_4bit = decide_4bit(cfg, n_params_b or 8.0)
    steps_per_epoch = max(1, len(train_rows) // (cfg.batch_size * cfg.grad_accum))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else int(steps_per_epoch * cfg.epochs)
    tokens_total = n_tokens * (cfg.epochs if cfg.max_steps <= 0 else total_steps / steps_per_epoch)
    tflops = measure_torch_tflops("cuda", "bf16") * 0.35
    hours = (4 * (n_params_b or 8.0) * 1e9 * tokens_total) / (tflops * 1e12) / 3600 if tflops else None
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[zenit] base {cfg.base_model} ({n_params_b:.1f}B), {'QLoRA 4-bit' if use_4bit else 'LoRA bf16'}, "
          f"GPU {torch.cuda.get_device_name(0)} {vram:.0f} GB")
    print(f"[zenit] {len(train_rows)} train examples, {total_steps} optimizer steps, about {tokens_total/1e6:.1f}M tokens, "
          f"estimated {hours:.1f} h" if hours else "")
    if hours and hours > 2 and not a.yes:
        print("[zenit] over 2 hours. Re-run with --yes to start, or lower max_examples / epochs.")
        return 2

    log = RunLogger(cfg.runs_dir, cfg.run_name)
    log.update(status="running", backend="torch+peft", device=torch.cuda.get_device_name(0), version=cfg.version,
               config=cfg.to_dict(), max_steps=total_steps, parameters=None, hardware=system_stats(),
               dataset={"tokens_train": n_tokens, "tokens_val": sum(len(r["input_ids"]) for r in eval_rows),
                        "examples": len(train_rows), "manifest_fingerprint": stats.get("manifest_fingerprint")})

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True,
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    if cfg.init_adapter:
        model = PeftModel.from_pretrained(model, cfg.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                                 lora_dropout=cfg.lora_dropout, target_modules=cfg.target_modules))
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.update(parameters=trainable)

    class ZenitLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            rec = {"step": state.global_step}
            if "loss" in logs:
                rec.update(loss=round(logs["loss"], 5), lr=logs.get("learning_rate"), grad_norm=logs.get("grad_norm"))
            if "eval_loss" in logs:
                rec.update(val_loss=round(logs["eval_loss"], 5))
                best = log.state.get("best_val_loss")
                if best is None or logs["eval_loss"] < best:
                    log.update(best_val_loss=round(logs["eval_loss"], 5))
                log.update(last_val_loss=round(logs["eval_loss"], 5))
            if len(rec) > 1:
                log.log(**rec, **system_stats())
                log.update(step=state.global_step, last_loss=rec.get("loss", log.state.get("last_loss")))

        def on_step_end(self, args, state, control, **kw):
            if log.stop_requested():
                control.should_training_stop = True

    ta = dict(output_dir=str(log.dir / "hf"), per_device_train_batch_size=cfg.batch_size,
              per_device_eval_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
              learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=cfg.warmup_ratio,
              num_train_epochs=cfg.epochs, max_steps=cfg.max_steps, weight_decay=cfg.weight_decay,
              logging_steps=cfg.logging_steps, eval_steps=cfg.eval_steps, save_steps=cfg.save_steps,
              save_total_limit=2, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
              gradient_checkpointing=cfg.gradient_checkpointing, report_to="none", seed=cfg.seed,
              remove_unused_columns=False)
    params = inspect.signature(TrainingArguments.__init__).parameters
    ta["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    if "gradient_checkpointing_kwargs" in params:
        ta["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    trainer = Trainer(model=model, args=TrainingArguments(**ta), train_dataset=train_rows, eval_dataset=eval_rows,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[ZenitLog()])
    t0 = time.time()
    status = "completed"
    try:
        trainer.train()
        if log.stop_requested():
            status = "stopped"
    except KeyboardInterrupt:
        status = "stopped"
    except Exception as err:
        log.update(status="failed", error=repr(err))
        raise

    out = Path(cfg.models_dir) / cfg.run_name
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    card = write_card(adapter_dir, cfg, {
        "run_id": log.run_id, "load_in_4bit": use_4bit, "trainable_parameters": trainable,
        "training_hours": round((time.time() - t0) / 3600, 2), "dataset": stats,
        "best_val_loss": log.state.get("best_val_loss"), "final_train_loss": log.state.get("last_loss"),
    })
    Registry(cfg.models_dir).register({
        "id": cfg.run_name, "name": "ZENIT", "version": cfg.version, "stage": cfg.stage, "engine": "hf",
        "path": cfg.base_model, "adapter": str(adapter_dir), "base_model": cfg.base_model,
        "parameters": None, "run_id": log.run_id,
        "metrics": {"best_val_loss": card["best_val_loss"], "final_train_loss": card["final_train_loss"]},
    })
    log.update(status=status, ended=True, export=str(adapter_dir))
    print(f"[zenit] {status}. adapter saved to {adapter_dir}")
    print("[zenit] serve it by adding this to configs/serve.json -> models:")
    print(json.dumps({"id": cfg.run_name, "engine": "hf", "path": cfg.base_model, "adapter": str(adapter_dir),
                      "load_in_4bit": use_4bit, "chat_template_kwargs": cfg.chat_template_kwargs}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
