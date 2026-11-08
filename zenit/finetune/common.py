"""Shared pieces for fine-tuning open-weight foundation models into ZENIT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Used only when a base tokenizer ships without a chat template.
FALLBACK_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '### ' + m['role'] | capitalize }}\n{{ m['content'] }}\n\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Assistant\n{% endif %}"
)


@dataclass
class FinetuneConfig:
    run_name: str = "zenit-2.0-sft"
    version: str = "2.0"
    stage: str = "sft"
    base_model: str = "Qwen/Qwen3-4B-Base"
    base_license: str = "Apache-2.0"
    manifest: str = "data/manifests/zenit-2.0-sft.json"
    persona_data: str = "data/sft/zenit_persona.jsonl"
    persona_weight: int = 3                 # repeat persona examples so identity sticks
    system_prompt_rate: float = 0.5         # share of examples that get the ZENIT system prompt
    init_adapter: str = ""                  # continue from an existing ZENIT adapter
    max_seq_len: int = 2048
    max_examples: int = 0                   # 0 = all
    eval_fraction: float = 0.02
    load_in_4bit: bool | str = "auto"       # QLoRA when VRAM is tight
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str | list = "all-linear"
    lr: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    seed: int = 1337
    runs_dir: str = "runs"
    models_dir: str = "models"
    # DPO only
    pref_manifest: str = ""
    pref_data: str = "data/pref/zenit_honesty.jsonl"
    beta: float = 0.1

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "FinetuneConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in overrides or []:
            k, _, raw = item.partition("=")
            try:
                data[k] = json.loads(raw)
            except json.JSONDecodeError:
                data[k] = raw
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown options: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def gpu_memory_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def decide_4bit(cfg: FinetuneConfig, n_params_b: float) -> bool:
    """bf16 LoRA needs about 2 bytes per base parameter plus activations;
    4-bit QLoRA needs about 0.6. Pick QLoRA when bf16 would not fit."""
    if cfg.load_in_4bit != "auto":
        return bool(cfg.load_in_4bit)
    vram = gpu_memory_gb()
    return vram == 0 or n_params_b * 2.0 + 4 > vram


def render_persona(path: str | Path, base_model: str, version: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            for m in rec["messages"]:
                m["content"] = m["content"].replace("{base_model}", base_model).replace("{version}", version)
            out.append(rec)
    return out


def tokenize_chat(tok, messages: list[dict], max_len: int, template_kwargs: dict) -> dict | None:
    """Assistant-only labels. Renders the conversation prefix by prefix, so
    whatever the base model's chat template adds, only assistant text (and
    its end marker) gets supervised."""
    ids: list[int] = []
    labels: list[int] = []
    done_text = ""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True, **template_kwargs)
        full = tok.apply_chat_template(messages[:i + 1], tokenize=False, add_generation_prompt=False, **template_kwargs)
        if not full.startswith(prefix) or not prefix.startswith(done_text):
            return None                                  # template not prefix-stable, skip example
        ctx = prefix[len(done_text):]
        answer = full[len(prefix):]
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        ids += c_ids + a_ids
        labels += [-100] * len(c_ids) + a_ids
        done_text = full
    if not ids or all(x == -100 for x in labels):
        return None
    if tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
        labels.append(tok.eos_token_id)
    ids, labels = ids[:max_len], labels[:max_len]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def write_card(out_dir: Path, cfg: FinetuneConfig, extra: dict) -> dict:
    card = {
        "id": cfg.run_name, "name": "ZENIT", "version": cfg.version, "stage": cfg.stage,
        "base_model": cfg.base_model, "base_license": cfg.base_license,
        "trained_from_scratch": False,
        "method": "QLoRA" if extra.get("load_in_4bit") else "LoRA",
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
                 "target_modules": cfg.target_modules},
        "independence": ("Runs entirely on local hardware with weights stored on disk. "
                         "No third-party AI API is involved in training or generation. "
                         "The base weights remain under the base model's license."),
        **extra,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (out_dir / "NOTICE.txt").write_text(
        f"ZENIT {cfg.version} is a fine-tuned derivative of {cfg.base_model}.\n"
        f"The base model is licensed under {cfg.base_license}. Keep this notice and the base\n"
        f"model's LICENSE file with any copy of these weights you distribute.\n", encoding="utf-8")
    return card
