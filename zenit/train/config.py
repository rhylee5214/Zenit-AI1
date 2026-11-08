"""Training configuration. Load from JSON, override with --set key=value."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..model.config import PRESETS, ZenitConfig


@dataclass
class TrainConfig:
    run_name: str = "zenit-0.1-nano"
    version: str = "0.1-nano"
    stage: str = "pretrain"              # pretrain | sft  (label for the registry)
    backend: str = "numpy"               # numpy (CPU) | torch (GPU or CPU)
    model: dict = field(default_factory=lambda: {"preset": "nano"})
    tokenizer: str = "models/tokenizers/zenit-bpe-1k.json"
    data_dir: str = "data/processed/zenit-seed"
    mode: str = "pretrain"               # pretrain = all tokens, sft = assistant tokens only
    init_from: str = ""                  # model dir to continue from (e.g. SFT after pretrain)
    batch_size: int = 16
    seq_len: int = 128
    grad_accum: int = 1
    max_steps: int = 1500
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 8
    log_interval: int = 10
    ckpt_interval: int = 250
    keep_checkpoints: int = 3
    seed: int = 1337
    device: str = "cuda"                 # torch only
    precision: str = "bf16"              # torch only: bf16 | fp16 | fp32
    compile: bool = False                # torch only
    grad_checkpointing: bool = False     # torch only
    runs_dir: str = "runs"
    models_dir: str = "models"
    register: bool = True
    notes: str = ""

    def model_config(self, vocab_size: int | None = None) -> ZenitConfig:
        spec = dict(self.model)
        name = spec.pop("preset", None)
        base = dict(PRESETS[name]) if name else {}
        base.update(spec)
        if vocab_size is not None:
            base["vocab_size"] = vocab_size
        base.setdefault("max_seq_len", self.seq_len)
        base["max_seq_len"] = max(base["max_seq_len"], self.seq_len)
        base["version"] = self.version
        return ZenitConfig(**base)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        for item in overrides or []:
            key, _, raw = item.partition("=")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            target = data
            parts = key.split(".")
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = value
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"unknown training options: {sorted(unknown)}")
        return cls(**known)
