"""Model configuration shared by the NumPy and PyTorch implementations.

Both implementations use the same parameter names and tensor layouts, so a
checkpoint trained by one loads into the other. The layout also matches the
Hugging Face Llama layout, which makes export to HF / GGUF possible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ZenitConfig:
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2          # grouped-query attention (GQA)
    ffn_hidden: int = 0          # 0 -> derived (SwiGLU 8/3 rule, rounded)
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    init_std: float = 0.02
    name: str = "zenit"
    version: str = "0.1-nano"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.ffn_hidden <= 0:
            hidden = int(8 * self.d_model / 3)
            multiple = 32 if self.d_model >= 256 else 8
            self.ffn_hidden = multiple * ((hidden + multiple - 1) // multiple)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_parameters(self) -> int:
        d, f, v = self.d_model, self.ffn_hidden, self.vocab_size
        kv = self.n_kv_heads * self.head_dim
        per_layer = d * d * 2 + d * kv * 2 + 3 * d * f + 2 * d
        total = v * d + self.n_layers * per_layer + d
        if not self.tie_embeddings:
            total += v * d
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "ZenitConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "ZenitConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Reference sizes. Parameter counts are computed, not guessed.
PRESETS: dict[str, dict] = {
    # Trains on a laptop CPU in minutes. Pipeline proof only.
    "nano": dict(vocab_size=1024, d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=128),
    # First real native run on a consumer GPU: 31,597,056 params with a 16k vocab.
    "small": dict(vocab_size=16384, d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    # 100,682,496 params with a 32k vocab. Needs a GPU with 12 GB+ for sensible batch sizes.
    "base": dict(vocab_size=32768, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=2048),
    # 299,418,624 params. Multi-day run on a single 24 GB GPU.
    "medium": dict(vocab_size=32768, d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, max_seq_len=2048),
}


def preset(name: str, **overrides) -> ZenitConfig:
    cfg = dict(PRESETS[name])
    cfg.update(overrides)
    return ZenitConfig(**cfg)
