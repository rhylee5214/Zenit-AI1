"""Batch samplers for packed token bins (pretraining) and SFT examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TokenBin:
    def __init__(self, path: str | Path, dtype: str = "uint16") -> None:
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, seq_len: int, rng: np.random.Generator):
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"dataset has {len(self.data)} tokens, need more than seq_len+1")
        starts = rng.integers(0, len(self.data) - seq_len - 1, size=batch_size)
        x = np.stack([self.data[s:s + seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1:s + 1 + seq_len] for s in starts]).astype(np.int64)
        return x, y


class SFTData:
    def __init__(self, path: str | Path, pad_id: int = 0) -> None:
        self.examples = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self.examples.append((rec["input_ids"], rec["labels"]))
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def num_tokens(self) -> int:
        return sum(len(x) for x, _ in self.examples)

    def batch(self, batch_size: int, seq_len: int, rng: np.random.Generator):
        picks = rng.integers(0, len(self.examples), size=batch_size)
        x = np.full((batch_size, seq_len), self.pad_id, dtype=np.int64)
        y = np.full((batch_size, seq_len), -1, dtype=np.int64)
        for row, i in enumerate(picks):
            ids, labels = self.examples[i]
            n = min(len(ids), seq_len)
            x[row, :n] = ids[:n]
            y[row, :n] = labels[:n]
        return x, y


def open_dataset(data_dir: str | Path, split: str, mode: str, pad_id: int = 0):
    d = Path(data_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if mode == "pretrain":
        return TokenBin(d / f"{split}.bin", meta["dtype"]), meta
    if mode == "sft":
        return SFTData(d / f"sft_{split}.jsonl", pad_id=pad_id), meta
    raise ValueError(f"unknown mode {mode}")
