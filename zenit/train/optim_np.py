"""AdamW, gradient clipping and the learning-rate schedule (NumPy)."""

from __future__ import annotations

import math

import numpy as np


def lr_at(step: int, max_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_grads(grads: dict, max_norm: float) -> float:
    total = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values()))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= scale
    return total


class AdamW:
    def __init__(self, params: dict, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1) -> None:
        self.params = params
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict, lr: float) -> None:
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            if p.ndim >= 2 and self.wd > 0:          # no decay on norm weights
                p -= lr * self.wd * p
            p -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)

    def state_dict(self) -> dict:
        out = {"t": np.array(self.t)}
        out.update({f"m::{k}": v for k, v in self.m.items()})
        out.update({f"v::{k}": v for k, v in self.v.items()})
        return out

    def load_state_dict(self, state) -> None:
        self.t = int(state["t"])
        for k in self.params:
            self.m[k] = state[f"m::{k}"].astype(self.params[k].dtype)
            self.v[k] = state[f"v::{k}"].astype(self.params[k].dtype)
