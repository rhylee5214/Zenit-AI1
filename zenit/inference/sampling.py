"""Token sampling: temperature, top-k, top-p (nucleus), repetition penalty."""

from __future__ import annotations

import numpy as np


def sample_next(logits: np.ndarray, recent: list[int], temperature: float, top_k: int,
                top_p: float, repetition_penalty: float, rng: np.random.Generator) -> int:
    logits = logits.astype(np.float64).copy()
    if repetition_penalty and repetition_penalty != 1.0 and recent:
        idx = np.unique(np.array(recent[-256:]))
        vals = logits[idx]
        logits[idx] = np.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 1e-5:
        return int(np.argmax(logits))
    logits /= temperature
    if top_k and 0 < top_k < logits.size:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    if 0 < top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cut]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()
    return int(rng.choice(probs.size, p=probs))
