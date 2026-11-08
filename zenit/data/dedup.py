"""Exact and near-duplicate removal.

Exact: SHA-1 of normalised text.
Near: MinHash over word 5-gram shingles with LSH banding. Two documents
whose estimated Jaccard similarity passes the threshold count as duplicates.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
_MERSENNE = (1 << 61) - 1


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


class ExactDeduper:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha1(normalise(text).encode()).hexdigest()
        if h in self.seen:
            return True
        self.seen.add(h)
        return False


class MinHashDeduper:
    def __init__(self, num_perm: int = 64, bands: int = 16, shingle: int = 5,
                 seed: int = 7) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.bands, self.rows, self.k = bands, num_perm // bands, shingle
        self.buckets: list[dict[bytes, int]] = [dict() for _ in range(bands)]
        self.count = 0

    def _signature(self, text: str) -> np.ndarray | None:
        words = _WORD.findall(text.lower())
        if len(words) < self.k:
            return None
        shingles = {" ".join(words[i:i + self.k]) for i in range(len(words) - self.k + 1)}
        hashes = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") >> 3
             for s in shingles), dtype=np.uint64, count=len(shingles))
        # (a*h + b) mod p, computed in float-safe pieces via Python ints is slow,
        # so use uint64 wraparound arithmetic: good enough for hashing purposes.
        perm = (hashes[:, None] * self.a[None, :] + self.b[None, :])
        return perm.min(axis=0)

    def is_duplicate(self, text: str) -> bool:
        sig = self._signature(text)
        if sig is None:
            return False
        keys = [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]
        dup = any(k in self.buckets[i] for i, k in enumerate(keys))
        if not dup:
            for i, k in enumerate(keys):
                self.buckets[i][k] = self.count
            self.count += 1
        return dup
