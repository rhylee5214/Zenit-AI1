"""Text embeddings for memory and document search.

Default: ZenitHashEmbedder. Dependency-free, deterministic, fast. It
captures word, word-pair and sub-word (character trigram) overlap through
feature hashing. Good lexical search, no deep semantic understanding.

Upgrade: set ZENIT_EMBEDDER=st:BAAI/bge-small-en-v1.5 (or any
sentence-transformers model) after `pip install sentence-transformers`.
Stored vectors record which embedder made them, and reindexing happens
automatically when the embedder changes.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from functools import lru_cache

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
STOPWORDS = set("""a an the and or but if of to in on at by for with from is are was were be been
being it its this that these those as do does did i you he she we they me my your our their what
which who whom how why when where not no so than then there here can will would should could""".split())


class ZenitHashEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"zenit-hash-{dim}-v1"

    def _features(self, text: str) -> Counter:
        words = [w for w in _WORD.findall(text.lower())]
        content = [w for w in words if w not in STOPWORDS]
        feats: Counter = Counter()
        for w in content:
            feats["w:" + w] += 1.0
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                feats["c:" + padded[i:i + 3]] += 0.35
        for a, b in zip(content, content[1:]):
            feats["b:" + a + "_" + b] += 0.6
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self._features(text).items():
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, idx] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
            n = np.linalg.norm(out[row])
            if n > 0:
                out[row] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


@lru_cache(maxsize=4)
def get_embedder(name: str | None = None):
    name = name or os.environ.get("ZENIT_EMBEDDER", "zenit-hash")
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    return ZenitHashEmbedder()


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
