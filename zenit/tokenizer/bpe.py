"""ZENIT byte-level BPE tokenizer.

Pure Python, no dependencies. Byte-level means every possible input string
round-trips exactly: unknown characters fall back to raw UTF-8 bytes.

Token id layout:
    [0, n_special)              special tokens (<|user|>, <|assistant|>, ...)
    [n_special, n_special+256)  raw bytes
    [n_special+256, vocab_size) learned merges, in merge order

The training algorithm keeps incremental pair counts, so it scales to corpora
of tens of megabytes. For bigger corpora, train on a representative sample.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|context|>",
    "<|memory|>",
    "<|tool|>",
] + [f"<|reserved_{i}|>" for i in range(6)]

# Words with optional leading space, single digits (helps arithmetic),
# underscores, punctuation runs and whitespace. [^\W\d_] matches unicode letters.
# The final [\s\S] alternative guarantees every character lands in some
# piece, so encoding is lossless for any input.
PRETOKEN_PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d| ?_+| ?[^\s\w]+|\s+(?!\S)|\s+|[\s\S]""",
    re.UNICODE,
)


class ZenitTokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 specials: list[str] | None = None) -> None:
        self.specials = list(specials or SPECIAL_TOKENS)
        self.special_to_id = {tok: i for i, tok in enumerate(self.specials)}
        self.byte_offset = len(self.specials)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.merge_to_id = {pair: self.byte_offset + 256 + i for i, pair in enumerate(self.merges)}
        self.id_to_bytes: list[bytes] = [s.encode() for s in self.specials]
        self.id_to_bytes += [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.id_to_bytes.append(self.id_to_bytes[a] + self.id_to_bytes[b])
        self._cache: dict[str, list[int]] = {}
        escaped = sorted((re.escape(s) for s in self.specials), key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(escaped) + ")")

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    def special_id(self, token: str) -> int:
        return self.special_to_id[token]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def end_id(self) -> int:
        return self.special_to_id["<|end|>"]

    # --------------------------------------------------------------- training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int,
              min_frequency: int = 2, verbose: bool = False) -> "ZenitTokenizer":
        tok = cls()
        n_merges = vocab_size - tok.byte_offset - 256
        if n_merges <= 0:
            raise ValueError(f"vocab_size must exceed {tok.byte_offset + 256}")

        word_freq: Counter[str] = Counter()
        for text in texts:
            for part in tok._special_re.split(text):
                if part in tok.special_to_id or not part:
                    continue
                word_freq.update(PRETOKEN_PATTERN.findall(part))

        words: list[list[int]] = []
        freqs: list[int] = []
        for w, f in word_freq.items():
            words.append([tok.byte_offset + b for b in w.encode("utf-8")])
            freqs.append(f)

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                pair_where[pair].add(wi)

        merges: list[tuple[int, int]] = []
        next_id = tok.byte_offset + 256
        for step in range(n_merges):
            if not pair_counts:
                break
            best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < min_frequency:
                break
            merges.append(best)
            affected = list(pair_where.pop(best, ()))
            del pair_counts[best]
            for wi in affected:
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):
                    if pair == best:
                        continue
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    s = pair_where.get(pair)
                    if s is not None:
                        s.discard(wi)
                new_w: list[int] = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == best[0] and w[i + 1] == best[1]:
                        new_w.append(next_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w
                for pair in zip(new_w, new_w[1:]):
                    pair_counts[pair] += f
                    pair_where[pair].add(wi)
            next_id += 1
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges} count={count}")
        return cls(merges=merges)

    # --------------------------------------------------------------- encoding
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [self.byte_offset + b for b in word.encode("utf-8")]
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            pair = (ids[best_i], ids[best_i + 1])
            ids = ids[:best_i] + [self.merge_to_id[pair]] + ids[best_i + 2:]
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        parts = self._special_re.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_to_id:
                out.append(self.special_to_id[part])
                continue
            for word in PRETOKEN_PATTERN.findall(part):
                out.extend(self._encode_word(word))
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        chunks = []
        for i in ids:
            if skip_special and i < self.byte_offset:
                continue
            chunks.append(self.id_to_bytes[i])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        return self.id_to_bytes[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        data = {"type": "zenit-bpe", "version": 1, "specials": self.specials,
                "merges": [list(m) for m in self.merges]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ZenitTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "zenit-bpe":
            raise ValueError(f"{path} is not a ZENIT tokenizer file")
        return cls(merges=[tuple(m) for m in data["merges"]], specials=data["specials"])


class IncrementalDecoder:
    """Turns a stream of token ids into clean text deltas for streaming.

    Byte-level tokens can split a multi-byte UTF-8 character. This decoder
    holds incomplete bytes until the character is complete.
    """

    def __init__(self, tokenizer: ZenitTokenizer, skip_special: bool = True) -> None:
        self.tok = tokenizer
        self.skip_special = skip_special
        self.buffer = b""

    def push(self, token_id: int) -> str:
        if self.skip_special and token_id < self.tok.byte_offset:
            return ""
        self.buffer += self.tok.id_to_bytes[token_id]
        try:
            text = self.buffer.decode("utf-8")
        except UnicodeDecodeError as err:
            if err.start > 0:
                text = self.buffer[:err.start].decode("utf-8")
                self.buffer = self.buffer[err.start:]
                return text
            if len(self.buffer) > 8:          # invalid sequence, give up on it
                text = self.buffer.decode("utf-8", errors="replace")
                self.buffer = b""
                return text
            return ""
        self.buffer = b""
        return text

    def flush(self) -> str:
        text = self.buffer.decode("utf-8", errors="replace")
        self.buffer = b""
        return text


def iter_stream(tokenizer: ZenitTokenizer, ids: Iterable[int]) -> Iterator[str]:
    dec = IncrementalDecoder(tokenizer)
    for i in ids:
        piece = dec.push(i)
        if piece:
            yield piece
    tail = dec.flush()
    if tail:
        yield tail
