"""Builds a training dataset from a manifest.

    python -m zenit.data.build --manifest data/manifests/seed.json \
        --out data/processed/seed --tokenizer models/tokenizers/zenit-1k.json \
        --train-tokenizer --vocab-size 1024

Pipeline: load -> clean -> quality filter -> exact + near dedup ->
train/val split -> (train tokenizer) -> tokenize -> pack to .bin / .jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..tokenizer.bpe import ZenitTokenizer
from ..tokenizer.chat_template import encode_for_training, render
from .cleaning import CleanOptions, clean_text, quality_reason
from .dedup import ExactDeduper, MinHashDeduper
from .loading import iter_records
from .manifest import load_manifest


def _split(text: str, val_fraction: float, seed: int) -> str:
    h = int(hashlib.md5(f"{seed}:{text}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def build(manifest_path: str, out_dir: str, tokenizer_path: str,
          train_tokenizer: bool = False, vocab_size: int = 1024,
          val_fraction: float = 0.05, seed: int = 0, near_dedup: bool = True,
          scrub_pii: bool = True, root: str = ".") -> dict:
    t0 = time.time()
    root_p = Path(root)
    manifest = load_manifest(manifest_path, root=root_p)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opt = CleanOptions(scrub_pii=scrub_pii)
    rng = random.Random(seed)

    exact, near = ExactDeduper(), MinHashDeduper() if near_dedup else None
    stats: dict = {"sources": {}, "filtered": Counter(), "dup_exact": 0, "dup_near": 0}
    pretrain = {"train": [], "val": []}
    sft = {"train": [], "val": []}

    for src in manifest.sources:
        kept = seen = 0
        src_opt = CleanOptions(**{**opt.__dict__, **src.filters})
        for rec in iter_records(src, root_p):
            seen += 1
            if "messages" in rec:
                msgs = [{"role": m["role"], "content": clean_text(m["content"], src_opt)} for m in rec["messages"]]
                flat = render(msgs, add_generation_prompt=False)
                check_text = " ".join(m["content"] for m in msgs)
            elif "text" in rec:
                msgs = None
                flat = clean_text(rec["text"], src_opt)
                check_text = flat
            else:
                stats["filtered"]["unsupported_record"] += 1
                continue
            reason = quality_reason(check_text, src_opt)
            if reason:
                stats["filtered"][reason] += 1
                continue
            if exact.is_duplicate(flat):
                stats["dup_exact"] += 1
                continue
            if near is not None and msgs is None and near.is_duplicate(flat):
                stats["dup_near"] += 1
                continue
            split = _split(flat, val_fraction, seed)
            repeats = int(src.weight) + (1 if rng.random() < src.weight - int(src.weight) else 0)
            for _ in range(max(repeats, 0)):
                pretrain[split].append(flat)
                if msgs is not None:
                    sft[split].append(msgs)
            kept += 1
        stats["sources"][src.name] = {"seen": seen, "kept": kept, "license": src.license,
                                      "origin": src.origin}

    tok_path = Path(tokenizer_path)
    if train_tokenizer or not tok_path.exists():
        print(f"training tokenizer (vocab {vocab_size}) on {len(pretrain['train'])} docs ...")
        tok = ZenitTokenizer.train(pretrain["train"], vocab_size=vocab_size)
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(tok_path)
    else:
        tok = ZenitTokenizer.load(tok_path)
    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32

    token_counts = {}
    for split in ("train", "val"):
        docs = pretrain[split]
        rng.shuffle(docs)
        ids: list[int] = []
        for d in docs:
            ids.extend(tok.encode(d))
            ids.append(tok.eos_id)
        np.array(ids, dtype=dtype).tofile(out / f"{split}.bin")
        token_counts[split] = len(ids)
        with (out / f"sft_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for msgs in sft[split]:
                x, y = encode_for_training(tok, msgs)
                fh.write(json.dumps({"input_ids": x, "labels": y}) + "\n")

    meta = {
        "manifest": manifest.path,
        "manifest_fingerprint": manifest.fingerprint(),
        "tokenizer": str(tok_path),
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "tokens": token_counts,
        "documents": {k: len(v) for k, v in pretrain.items()},
        "sft_examples": {k: len(v) for k, v in sft.items()},
        "filtered": dict(stats["filtered"]),
        "duplicates_removed": {"exact": stats["dup_exact"], "near": stats["dup_near"]},
        "sources": stats["sources"],
        "build_seconds": round(time.time() - t0, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train-tokenizer", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-near-dedup", action="store_true")
    ap.add_argument("--keep-pii", action="store_true", help="disable email/phone/IP scrubbing")
    a = ap.parse_args()
    meta = build(a.manifest, a.out, a.tokenizer, a.train_tokenizer, a.vocab_size,
                 a.val_fraction, a.seed, not a.no_near_dedup, not a.keep_pii)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
