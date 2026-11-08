"""Compares evaluation results across checkpoints.

    python -m zenit.evals.compare                      # table of every result, grouped by suite
    python -m zenit.evals.compare --diff A.json B.json # items that changed between two runs

Only results with the same suite hash appear in one table: they used
byte-identical test sets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COLS = ["general_knowledge", "math", "coding", "reasoning", "instruction_following", "conversation",
        "hallucination_rate", "first_token_ms", "tokens_per_sec"]


def fmt(k, v):
    if v is None or v == "skipped":
        return "skipped" if v == "skipped" else "n/a"
    if k in ("first_token_ms", "tokens_per_sec"):
        return f"{v:g}"
    return f"{100 * v:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="evals/results")
    ap.add_argument("--diff", nargs=2)
    a = ap.parse_args()
    if a.diff:
        x, y = (json.loads(Path(p).read_text(encoding="utf-8")) for p in a.diff)
        if x["suite_hash"] != y["suite_hash"]:
            print("warning: different suite hashes, items may not line up")
        before = {(i["category"], i["id"]): i["passed"] for i in x["items"]}
        for i in y["items"]:
            k = (i["category"], i["id"])
            if k in before and before[k] != i["passed"]:
                print(f"{i['category']:22s} {i['id']:18s} {before[k]} -> {i['passed']}")
        return
    groups = defaultdict(list)
    for f in sorted(Path(a.dir).glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        groups[r["suite_hash"]].append(r)
    for h, rows in groups.items():
        print(f"\nsuite {h}\n")
        print("| model | date | " + " | ".join(COLS) + " |")
        print("|" + "---|" * (len(COLS) + 2))
        for r in rows:
            s = r["summary"]
            print(f"| {r['model']} | {r['created'][:10]} | " + " | ".join(fmt(c, s.get(c)) for c in COLS) + " |")


if __name__ == "__main__":
    main()
