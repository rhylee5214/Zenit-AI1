"""ZENIT training entry point.

    python -m zenit.train --config configs/zenit-0.1-nano.json
    python -m zenit.train --config configs/zenit-1.0-small.json --set max_steps=20000 --yes
    python -m zenit.train --resume runs/<run_id>

The resource estimate prints first. Jobs over 2 hours, or that do not fit in
GPU memory or on disk, refuse to start without --yes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..tokenizer.bpe import ZenitTokenizer
from .config import TrainConfig
from .estimate import estimate, needs_confirmation


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set lr=1e-3")
    ap.add_argument("--resume", default="", help="run directory to resume")
    ap.add_argument("--yes", action="store_true", help="confirm large jobs")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--max-hours", type=float, default=2.0)
    a = ap.parse_args(argv)

    if a.resume:
        state = json.loads((Path(a.resume) / "run.json").read_text(encoding="utf-8"))
        tc = TrainConfig(**state["config"])
    else:
        if not a.config:
            ap.error("--config or --resume is required")
        tc = TrainConfig.load(a.config, a.set)

    vocab = ZenitTokenizer.load(tc.tokenizer).vocab_size if Path(tc.tokenizer).exists() else None
    est = estimate(tc, vocab)
    print("[zenit] resource estimate:")
    print(json.dumps(est, indent=2))
    if a.estimate_only:
        return 0
    reasons = needs_confirmation(est, a.max_hours)
    if reasons and not a.yes:
        print("[zenit] not starting. Re-run with --yes to confirm:\n  - " + "\n  - ".join(reasons))
        return 2

    if tc.backend == "numpy":
        from .trainer_np import train
    elif tc.backend == "torch":
        from .trainer_torch import train
    else:
        raise ValueError(f"unknown backend {tc.backend}")
    result = train(tc, resume_dir=a.resume or None)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
