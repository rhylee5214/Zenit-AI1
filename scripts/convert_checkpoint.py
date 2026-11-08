"""Converts native ZENIT checkpoints between the NumPy (.npz) and PyTorch
(safetensors / .pt) formats. Parameter names are identical in both.

    python scripts/convert_checkpoint.py --src models/zenit-0.1-nano --to torch --out models/zenit-0.1-nano-torch
    python scripts/convert_checkpoint.py --src models/zenit-1.0-small --to npz --out models/zenit-1.0-small-npz
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--to", choices=["torch", "npz"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.to == "torch":
        from zenit.model.torch_model import load_zenit, save_zenit
        save_zenit(load_zenit(src), out)
    else:
        import numpy as np
        from zenit.model.torch_model import load_state
        state = load_state(src)
        np.savez(out / "weights.npz", **{k: v.float().numpy() for k, v in state.items()})
        shutil.copy(src / "config.json", out / "config.json")
    for extra in ("tokenizer.json", "model_card.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
