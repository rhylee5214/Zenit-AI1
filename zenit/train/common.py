"""Shared trainer helpers: export, registry entry, loss plot."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .registry import Registry
from .runlog import RunLogger, read_metrics


def plot_losses(run_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = read_metrics(run_dir)
    tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    va = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    if not tr:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(*zip(*tr), label="train loss", color="#5B6CFF", linewidth=1.4)
    if va:
        ax.plot(*zip(*va), label="validation loss", color="#B36BFF", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(run_dir.name, fontsize=10)
    fig.tight_layout()
    out = run_dir / "loss.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def export_and_register(log: RunLogger, tc, src_dir: Path, model_meta: dict,
                        weights_format: str) -> dict:
    export = Path(tc.models_dir) / tc.run_name
    if export.exists():
        shutil.rmtree(export)
    shutil.copytree(src_dir, export, ignore=shutil.ignore_patterns("optimizer.*"))
    shutil.copy(tc.tokenizer, export / "tokenizer.json")
    data_meta = json.loads((Path(tc.data_dir) / "meta.json").read_text(encoding="utf-8"))
    card = {
        "id": tc.run_name,
        "name": "ZENIT",
        "version": tc.version,
        "stage": tc.stage,
        "architecture": "zenit-decoder (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings)",
        "weights_format": weights_format,
        "trained_from_scratch": not bool(tc.init_from),
        "init_from": tc.init_from or None,
        "run_id": log.run_id,
        "dataset": {"dir": tc.data_dir, "manifest_fingerprint": data_meta.get("manifest_fingerprint"),
                    "tokens": data_meta.get("tokens"), "sources": data_meta.get("sources")},
        **model_meta,
    }
    (export / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    entry = None
    if tc.register:
        entry = Registry(tc.models_dir).register({
            "id": tc.run_name, "name": "ZENIT", "version": tc.version, "stage": tc.stage,
            "engine": "numpy" if weights_format == "zenit-npz" else "torch",
            "path": str(export), "parameters": model_meta.get("parameters"),
            "run_id": log.run_id, "metrics": {k: model_meta.get(k) for k in
                                              ("final_train_loss", "best_val_loss", "steps")},
        })
    return {"export_dir": str(export), "registry": entry}
