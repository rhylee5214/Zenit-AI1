"""Run directories, metrics logging and hardware sampling.

runs/<run_id>/
    run.json          status, config, dataset, progress (dashboard reads this)
    metrics.jsonl     one JSON object per log or eval event
    checkpoints/      step_000250/, best/, latest -> text file naming the dir
    STOP              create this file to stop the run cleanly
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def system_stats() -> dict:
    out: dict = {}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["ram_used_gb"] = round(vm.used / 1e9, 2)
        out["ram_total_gb"] = round(vm.total / 1e9, 2)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            out["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            try:
                out["gpu_util_pct"] = torch.cuda.utilization()
            except Exception:
                pass
    except Exception:
        pass
    return out


class RunLogger:
    def __init__(self, runs_dir: str | Path, run_name: str, resume_dir: str | Path | None = None) -> None:
        if resume_dir:
            self.dir = Path(resume_dir)
            self.state = json.loads((self.dir / "run.json").read_text(encoding="utf-8"))
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.dir = Path(runs_dir) / f"{run_name}-{stamp}"
            self.dir.mkdir(parents=True, exist_ok=False)
            self.state = {"run_id": self.dir.name, "run_name": run_name, "status": "created",
                          "created": now_iso(), "step": 0, "best_val_loss": None}
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.t_start = time.time() - float(self.state.get("elapsed_s", 0.0))

    @property
    def run_id(self) -> str:
        return self.state["run_id"]

    def update(self, **fields) -> None:
        self.state.update(fields)
        self.state["updated"] = now_iso()
        self.state["elapsed_s"] = round(time.time() - self.t_start, 1)
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "run.json")

    def log(self, **record) -> None:
        record.setdefault("time", now_iso())
        record.setdefault("elapsed_s", round(time.time() - self.t_start, 1))
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    def checkpoint_dir(self, step: int) -> Path:
        return self.dir / "checkpoints" / f"step_{step:07d}"

    def mark_latest(self, path: Path) -> None:
        (self.dir / "checkpoints" / "latest").write_text(path.name, encoding="utf-8")

    def latest_checkpoint(self) -> Path | None:
        f = self.dir / "checkpoints" / "latest"
        if f.exists():
            p = self.dir / "checkpoints" / f.read_text(encoding="utf-8").strip()
            if p.exists():
                return p
        return None

    def prune(self, keep: int) -> None:
        ckpts = sorted(p for p in (self.dir / "checkpoints").glob("step_*") if p.is_dir())
        for old in ckpts[:-keep] if keep > 0 else []:
            shutil.rmtree(old, ignore_errors=True)


def read_metrics(run_dir: str | Path, limit: int | None = None) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
