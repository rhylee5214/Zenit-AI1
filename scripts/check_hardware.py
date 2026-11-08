"""Inspects this machine and recommends a ZENIT training plan.

    python scripts/check_hardware.py

Writes hardware_report.json next to this project. Run it on the PC you
plan to train on. Nothing here starts a training job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zenit.api.system_info import system_report  # noqa: E402


def best_gpu(rep: dict) -> tuple[str | None, float]:
    if rep.get("torch_gpus"):
        g = max(rep["torch_gpus"], key=lambda x: x["memory_gb"])
        return g["name"], g["memory_gb"]
    if rep.get("gpus"):
        g = max(rep["gpus"], key=lambda x: x["memory_total_gb"])
        return g["name"], g["memory_total_gb"]
    return None, 0.0


def plan(vram: float, ram: float) -> dict:
    if vram >= 22:
        return {"tier": "24 GB class", "finetune_base": "Qwen/Qwen3-8B-Base", "method": "LoRA bf16 (or QLoRA for 14B)",
                "seq_len": 4096, "batch": 2, "grad_accum": 8, "native_preset": "base (about 100M) or medium",
                "notes": "Comfortable for ZENIT 2.0 to 4.0 on an 8B base."}
    if vram >= 14:
        return {"tier": "16 GB class", "finetune_base": "Qwen/Qwen3-8B-Base", "method": "QLoRA 4-bit",
                "seq_len": 2048, "batch": 2, "grad_accum": 8, "native_preset": "base (about 100M)",
                "notes": "8B QLoRA fits. Use gradient checkpointing."}
    if vram >= 10:
        return {"tier": "12 GB class", "finetune_base": "Qwen/Qwen3-4B-Base", "method": "QLoRA 4-bit (bf16 LoRA is tight)",
                "seq_len": 2048, "batch": 1, "grad_accum": 16, "native_preset": "small (about 32M) or base with batch 8",
                "notes": "4B is the sweet spot. 8B QLoRA works with seq_len 1024."}
    if vram >= 6:
        return {"tier": "8 GB class", "finetune_base": "Qwen/Qwen3-4B-Base", "method": "QLoRA 4-bit",
                "seq_len": 1024, "batch": 1, "grad_accum": 16, "native_preset": "small (about 32M)",
                "notes": "Close other GPU apps (games, browsers with hardware acceleration) while training."}
    if vram > 0:
        return {"tier": "under 6 GB", "finetune_base": "Qwen/Qwen3-1.7B-Base", "method": "QLoRA 4-bit",
                "seq_len": 1024, "batch": 1, "grad_accum": 16, "native_preset": "nano or small with batch 4",
                "notes": "Small GPU. Consider renting a 24 GB cloud GPU for the fine-tuning stages."}
    return {"tier": "no CUDA GPU", "finetune_base": "none locally", "method": "rent a cloud GPU for fine-tuning",
            "seq_len": None, "batch": None, "grad_accum": None, "native_preset": "nano on CPU (NumPy backend)",
            "notes": "CPU works for the nano pipeline model only. Install the NVIDIA driver and CUDA PyTorch if you have an NVIDIA card."}


def main() -> int:
    rep = system_report()
    name, vram = best_gpu(rep)
    rec = plan(vram, rep.get("ram_total_gb", 0))
    rep["recommendation"] = rec
    print("ZENIT hardware check")
    print(f"  OS            {rep['os']} ({rep['machine']}), Python {rep['python']}")
    print(f"  CPU           {rep.get('cpu_cores_physical')} cores / {rep.get('cpu_cores_logical')} threads")
    print(f"  RAM           {rep.get('ram_total_gb')} GB total, {rep.get('ram_available_gb')} GB free")
    print(f"  Disk          {rep['disk_free_gb']} GB free of {rep['disk_total_gb']} GB")
    print(f"  GPU           {name or 'none detected'}{f' ({vram} GB)' if vram else ''}")
    print(f"  PyTorch       {rep.get('torch') or 'not installed'}"
          + (f", CUDA {rep.get('cuda_version')}, bf16={rep.get('bf16_supported')}" if rep.get("cuda_available") else ""))
    if rep.get("gpus") and not rep.get("cuda_available"):
        print("\n  An NVIDIA GPU is present but PyTorch cannot use it. Install the CUDA build of PyTorch:")
        print("  pick the pip command for your CUDA version at https://pytorch.org/get-started/locally/")
    print(f"\nRecommended plan ({rec['tier']}):")
    for k in ("finetune_base", "method", "seq_len", "batch", "grad_accum", "native_preset", "notes"):
        print(f"  {k:14s} {rec[k]}")
    if rep["disk_free_gb"] < 60:
        print("\n  Warning: under 60 GB free. A 4B base model needs about 8 GB, an 8B one about 16 GB,"
              " plus checkpoints and datasets.")
    out = ROOT / "hardware_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
