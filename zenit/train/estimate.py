"""Resource estimates before a run starts.

Throughput is measured on the actual machine with a short matmul benchmark,
then scaled by a utilisation factor. Estimates are rough (expect +-50%), but
they stop you from starting a multi-week job by accident.

    python -m zenit.train.estimate --config configs/zenit-1.0-small.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np

from .config import TrainConfig

UTILISATION = {"numpy": 0.30, "torch-cuda": 0.40, "torch-cpu": 0.30, "torch-mps": 0.30}


def measure_numpy_gflops(n: int = 384, reps: int = 6) -> float:
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b
    t = time.perf_counter()
    for _ in range(reps):
        a @ b
    dt = time.perf_counter() - t
    return 2 * n ** 3 * reps / dt / 1e9


def measure_torch_tflops(device: str, dtype_name: str = "bf16") -> float:
    import torch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    if device == "cpu":
        dtype = torch.float32
    n = 4096 if device.startswith("cuda") else 512
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    reps = 20 if device.startswith("cuda") else 5
    t = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return 2 * n ** 3 * reps / (time.perf_counter() - t) / 1e12


def estimate(tc: TrainConfig, vocab_size: int | None = None, measure: bool = True) -> dict:
    mc = tc.model_config(vocab_size)
    n_params = mc.num_parameters()
    n_embed = mc.vocab_size * mc.d_model
    tokens_per_step = tc.batch_size * tc.seq_len * tc.grad_accum
    total_tokens = tokens_per_step * tc.max_steps
    # 6N per token for forward+backward on weights, plus attention scores.
    flops_per_token = 6 * n_params + 12 * mc.n_layers * tc.seq_len * mc.d_model
    total_flops = flops_per_token * total_tokens

    bytes_per_act = 2 if (tc.backend == "torch" and tc.precision in ("bf16", "fp16")) else 4
    # Weights + grads + Adam moments (fp32 master copy).
    state_gb = n_params * 16 / 1e9
    # Activation memory per layer (Korthikanti et al. 2022), scaled for dtype.
    s, b, h, a = tc.seq_len, tc.batch_size, mc.d_model, mc.n_heads
    act_layer = (34 * s * b * h + 5 * a * s * s * b) * bytes_per_act / 2
    if tc.grad_checkpointing:
        act_layer = 2 * s * b * h * bytes_per_act
    act_gb = (act_layer * mc.n_layers + b * s * mc.vocab_size * 4 * 2) / 1e9
    ckpt_gb = n_params * (4 + 8) / 1e9   # weights + optimizer state per checkpoint

    out = {
        "parameters": n_params,
        "parameters_non_embedding": n_params - n_embed,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "total_pflops": round(total_flops / 1e15, 3),
        "memory_gb_estimate": round(state_gb + act_gb, 2),
        "memory_breakdown_gb": {"weights_grads_adam": round(state_gb, 3), "activations": round(act_gb, 3)},
        "disk_gb_per_checkpoint": round(ckpt_gb, 3),
        "disk_gb_total": round(ckpt_gb * (tc.keep_checkpoints + 1), 3),
        "chinchilla_optimal_tokens": 20 * n_params,
    }
    free_disk = shutil.disk_usage(".").free / 1e9
    out["disk_free_gb"] = round(free_disk, 1)

    if measure:
        key, rate = "numpy", None
        if tc.backend == "numpy":
            rate = measure_numpy_gflops() * 1e9
        else:
            try:
                import torch
                dev = tc.device if (tc.device != "cuda" or torch.cuda.is_available()) else "cpu"
                key = "torch-cuda" if dev.startswith("cuda") else ("torch-mps" if dev == "mps" else "torch-cpu")
                rate = measure_torch_tflops(dev, tc.precision) * 1e12
                out["device"] = dev
                if dev.startswith("cuda"):
                    props = torch.cuda.get_device_properties(0)
                    out["gpu"] = props.name
                    out["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            except ImportError:
                out["warning"] = "torch not installed; cannot measure GPU throughput"
        if rate:
            eff = rate * UTILISATION[key]
            secs = total_flops / eff
            out["measured_peak_tflops"] = round(rate / 1e12, 4)
            out["assumed_utilisation"] = UTILISATION[key]
            out["estimated_hours"] = round(secs / 3600, 3)
            out["estimated_tokens_per_sec"] = round(eff / flops_per_token)
    return out


def needs_confirmation(est: dict, max_hours: float = 2.0) -> list[str]:
    reasons = []
    if est.get("estimated_hours", 0) > max_hours:
        reasons.append(f"estimated {est['estimated_hours']:.1f} h of compute (limit without --yes: {max_hours} h)")
    gpu_mem = est.get("gpu_memory_gb")
    if gpu_mem and est["memory_gb_estimate"] > gpu_mem * 0.9:
        reasons.append(f"needs about {est['memory_gb_estimate']} GB but the GPU has {gpu_mem} GB")
    if est["disk_gb_total"] > est["disk_free_gb"] * 0.8:
        reasons.append(f"checkpoints need {est['disk_gb_total']} GB, only {est['disk_free_gb']} GB free")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    tc = TrainConfig.load(a.config, a.set)
    print(json.dumps(estimate(tc), indent=2))


if __name__ == "__main__":
    main()
