"""Hardware report used by the dashboard and scripts/check_hardware.py."""

from __future__ import annotations

import platform
import shutil
import subprocess


def nvidia_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({"name": parts[0], "memory_total_gb": round(float(parts[1]) / 1024, 1),
                         "memory_used_gb": round(float(parts[2]) / 1024, 1), "utilization_pct": float(parts[3]),
                         "driver": parts[4], "temperature_c": float(parts[5])})
    return gpus


def system_report() -> dict:
    rep: dict = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
                 "machine": platform.machine()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        rep.update(cpu_cores_logical=psutil.cpu_count(), cpu_cores_physical=psutil.cpu_count(logical=False),
                   cpu_pct=psutil.cpu_percent(interval=0.2), ram_total_gb=round(vm.total / 1e9, 1),
                   ram_available_gb=round(vm.available / 1e9, 1))
    except ImportError:
        pass
    du = shutil.disk_usage(".")
    rep.update(disk_total_gb=round(du.total / 1e9, 1), disk_free_gb=round(du.free / 1e9, 1))
    rep["gpus"] = nvidia_smi()
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["cuda_version"] = torch.version.cuda
            rep["bf16_supported"] = torch.cuda.is_bf16_supported()
            rep["torch_gpus"] = [{"name": torch.cuda.get_device_name(i),
                                  "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                                 for i in range(torch.cuda.device_count())]
        rep["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        rep["torch"] = None
    return rep
