"""Real hardware inspection.

Reports only measured or explicitly labeled values. Nothing is inferred from
package presence alone: GPU detection uses a real torch CUDA query (when torch
is installed) and never fabricates GPU utilization or memory readings.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from typing import Any

import psutil


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inspect_hardware() -> dict[str, Any]:
    from .config import get_config

    cfg = get_config()
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": psutil.cpu_count(logical=True) or None,
            "physical_cores": psutil.cpu_count(logical=False) or None,
            "freq_mhz": (psutil.cpu_freq().current if psutil.cpu_freq() else None),
        },
        "ram": {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
        },
        "disk": _disk_info(cfg.root),
        "gpu": detect_gpus(),
        "accelerators": _accelerator_backends(),
        "dependencies": inspect_dependencies(),
    }
    sw = info["dependencies"]
    info["training_ready"] = bool(sw.get("torch") and sw.get("transformers") and sw.get("peft"))
    info["datasets_lib"] = bool(sw.get("datasets"))
    return info


def _cpu_model() -> str:
    # Windows & Linux: platform.processor often returns ''; try platform-specific.
    if platform.system() == "Windows":
        return platform.processor() or "Unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.strip().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "Unknown"


def _disk_info(root) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(os.fspath(root) or ".")
        return {
            "target_path": os.fspath(root),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        }
    except OSError:
        return {"target_path": os.fspath(root), "total_bytes": None, "free_bytes": None}


def detect_gpus() -> dict[str, Any]:
    """GPU presence via real backend queries.

    Reports hardware *visible to the current process*, and separates that
    from 'supported by installed software'. GPU utilization is not reported
    (no fabricated numbers); process memory comes from psutil.
    """
    result: dict[str, Any] = {
        "cuda": {"available": False, "version": None, "devices": []},
        "mps": {"available": False},
        "instance_type": "cpu",
    }
    if not _module_available("torch"):
        result["source"] = "unavailable"
        result["note"] = (
            "PyTorch kurulu değil; GPU varlığı kontrol edilemedi. Eğitim motoru için "
            "torch kurulumu gereklidir."
        )
        return result
    try:
        import torch

        result["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            devices = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "capability": [props.major, props.minor],
                    }
                )
            result["cuda"] = {
                "available": True,
                "version": torch.version.cuda,
                "devices": devices,
            }
            result["instance_type"] = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            result["mps"] = {"available": True}
            result["instance_type"] = "mps"
        result["source"] = "torch"
    except Exception as exc:  # torch present but broken install: report truthfully
        result["source"] = "error"
        result["note"] = f"torch sorgusu başarısız: {exc}"
    return result


def _accelerator_backends() -> dict[str, bool]:
    return {
        "cuda": bool(detect_gpus()["cuda"]["available"]),
        "mps": bool(detect_gpus()["mps"]["available"]),
    }


def inspect_dependencies() -> dict[str, Any]:
    """Actual installed versions of the training/IR-relevant dependencies."""

    def ver(name: str):
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:
            return None

    deps = {
        "torch": ver("torch"),
        "transformers": ver("transformers"),
        "peft": ver("peft"),
        "accelerate": ver("accelerate"),
        "datasets": ver("datasets"),
        "bitsandbytes": ver("bitsandbytes"),
        "safetensors": ver("safetensors"),
        "gguf": ver("gguf"),
    }
    return deps


def process_resource_usage(pid: int | None = None) -> dict[str, Any]:
    """Real per-process metrics for the training workspace.

    GPU utilization is explicitly not reported: nvidia-smi sampling would be
    the only reliable source and is not universally available — the field is
    labeled unavailable instead of fabricated.
    """
    try:
        if pid:
            p = psutil.Process(pid)
        else:
            p = psutil.Process()
        return {
            "cpu_percent": p.cpu_percent(interval=0.1),
            "memory_rss_bytes": p.memory_info().rss,
            "gpu_utilization": None,  # not measured; labeled 'unavailable' in UI
            "gpu_memory_bytes": None,
        }
    except psutil.Error:
        return {
            "cpu_percent": None,
            "memory_rss_bytes": None,
            "gpu_utilization": None,
            "gpu_memory_bytes": None,
            "error": "process_unavailable",
        }
