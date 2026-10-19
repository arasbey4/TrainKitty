"""Resource preflight: measured hardware vs. estimated requirements.

Estimates are clearly labeled as estimates. The preflight blocks a run only on
definite blockers (missing deps, empty dataset); resource risks surface as
warnings with alternatives — the user can still explicitly approve a run.
"""

from __future__ import annotations

from typing import Any

from ..hardware import inspect_dependencies, inspect_hardware


def estimate_requirements(
    mode: str, model_metadata: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, Any]:
    """Rough, CLEARLY-LABELED estimates based on real safetensors size."""
    safetensors_bytes = (model_metadata or {}).get("safetensors_bytes") or None
    seq_len = config.get("sequence_length") or 512
    base_weight_bytes = (safetensors_bytes * 4) if safetensors_bytes else None
    disk_needed = int(safetensors_bytes * 2.5) if safetensors_bytes else None
    return {
        "estimate_basis": "safetensors_size",
        "model_disk_bytes": safetensors_bytes,
        "base_weight_bytes_fp32": base_weight_bytes,
        "peak_memory_label": "tahmin (kesin değil)",
        "disk_needed_bytes": disk_needed,
        "disk_label": "tahmin",
        "sequence_length": seq_len,
    }


def run_preflight(
    mode: str,
    config: dict[str, Any],
    model_metadata: dict[str, Any] | None,
    dataset_info: dict[str, Any] | None,
) -> dict[str, Any]:
    hw = inspect_hardware()
    deps = inspect_dependencies()
    est = estimate_requirements(mode, model_metadata, config)
    warnings: list[str] = []
    errors: list[str] = []

    missing = [k for k in ("torch", "transformers", "peft") if not deps.get(k)]
    if missing:
        errors.append(
            "Eğitim motoru bağımlılıkları eksik: " + ", ".join(missing) + ". Kurulum: "
            "pip install 'trainkit[training]' (torch için: pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu)"
        )
    elif mode == "qlora" and not deps.get("bitsandbytes"):
        errors.append("QLoRA için bitsandbytes kurulu değil; bu ortamda QLoRA kullanılamaz.")

    ram = hw.get("ram") or {}
    ram_available = ram.get("available_bytes") or 0
    peak = est.get("base_weight_bytes_fp32")
    if peak and ram_available < peak:
        warnings.append(
            f"Boş RAM ({ram_available / 1e9:.1f} GB), tahmini temel ağırlık bellek ihtiyacı "
            f"({peak / 1e9:.1f} GB) üzerinde olmayabilir. Alternatifler: daha küçük model, "
            "düşük sequence_length, gradyan checkpointing veya LoRA."
        )
    disk_free = (hw.get("disk") or {}).get("free_bytes")
    disk_needed = est.get("disk_needed_bytes")
    if disk_free is not None and disk_needed and disk_free < disk_needed:
        warnings.append(
            f"Hedef diskte tahminî alan ({disk_needed / 1e9:.1f} GB) yetersiz olabilir "
            f"(boş {disk_free / 1e9:.1f} GB)."
        )
    if mode == "full_finetune" and not (hw.get("gpu") or {}).get("cuda", {}).get("available"):
        warnings.append("GPU olmadan tam ince ayar pratikte muhtemelen imkânsız; LoRA önerilir.")
    if ((dataset_info or {}).get("train_count") or 0) == 0:
        errors.append("Veri kümesinde eğitim örneği yok (train_count = 0).")

    return {
        "mode": mode,
        "status": "blocked" if errors else ("warn" if warnings else "ok"),
        "errors": errors,
        "warnings": warnings,
        "hardware": hw,
        "dependencies": deps,
        "estimates": est,
    }
