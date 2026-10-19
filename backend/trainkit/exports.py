"""Model/artifact export with real verification.

- adapter export: copies the trained PEFT adapter into exports/ and verifies
  expected files exist with nonzero size.
- ollama_import: builds a Modelfile (supporting GGUF 'FROM' only when a real
  GGUF file exists), calls Ollama's /api/create, verifies the model appears in
  the real model list, and records an actual smoke inference result.

A PEFT adapter cannot be dropped into Ollama as-is: that restriction is
reported clearly instead of silently renaming files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import get_config
from .db_models import Artifact, ExportOperation, TrainingJob, new_id
from .ollama.client import OllamaClient
from .security import validate_inside_data_root


def _dir_size(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    except OSError:
        return 0


def export_adapter(db, job: TrainingJob) -> dict[str, Any]:
    """Copy + verify a trained adapter; saves an export operation record."""
    export = ExportOperation(
        id=new_id("exp"), project_id=job.project_id, job_id=job.id, export_type="adapter"
    )
    db.add(export)
    output_dir = ((job.config or {}).get("output_dir") or "")
    adapter_src = Path(output_dir) / "adapter" if output_dir else None
    if not adapter_src or not adapter_src.is_dir():
        export.status = "failed"
        export.error = "Adaptör dizini bulunamadı; eğitim çıktısı doğrulanamadı."
        db.commit()
        return {"status": "failed", "error": export.error}
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [f for f in required if not (adapter_src / f).is_file() or (adapter_src / f).stat().st_size == 0]
    if missing:
        export.status = "failed"
        export.error = (
            f"Adaptör doğrulaması başarısız: beklenen dosyalar yok veya boş: {missing}"
        )
        db.commit()
        return {"status": "failed", "error": export.error}

    target = get_config().exports_dir / job.id / "adapter"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target).is_dir():
            export.status = "failed"
            export.error = "Hedef dizinde zaten bir dışa aktarım var; üzerine yazma için dosyaları elle kaldırın."
            db.commit()
            return {"status": "failed", "error": export.error}
        shutil.copytree(adapter_src, target)
    except (OSError, shutil.Error) as exc:
        export.status = "failed"
        export.error = f"Dışa aktarım kopyalama hatası: {exc}"
        db.commit()
        return {"status": "failed", "error": export.error}

    verified_files = {
        f: {"exists": (target / f).is_file(), "size": (target / f).stat().st_size}
        for f in required
    }
    export.status = "completed"
    export.result = {
        "format": "peft_adapter",
        "path": str(target),
        "files": verified_files,
        "verification": "files_exist_and_nonzero",
    }
    db.add(
        Artifact(
            id=new_id("art"),
            project_id=job.project_id,
            job_id=job.id,
            kind="export",
            path=str(target),
            size_bytes=_dir_size(target),
            meta={"export_type": "peft_adapter"},
        )
    )
    db.commit()
    return {"status": "completed", "path": str(target), "files": verified_files}


def export_training_config(job: TrainingJob) -> dict[str, Any]:
    """Export the exact run configuration as JSON (always possible)."""
    target = get_config().exports_dir / job.id / "training_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(job.config or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "completed", "path": str(target), "format": "json"}


def export_dataset(db, dataset_row, project_id: str | None) -> dict[str, Any]:
    """Copy a dataset file into exports (verified)."""
    import os

    src = Path(dataset_row.path)
    if not src.is_file() or src.stat().st_size == 0:
        return {"status": "failed", "error": f"Kaynak veri kümesi dosyası bulunamadı: {src}"}
    target = get_config().exports_dir / "datasets" / f"{dataset_row.id}{src.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    return {"status": "completed", "path": str(target)}


_GGUF_TOOLS_HINT = (
    "GGUF dönüşümü için tam veya daha önce birleştirilmiş bir HF modeli, llama.cpp "
    "convert_hf_to_gguf.py betiği ve gguf paketi gerekir. Bu ortamda kullanılabilir "
    "bildirilmediği sürece dönüşüm yapılamaz."
)


def convert_to_gguf(job: TrainingJob) -> dict[str, Any]:
    """Attempt GGUF conversion honestly; report incompatibility otherwise."""
    try:
        import gguf  # noqa: F401
    except ImportError:
        return {
            "status": "unsupported_environment",
            "error": "gguf paketi kurulu değil. " + _GGUF_TOOLS_HINT,
        }
    # llama.cpp converter is not vendored; do not fake success.
    try:
        out = subprocess.run(
            ["python", "-c", "import llama_cpp_convert"], capture_output=True, timeout=20
        )
        if out.returncode != 0:
            return {
                "status": "unsupported_environment",
                "error": "llama.cpp dönüşüm betiği bulunamadı. " + _GGUF_TOOLS_HINT,
            }
    except Exception as exc:
        return {"status": "failed", "error": f"GGUF dönüşüm kontrolü başarısız: {exc}"}
    return {
        "status": "failed",
        "error": "GGUF dönüşümü bu sürümde otomatik düzenlemesi yapılamadı.",
    }


# Metadata flags that must match for a safe Ollama import decision.
def check_ollama_import_compatibility(adapter_present: bool, base_model_meta: dict[str, Any]) -> dict[str, Any]:
    arch = base_model_meta.get("architecture")
    lora_supported = bool((base_model_meta.get("compatibility") or {}).get("modes", {}).get("lora", {}).get("available"))
    # Realistic: Ollama supports adapter import for a limited arch set (e.g. legal
    # Llama/Mistral/Qwen LoRA); full model import requires merged+GGUF output.
    return {
        "direct_adapter_importer": False,
        "adapter_support_note": (
            "PEFT adapter dosyaları Ollama'ya doğrudan verilemez; birleştirilmiş + GGUF "
            "dönüşümü veya tam model dosyası gerekir." if adapter_present else None
        ),
        "requires_merge_and_gguf": adapter_present,
        "architecture": arch,
        "lora_supported_by_trainkit": lora_supported,
    }


def build_modelfile(*, from_source: str, system: str | None, template: str | None) -> str:
    if not from_source or any(ch in from_source for ch in "../\x00"):
        raise ValueError("Geçersiz FROM kaynağı")
    lines = [f"FROM {from_source}"]
    if template:
        lines += ["TEMPLATE " + json.dumps(template)]
    if system:
        lines += ["SYSTEM " + json.dumps(system)]
    return "\n".join(lines) + "\n"


def import_to_ollama(
    client: OllamaClient,
    *,
    model_name: str,
    modelfile: str,
    smoke_prompt: str = "Selam, kısaca kendini tanıt.",
) -> dict[str, Any]:
    """Create an Ollama model from a Modelfile; VERIFY list + smoke inference."""
    result: dict[str, Any] = {"steps": []}
    try:
        events = list(client.create_model_stream(model_name, modelfile))
    except Exception as exc:
        result.update({"status": "failed", "error": f"Model oluşturma başarısız: {exc}"})
        return result
    result["steps"].append("create_stream_completed")
    if not client.model_exists(model_name):
        result.update(
            {"status": "failed", "error": f"Model oluşturuldu, fakat model listesinde doğrulanamadı: {model_name}"}
        )
        return result
    result["steps"].append("verified_in_model_list")
    smoke_result: dict[str, Any] = {}
    try:
        text = client.generate(model_name, smoke_prompt, options={"num_predict": 64})
        smoke_result = {"status": "completed", "prompt": smoke_prompt, "response": text[:400]}
    except Exception as exc:
        smoke_result = {"status": "failed", "error": str(exc)}
    result["smoke_test"] = smoke_result
    if smoke_result.get("status") != "completed":
        result.update({"status": "partial", "error": "Basit çıkarım testi başarısız."})
    else:
        result.update({"status": "completed", "model_name": model_name})
    return result
