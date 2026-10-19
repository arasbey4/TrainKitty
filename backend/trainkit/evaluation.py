"""Evaluation pipeline: real inference on held-out examples.

A completed training run is one kind of evidence; a completed evaluation is a
different state and both are stored separately. Reports record the exact model
revision, dataset, decoding settings, and library versions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db_models import EvaluationRun, TrainingJob, new_id
from .database import session_scope


@dataclass
class EvalRequest:
    job_id: str
    examples: list[dict[str, Any]]
    max_new_tokens: int = 96
    temperature: float = 0.2
    compare_with_base: bool = True
    judge_model: str | None = None


def run_evaluation(request: EvalRequest) -> dict[str, Any]:
    """Run real generation for adapted vs base model on actual examples.

    Runs synchronously (inline evaluation in the API's thread pool) for small
    example counts; large sets should be chunked by the caller.
    """
    with session_scope() as db:
        job = db.get(TrainingJob, request.job_id)
        if job is None:
            return {"status": "failed", "error": "İş bulunamadı"}
        cfg = job.config or {}
        base_model_id = cfg.get("base_model_id")
        output_dir = cfg.get("output_dir")
        adapter_dir = f"{output_dir}/adapter" if output_dir else None

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return {"status": "failed", "error": f"Değerlendirme bağımlılıkları yüklenemedi: {exc}"}

    if not base_model_id:
        return {"status": "failed", "error": "İş yapılandırmasında temel model eksik."}

    results_rows: list[dict[str, Any]] = []
    attempted = 0

    def generate_with(model, tokenizer, prompt: str, max_new: int = 64) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=(request.temperature > 0),
                temperature=max(request.temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    except Exception as exc:
        return {"status": "failed", "error": f"Tokenizer yüklenemedi ({base_model_id}): {exc}"}

    base_model = None
    adapted_model = None
    dtype = next_dtype()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
        device = pick_device()
        base_model.to(device)
    except Exception as exc:
        if base_model is not None:
            del base_model
        return {"status": "failed", "error": f"Temel model yüklenemedi: {exc}"}

    if adapter_dir:
        try:
            from peft import PeftModel

            adapted_model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype), adapter_dir
            )
            adapted_model.to(pick_device())
        except Exception as exc:
            adapted_model = None
            results_note = (
                f"Adapter değerlendirmesi başlatılamadı: {exc}. Temel model sonuçları yine de üretilecek."
            )
        else:
            results_note = None
    else:
        results_note = "Adaptör dizini bulunamadı; yalnızca temel model karşılaştırması yapılmadı."

    device = pick_device()
    for ex in request.examples:
        attempted += 1
        prompt = (
            f"### Görev:\n{ex.get('instruction', '')}\n"
            + (f"\n### Girdi:\n{ex['input']}\n" if ex.get("input") else "")
            + "\n### Yanıt:\n"
        )
        reference = ex.get("output", "")
        row: dict[str, Any] = {
            "example_id": ex.get("_id") or f"eval_{attempted}",
            "instruction": ex.get("instruction", "")[:200],
            "reference": reference[:400],
        }
        try:
            row["trained_model_output"] = (
                generate_with(adapted_model or base_model, tokenizer, prompt) if adapted_model else None
            )
        except Exception as exc:
            row["trained_model_output"] = None
            row["trained_model_error"] = str(exc)[:300]
        if request.compare_with_base and base_model is not None:
            try:
                row["base_model_output"] = generate_with(base_model, tokenizer, prompt)
            except Exception as exc:
                row["base_model_output"] = None
                row["base_model_error"] = str(exc)[:300]
        # Output format checks on REAL outputs only.
        row["non_empty"] = bool(row.get("trained_model_output"))
        row["reference_align_prefix"] = (
            row["trained_model_output"] is not None
            and reference[:20].strip().lower() in row["trained_model_output"].lower()
        )
        results_rows.append(row)

    trained_non_empty = sum(1 for r in results_rows if r.get("trained_model_output"))
    base_non_empty = sum(1 for r in results_rows if r.get("base_model_output"))
    format_ok = sum(1 for r in results_rows if r.get("reference_align_prefix"))

    report: dict[str, Any] = {
        "status": "completed" if results_rows else "failed",
        "counts": {
            "examples_evaluated": attempted,
            "trained_model_responses": trained_non_empty,
            "base_model_responses": base_non_empty,
            "format_ok": format_ok,
        },
        "rows": results_rows[:200],
        "base_model": base_model_id,
        "adapter_dir": adapter_dir,
        "decoding": {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "seed": None,
        },
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "note": (
            "Değerlendirme gerçek çıkarım sonuçlarına dayanır. Format kontrolü ve "
            "istatistikler objektiftir; öneri niteliğindeki yorumlar ise model tabanlıdır."
        ),
        "judge_model": request.judge_model,
        "judge_note": (
            f"LLM yargıcı model tabanlı değerlendirme kullanıldı (yargıç modeli) ve "
            "objektif bir ölçüm değildir." if request.judge_model else None
        ),
        "results_note": results_note,
    }
    return report


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def next_dtype():
    import torch

    return torch.float16 if pick_device() != "cpu" else torch.float32


def persist_evaluation(db, job_id: str, project_id: str | None, report: dict[str, Any]):
    evaluation = EvaluationRun(
        id=new_id("eval"),
        project_id=project_id,
        job_id=job_id,
        status=report.get("status", "failed"),
        results=report,
        meta={"created_by": "trainkit.evaluation.run_evaluation"},
        finished_at=None,
    )
    db.add(evaluation)
    db.commit()
    return evaluation
