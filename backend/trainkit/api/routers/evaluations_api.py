"""Evaluation, export, artifact, and model-catalog endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database import db_session_dep
from ...datasets.schemas import normalize_example
from ...datasets.io import load_records
from ...db_models import Artifact, Dataset, EvaluationRun, TrainingJob, ExportOperation
from ...evaluation import EvalRequest, persist_evaluation, run_evaluation
from ...exports import (
    convert_to_gguf,
    export_adapter,
    export_dataset,
    export_training_config,
    import_to_ollama,
    build_modelfile,
)
from ...models_catalog import ModelCatalogError, inspect_model
from ...ollama.client import OllamaClient
from ...security import PathValidationError

router = APIRouter(prefix="/api", tags=["evaluation"])


class EvalRequest_(BaseModel):
    job_id: str
    max_examples: int = 5
    max_new_tokens: int = 96
    temperature: float = 0.2
    compare_with_base: bool = True


@router.post("/evaluations", status_code=201)
def create_evaluation(payload: EvalRequest_, db: Session = Depends(db_session_dep)):
    job = db.get(TrainingJob, payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı.")
    if job.status != "completed":
        raise HTTPException(
            status_code=428,
            detail="Değerlendirme yalnızca COMPLETED eğitim işinde çalışır. Mevcut durum: " + job.status,
        )
    dataset = db.get(Dataset, job.dataset_id or "")
    validation_path = ((job.config or {}).get("validation_dataset_path")) or None
    source_path = validation_path or (dataset.path if dataset else None)
    if not source_path:
        raise HTTPException(status_code=400, detail="Değerlendirilecek örnek veri kümesi bulunamadı.")
    try:
        records = list(load_records(source_path, max_records=payload.max_examples))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Örnek veri okunamadı: {exc}")
    examples = []
    for rec in records:
        norm = normalize_example(rec, index=len(examples), source_id="eval")
        if norm is None:
            continue
        kind, example = norm
        if kind == "instruction":
            examples.append(example)
        # chat examples kept out of the inline path for v0.1
    if not examples:
        raise HTTPException(status_code=400, detail="Örnek veri kümesinde değerlendirilebilir instruction örneği yok.")
    report = run_evaluation(
        EvalRequest(
            job_id=job.id,
            examples=examples,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            compare_with_base=payload.compare_with_base,
        )
    )
    evaluation = persist_evaluation(db, job.id, job.project_id, report)
    return {"id": evaluation.id, "status": evaluation.status, "report": report}


@router.get("/evaluations/{evaluation_id}")
def read_evaluation(evaluation_id: str, db: Session = Depends(db_session_dep)):
    ev = db.get(EvaluationRun, evaluation_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Değerlendirme bulunamadı.")
    return {
        "id": ev.id,
        "job_id": ev.job_id,
        "project_id": ev.project_id,
        "status": ev.status,
        "error": ev.error,
        "results": ev.results,
        "created_at": ev.created_at.isoformat(),
    }


@router.get("/projects/{project_id}/evaluations")
def project_evaluations(project_id: str, db: Session = Depends(db_session_dep)):
    rows = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.project_id == project_id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(50)
        .all()
    )
    return {
        "items": [
            {"id": e.id, "job_id": e.job_id, "status": e.status, "created_at": e.created_at.isoformat()}
            for e in rows
        ]
    }


class ExportRequest(BaseModel):
    job_id: str
    export_type: str  # adapter | config | dataset | gguf | ollama_import
    ollama_model_name: str | None = None
    system_prompt: str | None = None
    template: str | None = None


@router.post("/exports", status_code=201)
def create_export(payload: ExportRequest, db: Session = Depends(db_session_dep)):
    job = db.get(TrainingJob, payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı.")
    if payload.export_type == "adapter":
        result = export_adapter(db, job)
        return result
    if payload.export_type == "config":
        return export_training_config(job)
    if payload.export_type == "dataset":
        dataset = db.get(Dataset, job.dataset_id or "")
        if not dataset:
            raise HTTPException(status_code=404, detail="İşin veri kümesi bulunamadı.")
        return export_dataset(db, dataset, job.project_id)
    if payload.export_type == "gguf":
        return convert_to_gguf(job)
    if payload.export_type == "ollama_import":
        client = OllamaClient.from_settings(db)
        if not payload.ollama_model_name:
            raise HTTPException(status_code=400, detail="Ollama model adı gerekli.")
        # Real sources only: a GGUF path in run output, or an exported GGUF file.
        output_dir = ((job.config or {}).get("output_dir") or "")
        candidates = [
            f"{output_dir}/adapter.gguf",
            f"{output_dir}/model.gguf",
            f"{output_dir}/merged.gguf",
        ]
        from pathlib import Path

        gguf_path = next((p for p in candidates if Path(p).is_file()), None)
        if not gguf_path:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Uygun GGUF dosyası bulunamadı. PEFT adapter'ı Ollama'ya doğrudan "
                    "verilemez; önce birleştirme + GGUF dönüşümü (llama.cpp convert) gerekir."
                ),
            )
        modelfile = build_modelfile(
            from_source=gguf_path, system=payload.system_prompt, template=payload.template
        )
        result = import_to_ollama(client, model_name=payload.ollama_model_name, modelfile=modelfile)
        export = ExportOperation(
            id=_new_export_id(),
            project_id=job.project_id,
            job_id=job.id,
            export_type="ollama_import",
            status=result.get("status"),
            result=result,
        )
        db.add(export)
        db.commit()
        return result
    raise HTTPException(status_code=400, detail=f"Desteklenmeyen dışa aktarım türü: {payload.export_type}")


def _new_export_id() -> str:
    from ...db_models import new_id

    return new_id("exp")


@router.get("/exports/{export_id}")
def read_export(export_id: str, db: Session = Depends(db_session_dep)):
    ex = db.get(ExportOperation, export_id)
    if not ex:
        raise HTTPException(status_code=404, detail="Dışa aktarım bulunamadı.")
    return {
        "id": ex.id,
        "type": ex.export_type,
        "status": ex.status,
        "result": ex.result,
        "created_at": ex.created_at.isoformat(),
    }


@router.get("/artifacts")
def list_artifacts(
    project_id: str | None = None, limit: int = 50, offset: int = 0, db: Session = Depends(db_session_dep)
):
    q = db.query(Artifact).order_by(Artifact.created_at.desc())
    if project_id:
        q = q.filter(Artifact.project_id == project_id)
    total = q.count()
    items = q.offset(offset).limit(limit).all()
    return {
        "items": [
            {
                "id": a.id,
                "kind": a.kind,
                "path": a.path,
                "size_bytes": a.size_bytes,
                "project_id": a.project_id,
                "job_id": a.job_id,
                "created_at": a.created_at.isoformat(),
                "meta": a.meta,
            }
            for a in items
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/artifacts/{artifact_id}")
def read_artifact(artifact_id: str, db: Session = Depends(db_session_dep)):
    a = db.get(Artifact, artifact_id)
    if not a:
        raise HTTPException(status_code=404, detail="Artefakt bulunamadı.")
    return {
        "id": a.id,
        "kind": a.kind,
        "path": a.path,
        "size_bytes": a.size_bytes,
        "project_id": a.project_id,
        "job_id": a.job_id,
        "created_at": a.created_at.isoformat(),
        "meta": a.meta,
    }


class InspectModelRequest(BaseModel):
    repo_id: str
    revision: str = "main"


router_models = APIRouter(prefix="/api/models", tags=["models"])


@router_models.post("/inspect")
def inspect(payload: InspectModelRequest):
    try:
        meta = inspect_model(payload.repo_id, payload.revision)
    except ModelCatalogError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model metadata alınamadı: {exc}")
    return meta
