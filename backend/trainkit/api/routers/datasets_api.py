"""Datasets endpoints: generation via Ollama, HF import, preparation, preview."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database import db_session_dep
from ...datasets import hf_import as hf
from ...datasets import service as dssvc
from ...datasets.generate import GenerationOptions, generate_dataset
from ...datasets.service import (
    count_examples,
    delete_dataset,
    prepare,
    preview_records,
    register_generated,
    register_hf_import,
    register_uploaded_file,
)
from ...db_models import Dataset, DatasetSource
from ...ollama import OllamaError
from ...ollama.client import OllamaClient
from ...security import PathValidationError
from ...settings import get_setting

log = logging.getLogger("trainkit.api.datasets")
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

_generation_results: dict[str, Any] = {}
_generation_lock = threading.Lock()


@router.get("")
def read_datasets(limit: int = 50, offset: int = 0, db: Session = Depends(db_session_dep)):
    q = db.query(Dataset).order_by(Dataset.created_at.desc())
    total = q.count()
    items = q.offset(offset).limit(limit).all()
    return {
        "items": [_ds_dict(d) for d in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{dataset_id}")
def read_one(dataset_id: str, db: Session = Depends(db_session_dep)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Veri kümesi bulunamadı.")
    out = _ds_dict(dataset)
    out["provenance"] = dataset.provenance
    return out


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, limit: int = 20, offset: int = 0, db: Session = Depends(db_session_dep)):
    dataset = _get_fraction(db, dataset_id)
    limit = min(limit, 100)
    try:
        records = preview_records(dataset, limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Önizleme hatası: {exc}")
    return {"records": records, "limit": limit, "offset": offset}


class GenerateRequest(BaseModel):
    project_id: str | None = None
    name: str | None = None
    target_count: int = 20
    batch_size: int = 5
    temperature: float = 0.8
    schema_type: str = "instruction"
    seed: int | None = None
    async_run: bool = True


@router.post("/generate")
def generate(payload: GenerateRequest, db: Session = Depends(db_session_dep)):
    """Generate real examples with the local Ollama model (background + SSE-ish wait)."""
    model = get_setting(db, "ollama.orchestrator_model")
    if not model:
        raise HTTPException(
            status_code=409, detail="Orkestrasyon modeli seçilmedi; önce kurup seçin."
        )
    spec: dict[str, Any] = {}
    if payload.project_id:
        from ...projects import get_project

        project = get_project(db, payload.project_id)
        if not project:
            raise HTTPException(status_code=400, detail="Geçersiz proje.")
        if not project.spec_approved:
            raise HTTPException(
                status_code=428,
                detail="Veri üretimi için önce projeyi tanımlamalı ve spec'i onaymalısınız.",
            )
        spec = project.spec or {}
    client = OllamaClient.from_settings(db)
    options = GenerationOptions(
        target_count=payload.target_count,
        batch_size=min(payload.batch_size, 20),
        temperature=payload.temperature,
        schema_type=payload.schema_type if payload.schema_type in ("instruction", "chat") else "instruction",
        seed=payload.seed,
    )
    run_id = f"gen_{len(_generation_results) + 1}"
    progress: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "batches": [],
        "accepted_total": 0,
    }
    with _generation_lock:
        _generation_results[run_id] = progress

    def run() -> None:
        try:
            result = generate_dataset(
                client, model, spec, options,
                on_batch=lambda i, info: _on_batch(progress, i, info),
            )
            with db_session_dep_session() as gdb:
                dataset = register_generated(
                    gdb, result=result, model=model, options=options,
                    name=payload.name or f"uretilen-{run_id}", project_id=payload.project_id,
                )
            progress["dataset"] = _ds_dict(dataset)
            progress["result"] = result.to_dict()
            progress["status"] = "completed"
        except Exception as exc:
            log.exception("generation failed")
            progress["status"] = "failed"
            progress["error"] = str(exc)[:1000]

    if payload.async_run:
        threading.Thread(target=run, daemon=True).start()
        return {"run_id": run_id, "status": "running"}
    run()
    return progress


def _on_batch(progress: dict[str, Any], index: int, info: dict[str, Any]) -> None:
    progress["batches"].append({"batch": index, **info})
    if info.get("ok"):
        progress["accepted_total"] = info.get("accepted_total", progress["accepted_total"])


def db_session_dep_session():
    from ...database import session_scope

    return session_scope()


@router.get("/generation-status/{run_id}")
def generation_status(run_id: str):
    with _generation_lock:
        info = _generation_results.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Üretim çalışması bulunamadı.")
    return {k: v for k, v in info.items() if k not in ("batches",)} | {
        "recent_batches": info.get("batches", [])[-10:],
    }


class HFImportRequest(BaseModel):
    repo_id: str
    split: str = "train"
    config_name: str | None = None
    max_records: int = 1000
    name: str | None = None
    project_id: str | None = None
    column_mapping: dict[str, str | None] | None = None
    download_confirmed: bool | None = None


@router.post("/hf/discover")
def hf_discover(payload: HFImportRequest):
    try:
        meta = hf.discover_dataset(payload.repo_id)
    except hf.HFDatasetError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Metadata alınamadı: {exc}")
    meta["license_check"] = hf.license_check(meta)
    know_size = bool(meta.get("size_bytes_estimate"))
    meta["size_note"] = (
        f"Tahminî boyut: {int(meta['size_bytes_estimate']) / 1e9:.2f} GB"
        if know_size
        else "Boyut bilgisi bilinmiyor; beklenmedik indirmeleri onay vermeden yapma."
    )
    if know_size and (meta["size_bytes_estimate"] > 2 * (1024**3)) and payload.download_confirmed is not True:
        meta["size_note"] = (
            f"Boyut bilgisi bilinmiyor; beklenmedik indirmeleri onay vermeden yapma."
        )
        return {"requires_confirmation": True, **meta}
    return meta


@router.post("/hf/import")
def hf_import(payload: HFImportRequest, db: Session = Depends(db_session_dep)):
    try:
        meta = hf.discover_dataset(payload.repo_id)
        records, info = hf.load_dataset_records(
            payload.repo_id,
            split=payload.split,
            config_name=payload.config_name,
            max_records=min(payload.max_records, 100_000),
        )
    except hf.HFDatasetError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message)
    if payload.column_mapping:
        records = hf.map_records_to_schema(records, payload.column_mapping)
    lic = hf.license_check(meta)
    dataset = register_hf_import(
        db,
        records=records,
        info=info,
        name=payload.name or payload.repo_id.replace("/", "-")[:60],
        project_id=payload.project_id,
        license_meta={
            "license": meta.get("license"),
            "license_check": lic,
            "hf_metadata": meta,
        },
    )
    return {"dataset": _ds_dict(dataset), "info": info, "license": lic, "count": len(records)}


@router.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(db_session_dep)):
    import os as _os  # size check before buffering to disk

    size = file.size if getattr(file, "size", None) is not None else None
    try:
        contents = await file.read()
    finally:
        await file.close()
    if size is None:
        size = len(contents)
    if not contents:
        raise HTTPException(status_code=400, detail="Boş dosya.")
    try:
        dataset = register_uploaded_file(db, filename=file.filename or "upload", size_bytes=size, content=contents)
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _ds_dict(dataset)


class PrepareRequest(BaseModel):
    dataset_id: str
    validation_split: float = 0.1
    seed: int = 42
    dedupe: bool = True


@router.post("/prepare")
def prepare_endpoint(payload: PrepareRequest, db: Session = Depends(db_session_dep)):
    source = _get_fraction(db, payload.dataset_id)
    try:
        prepared_result = prepare(
            db, source,
            validation_split=payload.validation_split,
            seed=payload.seed,
            dedupe=payload.dedupe,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Hazırlama başarısız: {exc}")
    return {
        "prepared": _ds_dict(prepared_result[0]),
        "report": prepared_result[1],
    }


@router.get("/{dataset_id}/report")
def report(dataset_id: str, db: Session = Depends(db_session_dep)):
    dataset = _get_fraction(db, dataset_id)
    srcs = (
        db.query(DatasetSource)
        .filter(DatasetSource.dataset_id == dataset_id)
        .all()
    )
    return {
        "dataset": _ds_dict(dataset),
        "provenance": dataset.provenance,
        "sources": [
            {"path": s.path, "transformation": s.transformation, "created_at": s.created_at.isoformat()}
            for s in srcs
        ],
    }


@router.delete("/{dataset_id}")
def delete(dataset_id: str, confirm: str = "", db: Session = Depends(db_session_dep)):
    dataset = _get_fraction(db, dataset_id)
    if confirm != "true":
        raise HTTPException(status_code=428, detail="Bu işlem onay gerektirir (?confirm=true).")
    info = delete_dataset(db, dataset)
    return info


def _get_fraction(db: Session, dataset_id: str) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Veri kümesi bulunamadı.")
    return dataset


def _ds_dict(d: Dataset) -> dict[str, Any]:
    return {
        "id": d.id,
        "project_id": d.project_id,
        "name": d.name,
        "source": d.source,
        "format": d.format,
        "schema_type": d.schema_type,
        "example_count": d.example_count,
        "status": d.status,
        "created_at": d.created_at.isoformat(),
    }
