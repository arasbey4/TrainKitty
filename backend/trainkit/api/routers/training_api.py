"""Training endpoints: plan, preflight, jobs, logs, metrics, SSE updates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database import db_session_dep, session_scope
from ...db_models import Artifact, Dataset, EvaluationRun, TrainingJob, new_id
from ...hardware import process_resource_usage
from ...models_catalog import ModelCatalogError, inspect_model
from ...ollama import OllamaError
from ...projects import get_project
from ...settings import get_setting
from ...training.preflight import run_preflight
from ...training.runner import (
    append_event,
    cancel_job,
    start_job,
)
from ...training.runner import bus as job_bus

router = APIRouter(prefix="/api/training", tags=["training"])


class PlanRequest(BaseModel):
    project_id: str | None = None
    base_model_id: str
    base_model_revision: str = "main"
    dataset_id: str
    dataset_revision: str | None = None
    mode: str = "lora"
    overrides: dict[str, Any] | None = None


@router.post("/plan")
def plan(payload: PlanRequest, db: Session = Depends(db_session_dep)):
    """Build a validated training plan from a real dataset + real hardware."""
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Veri kümesi bulunamadı.")
    if dataset.status != "prepared":
        # prepared datasets get created by /datasets/prepare with status 'validated'
        if dataset.status not in ("validated", "prepared"):
            raise HTTPException(
                status_code=428,
                detail="Veri kümesi önce hazırlanmalı (/api/datasets/prepare).",
            )
    hw_probe = _hardware_safe()
    validation_path: str | None = None
    if dataset.provenance and dataset.provenance.get("train_path"):
        validation_path = dataset.provenance.get("validation_path")
    if validation_path is None and dataset.provenance:
        validation_path = dataset.provenance.get("report", {}).get("validation_path")
    config = default_plan_for_dataset(
        base_model_id=payload.base_model_id,
        dataset_id=dataset.id,
        dataset_path=dataset.path,
        validation_dataset_path=validation_path,
        dataset_schema=dataset.schema_type,
        hardware_summary=hw_probe,
    )
    if payload.base_model_revision != "main":
        config.base_model_revision = payload.base_model_revision
    if payload.mode in ("configuration_only", "lora", "full_finetune"):
        config.mode = payload.mode  # type: ignore[assignment]
    if payload.overrides:
        allowed = config.model_dump()
        for key, value in payload.overrides.items():
            if key not in allowed:
                continue
            try:
                allowed[key] = value
            except Exception:
                continue
        config = TrainingConfig(**allowed)
    issues = validate_plan(config, dataset.schema_type)
    return {"config": config.model_dump(), "issues": [i.model_dump() for i in issues]}


def _hardware_safe():
    from ...hardware import inspect_hardware

    try:
        return inspect_hardware()
    except Exception:
        return {}


class PreflightRequest(BaseModel):
    mode: str = "lora"
    config: dict[str, Any]


@router.post("/preflight")
def preflight(payload: PreflightRequest, db: Session = Depends(db_session_dep)):
    """Real preflight: measured env vs. labeled estimates; honest blockers."""
    cfg = payload.config
    dataset = db.get(Dataset, cfg.get("dataset_id") or "")
    dataset_info = {"train_count": dataset.example_count if dataset else 0}
    model_meta = None
    if cfg.get("base_model_id"):
        try:
            model_meta = inspect_model(cfg["base_model_id"], cfg.get("base_model_revision") or "main")
        except ModelCatalogError:
            model_meta = None  # offline: estimates based on measured deps only
    return run_preflight(payload.mode, cfg, model_meta, dataset_info)


class JobRequest(BaseModel):
    project_id: str | None = None
    config: dict[str, Any]
    approved: bool = False


@router.post("/jobs", status_code=201)
def create_job(payload: JobRequest, db: Session = Depends(db_session_dep)):
    """Create a training job. Explicit approval required (428 otherwise)."""
    if payload.approved is not True:
        raise HTTPException(
            status_code=428,
            detail="Bu işlem kaynak kullanımı gerektirir; 'approved': true onayı gerekli.",
        )
    try:
        config = TrainingConfig(**payload.config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Geçersiz eğitim yapılandırması: {exc}")
    dataset = db.get(Dataset, config.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Veri kümesi bulunamadı.")
    output_dir = get_config().checkpoints_dir / f"{config.base_model_id.replace('/', '_')[:40] }_job"
    job = TrainingJob(
        id=new_id("job"),
        project_id=payload.project_id or dataset.project_id,
        dataset_id=dataset.id,
        mode=config.mode,
        status="queued",
        config={**config.model_dump(), "output_dir": str(output_dir)},
        approved=True,
    )
    db.add(job)
    db.commit()
    append_event(job.id, "info", "İş sıraya alındı.", operation="create")
    return _job_dict(job)


@router.get("/jobs")
def read_jobs(limit: int = 50, offset: int = 0, db: Session = Depends(db_session_dep)):
    q = db.query(TrainingJob).order_by(TrainingJob.created_at.desc())
    total = q.count()
    jobs = q.offset(offset).limit(limit).all()
    return {"items": [_job_dict(j) for j in jobs], "total": total, "limit": limit, "offset": offset}


@router.get("/jobs/{job_id}")
def read_job(job_id: str, db: Session = Depends(db_session_dep)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı.")
    out = _job_dict(job)
    out["process"] = process_resource_usage(job.pid)
    out["events"] = [
        {
            "ts": e.ts.isoformat(),
            "level": e.level,
            "message": e.message,
            "operation": e.operation,
            "data": e.data,
        }
        for e in job.events[-100:]
    ]
    return out


@router.post("/jobs/{job_id}/start")
def start(job_id: str, db: Session = Depends(db_session_dep)):
    """Start the real training subprocess."""
    if start_job(job_id) is False:
        raise HTTPException(status_code=409, detail="İş başlatılamadı (zaten bir durumda).")
    return {"started": True, "job_id": job_id}


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, db: Session = Depends(db_session_dep)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı.")
    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"İş zaten sonlandırılmış: {job.status}")
    cancel_job(job_id)
    return {"cancelling": True, "job_id": job_id}


@router.get("/jobs/{job_id}/logs")
def logs(job_id: str, tail: int = 200, db: Session = Depends(db_session_dep)):
    job = db.get(TrainingJob, job_id)
    if not job or not job.run_dir:
        raise HTTPException(status_code=404, detail="Günlük bulunamadı.")
    log_dir = Path(job.run_dir)
    lines: list[str] = []
    for name in ("stdout.log", "log.jsonl"):
        p = log_dir / name
        if p.is_file():
            with open(p, "rb") as fh:
                chunk = fh.read()[-200_000:]
            lines.extend(chunk.decode(errors="replace").splitlines())
    return PlainTextResponse("\n".join(lines[-tail:]))


@router.get("/jobs/{job_id}/metrics")
def metrics(job_id: str, db: Session = Depends(db_session_dep)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı.")
    return {
        "metrics": [
            {
                "step": m.step,
                "epoch": m.epoch,
                "loss": m.loss,
                "eval_loss": m.eval_loss,
                "learning_rate": m.learning_rate,
                "ts": m.ts.isoformat(),
            }
            for m in job.metrics
        ],
        "total_steps": job.total_steps,
    }


@router.get("/jobs/{job_id}/stream")
def stream(job_id: str, db: Session = Depends(db_session_dep)):
    """SSE stream of real job events (pushed by the runner)."""
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı.")
    last_event_ts: list[str] = [job.updated_at.isoformat()] if job.updated_at else [""]

    def event_stream():
        last_m = 0
        last_event = 0
        idle = 0
        while idle < 60 and job_id in _active_or_recent_ids():
            latest = job_bus.poll_latest(job_id)
            if latest is not None:
                latest["ts"] = time.time()
                yield f"data: {json.dumps(latest, ensure_ascii=False)}\n\n"
                idle = 0
            else:
                idle += 1
            # also surface DB-persisted progress
            with session_scope() as gdb:
                j = gdb.get(TrainingJob, job_id)
                if j is None:
                    break
                yield f"data: {json.dumps({'type': 'job', 'status': j.status, 'step': j.progress_step, 'total_steps': j.total_steps})}\n\n"
                if j.status in ("completed", "failed", "cancelled"):
                    break
            time.sleep(1.0)
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _active_or_recent_ids():
    from ...training.runner import _processes

    return _processes.keys()


def _job_dict(job: TrainingJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "project_id": job.project_id,
        "dataset_id": job.dataset_id,
        "mode": job.mode,
        "status": job.status,
        "config": job.config,
        "error": job.error,
        "exit_code": job.exit_code,
        "progress_step": job.progress_step,
        "total_steps": job.total_steps,
        "approved": job.approved,
        "run_dir": job.run_dir,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "created_at": job.created_at.isoformat(),
        "trainer_output": job.trainer_output,
    }
