"""SQLAlchemy ORM models (metadata in DB; large files live on disk)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    # approved structured model specification (or latest draft)
    spec: Mapped[Any] = mapped_column(JSON, nullable=True)
    spec_approved: Mapped[bool] = mapped_column(default=False)
    base_model: Mapped[Any] = mapped_column(JSON, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    training_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    activity: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    datasets = relationship(
        "Dataset", back_populates="project", foreign_keys="Dataset.project_id"
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(32))  # ollama | huggingface | upload
    format: Mapped[str] = mapped_column(String(16))  # jsonl | json | csv | parquet
    path: Mapped[str] = mapped_column(Text)  # managed, absolute path on disk
    schema_type: Mapped[str | None] = mapped_column(String(32), nullable=True)  # instruction | chat | unknown
    example_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="raw")  # raw | validated | failed
    provenance: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="datasets", foreign_keys=[project_id])


class DatasetSource(Base):
    """Retained source file for a prepared dataset (history/logging)."""

    __tablename__ = "dataset_sources"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    path: Mapped[str] = mapped_column(Text)
    transformation: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    # "approve" prefix check reused by caller; job ids unique
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mode: Mapped[str] = mapped_column(String(32))  # lora | qlora | full | pretrain | configuration_only
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    config: Mapped[Any] = mapped_column(JSON, nullable=True)
    run_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_alive: Mapped[bool | None] = mapped_column(default=None, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved: Mapped[bool] = mapped_column(default=False)
    progress_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    trainer_output: Mapped[Any] = mapped_column(JSON, nullable=True)

    events = relationship(
        "JobEvent", back_populates="job", cascade="all, delete-orphan", order_by="JobEvent.seq"
    )
    metrics = relationship("TrainingMetric", back_populates="job", order_by="TrainingMetric.step")

    __table_args__ = (Index("ix_training_jobs_project_status", "project_id", "status"),)


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("training_jobs.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    level: Mapped[str] = mapped_column(String(16), default="info")
    operation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[Any] = mapped_column(JSON, nullable=True)

    job = relationship("TrainingJob", back_populates="events")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("training_jobs.id"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job = relationship("TrainingJob", back_populates="metrics")

    __table_args__ = (Index("ix_training_metrics_job_step", "job_id", "step"),)


class EvaluationRun(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    results: Mapped[Any] = mapped_column(JSON, nullable=True)
    meta: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ExportOperation(Base):
    __tablename__ = "export_operations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    export_type: Mapped[str] = mapped_column(String(32))  # adapter | merged | gguf | modelfile | ollama_import
    status: Mapped[str] = mapped_column(String(32), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(32))  # adapter | checkpoint | dataset | export | evaluation | modelfile | log
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (Index("ix_artifacts_project_kind", "project_id", "kind"),)


class AgentRun(Base):
    """A bounded agent interaction (workflow step) and its persisted state."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(64))  # e.g. spec_generation
    status: Mapped[str] = mapped_column(String(32), default="pending")
    input: Mapped[Any] = mapped_column(JSON, nullable=True)
    output: Mapped[Any] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def reconcile_orphan_jobs() -> None:
    """Mark jobs whose process vanished while state claimed it was running.

    On startup, any job in a running-ish state without a live process must be
    reconciled to ``failed`` (or unknown) instead of pretending to train.
    """
    from .session_utils import fallback_session  # local import to avoid cycles

    with fallback_session() as db:
        jobs = db.query(TrainingJob).filter(
            TrainingJob.status.in_(["queued", "preparing", "downloading", "validating", "training"])
        ).all()
        changed = False
        for job in jobs:
            import psutil

            alive = False
            if job.pid:
                try:
                    p = psutil.Process(job.pid)
                    alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    alive = False
            if not alive:
                job.status = "failed"
                job.error = job.error or (
                    "İşlem beklenmedik şekilde sonlandı (uygulama yeniden başlatıldı). "
                    "Süreç artık çalışmıyor; tamamlanmış kabul edilmedi."
                )
                job.finished_at = utcnow()
                changed = True
        if changed:
            db.commit()
