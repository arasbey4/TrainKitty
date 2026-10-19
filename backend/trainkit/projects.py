"""Project service: approved structured specification persistence."""

from __future__ import annotations

import pydantic
from sqlalchemy.orm import Session

from .db_models import Project, new_id, utcnow

PROJECT_STATUSES = ("draft", "spec_ready", "dataset_ready", "training", "trained", "failed")


class ProjectSpec(pydantic.BaseModel):
    """A structured model specification (mostly AI-drafted, user-approved)."""

    target_model_name: str | None = None
    purpose: str | None = None
    intended_users: str | None = None
    languages: list[str] = pydantic.Field(default_factory=list)
    personality: str | None = None
    system_instructions: str | None = None
    desired_behavior: list[str] = pydantic.Field(default_factory=list)
    unwanted_behavior: list[str] = pydantic.Field(default_factory=list)
    example_inputs: list[str] = pydantic.Field(default_factory=list)
    output_style: str | None = None
    dataset_requirements: str | None = None
    evaluation_criteria: list[str] = pydantic.Field(default_factory=list)
    base_model_preference: str | None = None
    training_method: str | None = None
    resource_constraints: str | None = None
    export_requirements: str | None = None


def create_project(db: Session, *, name: str, description: str = "") -> Project:
    project = Project(id=new_id("proj"), name=name, description=description, activity=[])
    db.add(project)
    db.commit()
    _log(project, "project_created", f"Proje oluşturuldu: {name}")
    return project


def get_project(db: Session, project_id: str) -> Project | None:
    return db.get(Project, project_id)


def list_projects(db: Session, *, limit: int = 100, offset: int = 0) -> tuple[list[Project], int]:
    q = db.query(Project).order_by(Project.updated_at.desc())
    total = q.count()
    return q.offset(offset).limit(limit).all(), total


def update_project(db: Session, project: Project, **fields) -> Project:
    special = {}
    for key in list(fields):
        if key == "spec" and isinstance(fields[key], dict):
            fields[key] = ProjectSpec(**{**((project.spec or {})), **fields[key]}).model_dump()
        if key == "spec" and isinstance(fields[key], ProjectSpec):
            fields[key] = fields[key].model_dump()
        if key in ("spec", "dataset_id"):
            special[key] = fields.pop(key)
    for key, value in fields.items():
        setattr(project, key, value)
    for key, value in special.items():
        setattr(project, key, value)
    if "spec" in special:
        project.spec = special["spec"]
    _log(project, "project_updated", "Proje güncellendi")
    db.commit()
    return project


def delete_project(db: Session, project: Project) -> None:
    db.delete(project)
    db.commit()


def _log(project: Project, kind: str, message: str) -> None:
    project.activity = (project.activity or []) + [
        {"kind": kind, "message": message, "ts": utcnow().isoformat()}
    ]
    project.activity = project.activity[-200:]
