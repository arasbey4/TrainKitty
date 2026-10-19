"""Projects endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...agent import draft_spec
from ...database import db_session_dep
from ...db_models import Dataset
from ...ollama import OllamaError
from ...ollama.client import OllamaClient
from ...projects import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    update_project,
)
from ...settings import get_setting

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    spec: dict[str, Any] | None = None
    spec_approved: bool | None = None
    base_model: dict[str, Any] | None = None
    dataset_id: str | None = None
    training_mode: str | None = None
    status: str | None = None


@router.get("")
def read_projects(limit: int = 50, offset: int = 0, db: Session = Depends(db_session_dep)):
    projects, total = list_projects(db, limit=limit, offset=offset)
    return {"items": [_project_dict(p) for p in projects], "total": total, "limit": limit, "offset": offset}


@router.post("", status_code=201)
def create(payload: ProjectCreate, db: Session = Depends(db_session_dep)):
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Proje adı boş olamaz.")
    if len(payload.name) > 200:
        raise HTTPException(status_code=400, detail="Proje adı çok uzun.")
    return _project_dict(create_project(db, name=payload.name.strip(), description=payload.description))


@router.get("/{project_id}")
def read_one(project_id: str, db: Session = Depends(db_session_dep)):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı.")
    out = _project_dict(project)
    out["spec_label"] = "AI tarafından önerilen taslak; kullanıcı onaylamadan bağlayıcı değildir." if not project.spec_approved else "kayıtlı"
    return out


@router.patch("/{project_id}")
def patch(project_id: str, payload: ProjectPatch, db: Session = Depends(db_session_dep)):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı.")
    fields = payload.model_dump(exclude_unset=True, exclude_none=True)
    # Deleting or replacing structured config is guarded: spec overrides only
    # when explicitly sent; dataset/base model links validated below.
    if "dataset_id" in fields and fields["dataset_id"] is not None:
        if not db.get(Dataset, fields["dataset_id"]):
            raise HTTPException(status_code=400, detail="Geçersiz veri kümesi kimliği.")
    return _project_dict(update_project(db, project, **fields))


@router.delete("/{project_id}")
def delete(project_id: str, confirm: str = "", db: Session = Depends(db_session_dep)):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı.")
    if confirm != "true":
        raise HTTPException(status_code=428, detail="Bu işlem onay gerektirir (?confirm=true).")
    delete_project(db, project)
    return {"deleted": True, "id": project_id}


class SpecDraftRequest(BaseModel):
    description: str


@router.post("/{project_id}/spec/draft")
def draft(project_id: str, payload: SpecDraftRequest, db: Session = Depends(db_session_dep)):
    """Use the REAL local Ollama model to draft a structured spec."""
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı.")
    model = get_setting(db, "ollama.orchestrator_model")
    if not model:
        raise HTTPException(
            status_code=409,
            detail="Orkestrasyon modeli seçilmedi. 'Ollama Bağlantısı' bölümünden kurulu bir model seçin.",
        )
    client = OllamaClient.from_settings(db)
    try:
        draft_data = draft_spec(db, client, model, payload.description, project.id)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=f"{exc.user_message} {exc.detail or ''}")
    except Exception as exc:
        detail = getattr(exc, "user_message", str(exc))
        raise HTTPException(status_code=502, detail=f"Spec taslağı üretilemedi: {detail}")
    return {"draft": draft_data, "note": "Bu bir tahmindir; kullanıcı onayı gerekir."}


def _project_dict(project) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "spec": project.spec,
        "spec_approved": project.spec_approved,
        "base_model": project.base_model,
        "dataset_id": project.dataset_id,
        "training_mode": project.training_mode,
        "status": project.status,
        "activity": (project.activity or [])[-50:],
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
    }
