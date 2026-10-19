"""Ollama endpoints: real connection status, model discovery, inference, pull."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database import db_session_dep
from ...ollama import (
    OllamaError,
    OllamaInvalidOutputError,
    OllamaModelNotFoundError,
)
from ...ollama.client import OllamaClient
from ...settings import get_setting, set_setting

router = APIRouter(prefix="/api/ollama", tags=["ollama"])


def get_client(db: Session) -> OllamaClient:
    return OllamaClient.from_settings(db)


@router.get("/status")
def status(db: Session = Depends(db_session_dep)):
    client = get_client(db)
    st = client.status()
    orchestrator = get_setting(db, "ollama.orchestrator_model")
    st["orchestrator_model"] = orchestrator
    if st.get("connected") and orchestrator:
        st["orchestrator_installed"] = any(
            (m.get("name") or "").split(":")[0] == orchestrator.split(":")[0]
            for m in st.get("models", [])
        )
    return st


@router.get("/models")
def list_models(db: Session = Depends(db_session_dep)):
    try:
        client = get_client(db)
        return {"models": client.list_models()}
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=exc.user_message)


class TestRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None


@router.post("/test")
def test_connection(payload: TestRequest, db: Session = Depends(db_session_dep)):
    client = get_client(db)
    try:
        models = client.list_models()
    except OllamaError as exc:
        return {"ok": False, "error": exc.user_message, "detail": exc.detail}
    model = payload.model or get_setting(db, "ollama.orchestrator_model")
    result: dict[str, Any] = {"ok": True, "models_found": len(models)}
    if model:
        try:
            text = client.generate(
                model,
                payload.prompt or "Merhaba de.",
                options={"num_predict": 32},
                timeout_s=60.0,
            )
            result["inference_ok"] = True
            result["sample_response"] = text[:200]
        except OllamaError as exc:
            result["ok"] = bool(models)
            result["inference_ok"] = False
            result["error"] = exc.user_message
            result["detail"] = exc.detail
    return result


class OrchestratorRequest(BaseModel):
    model: str


@router.post("/orchestrator")
def select_orchestrator(payload: OrchestratorRequest, db: Session = Depends(db_session_dep)):
    client = get_client(db)
    try:
        models = client.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=exc.user_message)
    names = {m.get("name") or "" for m in models}
    if payload.model not in names and payload.model.split(":")[0] not in {n.split(":")[0] for n in names}:
        raise HTTPException(
            status_code=400,
            detail=f"Model Ollama'da kurulu değil: {payload.model}. `ollama pull` çalıştırın.",
        )
    set_setting(db, "ollama.orchestrator_model", payload.model)
    return {"ok": True, "orchestrator_model": payload.model}


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    options: dict[str, Any] | None = None
    system: str | None = None


@router.post("/generate")
def generate(payload: GenerateRequest, db: Session = Depends(db_session_dep)):
    client = get_client(db)
    model = payload.model or get_setting(db, "ollama.orchestrator_model")
    if not model:
        raise HTTPException(status_code=400, detail="Model seçilmedi; önce orkestratör model ayarlayın.")
    try:
        text = client.generate(model, payload.prompt, system=payload.system, options=payload.options)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=f"{exc.user_message} ({exc.detail or ''})")
    return {"model": model, "response": text}


@router.post("/chat")
def chat(payload: GenerateRequest, db: Session = Depends(db_session_dep)):
    client = get_client(db)
    model = payload.model or get_setting(db, "ollama.orchestrator_model")
    if not model:
        raise HTTPException(status_code=400, detail="Model seçilmedi.")
    messages = [{"role": "user", "content": payload.prompt}]
    try:
        data = client.chat(model, messages, options=payload.options)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=f"{exc.user_message} ({exc.detail or ''})")
    msg = ((data.get("message") or {}).get("content")) if isinstance(data, dict) else None
    return {"model": model, "message": msg}


class PullRequest(BaseModel):
    model: str


@router.post("/pull")
def pull(payload: PullRequest, db: Session = Depends(db_session_dep)):
    """Stream a REAL model pull (SSE). The user must explicitly call this."""
    client = get_client(db)

    def event_stream():
        try:
            for event in client.pull(payload.model):
                yield f"data: {json.dumps({'type': 'pull', **event}, ensure_ascii=False)}\n\n"
        except OllamaError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': exc.user_message, 'detail': exc.detail})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
