"""FastAPI app assembly. Loopback-first; narrow CORS (dev origins only)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_config

logger = logging.getLogger("trainkit.app")

DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrainKit API",
        description=(
            "Local-first AI model creation and training platform. "
            "All data stays on-loopback; no cloud AI is required."
        ),
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )

    from ..database import init_db

    init_db()

    from .routers import (
        datasets_api,
        evaluations_api,
        ollama_api,
        projects_api,
        system,
        training_api,
    )

    app.include_router(system.router)
    app.include_router(ollama_api.router)
    app.include_router(projects_api.router)
    app.include_router(datasets_api.router)
    app.include_router(training_api.router)
    app.include_router(evaluations_api.router)
    app.include_router(evaluations_api.router_models)

    @app.get("/api/agent/tools")
    def agent_tools():
        from ..agent import SUPPORTED_TOOLS

        return {"tools": sorted(SUPPORTED_TOOLS), "policy": "model proposes; app executes"}

    return app


app = create_app()
