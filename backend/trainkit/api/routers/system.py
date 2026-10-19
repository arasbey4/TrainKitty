"""System endpoints: health, hardware, dependencies, settings."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ... import hardware
from ...database import db_session_dep
from ...settings import DEFAULTS, all_settings, get_setting, set_setting

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "app": "trainkit", "version": "0.1.0"}


@router.get("/system/info")
def system_info():
    from ..config import get_config

    cfg = get_config()
    return {
        "app": "trainkit",
        "version": "0.1.0",
        "data_dir": str(cfg.root),
        "os": hardware.inspect_hardware()["os"],
        "privacy": "Tüm veriler yerelde kalır; hiçbir telemetri veya bulut çıkarımı yoktur.",
    }


@router.get("/system/hardware")
def system_hardware():
    return hardware.inspect_hardware()


@router.get("/system/dependencies")
def system_dependencies():
    return hardware.inspect_dependencies()


class SettingsPayload(BaseModel):
    values: dict[str, Any]


@router.get("/settings")
def read_settings(db: Session = Depends(db_session_dep)):
    return {"settings": all_settings(db), "allowed_keys": sorted(DEFAULTS.keys())}


@router.patch("/settings")
def update_settings(payload: SettingsPayload, db: Session = Depends(db_session_dep)):
    updated: dict[str, Any] = {}
    for key, value in payload.values.items():
        try:
            updated[key] = set_setting(db, key, value)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Bilinmeyen ayar anahtarı: {key}")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return {"updated": updated}
