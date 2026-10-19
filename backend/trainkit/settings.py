"""Application settings persisted in the database (Ollama endpoint etc.)."""

from __future__ import annotations

import os
from typing import Any

from .database import get_sessionmaker
from .db_models import AppSetting

DEFAULTS: dict[str, Any] = {
    "ollama.base_url": os.environ.get("TRAINKIT_OLLAMA_URL") or "http://127.0.0.1:11434",
    "ollama.orchestrator_model": None,
    "ollama.request_timeout_s": 120.0,
    "ollama.num_ctx": 4096,
    "ollama.temperature": 0.4,
    "training.defaults": {},
    "ui.language": "tr",
    "data.data_dir": None,  # informational; actual root resolved at startup
}

_float_keys = {"ollama.request_timeout_s", "ollama.temperature"}
_int_keys = {"ollama.num_ctx"}


def get_setting(db, key: str, default: Any = None) -> Any:
    if key not in DEFAULTS:
        raise KeyError(f"Bilinmeyen ayar anahtarı: {key}")
    row = db.get(AppSetting, key)
    if row is None or row.value is None:
        return DEFAULTS[key] if DEFAULTS[key] is not None else default
    val = row.value
    if key in _float_keys:
        try:
            return float(val)
        except (TypeError, ValueError):
            return DEFAULTS[key]
    if key in _int_keys:
        try:
            return int(val)
        except (TypeError, ValueError):
            return DEFAULTS[key]
    return val


def set_setting(db, key: str, value: Any) -> Any:
    if key not in DEFAULTS:
        raise KeyError(f"Bilinmeyen ayar anahtarı: {key}")
    if key == "ollama.base_url" and isinstance(value, str):
        value = _validate_base_url(value)
    row = db.get(AppSetting, key)
    if row is None:
        row = AppSetting(key=key, value=value)
        db.add(row)
    else:
        row.value = value
    db.commit()
    return value


def all_settings(db) -> dict[str, Any]:
    out = dict(DEFAULTS)
    for row in db.query(AppSetting).all():
        if row.key in DEFAULTS:
            out[row.key] = row.value
    return out


def _validate_base_url(url: str) -> str:
    from urllib.parse import urlparse

    url = url.strip()
    if not url:
        raise ValueError("Boş adres")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Adres http(s) olmalı")
    if not parsed.hostname:
        raise ValueError("Adres host içermeli")
    return url.rstrip("/")
