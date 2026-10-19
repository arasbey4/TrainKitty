"""Database setup (SQLAlchemy 2.x, SQLite)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_config


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker | None = None
_init_lock = threading.Lock()


def _db_url() -> str:
    cfg = get_config()
    db_path: Path = cfg.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + str(db_path)


def get_engine():
    global _engine
    with _init_lock:
        if _engine is None:
            # check_same_thread=False: FastAPI uses a thread pool; SQLite runs
            # in a single file guarded by transactions.
            _engine = create_engine(
                _db_url(),
                connect_args={"check_same_thread": False, "timeout": 30},
                pool_pre_ping=True,
            )
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def _reset_for_tests() -> None:
    """Dispose engine/session so tests can point config at a new data dir."""
    global _engine, _SessionLocal
    with _init_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionLocal = None


def init_db() -> None:
    """Create all tables (and run lightweight reconciliation)."""
    import importlib

    # Ensure models register on Base.metadata.
    importlib.import_module("trainkit.db_models")
    from .db_models import reconcile_orphan_jobs

    Base.metadata.create_all(get_engine())
    reconcile_orphan_jobs()


def session_scope() -> Session:
    """Return a new session (caller closes or uses as context manager)."""
    return get_sessionmaker()()


def db_session_dep():
    """FastAPI dependency yielding a database session."""
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def db_file_path() -> Path:
    return Path(_db_url().split("sqlite:///", 1)[1])


def data_dir_root() -> Path:
    return get_config().root
