"""Small helpers around DB sessions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .database import get_sessionmaker


@contextmanager
def fallback_session() -> Iterator:
    """Yield a session; commit on success, roll back on error."""
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
