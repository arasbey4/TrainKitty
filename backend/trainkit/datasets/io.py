"""Dataset file IO helpers (bounded memory: JSONL/CSV streamed line by line)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {".json": "json", ".jsonl": "jsonl", ".csv": "csv", ".parquet": "parquet"}.get(
        suffix, "unknown"
    )


def load_records(path: str | Path, max_records: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream records from a local dataset file (JSON = array or object; JSONL; CSV)."""
    path = Path(path)
    fmt = detect_format(path)
    n = 0
    if fmt == "jsonl":
        import json

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    # Malformed lines are surfaced as parse errors instead of
                    # aborting the whole stream; callers count them as rejects.
                    yield exc
                n += 1
                if max_records and n >= max_records:
                    return
    elif fmt == "json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
        for rec in data:
            yield rec
            n += 1
            if max_records and n >= max_records:
                return
    elif fmt == "csv":
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield dict(row)
                n += 1
                if max_records and n >= max_records:
                    return
    else:
        raise ValueError(f"Desteklenmeyen veri kümesi biçimi: {fmt}")


def write_jsonl_atomic(path: str | Path, examples: list[dict[str, Any]]) -> None:
    """Write JSONL atomically (tempfile + os.replace); partial writes vanish."""
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def append_jsonl(path: str | Path, example: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(example, ensure_ascii=False) + "\n")


def file_size(path: str | Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def count_lines(path: str | Path) -> int:
    if Path(path).suffix.lower() in (".json", ".csv", ".parquet"):
        return sum(1 for _ in load_records(path, max_records=10**9))
    n = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        return -1
    return n
