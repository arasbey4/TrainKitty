"""Dataset service: lifecycle of a dataset (raw upload/generation -> prepared files).

Metadata lives in the database, example files live under the dataset dir.
Source files are retained separately from prepared outputs (never overwritten).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..config import get_config
from ..db_models import Dataset, DatasetSource, new_id
from ..security import PathValidationError, validate_upload_file
from .generate import GenerationOptions, generate_dataset
from .io import detect_format, load_records, write_jsonl_atomic
from .prepare import PreparationOptions, prepare_dataset
from .schemas import normalize_example as _normalize_example  # noqa: F401 (stable path)

log = logging.getLogger("trainkit.datasets")


def _safe_name(name: str) -> str:
    name = os.path.basename(name.strip())
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:60] or "dataset"
    return name


def create_dataset_dir(dataset_id: str) -> Path:
    directory = get_config().datasets_dir / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def register_uploaded_file(
    db, *, filename: str, size_bytes: int | None, content: bytes | None, project_id: str | None = None
) -> Dataset:
    """Validate + save a user-uploaded source file, register as raw dataset."""
    safe = validate_upload_file(filename, size_bytes)
    ds_id = new_id("ds")
    directory = create_dataset_dir(ds_id)
    # Store under a fixed, sanitized name (never the client-supplied name).
    target = directory / f"source{Path(safe).suffix.lower()}"
    with open(target, "wb") as fh:  # size already validated by caller/cap
        fh.write(content or b"")
    fmt = detect_format(target)
    dataset = Dataset(
        id=ds_id,
        project_id=project_id,
        name=Path(safe).stem or "upload",
        source="upload",
        format=fmt,
        path=str(target),
        schema_type=_detect_schema(target),
        example_count=0,
        status="raw",
        provenance={"uploaded_filename": safe, "size_bytes": size_bytes},
    )
    db.add(dataset)
    db.commit()
    dataset.example_count = count_examples(dataset)
    db.commit()
    return dataset


def register_generated(
    db,
    *,
    result: Any,
    model: str,
    options: GenerationOptions,
    name: str,
    project_id: str | None = None,
) -> Dataset:
    """Persist durable generated examples as a new raw dataset."""
    ds_id = new_id("ds")
    directory = create_dataset_dir(ds_id)
    target = directory / "source.jsonl"
    write_jsonl_atomic(target, result.examples)
    dataset = Dataset(
        id=ds_id,
        project_id=project_id,
        name=name or "generated",
        source="ollama",
        format="jsonl",
        path=str(target),
        schema_type=options.schema_type,
        example_count=len(result.examples),
        status="raw",
        provenance={
            "model": model,
            "generation": result.to_dict(),
            "options": vars(options),
        },
    )
    db.add(dataset)
    db.commit()
    return dataset


def register_hf_import(
    db,
    *,
    records: list[dict[str, Any]],
    info: dict[str, Any],
    name: str,
    project_id: str | None = None,
    license_meta: dict[str, Any] | None = None,
) -> Dataset:
    ds_id = new_id("ds")
    directory = create_dataset_dir(ds_id)
    target = directory / "source.jsonl"
    write_jsonl_atomic(target, records)
    dataset = Dataset(
        id=ds_id,
        project_id=project_id,
        name=name,
        source="huggingface",
        format="jsonl",
        path=str(target),
        schema_type=_detect_schema(target),
        example_count=len(records),
        status="raw",
        provenance={
            "hf": info,
            "license": license_meta,
        },
    )
    db.add(dataset)
    db.commit()
    return dataset


def count_examples(dataset: Dataset, max_records: int = 200_000) -> int:
    fmt = detect_format(dataset.path)
    if fmt in ("jsonl", "csv"):
        n = 0
        with open(dataset.path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n
    n = 0
    for _ in load_records(dataset.path, max_records=max_records):
        n += 1
    return n


def _detect_schema(path: str | Path) -> str | None:
    kinds: set[str] = set()
    try:
        for i, raw in enumerate(load_records(Path(path), max_records=50)):
            kind = normalize_example_classify(raw)
            if kind:
                kinds.add(kind)
            if len(kinds) == 2:
                break
    except Exception:
        return "unknown"
    if not kinds:
        return "unknown"
    return next(iter(kinds))


def normalize_example_classify(raw: Any) -> str | None:
    from .schemas import detect_example_kind

    kind = detect_example_kind(raw)
    return kind


def prepare(
    db,
    dataset: Dataset,
    *,
    validation_split: float,
    seed: int,
    dedupe: bool,
    name_suffix: str = "egitim",
) -> tuple[Dataset, dict[str, Any]]:
    """Run real preparation. Creates a NEW prepared dataset row."""
    ds_id = new_id("ds")
    directory = create_dataset_dir(ds_id)
    train_target = directory / "train.jsonl"
    validation_target = directory / "validation.jsonl"
    report = prepare_dataset(
        dataset.path,
        train_target,
        validation_target if validation_split > 0 else None,
        PreparationOptions(
            dedupe=dedupe,
            validation_split=validation_split,
            seed=seed,
        ),
    )
    prepared = Dataset(
        id=ds_id,
        project_id=dataset.project_id,
        name=f"{dataset.name}-{name_suffix}",
        source=dataset.source,
        format="jsonl",
        path=str(train_target),
        schema_type=report.schema,
        example_count=report.train_count,
        status="validated",
        provenance={
            "derived_from_dataset_id": dataset.id,
            "report": report.to_dict(),
            "validation_split": validation_split,
            "seed": seed,
        },
    )
    db.add(prepared)
    db.add(
        DatasetSource(
            id=new_id("dssrc"),
            dataset_id=prepared.id,
            path=dataset.path,
            transformation=report.to_dict(),
        )
    )
    dataset.status = "validated"
    db.commit()

    validation_path = None
    if validation_split > 0 and report.validation_count > 0:
        validation_path = str(validation_target)
    return prepared, {
        "report": report.to_dict(),
        "train_path": str(train_target),
        "validation_path": validation_path,
    }


def preview_records(dataset: Dataset, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(load_records(dataset.path, max_records=None if limit else 0)):
        if isinstance(raw, Exception):
            out.append({"_parse_error": str(raw)[:200], "_line": i + 1})
            continue
        if i < offset:
            continue
        if len(out) >= limit:
            break
        out.append(raw)
    return out


def delete_dataset(db, dataset: Dataset) -> dict[str, Any]:
    """Delete DB row; files removed only from the dataset's managed dir.

    Failure of physical file-unlink alone aborts nothing destructive;
    deletion of files requires the directory to exist inside the data root.
    """
    from ..db_models import DatasetSource

    for src in db.query(DatasetSource).filter(DatasetSource.dataset_id == dataset.id):
        db.delete(src)
    db.delete(dataset)
    db.commit()
    directory = get_config().datasets_dir / dataset.id
    removed = False
    if directory.is_dir():
        import shutil

        shutil.rmtree(directory)
        removed = True
    return {"deleted": True, "dataset_id": dataset.id, "files_removed": removed}


def dataset_report(db, dataset: Dataset, source: Any, transformation: Any) -> dict[str, Any]:
    return {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "source": dataset.source,
            "format": dataset.format,
            "schema": dataset.schema_type,
            "example_count": dataset.example_count,
            "status": dataset.status,
            "path": dataset.path,
        },
        "source_references": source,
        "transformations": transformation,
    }
