"""Dataset preparation: normalize -> validate -> dedupe -> split -> stats.

Produces a *prepared* JSONL dataset plus a transformation report; source
datasets are never overwritten.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import load_records, write_jsonl_atomic
from .schemas import examples_hash, normalize_example


@dataclass
class PreparationOptions:
    dedupe: bool = True
    min_text_len: int = 1
    max_text_len: int = 20000
    validation_split: float = 0.0
    seed: int = 42


@dataclass
class PreparationReport:
    source_records: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    schema: str | None = None
    train_count: int = 0
    validation_count: int = 0
    avg_output_len: float | None = None
    rejection_breakdown: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejection_breakdown[reason] = self.rejection_breakdown.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_records": self.source_records,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "detected_schema": self.schema,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "avg_output_len": self.avg_output_len,
            "rejection_breakdown": self.rejection_breakdown,
        }


def prepare_dataset(
    source_path: str | Path,
    out_train_path: str | Path,
    out_validation_path: str | Path | None = None,
    options: PreparationOptions | None = None,
) -> PreparationReport:
    """Run real normalization and produce separate train/validation files."""
    opts = options or PreparationOptions()
    report = PreparationReport()
    seen: set[str] = set()
    train_examples: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []

    kinds: set[str] = set()
    output_lens: list[int] = []

    for entry in load_records(source_path):
        if isinstance(entry, Exception):
            report.source_records += 1
            report.rejected += 1
            report.reject("parse_error")
            continue
        raw = entry
        report.source_records += 1
        normalized = normalize_example(raw, index=report.source_records, source_id=str(source_path))
        if normalized is None:
            report.rejected += 1
            report.reject("schema_mismatch")
            continue
        kind, example = normalized
        kinds.add(kind)
        text_paths = (
            [example.get("instruction", ""), example.get("input", ""), example.get("output", "")]
            if kind == "instruction"
            else [m.get("content", "") for m in example.get("messages", [])]
        )
        text_lens = [len(t) for t in text_paths]
        if any(t and len(t) < opts.min_text_len for t in text_paths):
            report.rejected += 1
            report.reject("too_short")
            continue
        if any(l > opts.max_text_len for l in text_lens):
            report.rejected += 1
            report.reject("too_long")
            continue
        if opts.dedupe:
            h = examples_hash(example)
            if h in seen:
                report.duplicates += 1
                continue
            seen.add(h)
        if kind == "instruction":
            output_lens.append(len(example.get("output", "")))

        report.accepted += 1
        train_examples.append(example)

    if opts.validation_split > 0 and train_examples:
        rng = random.Random(opts.seed)
        n_val = int(round(len(train_examples) * opts.validation_split))
        if n_val:
            idx = rng.sample(range(len(train_examples)), n_val)
            idx_set = set(idx)
            validation_examples = [
                train_examples[i] for i in range(len(train_examples)) if (i in idx_set) or False
            ]
            train_examples = [
                ex for i, ex in enumerate(train_examples) if i not in idx_set
            ]

    report.schema = next(iter(kinds), None)
    write_jsonl_atomic(out_train_path, train_examples)
    if out_validation_path is not None and validation_examples:
        write_jsonl_atomic(out_validation_path, validation_examples)
    report.train_count = len(train_examples)
    report.validation_count = len(validation_examples)
    if output_lens:
        report.avg_output_len = float(
            round(statistics.mean(output_lens), 1)
        )
    return report
