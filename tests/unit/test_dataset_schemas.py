"""Dataset schema detection / normalization / preparation unit tests."""

import json

import pytest

from trainkit.datasets.prepare import PreparationOptions, prepare_dataset
from trainkit.datasets.schemas import normalize_example


def test_normalize_instruction():
    kind, ex = normalize_example(
        {"instruction": "Explain a TypeError.", "input": "", "output": "A TypeError occurs ..."},
        index=0,
        source_id="t",
    )
    assert kind == "instruction"
    assert ex["instruction"] == "Explain a TypeError."
    assert ex["_index"] == 0


def test_normalize_chat_blocks_missing_content():
    bad = {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]}
    assert normalize_example(bad, index=0, source_id="t") is None


def test_normalize_role_aliases():
    kind, ex = normalize_example(
        {"messages": [{"role": "human", "content": "hi"}, {"role": "GPT", "content": "hello"}]},
        index=0,
        source_id="t",
    )
    assert kind == "chat"
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["user", "assistant"]


def test_reject_empty_and_malformed():
    assert normalize_example("nope", index=0, source_id="t") is None
    assert normalize_example({}, index=0, source_id="t") is None
    assert normalize_example({"instruction": "x", "output": 42}, index=0, source_id="t") is None


def test_prepare_dedupes_and_splits(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = []
    for i in range(10):
        rows.append({"instruction": f"Q{i}: what?", "output": f"A{i}."})
    rows.append({"instruction": "Q0: what?", "output": "A0."})  # exact duplicate
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    report = prepare_dataset(
        src,
        tmp_path / "train.jsonl",
        tmp_path / "val.jsonl",
        PreparationOptions(validation_split=0.2, seed=7),
    )
    assert report.accepted == 10
    assert report.duplicates == 1
    assert report.train_count == 8
    assert report.validation_count == 2

    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (tmp_path / "val.jsonl").read_text().splitlines()]
    train_keys = {e["_id"] for e in train}
    val_keys = {e["_id"] for e in val}
    assert not (train_keys & val_keys)


def test_prepare_rejects_bad_schema_counts(tmp_path):
    src = tmp_path / "bad.jsonl"
    src.write_text(
        'not json\n{"instruction":"","output":""}\n', encoding="utf-8"
    )
    report = prepare_dataset(src, tmp_path / "train.jsonl", None)
    assert report.source_records == 2  # parse error counted + empty record
    assert report.rejection_breakdown.get("parse_error") == 1
    assert report.rejection_breakdown.get("schema_mismatch") == 1
