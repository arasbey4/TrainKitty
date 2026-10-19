"""Security and training-plan validation unit tests."""

import pytest

from trainkit.security import (
    PathValidationError,
    validate_upload_file,
    validate_inside_data_root,
    validate_safe_hf_repo_id,
)
from trainkit.training.plan import TrainingConfig, validate_plan


def test_upload_rejects_traversal(tmp_path):
    with pytest.raises(PathValidationError):
        validate_upload_file("../../etc/passwd", 10)
    with pytest.raises(PathValidationError):
        validate_upload_file("data.exe", 10)


def test_upload_rejects_oversize():
    with pytest.raises(PathValidationError):
        validate_upload_file("big.jsonl", 600 * 1024 * 1024)


def test_path_validation_blocks_escape(tmp_data_dir):
    root = tmp_data_dir.root
    ok = validate_inside_data_root(root / "models" / "x", root)
    assert ok.is_absolute()
    with pytest.raises(PathValidationError):
        validate_inside_data_root("/etc/passwd", root)


def test_hf_repo_id_rejects_url_and_traversal():
    assert validate_safe_hf_repo_id("tatsu-lab/alpaca") == "tatsu-lab/alpaca"
    with pytest.raises(ValueError):
        validate_safe_hf_repo_id("http://evil.com/x")
    with pytest.raises(ValueError):
        validate_safe_hf_repo_id("a/b/../c")


def test_training_config_rejects_bad_hyperparams():
    with pytest.raises(Exception):
        TrainingConfig(base_model_id="m", dataset_id="d", dataset_path="p", learning_rate=-1)
    with pytest.raises(Exception):
        TrainingConfig(base_model_id="m", dataset_id="d", dataset_path="p", sequence_length=10)


def test_validate_plan_reports_compute_estimate():
    cfg = TrainingConfig(base_model_id="m", dataset_id="d", dataset_path="p")
    issues = validate_plan(cfg, "instruction")
    assert any(i.field == "compute" for i in issues)
