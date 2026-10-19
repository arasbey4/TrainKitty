import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    """Point TrainKit at a temp data dir and reset the engine/config."""
    from trainkit import config as tk_config
    from trainkit import database as tk_db

    data_dir = tmp_path / "trainkit-data"
    monkeypatch.setenv("TRAINKIT_DATA_DIR", str(data_dir))
    tk_config._config = None
    tk_db._reset_for_tests()
    cfg = tk_config.get_config()
    cfg.ensure_dirs()
    yield cfg
    monkeypatch.delenv("TRAINKIT_DATA_DIR", raising=False)
    tk_config._config = None
    tk_db._reset_for_tests()
