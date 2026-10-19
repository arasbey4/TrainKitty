"""Application configuration.

All paths in this file resolve inside the TrainKit data directory. The data
 root can be overridden with the ``TRAINKIT_DATA_DIR`` environment variable.
The default Ollama endpoint can be overridden with ``TRAINKIT_OLLAMA_URL``
(it is only a *default*; the effective endpoint is persisted in app settings
so users can change it from the UI without touching environment variables).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def default_data_dir() -> Path:
    """Resolve the default application data directory."""
    env = os.environ.get("TRAINKIT_DATA_DIR")
    if env:
        return Path(os.path.expanduser(env))
    # ~/.trainkit on every supported OS.
    home = os.path.expanduser("~")
    if not home or home == "~":  # expanduser failed (no HOME)
        home = Path.cwd().as_posix()
    return Path(os.path.join(os.path.expanduser("~"), ".trainkit"))


class Config:
    """Holds resolved paths for the TrainKit data layout."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_data_dir()).expanduser()
        self.subdirs = [
            "database",
            "projects",
            "datasets",
            "models",
            "runs",
            "checkpoints",
            "exports",
            "logs",
            "cache",
        ]

    @property
    def database_dir(self) -> Path:
        return self.root / "database"

    @property
    def database_path(self) -> Path:
        return self.database_dir / "trainkit.db"

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in self.subdirs:
            (self.root / sub).mkdir(parents=True, exist_ok=True)


_config: Config | None = None


def get_config() -> Config:
    """Return the process-wide configuration singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config_dir(root: Path | None = None, *, env_data_dir: str | None = None) -> Config:
    """Reset the configuration singleton (used by tests) and ensure dirs exist."""
    global _config
    if env_data_dir:
        os.environ["TRAINKIT_DATA_DIR"] = env_data_dir
    cfg = Config(root if root is not None else Path(os.environ.get("TRAINKIT_DATA_DIR")))
    cfg.ensure_dirs()
    _config = cfg
    return cfg


def os_name() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"
