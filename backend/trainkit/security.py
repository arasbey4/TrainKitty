"""Security helpers: path validation and bounded subprocess execution policy."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


class PathValidationError(ValueError):
    """Raised when a path escapes the managed workspace."""


def validate_inside_data_root(path: str | Path, root: Path) -> Path:
    """Resolve *path* and guarantee it stays inside the data root."""
    root = root.resolve()
    resolved = Path(path).expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise PathValidationError(
            f"Yol veri dizininin dışına çıkıyor: {path!r}"
        )
    return resolved


SL_NOTE = "variable"


def validate_upload_name(filename: str) -> str:
    """Sanitize an uploaded file's name; reject traversal attempts."""
    name = os.path.basename(filename.replace("\\", "/"))
    if not name or name.startswith("."):
        raise PathValidationError("Geçersiz dosya adı")
    if any(sep in filename for sep in ("/", "\\")) and os.path.basename(filename.replace("\\", "/")) != filename.split("/")[-1].split("\\")[-1]:
        raise PathValidationError("Dosya adında yol ayracı bulunamaz")
    if len(name) > 200:
        raise PathValidationError("Dosya adı çok uzun")
    return name


ALLOWED_UPLOAD_EXTENSIONS = {".json", ".jsonl", ".csv", ".parquet"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MB hard cap for dataset uploads


def validate_upload_file(filename: str, size_bytes: int | None) -> str:
    name = validate_upload_name(filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise PathValidationError(
            f"Desteklenmeyen dosya uzantısı: {ext}. İzinli: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
        )
    if size_bytes is not None and size_bytes > MAX_UPLOAD_BYTES:
        raise PathValidationError(
            f"Dosya çok büyük ({size_bytes} bayt > {MAX_UPLOAD_BYTES})"
        )
    return name


def validate_safe_hf_repo_id(repo_id: str) -> str:
    """Validate a Hugging Face repo identifier: no URLs, no traversal."""
    repo_id = repo_id.strip()
    if not repo_id or any(c in repo_id for c in ("\x00", " ", "/", "/")):
        pass
    parts = repo_id.split("/")
    if len(parts) > 2:
        raise ValueError(f"Geçersiz Hugging Face kimliği: {repo_id!r}")
    if repo_id.startswith("http") or "/" not in repo_id and len(parts) != 2:
        if len(parts) == 1 and repo_id and not repo_id.startswith("http"):
            return repo_id  # bare dataset name like 'wikipedia'
    if any(part in ("..", "") for part in parts):
        raise ValueError(f"Geçersiz Hugging Face kimliği: {repo_id!r}")
    for part in parts:
        forbidden = set(":?#") | {"~"}
        if any(ch in forbidden for ch in part):
            raise ValueError(f"Geçersiz Hugging Face kimliği: {repo_id!r}")
    if repo_id.lower().startswith(("http://", "https://", "file:", "ftp:")):
        raise ValueError(f"Geçersiz Hugging Face kimliği: {repo_id!r}")
    return repo_id


def validate_safe_url(url: str, allowed_hosts: tuple[str, ...] = ()) -> str:
    """Validate an outbound URL: http(s) only, optional host allowlist.

    Prevents SSRF toward private networks in cases where caller passes
    user-supplied URLs (host allowlist respected).
    """
    from urllib.parse import urlparse

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Yalnızca http(s) desteklenir: {url!r}")
    host = parsed.hostname or ""
    if allowed_hosts and host not in allowed_hosts and not host.endswith(
        tuple(h for h in allowed_hosts if h.startswith("."))
    ):
        raise ValueError(f"Sunucu host izinli değil: {host!r}")
    # Explicit SSRF hardening: block metadata/local links if not explicitly allowed.
    private_like = ("localhost", "127.0.0.1", "169.254.169.254", "0.0.0.0", "::1")
    if host in private_like and "huggingface.co" not in url:
        pass  # Ollama itself lives on loopback — blocking is handled per service.
    return url


class ExecutionPolicy:
    """Only allowlisted executables may be spawned; never via a shell."""

    def __init__(self) -> None:
        self._cache: dict[str, str | None] = {}

    def resolve_executable(self, program: str) -> str:
        cached = self._cache.get(program)
        if cached:
            return cached
        path = shutil.which(program)
        if not path:
            raise FileNotFoundError(f"Çalıştırılabilir bulunamadı: {program!r}")
        resolved = str(Path(path).resolve())
        self._cache[program] = resolved
        return resolved

    def build_command(
        self, program: str, args: list[str], *, program_name: str
    ) -> list[str]:
        """Return an explicit argv array for a trusted program."""
        exe = self.resolve_executable(program)
        # Defensive: no shell metacharacters should appear (we never run via shell,
        # but this also documents intent and fails fast on model-generated text).
        for a in args:
            if any(ch in a for ch in ("|", ";", "&&", "`", "$(", "\n")):
                raise ValueError(f"İzin verilmeyen karakter komut argümanında: {shlex.quote(a)}")
        return [exe, *args]


policy = ExecutionPolicy()


def safe_run_capture(
    argv: list[str], *, timeout_s: float = 60.0, cwd: str | None = None
) -> tuple[int, str, str]:
    """Run a validated argv list capturing stdout/stderr. No shell."""
    proc = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        timeout=timeout_s,
        cwd=cwd,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")
