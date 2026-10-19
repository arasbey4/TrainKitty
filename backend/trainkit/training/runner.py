"""Training job runner.

Runs training as a real backend-managed subprocess; records stdout/stderr,
parses JSON-line progress, writes real metrics to the DB, appends job events,
verifiers artifacts on failure/completion, and drives an in-process event bus
for SSE. A crashed process is never marked completed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from typing import Any

from ..config import get_config
from ..database import get_sessionmaker, session_scope
from ..db_models import Artifact, JobEvent, TrainingJob, TrainingMetric, new_id, utcnow

JOB_STATES = (
    "queued",
    "preparing",
    "downloading",
    "validating",
    "training",
    "evaluating",
    "packaging",
    "completed",
    "failed",
    "cancelled",
)
ACTIVE_STATES = ("queued", "preparing", "downloading", "validating", "training", "evaluating", "packaging")

RUNNING_STATES = ("preparing", "downloading", "validating", "training", "evaluating", "packaging")


class EventBus:
    """Simple in-process pub/sub for job updates (SSE backend)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[threading.Event]] = {}
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {}

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._latest[topic] = payload
            events = list(self._subscribers.get(topic, []))
        for ev in events:
            ev.set()

    def poll_latest(self, topic: str) -> dict[str, Any] | None:
        with self._lock:
            return self._latest.get(topic)

    def wait_for(self, topic: str, timeout: float = 1.0) -> dict[str, Any] | None:
        ev = threading.Event()
        with self._lock:
            self._subscribers.setdefault(topic, []).append(ev)
        try:
            ev.wait(timeout)
        finally:
            with self._lock:
                subs = self._subscribers.get(topic, [])
                if ev in subs:
                    subs.remove(ev)
        return self.poll_latest(topic)


bus = EventBus()
_processes: dict[str, subprocess.Popen] = {}
_cancel_requested: set[str] = set()


def append_event(
    job_id: str,
    level: str,
    message: str,
    *,
    operation: str | None = None,
    data: Any = None,
) -> None:
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        db.add(
            JobEvent(
                job_id=job_id,
                level=level,
                message=message[:4000],
                operation=operation,
                data=data,
            )
        )

def start_job(job_id: str) -> bool:
    """Spawn the training subprocess for a queued job."""
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if job is None or job.status not in ("queued",):
            return False
        cfg = job.config or {}
        run_dir = get_config().runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        job.run_dir = str(run_dir)
        job.status = "preparing"
        job.started_at = utcnow()
        run_config = {**cfg, "run_dir": str(run_dir)}
        (run_dir / "config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    cmd = [sys.executable, "-m", "trainkit.training.train_script", str(run_dir / "config.json")]

    log_file = open(run_dir / "stdout.log", "ab")
    env = {**get_config_env(), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=log_file,
        cwd=str(get_config().root),
        env=env,
        shell=False,
    )
    _processes[job_id] = proc
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if job:
                job.pid = proc.pid
                db.add(
                    JobEvent(
                        job_id=job_id,
                        level="info",
                        message=f"Eğitim süreci başlatıldı (PID {proc.pid}).",
                        operation="start",
                    )
                )
    bus.publish(job_id, {"type": "state", "state": "preparing", "pid": proc.pid})
    threading.Thread(target=_watch_job, args=(job_id, proc, run_dir), daemon=True).start()
    return True


def get_config_env() -> dict[str, str]:
    """Child environment: pass current env through (HF_TOKEN may be needed for
    gated models; it is the user's own token and stays on this machine)."""
    import os

    return dict(os.environ)


def cancel_job(job_id: str) -> bool:
    _cancel_requested.add(job_id)
    proc = _processes.get(job_id)
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return False
        append_event(job_id, "warning", "İptal talebi alındı; süreç sonlandırılıyor.", operation="cancel")
        if proc is None and job.status not in ("queued",):
            return False
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
    _set_terminal_state(job_id, "cancelled", "Kullanıcı tarafından iptal edildi.")
    return True


def _watch_job(job_id: str, proc: subprocess.Popen, run_dir) -> None:
    """Consume subprocess stdout JSON lines and update real state."""
    assert proc.stdout is not None
    job = TrainingJob(id=job_id)
    errors: list[str] = []
    state = "preparing"
    for raw in proc.stdout:
        line = raw.decode(errors="replace").strip()
        if not line:
            continue
        with open(run_dir / "log.jsonl", "ab") as logf:
            logf.write((line + "\n").encode())
        try:
            parsed = json.loads(line)
        except ValueError:
            parsed = {"type": "log", "message": line[:300]}
        ptype = parsed.get("type")
        if ptype == "state":
            state = parsed.get("state", state)
            total = parsed.get("total_steps")
            with session_scope() as db:
                j = db.get(TrainingJob, job_id)
                if j:
                    if state in RUNNING_STATES:
                        j.status = state
                    if total is not None:
                        j.total_steps = int(total)
            bus.publish(job_id, {"type": "state", "state": state, "total_steps": total})
        elif ptype == "metric":
            with session_scope() as db:
                j = db.get(TrainingJob, job_id)
                if j:
                    db.add(
                        TrainingMetric(
                            job_id=job_id,
                            step=int(parsed.get("step", 0)),
                            epoch=parsed.get("epoch"),
                            loss=parsed.get("loss"),
                            eval_loss=parsed.get("eval_loss"),
                            learning_rate=parsed.get("learning_rate"),
                            elapsed_s=None,
                        )
                    )
                    j.progress_step = int(parsed.get("step", 0))
                    if parsed.get("total_steps"):
                        j.total_steps = int(parsed["total_steps"])
            bus.publish(job_id, {"type": "metric", **parsed})
        elif ptype == "log":
            append_event(job_id, "info", parsed.get("message", ""), operation="train")
            bus.publish(job_id, {"type": "log", "message": parsed.get("message", "")})
        elif ptype == "error":
            errors.append(str(parsed.get("error", "")))
            append_event(job_id, "error", str(parsed.get("error", "")), operation="train")
            bus.publish(job_id, {"type": "error", "error": parsed.get("error", "")})
        elif ptype == "result":
            with session_scope() as db:
                j = db.get(TrainingJob, job_id)
                if j:
                    j.trainer_output = parsed

    proc.wait()
    _processes.pop(job_id, None)
    exit_code = proc.returncode
    with session_scope() as db:
        j = db.get(TrainingJob, job_id)
        if j:
            j.exit_code = exit_code

    cancel = job_id in _cancel_requested
    if cancel:
        _set_terminal_state(job_id, "cancelled", "Kullanıcı tarafından iptal edildi.")
        return
    if exit_code == 0 and verify_run_artifacts(job_id):
        _set_terminal_state(job_id, "completed", None, trainer_state="completed")
        bus.publish(job_id, {"type": "state", "state": "completed"})
    else:
        msg = errors[-1] if errors else f"Eğitim süreci {exit_code} koduyla sonlandı."
        _set_terminal_state(job_id, "failed", msg)
        bus.publish(job_id, {"type": "state", "state": "failed", "error": msg})


def verify_run_artifacts(job_id: str) -> bool:
    """Verify real outputs before marking a run completed."""
    from pathlib import Path

    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return False
        run_dir = Path(job.run_dir or "")
        output_dir = ((job.config or {}).get("output_dir") or "")
        checks_ok = False
        if not run_dir.is_dir():
            return False
        result_file = run_dir / "result.json"
        if not result_file.is_file():
            return False
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except ValueError:
            return False
        if data.get("mode") == "configuration_only":
            checks_ok = bool(data.get("sample_generated"))
            artifact_dir = run_dir
        else:
            required = (
                ("adapter_config.json", "adapter_model.safetensors")
                if job.mode == "lora"
                else ("config.json", "model.safetensors")
            )
            dir_path = Path(output_dir) / "adapter" if job.mode == "lora" else Path(output_dir)
            for fname in required:
                f = dir_path / fname
                if not f.is_file() or f.stat().st_size == 0:
                    return False
            checks_ok = True
            artifact_dir = str(dir_path)
        db.add(
            Artifact(
                id=new_id("art"),
                project_id=job.project_id,
                job_id=job.id,
                kind="adapter" if job.mode == "lora" else "model",
                path=artifact_dir,
                size_bytes=_dir_size(artifact_dir),
                meta={"mode": job.mode},
            )
        )
    return checks_ok


def _dir_size(path: str) -> int:
    from pathlib import Path

    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    except OSError:
        return 0


def _set_terminal_state(
    job_id: str, state: str, error: str | None = None, *, trainer_state: str | None = None
) -> None:
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        job.status = state
        if error:
            job.error = error[:2000]
        if state in ("completed", "failed", "cancelled"):
            job.finished_at = utcnow()
            job.process_alive = False
    append_event(
        job_id,
        "info" if state == "completed" else "error",
        f"İş durumu: {state}" + (f" — {error}" if error else ""),
        operation="finalize",
    )
