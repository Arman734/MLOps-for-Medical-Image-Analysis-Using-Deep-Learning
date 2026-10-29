"""Background training jobs launched from the console."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from medmlops.config import PROJECT_ROOT

ALLOWED_CONFIGS = {
    "configs/isic_unet.yaml",
    "configs/pneumonia_densenet.yaml",
}


@dataclass
class Job:
    id: str
    config: str
    overrides: list[str]
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    log: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None


class JobManager:
    def __init__(self, on_finish: Callable[[], None] | None = None, max_lines: int = 4000) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._on_finish = on_finish
        self._max_lines = max_lines

    def _running(self) -> Job | None:
        for job in self.jobs.values():
            if job.status == "running":
                return job
        return None

    def start(self, config: str, overrides: list[str] | None = None) -> dict:
        config = config.strip()
        if config not in ALLOWED_CONFIGS:
            raise ValueError(f"config must be one of {sorted(ALLOWED_CONFIGS)}")
        overrides = [o for o in (overrides or []) if o]
        with self._lock:
            running = self._running()
            if running is not None:
                return {"error": f"job {running.id} is already {running.status}", "job_id": running.id}
            finished = sorted(
                (job for job in self.jobs.values() if job.status != "running"),
                key=lambda job: job.started_at,
            )
            for stale in finished[:-20]:
                self.jobs.pop(stale.id, None)
            job_id = uuid.uuid4().hex[:8]
            cmd = [sys.executable, "-m", "medmlops.cli", "train", "--config", config, "--device", "auto"]
            for override in overrides:
                cmd.extend(["--set", override])
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            job = Job(id=job_id, config=config, overrides=overrides, process=process)
            self.jobs[job_id] = job
            threading.Thread(target=self._reader, args=(job,), daemon=True).start()
        return self._public(job, full_log=False)

    def _reader(self, job: Job) -> None:
        assert job.process is not None
        assert job.process.stdout is not None
        for line in job.process.stdout:
            with self._lock:
                job.log.append(line.rstrip("\n"))
                if len(job.log) > self._max_lines:
                    del job.log[: len(job.log) - self._max_lines]
        job.returncode = job.process.wait()
        with self._lock:
            # A cancelled job must stay cancelled: the reader can finish after
            # the cancel endpoint already changed the status.
            if job.status == "running":
                job.status = "succeeded" if job.returncode == 0 else "failed"
                job.finished_at = time.time()
                if job.returncode == 0:
                    job.log.append("[job] training finished - refreshing loaded models")
        if job.returncode == 0 and job.status == "succeeded" and self._on_finish is not None:
            try:
                self._on_finish()
            except Exception as exc:  # pragma: no cover - refresh is best effort
                with self._lock:
                    job.log.append(f"[job] model refresh failed: {exc}")

    def _public(self, job: Job, full_log: bool = True) -> dict:
        payload = {
            "id": job.id,
            "config": job.config,
            "overrides": job.overrides,
            "status": job.status,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "returncode": job.returncode,
            "log_lines": len(job.log),
        }
        if full_log:
            payload["log"] = job.log
        return payload

    def get(self, job_id: str, log_offset: int = 0) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            payload = self._public(job)
            payload["log"] = job.log[log_offset:]
            payload["log_offset"] = log_offset
            return payload

    def list(self) -> list[dict]:
        with self._lock:
            return [self._public(job, full_log=False) for job in sorted(self.jobs.values(), key=lambda j: j.started_at, reverse=True)]

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status == "running" and job.process is not None:
                job.status = "cancelled"
                job.finished_at = time.time()
                job.log.append("[job] cancelled by user")
                process = job.process
            else:
                process = None
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        return self._public(job)

    def shutdown(self) -> None:
        """Stop any training subprocess when the API shuts down."""
        with self._lock:
            running = [job for job in self.jobs.values() if job.status == "running" and job.process is not None]
            for job in running:
                job.status = "cancelled"
                job.log.append("[job] cancelled (server shutdown)")
        for job in running:
            assert job.process is not None
            job.process.terminate()
        for job in running:
            if job.process is None:
                continue
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()


_manager: JobManager | None = None


def get_manager(on_finish: Callable[[], None]) -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager(on_finish=on_finish)
    return _manager


def reset_manager() -> None:
    global _manager
    _manager = None


def shutdown_manager() -> None:
    global _manager
    if _manager is not None:
        _manager.shutdown()
        _manager = None
