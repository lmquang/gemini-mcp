"""Background job manager for long-running Gemini operations."""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("gemini-mcp")
ARTIFACT_SCHEMA_VERSION = 1
SESSION_DIR_FALLBACK = "no-session"
SESSION_DIR_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    tool_name: str
    status: JobStatus = JobStatus.PENDING
    status_message: str = ""
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    result: Optional[dict] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None
    task: Optional[asyncio.Task] = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    def to_dict(self) -> dict:
        payload = {
            "jobID": self.job_id,
            "tool": self.tool_name,
            "status": self.status.value,
            "statusMessage": self.status_message,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at,
            "elapsedSeconds": int(time.time() - self.created_at),
        }
        if self.session_id:
            payload["sessionID"] = self.session_id
        if self.model:
            payload["model"] = self.model
        if self.output_path:
            payload["outputPath"] = self.output_path
        return payload

    def build_artifact_metadata(self) -> dict[str, Any]:
        return {
            "schemaVersion": ARTIFACT_SCHEMA_VERSION,
            "jobID": self.job_id,
            "tool": self.tool_name,
            "status": self.status.value,
            "statusMessage": self.status_message,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at,
            "durationSeconds": int(self.last_updated_at - self.created_at),
            "sessionID": self.session_id,
            "model": self.model,
            "outputPath": self.output_path,
        }


class JobManager:
    JOB_OUTPUT_DIR = ".gemini-mcp/jobs"

    def __init__(self, ttl_seconds: float = 3600.0):
        self._jobs: dict[str, Job] = {}
        self._ttl_seconds = ttl_seconds
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup(self):
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            self._expire_old_jobs()

    def _expire_old_jobs(self):
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.is_terminal and (now - job.last_updated_at) > self._ttl_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]
            logger.debug("Expired completed job", extra={"job_id": job_id})

    def create_job(self, tool_name: str) -> Job:
        job_id = uuid.uuid4().hex[:16]
        job = Job(job_id=job_id, tool_name=tool_name)
        self._jobs[job_id] = job
        logger.debug("Created job", extra={"job_id": job_id, "tool": tool_name})
        return job

    def _artifact_date_segment(self, created_at: float) -> str:
        return datetime.fromtimestamp(created_at, tz=timezone.utc).strftime("%Y%m%d")

    def _artifact_job_segment(self, job: Job) -> str:
        time_prefix = datetime.fromtimestamp(job.created_at, tz=timezone.utc).strftime("%H%M%S")
        return f"{time_prefix}-{job.job_id}"

    def _artifact_session_segment(self, session_id: Optional[str]) -> str:
        if not session_id or not session_id.strip():
            return SESSION_DIR_FALLBACK
        sanitized = SESSION_DIR_SANITIZE_RE.sub("-", session_id.strip()).strip(".-")
        return sanitized or SESSION_DIR_FALLBACK

    def build_artifact_dir(self, job: Job, cwd: Optional[str] = None) -> Path:
        base = Path(cwd) if cwd else Path.cwd()
        artifact_dir = (
            base
            / self.JOB_OUTPUT_DIR
            / self._artifact_date_segment(job.created_at)
            / self._artifact_session_segment(job.session_id)
            / self._artifact_job_segment(job)
        )
        logger.debug(
            "Resolved job artifact directory",
            extra={
                "job_id": job.job_id,
                "tool": job.tool_name,
                "date_segment": artifact_dir.parent.parent.name,
                "session_segment": artifact_dir.parent.name,
                "job_segment": artifact_dir.name,
                "artifact_dir": str(artifact_dir),
            },
        )
        return artifact_dir

    def _build_result_markdown(self, job: Job) -> str:
        response = job.result.get("response", "") if job.result else ""
        header = f"# {job.tool_name}: {job.job_id}\n\n"
        meta = []
        if job.model:
            meta.append(f"- **Model**: {job.model}")
        if job.session_id:
            meta.append(f"- **Session**: {job.session_id}")
        meta.append(f"- **Duration**: {int(job.last_updated_at - job.created_at)}s")
        tools_used = job.result.get("tools_used", []) if job.result else []
        if tools_used:
            meta.append(f"- **Tools used**: {', '.join(tools_used)}")
        files_touched = job.result.get("files_touched", []) if job.result else []
        if files_touched:
            meta.append(f"- **Files touched**: {', '.join(files_touched)}")
        meta_section = "\n".join(meta) + "\n\n" if meta else ""
        return header + meta_section + response + "\n"

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [job.to_dict() for job in self._jobs.values()]

    async def cancel_job(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job.is_terminal:
            return job
        job._cancel_event.set()
        if job.task and not job.task.done():
            job.task.cancel()
            try:
                await asyncio.wait_for(job.task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        job.status = JobStatus.CANCELLED
        job.status_message = "Cancelled by request"
        job.last_updated_at = time.time()
        logger.info("Cancelled job", extra={"job_id": job_id, "tool": job.tool_name})
        return job

    def update_job(self, job_id: str, **kwargs) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.last_updated_at = time.time()
        logger.debug(
            "Updated job",
            extra={"job_id": job_id, "updates": list(kwargs.keys()), "status": job.status.value},
        )
        return job

    def persist_result(self, job_id: str, cwd: Optional[str] = None) -> Optional[str]:
        job = self._jobs.get(job_id)
        if not job or not job.result or not job.result.get("response"):
            return None
        artifact_dir = self.build_artifact_dir(job, cwd=cwd)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        result_markdown_path = artifact_dir / "result.md"
        result_json_path = artifact_dir / "result.json"
        meta_json_path = artifact_dir / "meta.json"

        result_markdown_path.write_text(self._build_result_markdown(job), encoding="utf-8")
        result_json_path.write_text(json.dumps(job.result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        job.output_path = str(result_markdown_path)
        meta_json_path.write_text(
            json.dumps(job.build_artifact_metadata(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logger.info(
            "Persisted job result bundle",
            extra={
                "job_id": job_id,
                "tool": job.tool_name,
                "artifact_dir": str(artifact_dir),
                "output_path": str(result_markdown_path),
            },
        )
        return str(result_markdown_path)


job_manager = JobManager()
