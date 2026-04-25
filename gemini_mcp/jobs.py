"""Artifact persistence for Gemini tool runs."""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("gemini-mcp")
ARTIFACT_SCHEMA_VERSION = 2
SESSION_DIR_FALLBACK = "no-session"
SESSION_DIR_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunRecord:
    run_id: str
    tool_name: str
    status: RunStatus = RunStatus.COMPLETED
    status_message: str = ""
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    result: Optional[dict] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None
    reasoning_trace_path: Optional[str] = None
    reasoning_trace: Optional[str] = None

    def to_dict(self) -> dict:
        payload = {
            "runID": self.run_id,
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
        if self.reasoning_trace_path:
            payload["reasoningTracePath"] = self.reasoning_trace_path
        return payload

    def build_artifact_metadata(self) -> dict[str, Any]:
        return {
            "schemaVersion": ARTIFACT_SCHEMA_VERSION,
            "runID": self.run_id,
            "tool": self.tool_name,
            "status": self.status.value,
            "statusMessage": self.status_message,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at,
            "durationSeconds": int(self.last_updated_at - self.created_at),
            "sessionID": self.session_id,
            "model": self.model,
            "outputPath": self.output_path,
            "reasoningTracePath": self.reasoning_trace_path,
        }


class ArtifactStore:
    JOB_OUTPUT_DIR = ".gemini-mcp/jobs"

    def create_run(self, tool_name: str) -> RunRecord:
        run_id = uuid.uuid4().hex[:16]
        run = RunRecord(run_id=run_id, tool_name=tool_name)
        logger.debug("Created run record", extra={"run_id": run_id, "tool": tool_name})
        return run

    def _artifact_date_segment(self, created_at: float) -> str:
        return datetime.fromtimestamp(created_at, tz=timezone.utc).strftime("%Y%m%d")

    def _artifact_run_segment(self, run: RunRecord) -> str:
        time_prefix = datetime.fromtimestamp(run.created_at, tz=timezone.utc).strftime("%H%M%S")
        return f"{time_prefix}-{run.run_id}"

    def _artifact_session_segment(self, session_id: Optional[str]) -> str:
        if not session_id or not session_id.strip():
            return SESSION_DIR_FALLBACK
        sanitized = SESSION_DIR_SANITIZE_RE.sub("-", session_id.strip()).strip(".-")
        return sanitized or SESSION_DIR_FALLBACK

    def build_artifact_dir(self, run: RunRecord, cwd: Optional[str] = None) -> Path:
        base = Path(cwd) if cwd else Path.cwd()
        artifact_dir = (
            base
            / self.JOB_OUTPUT_DIR
            / self._artifact_date_segment(run.created_at)
            / self._artifact_session_segment(run.session_id)
            / self._artifact_run_segment(run)
        )
        logger.debug(
            "Resolved run artifact directory",
            extra={
                "run_id": run.run_id,
                "tool": run.tool_name,
                "date_segment": artifact_dir.parent.parent.name,
                "session_segment": artifact_dir.parent.name,
                "run_segment": artifact_dir.name,
                "artifact_dir": str(artifact_dir),
            },
        )
        return artifact_dir

    def _build_result_markdown(self, run: RunRecord) -> str:
        response = run.result.get("response", "") if run.result else ""
        header = f"# {run.tool_name}: {run.run_id}\n\n"
        meta = []
        if run.model:
            meta.append(f"- **Model**: {run.model}")
        if run.session_id:
            meta.append(f"- **Session**: {run.session_id}")
        meta.append(f"- **Duration**: {int(run.last_updated_at - run.created_at)}s")
        tools_used = run.result.get("tools_used", []) if run.result else []
        if tools_used:
            meta.append(f"- **Tools used**: {', '.join(tools_used)}")
        files_touched = run.result.get("files_touched", []) if run.result else []
        if files_touched:
            meta.append(f"- **Files touched**: {', '.join(files_touched)}")
        meta_section = "\n".join(meta) + "\n\n" if meta else ""
        return "".join([header, meta_section, "## Final Answer\n\n", response, "\n"])

    def _build_reasoning_markdown(self, run: RunRecord) -> str:
        reasoning_trace = (run.reasoning_trace or "").strip()
        if not reasoning_trace:
            return ""
        header = f"# {run.tool_name} Reasoning Trace: {run.run_id}\n\n"
        meta = []
        if run.model:
            meta.append(f"- **Model**: {run.model}")
        if run.session_id:
            meta.append(f"- **Session**: {run.session_id}")
        meta.append(f"- **Duration**: {int(run.last_updated_at - run.created_at)}s")
        meta_section = "\n".join(meta) + "\n\n" if meta else ""
        return "".join([header, meta_section, reasoning_trace, "\n"])

    def persist_run(self, run: RunRecord, cwd: Optional[str] = None) -> dict:
        artifact_dir = self.build_artifact_dir(run, cwd=cwd)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        result_markdown_path = artifact_dir / "result.md"
        reasoning_markdown_path = artifact_dir / "reasoning.md"
        result_json_path = artifact_dir / "result.json"
        meta_json_path = artifact_dir / "meta.json"

        response = run.result.get("response") if run.result else None
        if response:
            result_markdown_path.write_text(self._build_result_markdown(run), encoding="utf-8")
            run.output_path = str(result_markdown_path)
        else:
            run.output_path = None

        if run.result is not None:
            result_json_path.write_text(json.dumps(run.result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if run.reasoning_trace:
            reasoning_markdown_path.write_text(self._build_reasoning_markdown(run), encoding="utf-8")
            run.reasoning_trace_path = str(reasoning_markdown_path)
        else:
            run.reasoning_trace_path = None

        meta_json_path.write_text(
            json.dumps(run.build_artifact_metadata(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logger.info(
            "Persisted run artifact bundle",
            extra={
                "run_id": run.run_id,
                "tool": run.tool_name,
                "artifact_dir": str(artifact_dir),
                "output_path": run.output_path,
            },
        )
        return {
            "runID": run.run_id,
            "outputPath": run.output_path,
            "reasoningTracePath": run.reasoning_trace_path,
            "artifactDir": str(artifact_dir),
        }

    def list_runs(self, cwd: Optional[str] = None, limit: int = 20) -> list[dict]:
        base = Path(cwd) if cwd else Path.cwd()
        jobs_dir = base / self.JOB_OUTPUT_DIR
        if not jobs_dir.exists():
            return []
        runs = []
        for meta_path in jobs_dir.glob("*/*/*/meta.json"):
            try:
                runs.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                logger.debug("Failed to read run metadata", extra={"path": str(meta_path)}, exc_info=True)
        runs.sort(key=lambda entry: entry.get("createdAt", 0), reverse=True)
        return runs[:limit]


artifact_store = ArtifactStore()
