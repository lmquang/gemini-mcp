import json
from datetime import datetime, timezone

from gemini_mcp.jobs import ARTIFACT_SCHEMA_VERSION, SESSION_DIR_FALLBACK, JobManager, JobStatus


def _timestamp(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def test_persist_result_writes_date_session_job_bundle(tmp_path):
    manager = JobManager()
    job = manager.create_job("explore")
    job.created_at = _timestamp(2026, 4, 18)
    job.last_updated_at = job.created_at + 12
    job.status = JobStatus.COMPLETED
    job.status_message = "Done"
    job.session_id = "e40acfd3-364d-4822-b8ab-487688625e35"
    job.model = "gemini-3-flash-preview"
    job.reasoning_trace = "First inspect files\n\nThen summarize structure"
    job.result = {
        "response": "Artifact body\n",
        "tools_used": ["glob"],
        "files_touched": ["gemini_mcp/jobs.py"],
    }

    output_path = manager.persist_result(job.job_id, cwd=str(tmp_path))

    expected_dir = tmp_path / ".gemini-mcp" / "jobs" / "20260418" / job.session_id / f"000000-{job.job_id}"
    assert output_path == str(expected_dir / "result.md")
    assert job.output_path == output_path
    assert job.reasoning_trace_path == str(expected_dir / "reasoning.md")
    assert (expected_dir / "result.md").is_file()
    assert (expected_dir / "reasoning.md").is_file()
    assert (expected_dir / "result.json").is_file()
    assert (expected_dir / "meta.json").is_file()

    result_markdown = (expected_dir / "result.md").read_text(encoding="utf-8")
    assert "# explore:" in result_markdown
    assert "- **Session**: e40acfd3-364d-4822-b8ab-487688625e35" in result_markdown
    assert "- **Tools used**: glob" in result_markdown
    assert "Artifact body" in result_markdown
    assert "Reasoning Trace" not in result_markdown

    reasoning_markdown = (expected_dir / "reasoning.md").read_text(encoding="utf-8")
    assert "# explore Reasoning Trace:" in reasoning_markdown
    assert "First inspect files" in reasoning_markdown
    assert "Then summarize structure" in reasoning_markdown

    result_json = json.loads((expected_dir / "result.json").read_text(encoding="utf-8"))
    assert result_json == job.result

    meta_json = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_json["schemaVersion"] == ARTIFACT_SCHEMA_VERSION
    assert meta_json["jobID"] == job.job_id
    assert meta_json["tool"] == "explore"
    assert meta_json["status"] == JobStatus.COMPLETED.value
    assert meta_json["sessionID"] == job.session_id
    assert meta_json["outputPath"] == output_path
    assert meta_json["reasoningTracePath"] == str(expected_dir / "reasoning.md")


def test_persist_result_uses_no_session_fallback_directory(tmp_path):
    manager = JobManager()
    job = manager.create_job("document")
    job.created_at = _timestamp(2026, 4, 19)
    job.last_updated_at = job.created_at + 3
    job.status = JobStatus.COMPLETED
    job.status_message = "Done"
    job.result = {"response": "No session bundle"}

    output_path = manager.persist_result(job.job_id, cwd=str(tmp_path))

    expected_dir = tmp_path / ".gemini-mcp" / "jobs" / "20260419" / SESSION_DIR_FALLBACK / f"000000-{job.job_id}"
    assert output_path == str(expected_dir / "result.md")
    meta_json = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_json["sessionID"] is None
    assert meta_json["outputPath"] == output_path
    assert meta_json["reasoningTracePath"] is None
