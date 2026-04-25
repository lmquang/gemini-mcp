import json
from datetime import datetime, timezone

from gemini_mcp.jobs import ARTIFACT_SCHEMA_VERSION, SESSION_DIR_FALLBACK, ArtifactStore, RunRecord, RunStatus


def _timestamp(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def test_persist_run_writes_date_session_run_bundle(tmp_path):
    store = ArtifactStore()
    run = RunRecord(
        run_id="run-success",
        tool_name="explore",
        created_at=_timestamp(2026, 4, 18),
        last_updated_at=_timestamp(2026, 4, 18) + 12,
        status=RunStatus.COMPLETED,
        status_message="Done",
        session_id="e40acfd3-364d-4822-b8ab-487688625e35",
        model="gemini-3-flash-preview",
        reasoning_trace="First inspect files\n\nThen summarize structure",
        result={"response": "Artifact body\n", "tools_used": ["glob"], "files_touched": ["gemini_mcp/jobs.py"]},
    )

    persisted = store.persist_run(run, cwd=str(tmp_path))

    expected_dir = tmp_path / ".gemini-mcp" / "jobs" / "20260418" / run.session_id / "000000-run-success"
    assert persisted["runID"] == "run-success"
    assert persisted["outputPath"] == str(expected_dir / "result.md")
    assert persisted["reasoningTracePath"] == str(expected_dir / "reasoning.md")
    assert (expected_dir / "result.md").is_file()
    assert (expected_dir / "reasoning.md").is_file()
    assert (expected_dir / "result.json").is_file()
    assert (expected_dir / "meta.json").is_file()

    result_markdown = (expected_dir / "result.md").read_text(encoding="utf-8")
    assert "# explore:" in result_markdown
    assert "- **Session**: e40acfd3-364d-4822-b8ab-487688625e35" in result_markdown
    assert "- **Tools used**: glob" in result_markdown
    assert "Artifact body" in result_markdown

    reasoning_markdown = (expected_dir / "reasoning.md").read_text(encoding="utf-8")
    assert "# explore Reasoning Trace:" in reasoning_markdown
    assert "First inspect files" in reasoning_markdown

    result_json = json.loads((expected_dir / "result.json").read_text(encoding="utf-8"))
    assert result_json == run.result

    meta_json = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_json["schemaVersion"] == ARTIFACT_SCHEMA_VERSION
    assert meta_json["runID"] == run.run_id
    assert "jobID" not in meta_json
    assert meta_json["tool"] == "explore"
    assert meta_json["status"] == RunStatus.COMPLETED.value
    assert meta_json["sessionID"] == run.session_id
    assert meta_json["outputPath"] == persisted["outputPath"]
    assert meta_json["reasoningTracePath"] == persisted["reasoningTracePath"]


def test_persist_run_uses_no_session_fallback_directory(tmp_path):
    store = ArtifactStore()
    run = RunRecord(
        run_id="run-no-session",
        tool_name="document",
        created_at=_timestamp(2026, 4, 19),
        last_updated_at=_timestamp(2026, 4, 19) + 3,
        status=RunStatus.COMPLETED,
        status_message="Done",
        result={"response": "No session bundle"},
    )

    persisted = store.persist_run(run, cwd=str(tmp_path))

    expected_dir = tmp_path / ".gemini-mcp" / "jobs" / "20260419" / SESSION_DIR_FALLBACK / "000000-run-no-session"
    assert persisted["outputPath"] == str(expected_dir / "result.md")
    meta_json = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_json["sessionID"] is None
    assert meta_json["outputPath"] == persisted["outputPath"]
    assert meta_json["reasoningTracePath"] is None


def test_persist_run_writes_failed_artifacts_without_response_markdown(tmp_path):
    store = ArtifactStore()
    run = RunRecord(
        run_id="run-failed",
        tool_name="analyze",
        created_at=_timestamp(2026, 4, 20),
        last_updated_at=_timestamp(2026, 4, 20) + 5,
        status=RunStatus.FAILED,
        status_message="boom",
        result={"ok": False, "error": "boom"},
    )

    persisted = store.persist_run(run, cwd=str(tmp_path))

    expected_dir = tmp_path / ".gemini-mcp" / "jobs" / "20260420" / SESSION_DIR_FALLBACK / "000000-run-failed"
    assert persisted["outputPath"] is None
    assert (expected_dir / "result.json").is_file()
    assert (expected_dir / "meta.json").is_file()
    assert not (expected_dir / "result.md").exists()

    meta_json = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_json["status"] == RunStatus.FAILED.value
    assert meta_json["statusMessage"] == "boom"
    assert meta_json["outputPath"] is None


def test_persist_run_writes_cancelled_artifacts_without_response_markdown(tmp_path):
    store = ArtifactStore()
    run = RunRecord(
        run_id="run-cancelled",
        tool_name="plan",
        created_at=_timestamp(2026, 4, 21),
        last_updated_at=_timestamp(2026, 4, 21) + 4,
        status=RunStatus.CANCELLED,
        status_message="Cancelled by request",
        result={"ok": False, "error": "Cancelled by request"},
    )

    persisted = store.persist_run(run, cwd=str(tmp_path))

    expected_dir = tmp_path / ".gemini-mcp" / "jobs" / "20260421" / SESSION_DIR_FALLBACK / "000000-run-cancelled"
    assert persisted["outputPath"] is None
    assert (expected_dir / "result.json").is_file()
    assert (expected_dir / "meta.json").is_file()
    assert not (expected_dir / "result.md").exists()

    meta_json = json.loads((expected_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_json["status"] == RunStatus.CANCELLED.value
    assert meta_json["statusMessage"] == "Cancelled by request"


def test_list_runs_reads_persisted_metadata(tmp_path):
    store = ArtifactStore()
    run = RunRecord(
        run_id="run-listed",
        tool_name="explore",
        created_at=_timestamp(2026, 4, 22),
        last_updated_at=_timestamp(2026, 4, 22) + 1,
        status=RunStatus.COMPLETED,
        status_message="Done",
        result={"response": "Listed"},
    )
    store.persist_run(run, cwd=str(tmp_path))

    runs = store.list_runs(cwd=str(tmp_path))

    assert [entry["runID"] for entry in runs] == ["run-listed"]
