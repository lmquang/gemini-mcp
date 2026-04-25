import json
import asyncio
from types import SimpleNamespace

import pytest

from gemini_mcp import tools


class RecordingContext:
    def __init__(self, session_id: str = "client-native"):
        self.session = SimpleNamespace(id=session_id)
        self.progress_messages = []
        self.progress_values = []
        self.info_messages = []

    async def report_progress(self, progress: float, total=None, message: str | None = None):
        self.progress_values.append(progress)
        self.progress_messages.append(message)

    async def info(self, message: str):
        self.info_messages.append(message)


@pytest.mark.asyncio
async def test_analyze_returns_final_payload_with_run_artifacts(monkeypatch, tmp_path):
    async def fake_run_agentic_tool(*args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            await progress_callback("fake runner progress")
        return json.dumps(
            {
                "ok": True,
                "response": "review complete",
                "reasoning_trace": "checked files",
                "tools_used": ["read_file"],
                "files_touched": ["gemini_mcp/tools/__init__.py"],
                "model": "gemini-test",
                "sessionID": "session-test",
            }
        )

    monkeypatch.setattr(tools, "run_agentic_tool", fake_run_agentic_tool)

    payload = json.loads(await tools.analyze("review", RecordingContext(), cwd=str(tmp_path)))

    assert payload["ok"] is True
    assert payload["response"] == "review complete"
    assert payload["runID"]
    assert payload["outputPath"].endswith("result.md")
    assert payload["reasoningTracePath"].endswith("reasoning.md")
    assert "jobID" not in payload


@pytest.mark.asyncio
async def test_agentic_tool_reports_progress(monkeypatch, tmp_path):
    async def fake_run_agentic_tool(*args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            await progress_callback("runner stage")
        return json.dumps({"ok": True, "response": "done", "model": "gemini-test"})

    context = RecordingContext()
    monkeypatch.setattr(tools, "run_agentic_tool", fake_run_agentic_tool)

    await tools.explore("map", context, cwd=str(tmp_path))

    assert "Preparing explore run" in context.progress_messages
    assert "runner stage" in context.progress_messages
    assert "Persisting explore artifact" in context.progress_messages
    assert context.progress_values == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_agentic_tool_failure_returns_failed_payload_and_artifact(monkeypatch, tmp_path):
    async def fake_run_agentic_tool(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "run_agentic_tool", fake_run_agentic_tool)

    payload = json.loads(await tools.plan("plan", RecordingContext(), cwd=str(tmp_path)))

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error"] == "boom"
    assert payload["runID"]
    assert payload["outputPath"] is None


@pytest.mark.asyncio
async def test_native_task_cancellation_persists_cancelled_artifact(tmp_path):
    async def cancelled_run(progress_callback):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await tools._execute_agentic_run("analyze", RecordingContext(), str(tmp_path), cancelled_run)

    runs = tools.artifact_store.list_runs(cwd=str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["status"] == "cancelled"
    assert runs[0]["statusMessage"] == "Cancelled by request"
    assert runs[0]["runID"]
