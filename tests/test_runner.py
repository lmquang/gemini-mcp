from types import SimpleNamespace

import pytest

from gemini_mcp.config import SUBPROCESS_STREAM_LIMIT_BYTES
from gemini_mcp.runner import run_gemini_cli


class _FakeStdout:
    def __init__(self):
        self._lines = [b'{"type":"result","status":"success"}\n', b""]

    async def readline(self):
        return self._lines.pop(0)


class _FakeProcess:
    def __init__(self):
        self.stdout = _FakeStdout()
        self.returncode = None

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


@pytest.mark.asyncio
async def test_run_gemini_cli_uses_raised_stream_limit(monkeypatch):
    captured = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["limit"] = kwargs.get("limit")
        return _FakeProcess()

    async def fake_setup_isolated_home(approval_mode=None):
        return "/tmp/gemini-home"

    async def fake_resolve_session_reference(**kwargs):
        return None

    async def fake_discover_session_id(**kwargs):
        return None

    monkeypatch.setattr("gemini_mcp.runner.setup_isolated_home", fake_setup_isolated_home)
    monkeypatch.setattr("gemini_mcp.runner.resolve_managed_session", lambda **kwargs: ("registry", None))
    monkeypatch.setattr("gemini_mcp.runner.resolve_session_reference", fake_resolve_session_reference)
    monkeypatch.setattr("gemini_mcp.runner.discover_session_id", fake_discover_session_id)
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    result = await run_gemini_cli("hello", mode="inspect", model="gemini-3-flash-preview", cwd="/tmp")

    assert result["ok"] is True
    assert captured["limit"] == SUBPROCESS_STREAM_LIMIT_BYTES
