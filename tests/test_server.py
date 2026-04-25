from types import SimpleNamespace

from gemini_mcp import server


def test_cleanup_kills_active_processes():
    killed = []

    class Process:
        def __init__(self, name: str):
            self.name = name

        def kill(self):
            killed.append(self.name)

    original = set(server.active_processes)
    server.active_processes.clear()
    server.active_processes.update({Process("a"), Process("b")})

    try:
        server.cleanup()
    finally:
        server.active_processes.clear()
        server.active_processes.update(original)

    assert sorted(killed) == ["a", "b"]
    assert not server.active_processes


def test_main_runs_stdio_transport(monkeypatch):
    calls = []

    monkeypatch.setattr(server, "configure_logging", lambda: calls.append(("logging", None)))
    monkeypatch.setattr(server, "install_signal_handlers", lambda: calls.append(("signals", None)))
    monkeypatch.setattr(server, "mcp", SimpleNamespace(run=lambda **kwargs: calls.append(("run", kwargs))))

    server.main([])

    assert calls == [("logging", None), ("signals", None), ("run", {})]


def test_main_runs_streamable_http_transport(monkeypatch):
    calls = []

    monkeypatch.setattr(server, "configure_logging", lambda: calls.append(("logging", None)))
    monkeypatch.setattr(server, "install_signal_handlers", lambda: calls.append(("signals", None)))
    monkeypatch.setattr(server, "mcp", SimpleNamespace(run=lambda **kwargs: calls.append(("run", kwargs))))

    server.main(["--transport", "streamable-http", "--host", "127.0.0.1", "--port", "9000"])

    assert calls == [
        ("logging", None),
        ("signals", None),
        ("run", {"transport": "streamable-http", "host": "127.0.0.1", "port": 9000}),
    ]
