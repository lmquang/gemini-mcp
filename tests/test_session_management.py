from types import SimpleNamespace

from gemini_mcp.runner import (
    build_managed_session_key,
    clear_managed_session,
    get_most_recent_session_id,
    merge_session_sources,
    resolve_managed_session,
    store_managed_session,
)


def _context(session_id: str = "client-1"):
    return SimpleNamespace(session=SimpleNamespace(id=session_id))


def test_resolve_managed_session_reuses_managed_session_in_auto_mode():
    context = _context()
    cwd = "/tmp/project-auto"
    registry_key = build_managed_session_key(context, cwd)
    try:
        store_managed_session(registry_key, "managed-session-123")

        _, session_id = resolve_managed_session(context, cwd, None, "auto")

        assert session_id == "managed-session-123"
    finally:
        clear_managed_session(registry_key)


def test_resolve_managed_session_does_not_reuse_session_in_new_mode():
    context = _context()
    cwd = "/tmp/project-new"
    registry_key = build_managed_session_key(context, cwd)
    try:
        store_managed_session(registry_key, "managed-session-456")

        _, session_id = resolve_managed_session(context, cwd, None, "new")

        assert session_id is None
    finally:
        clear_managed_session(registry_key)


def test_merge_session_sources_orders_by_recency_and_prefers_persisted_metadata():
    cli_sessions = [
        {"id": "older-cli", "index": 1, "title": "Older CLI", "age": "2h"},
        {"id": "newer-cli", "index": 2, "title": "Newer CLI", "age": "1h"},
    ]
    persisted_sessions = [
        {
            "id": "newer-cli",
            "index": 99,
            "title": "Persisted Title Wins",
            "lastUpdated": "2026-04-24T08:00:00Z",
            "source": "chat_file",
        },
        {
            "id": "persisted-latest",
            "index": 3,
            "title": "Newest Persisted",
            "lastUpdated": "2026-04-24T09:00:00Z",
            "source": "chat_file",
        },
    ]

    merged = merge_session_sources(cli_sessions, persisted_sessions)

    assert [session["id"] for session in merged] == ["persisted-latest", "newer-cli", "older-cli"]
    assert merged[1]["title"] == "Persisted Title Wins"
    assert merged[1]["source"] == "chat_file"
    assert [session["index"] for session in merged] == [1, 2, 3]


def test_get_most_recent_session_id_uses_sorted_recency_not_largest_index():
    sessions = [
        {"id": "most-recent", "index": 1, "lastUpdated": "2026-04-24T09:00:00Z"},
        {"id": "oldest", "index": 2, "lastUpdated": "2026-04-24T07:00:00Z"},
    ]

    assert get_most_recent_session_id(sessions) == "most-recent"
