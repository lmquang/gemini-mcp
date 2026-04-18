from types import SimpleNamespace

from gemini_mcp.jobs import job_manager
from gemini_mcp.runner import build_managed_session_key, clear_managed_session, store_managed_session
from gemini_mcp.tools import _prepare_job_session


def _context(session_id: str = "client-1"):
    return SimpleNamespace(session=SimpleNamespace(id=session_id))


def test_prepare_job_session_reuses_managed_session_in_auto_mode():
    context = _context()
    cwd = "/tmp/project-auto"
    registry_key = build_managed_session_key(context, cwd)
    job = job_manager.create_job("explore")
    try:
        store_managed_session(registry_key, "managed-session-123")

        _prepare_job_session(job.job_id, context, cwd, None, "auto")

        prepared_job = job_manager.get_job(job.job_id)
        assert prepared_job is not None
        assert prepared_job.session_id == "managed-session-123"
    finally:
        job_manager._jobs.pop(job.job_id, None)
        clear_managed_session(registry_key)


def test_prepare_job_session_does_not_reuse_session_in_new_mode():
    context = _context()
    cwd = "/tmp/project-new"
    registry_key = build_managed_session_key(context, cwd)
    job = job_manager.create_job("analyze")
    try:
        store_managed_session(registry_key, "managed-session-456")

        _prepare_job_session(job.job_id, context, cwd, None, "new")

        prepared_job = job_manager.get_job(job.job_id)
        assert prepared_job is not None
        assert prepared_job.session_id is None
    finally:
        job_manager._jobs.pop(job.job_id, None)
        clear_managed_session(registry_key)
