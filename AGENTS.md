# AGENTS

## Runbook
- Use `uv`-based commands. The executable sources of truth are in `Makefile`, not the simpler `README` examples.
- Start the stdio server with `uv run python -m gemini_mcp.server` or `make run`.
- Start the streamable HTTP server with `uv run python -m gemini_mcp.server --transport streamable-http` or `make run-http`.
- Use `make dev` for FastMCP Inspector with hot reload. It runs `uv run fastmcp dev inspector server.py` against the standalone root `server.py`.
- `gemini-mcp` is the installed console entrypoint and resolves to `gemini_mcp.server:main`.

## Verification
- There is no repo-local lint, formatter, or typecheck config. Do not invent extra verification steps.
- Current focused test command: `pytest tests/test_parsers.py tests/test_jobs.py tests/test_session_management.py tests/test_tools_native_tasks.py tests/test_server.py`.
- Tests import the package via `tests/conftest.py`, so running the focused pytest command from repo root works without installation.
- Pytest currently emits a `pytest-asyncio` deprecation warning about `asyncio_default_fixture_loop_scope`; this is expected in the current config.

## Architecture
- `gemini_mcp/server.py` is the real package entrypoint. It sets DEBUG logging, installs signal/atexit cleanup, and runs the FastMCP app.
- Root `server.py` is only a standalone wrapper with inline script dependencies that forwards to `gemini_mcp.server:main`.
- `gemini_mcp/tools/__init__.py` is the tool surface and registration point for MCP tools.
- `gemini_mcp/runner.py` owns Gemini CLI execution, isolated home setup, retries, session reuse, and subprocess lifecycle.
- `gemini_mcp/jobs.py` owns persisted run artifacts/history only. Native FastMCP tasks are the live execution source of truth.

## Session Rules
- Session reuse is designed to be the default. Use `sessionMode="auto"` or omit it for normal follow-up calls on the same MCP connection and `cwd`.
- Use `sessionMode="new"` only when you explicitly want a fresh Gemini conversation boundary.
- Agentic tool final payloads include the final `sessionID` when Gemini reports one.

## Persistent Artifacts
- Native task run outputs are written under `.gemini-mcp/jobs/YYYYMMDD/<session-id|no-session>/HHMMSS-<run_id>/`.
- Each persisted run bundle contains `meta.json`, `result.json`, and `result.md` when a final response exists. Runs with no final response still persist `meta.json` and `result.json`.
- `outputPath` points to the bundle's `result.md`, not to the job directory.
- Legacy flat `.gemini-mcp/jobs/*.md` files may still exist from older runs; do not assume the directory contains only the new layout.

## Environment And Logging
- Runtime logs go to `/tmp/gemini-mcp.log` unless `GEMINI_LOG_FILE` overrides the path.
- The server expects the Gemini CLI to already be installed and authenticated outside this repo; the Python package does not vendor that dependency.
