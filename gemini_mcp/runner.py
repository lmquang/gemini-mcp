"""Gemini CLI execution runner."""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import asyncio

from gemini_mcp.config import DEFAULT_READ_TIMEOUT, DEFAULT_TIMEOUT_SECONDS, HEARTBEAT_INTERVAL_SECONDS, MODEL_TIERS
from gemini_mcp.exceptions import GeminiAuthError, GeminiError, GeminiRateLimitError, GeminiTimeoutError
from gemini_mcp.parsers import extract_session_id, parse_and_summarize, parse_stream_json

try:
    import aiofiles

    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

logger = logging.getLogger("gemini-mcp")

active_processes: set = set()
_home_cache: dict[str, str] = {}
_managed_sessions: dict[str, str] = {}
SESSION_LIST_ENTRY_RE = re.compile(r"^\s*(?P<index>\d+)\.\s+(?P<title>.*?)\s+\((?P<age>.*?)\)\s+\[(?P<id>[^\]]+)\]\s*$")
UUID_LIKE_RE = re.compile(r"^[0-9a-fA-F-]{8,}$")


def validate_session_mode(session_mode: str) -> str:
    """Validate and normalize session mode."""
    normalized = session_mode or "auto"
    if normalized not in {"auto", "new"}:
        raise GeminiError(f"Unsupported sessionMode '{session_mode}'. Expected 'auto' or 'new'.")
    return normalized


def get_context_session_identifier(context: Optional["Context"]) -> str:
    """Build a stable identifier for the current MCP connection."""
    if not context:
        return "detached"
    session = getattr(context, "session", None)
    if session is None:
        return "detached"
    for attr in ("id", "session_id", "client_id"):
        value = getattr(session, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return f"session-{id(session)}"


def build_managed_session_key(context: Optional["Context"], cwd: Optional[str]) -> str:
    """Build the registry key for the global managed Gemini session."""
    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    return f"{get_context_session_identifier(context)}::{resolved_cwd}"


def resolve_managed_session(
    context: Optional["Context"],
    cwd: Optional[str],
    session: Optional[str],
    session_mode: str,
) -> tuple[str, Optional[str]]:
    """Resolve the effective Gemini session for this tool call."""
    normalized_mode = validate_session_mode(session_mode)
    registry_key = build_managed_session_key(context, cwd)
    if session:
        logger.debug(
            "Using explicit Gemini session override",
            extra={"registry_key": registry_key, "session_id": session, "session_mode": normalized_mode},
        )
        return registry_key, session
    if normalized_mode == "new":
        logger.debug("Forcing new managed Gemini session", extra={"registry_key": registry_key})
        return registry_key, None
    stored_session = _managed_sessions.get(registry_key)
    logger.debug(
        "Resolved managed Gemini session",
        extra={"registry_key": registry_key, "stored_session_id": stored_session, "session_mode": normalized_mode},
    )
    return registry_key, stored_session


def store_managed_session(registry_key: str, session_id: str):
    """Persist the active Gemini session for seamless resume."""
    _managed_sessions[registry_key] = session_id
    logger.debug("Stored managed Gemini session", extra={"registry_key": registry_key, "session_id": session_id})


def clear_managed_session(registry_key: str):
    """Remove a managed Gemini session mapping."""
    removed = _managed_sessions.pop(registry_key, None)
    logger.debug("Cleared managed Gemini session", extra={"registry_key": registry_key, "session_id": removed})


def build_isolated_settings(approval_mode: Optional[str] = None) -> dict:
    """Build isolated Gemini settings dict."""
    settings = {
        "skills": {"enabled": False},
        "hooksConfig": {"enabled": False},
        "tools": {"enableHooks": False},
        "admin": {"mcp": {"enabled": False}, "skills": {"enabled": False}, "extensions": {"enabled": False}},
        "context": {"includeDirectoryTree": False, "loadMemoryFromIncludeDirectories": False},
        "security": {"folderTrust": {"enabled": False}, "auth": {"selectedType": "oauth-personal"}},
        "ui": {"hideBanner": True, "hideContextSummary": True, "showCompatibilityWarnings": False, "loadingPhrases": "off"},
    }
    if approval_mode:
        settings["general"] = {"defaultApprovalMode": approval_mode}
    return settings


async def _copy_file_async(src: Path, dst: Path):
    if HAS_AIOFILES:
        async with aiofiles.open(src, "rb") as f:
            content = await f.read()
        async with aiofiles.open(dst, "wb") as f:
            await f.write(content)
    else:
        shutil.copy2(src, dst)


async def _write_json_async(path: Path, data: dict):
    content = json.dumps(data, indent=2)
    if HAS_AIOFILES:
        async with aiofiles.open(path, "w") as f:
            await f.write(content)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


async def _rmtree_async(path: str):
    if HAS_AIOFILES:
        await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


async def _symlink_or_copy_path(src: Path, dst: Path):
    """Expose Gemini CLI state into the isolated home."""
    if dst.exists() or dst.is_symlink():
        return
    if not src.exists():
        return
    try:
        if src.is_dir():
            os.symlink(src, dst, target_is_directory=True)
        else:
            os.symlink(src, dst)
        logger.debug("Linked Gemini state into isolated home", extra={"source": str(src), "destination": str(dst)})
    except OSError:
        if src.is_dir():
            await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)
        else:
            await _copy_file_async(src, dst)
        logger.debug("Copied Gemini state into isolated home", extra={"source": str(src), "destination": str(dst)})


def parse_session_listing(stdout: str) -> list[dict]:
    """Parse `gemini --list-sessions` text output into structured entries."""
    sessions = []
    for line in stdout.splitlines():
        match = SESSION_LIST_ENTRY_RE.match(line)
        if not match:
            continue
        sessions.append(
            {
                "index": int(match.group("index")),
                "title": match.group("title"),
                "age": match.group("age"),
                "id": match.group("id"),
            }
        )
    return sessions


def _read_json_file(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.debug("Failed to read JSON file", extra={"path": str(path)}, exc_info=True)
        return None


def _truncate_session_title(title: str, limit: int = 80) -> str:
    cleaned = " ".join(title.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _session_sort_key(entry: dict) -> str:
    return entry.get("lastUpdated") or entry.get("startTime") or f"{entry.get('index', 0):08d}"


def _project_chat_dirs(source_home: Path, cwd: Optional[str]) -> list[Path]:
    target_cwd = os.path.abspath(cwd or os.getcwd())
    tmp_root = source_home / "tmp"
    if not tmp_root.exists():
        return []
    matches = []
    for project_dir in tmp_root.iterdir():
        project_root_file = project_dir / ".project_root"
        chats_dir = project_dir / "chats"
        if not project_root_file.exists() or not chats_dir.exists():
            continue
        try:
            project_root = project_root_file.read_text().strip()
        except Exception:
            logger.debug("Failed to read Gemini project root", extra={"path": str(project_root_file)}, exc_info=True)
            continue
        if os.path.abspath(project_root) == target_cwd:
            matches.append(chats_dir)
    return matches


def load_persisted_chat_sessions(source_home: Path, cwd: Optional[str]) -> list[dict]:
    """Load persisted Gemini sessions directly from chat files for the current project."""
    sessions = []
    for chats_dir in _project_chat_dirs(source_home, cwd):
        for chat_file in sorted(chats_dir.glob("session-*.json")):
            payload = _read_json_file(chat_file)
            if not payload:
                continue
            session_id = payload.get("sessionId")
            if not isinstance(session_id, str) or not session_id.strip():
                continue
            messages = payload.get("messages") or []
            first_user_text = None
            for message in messages:
                if message.get("type") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    for part in content:
                        text = part.get("text") if isinstance(part, dict) else None
                        if isinstance(text, str) and text.strip():
                            first_user_text = text.strip()
                            break
                elif isinstance(content, str) and content.strip():
                    first_user_text = content.strip()
                if first_user_text:
                    break
            sessions.append(
                {
                    "index": 0,
                    "title": _truncate_session_title(first_user_text or session_id),
                    "age": payload.get("lastUpdated") or payload.get("startTime") or "unknown",
                    "id": session_id,
                    "source": "chat_file",
                    "startTime": payload.get("startTime"),
                    "lastUpdated": payload.get("lastUpdated"),
                    "kind": payload.get("kind"),
                    "path": str(chat_file),
                }
            )
    sessions.sort(key=lambda entry: entry.get("lastUpdated") or entry.get("startTime") or "", reverse=True)
    for idx, session in enumerate(sessions, start=1):
        session["index"] = idx
    return sessions


def merge_session_sources(cli_sessions: list[dict], persisted_sessions: list[dict]) -> list[dict]:
    """Merge CLI-listed and chat-file sessions, preferring persisted metadata."""
    merged: dict[str, dict] = {}
    for session in cli_sessions:
        merged[session["id"]] = {**session, "source": "cli"}
    for session in persisted_sessions:
        existing = merged.get(session["id"], {})
        merged[session["id"]] = {**existing, **session}
    ordered = sorted(
        merged.values(),
        key=_session_sort_key,
        reverse=True,
    )
    for idx, session in enumerate(ordered, start=1):
        session["index"] = idx
    return ordered


def resolve_session_reference_value(session: Optional[str], sessions: list[dict]) -> Optional[str]:
    """Normalize a session reference into the Gemini CLI resume value."""
    if not session:
        return None
    if session == "latest" or session.isdigit():
        return session
    if UUID_LIKE_RE.match(session):
        return session
    for entry in sessions:
        if entry["id"] == session:
            return entry["id"]
    raise GeminiError(f"Session '{session}' was not found for the current project")


def get_most_recent_session_id(sessions: list[dict]) -> Optional[str]:
    """Return the newest Gemini session ID from a parsed session listing."""
    if not sessions:
        return None
    return sessions[0]["id"]


async def list_gemini_sessions(cwd: Optional[str], approval_mode: Optional[str] = None) -> list[dict]:
    """List Gemini CLI sessions for the current project."""
    isolated_home = await setup_isolated_home(approval_mode)
    env = os.environ.copy()
    env["HOME"] = isolated_home
    env["PYTHONUNBUFFERED"] = "1"
    env["PAGER"] = "cat"
    command = ["gemini", "--list-sessions"]
    logger.debug("Listing Gemini sessions", extra={"cwd": cwd or os.getcwd(), "home": isolated_home})
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd or os.getcwd(),
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    stdout, _ = await process.communicate()
    output = stdout.decode(errors="replace")
    if process.returncode != 0:
        logger.debug(
            "Gemini session listing failed",
            extra={"cwd": cwd or os.getcwd(), "returncode": process.returncode, "output": output[-1000:]},
        )
        raise GeminiError(output.strip() or "Failed to list Gemini sessions", exit_code=process.returncode, stdout=output)
    cli_sessions = parse_session_listing(output)
    persisted_sessions = load_persisted_chat_sessions(Path.home() / ".gemini", cwd)
    sessions = merge_session_sources(cli_sessions, persisted_sessions)
    logger.debug(
        "Parsed Gemini sessions",
        extra={
            "cwd": cwd or os.getcwd(),
            "cli_session_count": len(cli_sessions),
            "persisted_session_count": len(persisted_sessions),
            "merged_session_count": len(sessions),
        },
    )
    return sessions


async def resolve_session_reference(session: Optional[str], cwd: Optional[str], approval_mode: Optional[str] = None) -> Optional[str]:
    """Resolve Gemini session IDs to the CLI's numeric resume format."""
    if not session or session == "latest" or session.isdigit() or UUID_LIKE_RE.match(session):
        return session
    sessions = await list_gemini_sessions(cwd=cwd, approval_mode=approval_mode)
    resolved_session = resolve_session_reference_value(session, sessions)
    logger.debug(
        "Resolved Gemini session reference",
        extra={"cwd": cwd or os.getcwd(), "requested_session": session, "resolved_session": resolved_session},
    )
    return resolved_session


async def discover_session_id(
    stdout: str,
    output_format: str,
    cwd: Optional[str],
    approval_mode: Optional[str],
    requested_session_id: Optional[str],
) -> Optional[str]:
    """Determine the stable Gemini session ID for the completed run."""
    if requested_session_id and requested_session_id != "latest" and not requested_session_id.isdigit():
        return requested_session_id

    session_id = None
    if output_format == "json":
        json_start = -1
        for i, char in enumerate(stdout):
            if char in "[{":
                json_start = i
                break
        if json_start >= 0:
            try:
                session_id = extract_session_id(json.loads(stdout[json_start:]))
            except json.JSONDecodeError:
                session_id = None
    else:
        session_id = extract_session_id(parse_stream_json(stdout))

    if session_id:
        return session_id

    sessions = await list_gemini_sessions(cwd=cwd, approval_mode=approval_mode)
    latest_session_id = get_most_recent_session_id(sessions)
    logger.debug(
        "Inferred latest Gemini session after execution",
        extra={"cwd": cwd or os.getcwd(), "session_id": latest_session_id, "session_count": len(sessions)},
    )
    return latest_session_id


async def setup_isolated_home(approval_mode: Optional[str] = None) -> str:
    """Create isolated Gemini home directory with auth files."""
    cache_key = approval_mode or "__default__"
    if cache_key in _home_cache and Path(_home_cache[cache_key]).exists():
        logger.debug("Reusing cached isolated home: %s", _home_cache[cache_key])
        return _home_cache[cache_key]

    source_home = Path.home() / ".gemini"
    temp_dir = tempfile.mkdtemp(prefix="gemini-mcp-")
    try:
        isolated_gemini_home = Path(temp_dir) / ".gemini"
        isolated_gemini_home.mkdir(parents=True, exist_ok=True)

        auth_files = ["google_accounts.json", "google_accounts.json.bak", "oauth_cred_gemini.json", "oauth_creds.json"]
        if source_home.exists():
            for file_name in auth_files:
                source = source_home / file_name
                if source.exists():
                    try:
                        await _copy_file_async(source, isolated_gemini_home / file_name)
                        logger.debug("Copied auth file: %s", file_name)
                    except Exception as e:
                        logger.warning("Failed to copy auth file %s: %s", file_name, e)

            await _symlink_or_copy_path(source_home / "projects.json", isolated_gemini_home / "projects.json")
            await _symlink_or_copy_path(source_home / "tmp", isolated_gemini_home / "tmp")
            await _symlink_or_copy_path(source_home / "history", isolated_gemini_home / "history")
            await _symlink_or_copy_path(source_home / "commands", isolated_gemini_home / "commands")

        await _write_json_async(isolated_gemini_home / "settings.json", build_isolated_settings(approval_mode))

        _home_cache[cache_key] = temp_dir
        logger.debug("Created isolated home: %s (approval_mode=%s)", temp_dir, approval_mode)
        return temp_dir
    except Exception as e:
        logger.exception("Failed to setup isolated home")
        await _rmtree_async(temp_dir)
        raise GeminiError(f"Failed to setup execution environment: {e}") from e


def build_gemini_command(
    mode: str,
    model: str,
    output_format: str,
    approval_mode: Optional[str],
    include_directories: Optional[list[str]],
    include_files: Optional[list[str]],
    session: Optional[str],
    sandbox: bool,
) -> tuple[list, Optional[str]]:
    """Build the Gemini CLI command and optional file hints."""
    cmd = ["gemini", "--output-format", output_format, "--model", model]
    if session:
        cmd.extend(["--resume", session])
    if approval_mode:
        cmd.extend(["--approval-mode", approval_mode])
    if sandbox:
        cmd.append("--sandbox")
    if include_directories:
        for directory in include_directories:
            cmd.extend(["--include-directories", directory])
    if include_files:
        file_hints = "Focus on these specific files: " + ", ".join(include_files)
    else:
        file_hints = None
    return cmd, file_hints


async def run_gemini_cli(
    prompt: str,
    mode: str,
    model: str,
    context: Optional["Context"] = None,
    cwd: Optional[str] = None,
    include_directories: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    approval_mode: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    session: Optional[str] = None,
    session_mode: str = "auto",
    sandbox: Optional[bool] = None,
    progress_callback=None,
) -> dict:
    """Execute Gemini CLI with retries, isolated env, and heartbeat."""
    max_retries = 3
    retry_delay = 2.0
    last_res = {"ok": False, "error": "Failed after max retries"}

    for attempt in range(max_retries):
        isolated_home = None
        process = None
        heartbeat_task = None
        try:
            isolated_home = await setup_isolated_home(approval_mode)
            env = os.environ.copy()
            env["HOME"] = isolated_home
            env["PYTHONUNBUFFERED"] = "1"
            env["PAGER"] = "cat"

            output_format = "stream-json" if mode in ["inspect", "edit"] else "json"

            if sandbox is None:
                sandbox = mode in ["inspect", "edit"]
            if not approval_mode:
                if mode == "inspect":
                    approval_mode = "plan"
                elif mode == "edit":
                    approval_mode = "auto_edit"
                elif mode == "chat":
                    approval_mode = "yolo"

            registry_key, managed_session = resolve_managed_session(context=context, cwd=cwd, session=session, session_mode=session_mode)
            resolved_session = await resolve_session_reference(session=managed_session, cwd=cwd, approval_mode=approval_mode)

            cmd, file_hints = build_gemini_command(
                mode=mode,
                model=model,
                output_format=output_format,
                approval_mode=approval_mode,
                include_directories=include_directories,
                include_files=include_files,
                session=resolved_session,
                sandbox=sandbox,
            )

            full_prompt = ""
            if file_hints:
                full_prompt += file_hints + "\n\n"
            full_prompt += prompt

            logger.debug(
                "Executing Gemini CLI (attempt %d/%d)",
                attempt + 1,
                max_retries,
                extra={
                    "mode": mode,
                    "model": model,
                    "cwd": cwd or os.getcwd(),
                    "timeout": timeout,
                    "read_timeout": read_timeout,
                    "include_directories": include_directories or [],
                    "include_files": include_files or [],
                    "output_format": output_format,
                    "approval_mode": approval_mode,
                    "sandbox": sandbox,
                    "session": session,
                    "managed_session": managed_session,
                    "session_mode": session_mode,
                    "registry_key": registry_key,
                    "resolved_session": resolved_session,
                },
            )

            process = await asyncio.create_subprocess_exec(
                *cmd,
                "--prompt", full_prompt,
                cwd=cwd or os.getcwd(),
                stdin=subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            active_processes.add(process)

            stdout_chunks = []
            start_time = time.time()
            heartbeat_state = {"sequence": 0}
            stream_state = {"saw_init": False, "saw_assistant": False, "completed": False}

            await _notify_progress(progress_callback, f"Starting Gemini CLI with {model}")
            await _notify_context(context, f"Starting Gemini CLI with {model}...")
            await _notify_context(context, "Gemini process started, waiting for first output...")

            async def heartbeat_loop():
                while process.returncode is None:
                    await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                    if process.returncode is not None:
                        break
                    heartbeat_state["sequence"] += 1
                    elapsed = int(time.time() - start_time)
                    await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], elapsed, "running", f"Gemini still running ({elapsed}s elapsed)")

            heartbeat_task = asyncio.create_task(heartbeat_loop())
            await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], 0, "started", f"Starting Gemini CLI with {model}")

            while True:
                try:
                    if read_timeout > 0:
                        line_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=read_timeout)
                    else:
                        line_bytes = await process.stdout.readline()
                    if not line_bytes:
                        break

                    line = line_bytes.decode(errors="replace")
                    stdout_chunks.append(line)

                    if context:
                        events = parse_stream_json(line)
                        for event in events:
                            event_type = event.get("type")
                            if event_type == "init" and not stream_state["saw_init"]:
                                stream_state["saw_init"] = True
                                heartbeat_state["sequence"] += 1
                                await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], int(time.time() - start_time), "initialized", "Gemini initialized")
                                await _notify_context(context, "Gemini initialized and is processing the request...")
                            elif event_type == "tool_use":
                                heartbeat_state["sequence"] += 1
                                await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], int(time.time() - start_time), "using tools", f"Gemini is using {event.get('tool_name', 'a tool')}")
                                await _notify_context(context, f"Gemini is using {event.get('tool_name', 'a tool')}...")
                            elif event_type == "message" and event.get("role") == "assistant" and not stream_state["saw_assistant"]:
                                stream_state["saw_assistant"] = True
                                heartbeat_state["sequence"] += 1
                                await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], int(time.time() - start_time), "responding", "Gemini started responding")
                                await _notify_context(context, "Gemini started responding.")
                            elif event_type == "result" and not stream_state["completed"]:
                                stream_state["completed"] = True
                                heartbeat_state["sequence"] += 1
                                await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], int(time.time() - start_time), "completed", "Gemini run completed")
                                await _notify_context(context, "Gemini run completed.")

                        if "429" in line:
                            heartbeat_state["sequence"] += 1
                            await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], int(time.time() - start_time), "retrying", "Rate limit hit, retrying")
                            await _notify_context(context, "Rate limit hit, retrying...")

                except asyncio.TimeoutError:
                    elapsed = int(time.time() - start_time)
                    heartbeat_state["sequence"] += 1
                    await _emit_progress_update(context, progress_callback, heartbeat_state["sequence"], elapsed, "running", f"Gemini still running ({elapsed}s elapsed)")
                    await _notify_context(context, f"Still working... ({elapsed}s elapsed)")

                    if timeout > 0 and time.time() - start_time > timeout:
                        logger.error("Gemini task timed out", extra={"mode": mode, "model": model, "timeout": timeout})
                        process.kill()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.debug("Timed out waiting for Gemini process to exit after kill", extra={"mode": mode, "model": model})
                        timed_out_stdout = "".join(stdout_chunks)
                        active_session_id = await discover_session_id(
                            stdout=timed_out_stdout,
                            output_format=output_format,
                            cwd=cwd,
                            approval_mode=approval_mode,
                            requested_session_id=managed_session,
                        )
                        if active_session_id:
                            store_managed_session(registry_key, active_session_id)
                        raise GeminiTimeoutError(
                            f"Gemini CLI timed out after {timeout}s",
                            exit_code=-9,
                            stdout=timed_out_stdout,
                            session_id=active_session_id,
                        )

            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            stdout = "".join(stdout_chunks)

            is_ok = process.returncode == 0 and (stdout.strip() != "" or "result" in stdout)

            output_lower = stdout.lower()
            is_transient = any(err in output_lower for err in ["429", "resource_exhausted", "capacity"])

            if is_transient and not is_ok:
                logger.warning("Transient error detected (attempt %d). Retrying in %.1fs...", attempt + 1, retry_delay)
                await _notify_context(context, f"Rate limit or capacity issue, retrying in {retry_delay:.0f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue

            error_text = stdout if not is_ok else None

            if not is_ok:
                err_cls = type(GeminiError(error_text or "", exit_code=process.returncode, stdout=stdout))
                if any(kw in stdout.lower() for kw in ["auth", "credential", "oauth", "unauthorized", "login"]):
                    err_cls = GeminiAuthError
                elif any(kw in stdout.lower() for kw in ["429", "resource_exhausted", "capacity", "rate limit"]):
                    err_cls = GeminiRateLimitError
                elif any(kw in stdout.lower() for kw in ["timeout", "timed out", "deadline"]):
                    err_cls = GeminiTimeoutError
                raise err_cls(error_text or "Unknown error", exit_code=process.returncode, stdout=stdout)

            active_session_id = await discover_session_id(
                stdout=stdout,
                output_format=output_format,
                cwd=cwd,
                approval_mode=approval_mode,
                requested_session_id=managed_session,
            )
            if active_session_id:
                store_managed_session(registry_key, active_session_id)

            result = {"ok": is_ok, "exit_code": process.returncode, "stdout": stdout, "stderr": ""}
            if active_session_id:
                result["active_session_id"] = active_session_id
            return result

        except GeminiTimeoutError:
            raise
        except GeminiAuthError:
            raise
        except asyncio.CancelledError:
            await _notify_progress(progress_callback, "Gemini run cancelled")
            raise
        except Exception as e:
            logger.exception("Execution error on attempt %d", attempt + 1)
            last_res = {"ok": False, "error": str(e), "stderr": str(e), "stdout": ""}
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            if process:
                active_processes.discard(process)
                if process.returncode is None:
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.debug("Timed out waiting for Gemini process cleanup", extra={"mode": mode})
            if isolated_home and not _home_cache.values():
                await _rmtree_async(isolated_home)

    return last_res


async def _notify_context(context: Optional["Context"], message: str):
    """Send a message to MCP context."""
    if not context:
        return
    try:
        await context.info(message)
    except Exception:
        logger.debug("Failed to send MCP context update", extra={"message": message}, exc_info=True)


async def _notify_progress(progress_callback, message: str):
    if not progress_callback:
        return
    try:
        await progress_callback(message)
    except Exception:
        logger.debug("Failed to send progress callback", extra={"message": message}, exc_info=True)


async def _emit_progress_update(
    context: Optional["Context"],
    progress_callback,
    sequence: int,
    elapsed_seconds: int,
    stage: str,
    message: str,
):
    if progress_callback:
        await _notify_progress(progress_callback, message)
        return
    await _report_keepalive(context, sequence, elapsed_seconds, stage)


async def _report_keepalive(context: Optional["Context"], sequence: int, elapsed_seconds: int, stage: str):
    """Send progress notification to MCP session."""
    if not context:
        return
    message = f"Gemini {stage} ({elapsed_seconds}s elapsed)"
    try:
        await context.report_progress(progress=float(sequence), total=None, message=message)
        logger.debug(
            "Sent MCP keepalive progress notification",
            extra={
                "request_id": getattr(context, "request_id", None),
                "sequence": sequence,
                "elapsed_seconds": elapsed_seconds,
                "stage": stage,
            },
        )
    except Exception:
        logger.debug(
            "Failed to send MCP keepalive progress notification",
            extra={"sequence": sequence, "elapsed_seconds": elapsed_seconds, "stage": stage},
            exc_info=True,
        )


async def run_with_fallback(
    prompt: str,
    mode: str,
    models: list[str],
    context: Optional["Context"] = None,
    progress_callback=None,
    **kwargs,
) -> dict:
    """Run Gemini CLI with model fallback chain."""
    last_res = {"ok": False, "error": "No models available"}
    for model in models:
        await _notify_context(context, f"Attempting with {model}...")
        await _notify_progress(progress_callback, f"Attempting with {model}")
        try:
            res = await run_gemini_cli(prompt, mode=mode, model=model, context=context, progress_callback=progress_callback, **kwargs)
        except GeminiAuthError as e:
            result = {"ok": False, "error": f"Authentication failed: {e}", "stdout": e.stdout, "stderr": "", "model": model}
            if e.session_id:
                result["active_session_id"] = e.session_id
            return result
        except GeminiTimeoutError as e:
            result = {"ok": False, "error": f"Execution timed out: {e}", "stdout": e.stdout, "stderr": "", "model": model}
            if e.session_id:
                result["active_session_id"] = e.session_id
            return result
        except GeminiError as e:
            result = {"ok": False, "error": str(e), "stdout": e.stdout, "stderr": "", "model": model}
            if e.session_id:
                result["active_session_id"] = e.session_id
            return result

        if res.get("ok"):
            res["model"] = model
            return res

        error_text = (res.get("stdout", "") + res.get("error", "") + res.get("stderr", "")).lower()
        is_transient = any(err in error_text for err in ["429", "capacity", "exhausted", "timeout", "internal error"])

        if is_transient:
            await _notify_context(context, f"Model {model} failed or busy. Falling back...")
            await _notify_progress(progress_callback, f"Model {model} failed or busy; falling back")
            last_res = res
            last_res["model"] = model
            continue

        res["model"] = model
        return res
    return last_res


async def run_agentic_tool(
    default_instruction: str,
    prompt: str,
    context: "Context",
    mode: str = "inspect",
    tier: str = "balanced",
    models: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    include_directories: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    session: Optional[str] = None,
    session_mode: str = "auto",
    apply: bool = False,
    sandbox_override: Optional[bool] = None,
    system_prompt: Optional[str] = None,
    progress_callback=None,
) -> str:
    """Helper for standard agentic tools (inspect/edit)."""
    instruction = system_prompt if system_prompt else default_instruction
    selected_models = models if models is not None else MODEL_TIERS[tier]

    if mode == "edit":
        sandbox = not apply
        approval = "auto_edit" if apply else "plan"
    else:
        sandbox = True
        approval = "plan"

    if sandbox_override is not None:
        sandbox = sandbox_override

    logger.debug(
        "Configured agentic tool execution",
        extra={
            "mode": mode,
            "tier": tier,
            "models": selected_models,
            "cwd": cwd or os.getcwd(),
            "approval_mode": approval,
            "sandbox": sandbox,
            "apply": apply,
        },
    )

    res = await run_with_fallback(
        f"{instruction}\n\nTask: {prompt}", mode, selected_models, context,
        cwd=cwd, include_directories=include_directories, include_files=include_files,
        timeout=timeout, read_timeout=read_timeout, session=session, session_mode=session_mode,
        sandbox=sandbox, approval_mode=approval, progress_callback=progress_callback,
    )
    data = parse_and_summarize(res, "stream-json")
    if mode == "edit" and apply and data.get("ok"):
        data["mode"] = "applied"
    elif mode == "edit" and data.get("ok"):
        data["mode"] = "proposed"
        data["note"] = "Changes were proposed in read-only mode and NOT written to disk. Set apply=true to write for real."
    return json.dumps(data, indent=2)
