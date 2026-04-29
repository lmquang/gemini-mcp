"""Gemini MCP tool definitions."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

_MAX_RESPONSE_CHARS = 25000

from fastmcp import FastMCP, Context

from gemini_mcp.config import AVAILABLE_MODELS, DEFAULT_READ_TIMEOUT, DEFAULT_TIMEOUT_SECONDS, MODEL_TIERS
from gemini_mcp.jobs import RunStatus, artifact_store
from gemini_mcp.runner import list_gemini_sessions, run_agentic_tool, run_with_fallback
from gemini_mcp.parsers import parse_and_summarize

logger = logging.getLogger("gemini-mcp")

mcp = FastMCP("Gemini Expert Assistant")


def _coerce_tool_result(data):
    if isinstance(data, dict):
        return data
    return {
        "ok": True,
        "response": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False),
        "raw": data,
        "tools_used": [],
        "files_touched": [],
    }


def _merge_target_directory(include_directories: Optional[list[str]], target_directory: Optional[str]) -> Optional[list[str]]:
    if not target_directory:
        return include_directories
    merged = list(include_directories or [])
    if os.path.isdir(target_directory) and target_directory not in merged:
        merged.append(target_directory)
    return merged


def _build_document_instruction(
    system_prompt: Optional[str],
    goal: Optional[str],
    tone: Optional[str],
    target_directory: Optional[str],
) -> str:
    base_instruction = system_prompt if system_prompt else (
        "You are a technical writer. Write clear, accurate, well-structured documentation. "
        "Use proper Markdown formatting with headers, code blocks, tables, and lists where appropriate. "
        "Include concrete examples and usage snippets. "
        "Define terms on first use. Write for the intended audience — developers who need to understand and use the subject matter. "
        "Be concise but complete. Avoid filler text. Prioritize clarity and usefulness."
    )
    additions = []
    if goal:
        additions.append(f"Primary goal: {goal}")
    if tone:
        additions.append(f"Tone: {tone}")
    if target_directory:
        additions.append(f"When creating new Markdown files, write them under: {target_directory}")
    if additions:
        return base_instruction + " " + " ".join(additions)
    return base_instruction


async def _report_tool_progress(context: Context, sequence: int, message: str) -> None:
    try:
        await context.report_progress(progress=float(sequence), total=None, message=message)
    except Exception:
        logger.debug("Failed to report tool progress", extra={"sequence": sequence, "message": message}, exc_info=True)


async def _execute_agentic_run(tool_name: str, context: Context, cwd: Optional[str], coro_factory) -> str:
    """Run an agentic tool as a native FastMCP task and persist its artifact bundle."""
    run = artifact_store.create_run(tool_name)
    progress_state = {"sequence": 0}

    async def progress_callback(message: str) -> None:
        progress_state["sequence"] += 1
        await _report_tool_progress(context, progress_state["sequence"], message)

    await progress_callback(f"Preparing {tool_name} run")
    try:
        result_str = await coro_factory(progress_callback)
        data = _coerce_tool_result(json.loads(result_str))
        run.status = RunStatus.COMPLETED if data.get("ok") else RunStatus.FAILED
        run.status_message = "Done" if data.get("ok") else data.get("error", "Failed")
        run.result = data
        run.session_id = data.get("sessionID")
        run.model = data.get("model")
        run.reasoning_trace = data.get("reasoning_trace", "")
        run.last_updated_at = time.time()
        await progress_callback(f"Persisting {tool_name} artifact")
        artifact = artifact_store.persist_run(run, cwd=cwd)

        output_path = artifact.get("outputPath")
        if output_path:
            try:
                response_text = Path(output_path).read_text("utf-8")
            except Exception:
                response_text = data.get("response", "")
        else:
            response_text = data.get("response", "")

        if len(response_text) > _MAX_RESPONSE_CHARS:
            response_text = response_text[:_MAX_RESPONSE_CHARS] + f"\n\n[truncated at {_MAX_RESPONSE_CHARS} chars. Full output: {output_path}]"

        payload = {"response": response_text}
        if run.session_id:
            payload["sessionID"] = run.session_id
        if output_path:
            payload["outputPath"] = output_path

        return json.dumps(payload, indent=2)
    except asyncio.CancelledError:
        run.status = RunStatus.CANCELLED
        run.status_message = "Cancelled by request"
        run.result = {"ok": False, "error": "Cancelled by request", "status": RunStatus.CANCELLED.value}
        run.last_updated_at = time.time()
        artifact_store.persist_run(run, cwd=cwd)
        logger.info("Native task cancelled", extra={"run_id": run.run_id, "tool": tool_name})
        raise
    except Exception as e:
        run.status = RunStatus.FAILED
        run.status_message = str(e)
        run.result = {"ok": False, "error": str(e), "status": RunStatus.FAILED.value}
        run.last_updated_at = time.time()
        await progress_callback(f"Persisting failed {tool_name} artifact")
        artifact = artifact_store.persist_run(run, cwd=cwd)
        payload = {
            "response": f"Error: {e}",
            "error": True,
        }
        if artifact.get("outputPath"):
            payload["outputPath"] = artifact["outputPath"]
        logger.exception("Native task failed", extra={"run_id": run.run_id, "tool": tool_name})
        return json.dumps(payload, indent=2)


@mcp.tool()
async def list_runs(cwd: Optional[str] = None, limit: int = 20) -> str:
    """List persisted Gemini run artifacts. Live execution is handled by native FastMCP tasks, not polling."""
    return json.dumps(artifact_store.list_runs(cwd=cwd, limit=limit), indent=2)


# ---------------------------------------------------------------------------
# Gemini model / session tools (fast)
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_models() -> str:
    """List available Gemini models with their tier and usage."""
    return json.dumps(AVAILABLE_MODELS, indent=2)


@mcp.tool()
async def list_sessions(cwd: Optional[str] = None) -> str:
    """List Gemini CLI sessions for the current project, including numeric resume indexes and stable session IDs."""
    sessions = await list_gemini_sessions(cwd=cwd)
    return json.dumps(sessions, indent=2)


# ---------------------------------------------------------------------------
# Chat (fast, no background needed)
# ---------------------------------------------------------------------------

@mcp.tool()
async def chat(
    prompt: str,
    context: Context,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    model: Optional[str] = None,
    sessionID: Optional[str] = None,
    sessionMode: str = "auto",
    system_prompt: Optional[str] = None,
) -> str:
    """Lightweight chat with Gemini — no repo indexing, no sandbox.

    Session behavior: default to `sessionMode='auto'` to reuse the managed session for the same MCP connection and cwd.
    Use `sessionMode='new'` only when you need a fresh conversation boundary. Pass `sessionID` to force a specific Gemini session.
    """
    full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
    full_prompt += prompt
    models = [model] if model else MODEL_TIERS["cheap"]
    res = await run_with_fallback(full_prompt, "chat", models, context, cwd=cwd, timeout=timeout, session=sessionID, session_mode=sessionMode, sandbox=False)
    parsed = parse_and_summarize(res, "json")
    payload = {"response": parsed.get("response") or parsed.get("error", "")}
    if parsed.get("sessionID"):
        payload["sessionID"] = parsed["sessionID"]
    if not parsed.get("ok"):
        payload["error"] = True
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Agentic tools (native FastMCP tasks)
# ---------------------------------------------------------------------------

@mcp.tool(task=True)
async def explore(
    prompt: str,
    context: Context,
    cwd: Optional[str] = None,
    include_directories: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    sessionID: Optional[str] = None,
    sessionMode: str = "auto",
    system_prompt: Optional[str] = None,
) -> str:
    """Explore codebase with Gemini as a native FastMCP task. Await the task result for the final structured payload: `response` (Markdown content), `sessionID`, and `outputPath` for persisted artifacts.

    Runs in plan mode without full-process sandboxing. Default to `sessionMode='auto'` to continue the current managed Gemini session.
    Use `sessionMode='new'` only when you want a clean exploration thread or need to avoid prior conversation context.
    """
    async def run(progress_callback):
        default_instruction = "You are a codebase researcher. Investigate the requested topic thoroughly. Map relationships between files, functions, and modules. Report findings with exact file paths and line references. Be precise — only report what you can verify from the source code."
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="inspect", tier="balanced", cwd=cwd,
            include_directories=include_directories, include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode,
            sandbox_override=False, system_prompt=system_prompt, progress_callback=progress_callback,
        )

    return await _execute_agentic_run("explore", context, cwd, run)


@mcp.tool(task=True)
async def analyze(
    prompt: str,
    context: Context,
    cwd: Optional[str] = None,
    include_directories: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    sessionID: Optional[str] = None,
    sessionMode: str = "auto",
    system_prompt: Optional[str] = None,
) -> str:
    """Technical code review with Gemini as a native FastMCP task. Await the task result for the final structured payload: `response` (Markdown content), `sessionID`, and `outputPath` for persisted artifacts.

    Runs in plan mode without full-process sandboxing. Default to `sessionMode='auto'` to reuse the current managed review session.
    Use `sessionMode='new'` when the review should ignore prior context or start a fresh investigation.
    """
    async def run(progress_callback):
        default_instruction = "You are a senior code reviewer. Analyze the code for bugs, performance issues, security vulnerabilities, and architectural problems. Be specific — reference exact file paths, function names, and line numbers. Prioritize findings by severity. Suggest concrete fixes, not vague improvements."
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="inspect", tier="smart", cwd=cwd,
            include_directories=include_directories, include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode,
            sandbox_override=False, system_prompt=system_prompt, progress_callback=progress_callback,
        )

    return await _execute_agentic_run("analyze", context, cwd, run)


@mcp.tool(task=True)
async def plan(
    prompt: str,
    context: Context,
    cwd: Optional[str] = None,
    include_directories: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    sessionID: Optional[str] = None,
    sessionMode: str = "auto",
    system_prompt: Optional[str] = None,
) -> str:
    """Generate architecture plans with Gemini as a native FastMCP task. Await the task result for the final structured payload: `response` (Markdown content), `sessionID`, and `outputPath` for persisted artifacts.

    Runs in plan mode without full-process sandboxing. Default to `sessionMode='auto'` to continue the current planning session.
    Use `sessionMode='new'` when you want an isolated plan not influenced by earlier prompts.
    """
    async def run(progress_callback):
        default_instruction = "You are a software architect. Create a clear, actionable implementation plan. Break down the work into numbered steps with file-level detail. Identify dependencies between steps. Consider edge cases and potential risks. Output in well-structured Markdown."
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="inspect", tier="smart", cwd=cwd,
            include_directories=include_directories, include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode,
            sandbox_override=False, system_prompt=system_prompt, progress_callback=progress_callback,
        )

    return await _execute_agentic_run("plan", context, cwd, run)


@mcp.tool(task=True)
async def document(
    prompt: str,
    context: Context,
    cwd: Optional[str] = None,
    include_directories: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    model: Optional[str] = None,
    sessionID: Optional[str] = None,
    sessionMode: str = "auto",
    apply: bool = False,
    goal: Optional[str] = None,
    tone: Optional[str] = None,
    target_directory: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Write documentation with Gemini as a native FastMCP task. Await the task result for the final structured payload: `response` (Markdown content), `sessionID`, and `outputPath` for persisted artifacts.

    Default to `sessionMode='auto'` to reuse the current managed documentation session.
    Use `sessionMode='new'` when you want a fresh documentation thread. Pass `sessionID` to force a specific Gemini session.
    Uses Flash-Lite first by default, or a caller-specified model when provided.
    """
    default_instruction = _build_document_instruction(system_prompt, goal, tone, target_directory)
    doc_models = [model] if model else MODEL_TIERS["cheap"]

    async def run(progress_callback):
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="edit", models=doc_models, cwd=cwd,
            include_directories=_merge_target_directory(include_directories, target_directory),
            include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode, apply=apply,
            sandbox_override=False if not apply else None,
            system_prompt=default_instruction, progress_callback=progress_callback,
        )

    return await _execute_agentic_run("document", context, cwd, run)
