"""Gemini MCP tool definitions."""

import asyncio
import json
import logging
import os
from typing import Optional

from fastmcp import FastMCP, Context

from gemini_mcp.config import AVAILABLE_MODELS, DEFAULT_READ_TIMEOUT, DEFAULT_TIMEOUT_SECONDS, MODEL_TIERS
from gemini_mcp.jobs import JobStatus, job_manager
from gemini_mcp.runner import list_gemini_sessions, resolve_managed_session, run_agentic_tool, run_with_fallback
from gemini_mcp.parsers import parse_and_summarize

logger = logging.getLogger("gemini-mcp")

mcp = FastMCP("Gemini Expert Assistant")


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


def _prepare_job_session(
    job_id: str,
    context: Context,
    cwd: Optional[str],
    session_id: Optional[str],
    session_mode: str,
) -> None:
    """Pre-resolve the managed session so job metadata reflects reuse immediately."""
    registry_key, preview_session_id = resolve_managed_session(
        context=context,
        cwd=cwd,
        session=session_id,
        session_mode=session_mode,
    )
    updates = {}
    if preview_session_id:
        updates["session_id"] = preview_session_id
    if updates:
        job_manager.update_job(job_id, **updates)
    logger.debug(
        "Prepared Gemini job session",
        extra={
            "job_id": job_id,
            "registry_key": registry_key,
            "requested_session_id": session_id,
            "preview_session_id": preview_session_id,
            "session_mode": session_mode or "auto",
        },
    )


async def _run_job_background(
    job_id: str,
    tool_name: str,
    coro_factory,
    cwd: Optional[str] = None,
) -> None:
    """Execute a long-running operation in the background, updating the job."""
    job = job_manager.get_job(job_id)
    if not job:
        return
    job_manager.update_job(job_id, status=JobStatus.RUNNING, status_message="Gemini process starting")
    try:
        result_str = await coro_factory()
        data = json.loads(result_str)
        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            status_message="Done",
            result=data,
            session_id=data.get("sessionID"),
            model=data.get("model"),
            reasoning_trace=data.get("reasoning_trace", ""),
        )
        job_manager.persist_result(job_id, cwd=cwd)
        logger.info("Job completed", extra={"job_id": job_id, "tool": tool_name})
    except asyncio.CancelledError:
        job_manager.update_job(job_id, status=JobStatus.CANCELLED, status_message="Cancelled")
        logger.info("Job cancelled", extra={"job_id": job_id, "tool": tool_name})
    except Exception as e:
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            status_message=str(e),
            result={"ok": False, "error": str(e)},
        )
        logger.exception("Job failed", extra={"job_id": job_id, "tool": tool_name})


# ---------------------------------------------------------------------------
# Job management tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def job_status(jobID: str) -> str:
    """Check the status of a background Gemini job. Poll this until status is completed/failed/cancelled. When the job completes, the outputPath field points to a persisted Markdown file with a clean answer and collapsible reasoning trace."""
    job = job_manager.get_job(jobID)
    if not job:
        return json.dumps({"ok": False, "error": f"Job {jobID} not found"}, indent=2)
    return json.dumps(job.to_dict(), indent=2)


@mcp.tool()
async def job_result(jobID: str) -> str:
    """Get the result of a completed background Gemini job. Only call after job_status shows completed. Results are cleaned: intermediate thinking steps are separated from the final answer. The `response` field contains the substantive answer; `reasoning_trace` contains intermediate reasoning (collapsed in the persisted markdown). The `outputPath` from job_status points to a formatted Markdown file suitable for agent handoff."""
    job = job_manager.get_job(jobID)
    if not job:
        return json.dumps({"ok": False, "error": f"Job {jobID} not found"}, indent=2)
    if not job.is_terminal:
        return json.dumps({"ok": False, "error": f"Job {jobID} is still {job.status.value}. Poll job_status first."}, indent=2)
    result = job.result or {"ok": False, "error": "No result available"}
    result["jobID"] = job.job_id
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_jobs() -> str:
    """List all background Gemini jobs and their statuses."""
    return json.dumps(job_manager.list_jobs(), indent=2)


@mcp.tool()
async def cancel_job(jobID: str) -> str:
    """Cancel a running background Gemini job."""
    job = await job_manager.cancel_job(jobID)
    if not job:
        return json.dumps({"ok": False, "error": f"Job {jobID} not found"}, indent=2)
    return json.dumps({"ok": True, **job.to_dict()}, indent=2)


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
    data = parse_and_summarize(res, "json")
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Agentic tools (background jobs)
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
    """Explore codebase with Gemini. Starts a background job — returns a jobID immediately. Poll job_status, then get results with job_result. Responses are cleaned: thinking steps are separated from the final answer in job_result.

    Runs in plan mode without full-process sandboxing. Default to `sessionMode='auto'` to continue the current managed Gemini session.
    Use `sessionMode='new'` only when you want a clean exploration thread or need to avoid prior conversation context.
    """
    job = job_manager.create_job("explore")
    _prepare_job_session(job.job_id, context, cwd, sessionID, sessionMode)

    async def run():
        default_instruction = "You are a codebase researcher. Investigate the requested topic thoroughly. Map relationships between files, functions, and modules. Report findings with exact file paths and line references. Be precise — only report what you can verify from the source code."
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="inspect", tier="balanced", cwd=cwd,
            include_directories=include_directories, include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode,
            sandbox_override=False, system_prompt=system_prompt,
        )

    task = asyncio.create_task(_run_job_background(job.job_id, "explore", run, cwd=cwd))
    job_manager.update_job(job.job_id, task=task)
    return json.dumps({"ok": True, "jobID": job.job_id, "status": "running", "statusMessage": "Job started. Poll job_status to track progress.", "sessionID": job.session_id}, indent=2)


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
    """Technical code review with Gemini. Starts a background job — returns a jobID immediately. Poll job_status, then get results with job_result. Responses are cleaned: thinking steps are separated from the final answer in job_result.

    Runs in plan mode without full-process sandboxing. Default to `sessionMode='auto'` to reuse the current managed review session.
    Use `sessionMode='new'` when the review should ignore prior context or start a fresh investigation.
    """
    job = job_manager.create_job("analyze")
    _prepare_job_session(job.job_id, context, cwd, sessionID, sessionMode)

    async def run():
        default_instruction = "You are a senior code reviewer. Analyze the code for bugs, performance issues, security vulnerabilities, and architectural problems. Be specific — reference exact file paths, function names, and line numbers. Prioritize findings by severity. Suggest concrete fixes, not vague improvements."
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="inspect", tier="smart", cwd=cwd,
            include_directories=include_directories, include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode,
            sandbox_override=False, system_prompt=system_prompt,
        )

    task = asyncio.create_task(_run_job_background(job.job_id, "analyze", run, cwd=cwd))
    job_manager.update_job(job.job_id, task=task)
    return json.dumps({"ok": True, "jobID": job.job_id, "status": "running", "statusMessage": "Job started. Poll job_status to track progress.", "sessionID": job.session_id}, indent=2)


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
    """Generate architecture plans with Gemini. Starts a background job — returns a jobID immediately. Poll job_status, then get results with job_result. Responses are cleaned: thinking steps are separated from the final answer in job_result.

    Runs in plan mode without full-process sandboxing. Default to `sessionMode='auto'` to continue the current planning session.
    Use `sessionMode='new'` when you want an isolated plan not influenced by earlier prompts.
    """
    job = job_manager.create_job("plan")
    _prepare_job_session(job.job_id, context, cwd, sessionID, sessionMode)

    async def run():
        default_instruction = "You are a software architect. Create a clear, actionable implementation plan. Break down the work into numbered steps with file-level detail. Identify dependencies between steps. Consider edge cases and potential risks. Output in well-structured Markdown."
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="inspect", tier="smart", cwd=cwd,
            include_directories=include_directories, include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode,
            sandbox_override=False, system_prompt=system_prompt,
        )

    task = asyncio.create_task(_run_job_background(job.job_id, "plan", run, cwd=cwd))
    job_manager.update_job(job.job_id, task=task)
    return json.dumps({"ok": True, "jobID": job.job_id, "status": "running", "statusMessage": "Job started. Poll job_status to track progress.", "sessionID": job.session_id}, indent=2)


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
    """Write documentation with Gemini. Starts a background job — returns a jobID immediately. Poll job_status, then get results with job_result. Responses are cleaned: thinking steps are separated from the final answer in job_result.

    Default to `sessionMode='auto'` to reuse the current managed documentation session.
    Use `sessionMode='new'` when you want a fresh documentation thread. Pass `sessionID` to force a specific Gemini session.
    Uses Flash-Lite first by default, or a caller-specified model when provided.
    """
    job = job_manager.create_job("document")
    _prepare_job_session(job.job_id, context, cwd, sessionID, sessionMode)
    default_instruction = _build_document_instruction(system_prompt, goal, tone, target_directory)
    doc_models = [model] if model else MODEL_TIERS["cheap"]

    async def run():
        return await run_agentic_tool(
            default_instruction, prompt, context, mode="edit", models=doc_models, cwd=cwd,
            include_directories=_merge_target_directory(include_directories, target_directory),
            include_files=include_files,
            timeout=timeout, read_timeout=read_timeout, session=sessionID, session_mode=sessionMode, apply=apply,
            sandbox_override=False if not apply else None,
            system_prompt=default_instruction,
        )

    task = asyncio.create_task(_run_job_background(job.job_id, "document", run, cwd=cwd))
    job_manager.update_job(job.job_id, task=task)
    return json.dumps({"ok": True, "jobID": job.job_id, "status": "running", "statusMessage": "Job started. Poll job_status to track progress.", "sessionID": job.session_id}, indent=2)
