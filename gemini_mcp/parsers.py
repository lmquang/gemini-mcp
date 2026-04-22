"""Output parsers for Gemini CLI responses."""

import json
import logging
import re

logger = logging.getLogger("gemini-mcp")

THOUGHT_MARKER_RE = re.compile(r"\[Thought:\s*(?:true|false)\]\s*", re.IGNORECASE)
WORD_BREAK_RE = re.compile(r"(\w)-\n(\w)")
THINKING_VERBS = (
    "Analyzing", "Investigating", "Correcting", "Planning", "Examining",
    "Processing", "Refining", "Looking", "Searching", "Checking",
    "Evaluating", "Comparing", "Reviewing", "Considering",
)


def extract_session_id(payload) -> str | None:
    """Extract Gemini session ID from nested response payloads."""
    if isinstance(payload, dict):
        session_id = payload.get("session_id") or payload.get("sessionID")
        if isinstance(session_id, str) and session_id.strip():
            return session_id
        for value in payload.values():
            nested_session_id = extract_session_id(value)
            if nested_session_id:
                return nested_session_id
    elif isinstance(payload, list):
        for item in payload:
            nested_session_id = extract_session_id(item)
            if nested_session_id:
                return nested_session_id
    return None


def parse_stream_json(stdout: str) -> list[dict]:
    """Parse a string containing potentially multiple JSON objects or streams."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pos = 0
        while pos < len(line):
            brace = line.find('{', pos)
            bracket = line.find('[', pos)
            if brace == -1 and bracket == -1:
                break
            start = brace if brace != -1 and (bracket == -1 or brace < bracket) else bracket
            if start == -1:
                break
            end_char = '}' if line[start] == '{' else ']'
            depth = 0
            end = start
            for i in range(start, len(line)):
                if line[i] == line[start]:
                    depth += 1
                elif line[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if depth == 0:
                candidate = line[start:end]
                try:
                    events.append(json.loads(candidate))
                    pos = end
                except json.JSONDecodeError:
                    pos = start + 1
            else:
                break
    return events


def clean_stream_response(response_parts: list[str]) -> tuple[str, str]:
    """Separate thinking steps from the final answer and clean formatting.

    Strips [Thought: true/false] markers, rejoins word-broken lines, and
    classifies each part as either an intermediate reasoning step or the
    substantive final answer.

    Returns (final_answer, reasoning_trace).
    """
    if not response_parts:
        return "", ""

    thinking_parts: list[str] = []
    answer_parts: list[str] = []

    for part in response_parts:
        has_thought_marker = bool(THOUGHT_MARKER_RE.search(part))
        cleaned = THOUGHT_MARKER_RE.sub("", part)
        cleaned = WORD_BREAK_RE.sub(r"\1\2", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            continue

        is_short_interjection = len(cleaned) < 120 and cleaned.startswith(THINKING_VERBS)

        if has_thought_marker or is_short_interjection:
            thinking_parts.append(cleaned)
        else:
            answer_parts.append(cleaned)

    if not answer_parts and thinking_parts:
        answer_parts = [thinking_parts.pop()]

    final_answer = "\n\n".join(answer_parts).strip()
    reasoning_trace = "\n\n".join(thinking_parts).strip()
    return final_answer, reasoning_trace


def normalize_stream_text(text: str) -> str:
    """Normalize reconstructed assistant text for handoff output."""
    cleaned = THOUGHT_MARKER_RE.sub("", text)
    cleaned = WORD_BREAK_RE.sub(r"\1\2", cleaned)
    return cleaned.strip()


def collect_assistant_response_parts(events: list[dict]) -> list[str]:
    """Reconstruct assistant responses from stream-json events.

    Gemini `stream-json` emits assistant text as incremental delta chunks. Those
    chunks must be concatenated before any reasoning-vs-answer cleanup happens.
    """
    response_parts: list[str] = []
    delta_buffer: list[str] = []

    def flush_delta_buffer() -> None:
        if not delta_buffer:
            return
        combined = "".join(delta_buffer).strip()
        delta_buffer.clear()
        if combined:
            response_parts.append(combined)

    for event in events:
        if event.get("type") != "message" or event.get("role") != "assistant":
            flush_delta_buffer()
            continue

        content = event.get("content")
        text_parts: list[str] = []
        if isinstance(content, str) and content.strip():
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if isinstance(text, str) and text:
                        text_parts.append(text)

        if not text_parts:
            continue

        text = "".join(text_parts)
        if event.get("delta") is True:
            delta_buffer.append(text)
            continue

        flushed_delta = "".join(delta_buffer).strip() if delta_buffer else ""
        flush_delta_buffer()
        normalized_text = text.strip()
        if normalized_text and normalized_text != flushed_delta:
            response_parts.append(normalized_text)

    flush_delta_buffer()
    logger.debug(
        "Collected assistant response parts",
        extra={
            "event_count": len(events),
            "response_part_count": len(response_parts),
            "delta_detected": any(event.get("delta") is True for event in events),
        },
    )
    return response_parts


def extract_touched_paths(events: list[dict]) -> list[str]:
    """Extract file paths that Gemini tools interacted with."""
    paths = set()
    path_keys = ["file_path", "path", "filePath", "fileName", "dir_path", "directory"]
    for e in events:
        if e.get("type") in ("tool_use", "tool_result"):
            params = e.get("parameters") or e.get("output") or {}
            if isinstance(params, dict):
                for key in path_keys:
                    if key in params and isinstance(params[key], str):
                        p = params[key]
                        if "/" in p or p.endswith((".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml", ".toml", ".go", ".rs")):
                            paths.add(p)
            output_text = e.get("output") or ""
            if isinstance(output_text, str):
                cleaned = re.sub(r'https?://\S+', '', output_text)
                for match in re.findall(r'[\w/.-]+\.\w{1,10}', cleaned):
                    if "/" in match:
                        paths.add(match)
    return sorted(paths)


def parse_and_summarize(output: dict, format: str) -> dict:
    """Parse Gemini CLI output into structured result for agent consumption."""
    stdout = output.get("stdout", "")
    error = output.get("error")
    exit_code = output.get("exit_code")
    model = output.get("model")
    requested_session_id = output.get("active_session_id") or output.get("requested_session_id")

    if not output.get("ok"):
        if not error and stdout:
            error_lines = [l for l in stdout.splitlines() if "error" in l.lower() or "exception" in l.lower()]
            error = "\n".join(error_lines) if error_lines else stdout[-500:].strip()
        result = {
            "ok": False,
            "error": error or "Unknown execution error",
            "exit_code": exit_code,
            "response": "",
            "tools_used": [],
            "files_touched": [],
        }
        if model:
            result["model"] = model
        if requested_session_id:
            result["sessionID"] = requested_session_id
        return result

    if format == "json":
        # Try to find and parse the main JSON response object
        # Look for JSON starting from the first { or [ after any log messages
        json_start = -1
        for i, char in enumerate(stdout):
            if char in '{[':
                json_start = i
                break

        if json_start >= 0:
            try:
                parsed = json.loads(stdout[json_start:])
                result = {
                    "ok": True,
                    "response": parsed.get("response", parsed.get("content", str(parsed))),
                    "tools_used": [],
                    "files_touched": [],
                    "raw": parsed,
                }
                if model:
                    result["model"] = model
                session_id = extract_session_id(parsed) or requested_session_id
                if session_id:
                    result["sessionID"] = session_id
                return result
            except json.JSONDecodeError:
                pass

        # Fallback: return raw stdout
        result = {
            "ok": True,
            "response": stdout.strip(),
            "tools_used": [],
            "files_touched": [],
            "raw": stdout,
        }
        if model:
            result["model"] = model
        if requested_session_id:
            result["sessionID"] = requested_session_id
        return result

    events = parse_stream_json(stdout)

    response_parts = collect_assistant_response_parts(events)

    tools = sorted({
        e.get("tool_name") or e.get("name") or "unknown"
        for e in events if e.get("type") == "tool_use"
    })
    files_touched = extract_touched_paths(events)

    if len(response_parts) > 1:
        normalized_parts = [normalize_stream_text(part) for part in response_parts]
        normalized_parts = [part for part in normalized_parts if part]
        final_answer = normalized_parts[-1] if normalized_parts else ""
        reasoning_trace = "\n\n".join(normalized_parts[:-1]).strip()
    else:
        final_answer, reasoning_trace = clean_stream_response(response_parts)

    if not final_answer and not tools:
        if stdout.strip():
            final_answer = f"No structured response found. Raw output:\n{stdout[:1000]}"
        else:
            final_answer = "No response received from Gemini CLI."

    result = {
        "ok": True,
        "response": final_answer,
        "reasoning_trace": reasoning_trace,
        "tools_used": tools,
        "files_touched": files_touched,
        "event_count": len(events),
    }
    if model:
        result["model"] = model
    session_id = extract_session_id(events) or requested_session_id
    if session_id:
        result["sessionID"] = session_id
    return result


def classify_error(stdout: str) -> type | None:
    """Classify error type from Gemini CLI output."""
    from gemini_mcp.exceptions import GeminiAuthError, GeminiRateLimitError, GeminiTimeoutError

    lower = stdout.lower()
    if any(kw in lower for kw in ["auth", "credential", "oauth", "unauthorized", "login"]):
        return GeminiAuthError
    if any(kw in lower for kw in ["429", "resource_exhausted", "capacity", "rate limit"]):
        return GeminiRateLimitError
    if any(kw in lower for kw in ["timeout", "timed out", "deadline"]):
        return GeminiTimeoutError
    return None
