"""Output parsers for Gemini CLI responses."""

import json
import re


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

    response_parts = []
    for e in events:
        if e.get("type") == "message" and e.get("role") == "assistant":
            content = e.get("content")
            if isinstance(content, str) and content.strip():
                response_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        response_parts.append(part.get("text", ""))

    tools = sorted({
        e.get("tool_name") or e.get("name") or "unknown"
        for e in events if e.get("type") == "tool_use"
    })
    files_touched = extract_touched_paths(events)

    response = "\n".join(response_parts).strip()

    if not response and not tools:
        if stdout.strip():
            response = f"No structured response found. Raw output:\n{stdout[:1000]}"
        else:
            response = "No response received from Gemini CLI."

    result = {
        "ok": True,
        "response": response,
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
