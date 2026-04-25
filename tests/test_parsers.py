import json

from gemini_mcp.jobs import ArtifactStore, RunStatus
from gemini_mcp.parsers import (
    THOUGHT_MARKER_RE,
    WORD_BREAK_RE,
    clean_stream_response,
    collect_assistant_response_parts,
    normalize_stream_text,
    parse_and_summarize,
)


def test_thought_marker_re_strips_markers():
    assert THOUGHT_MARKER_RE.sub("", "**Analyzing**[Thought: true]Result") == "**Analyzing**Result"
    assert THOUGHT_MARKER_RE.sub("", "No marker here") == "No marker here"
    assert THOUGHT_MARKER_RE.sub("", "[Thought: false]clean") == "clean"
    assert THOUGHT_MARKER_RE.sub("", "[Thought:True]clean") == "clean"
    assert THOUGHT_MARKER_RE.sub("", "[thought: TRUE]clean") == "clean"


def test_word_break_re_rejoins():
    assert WORD_BREAK_RE.sub(r"\1\2", "word-\nbreak") == "wordbreak"
    assert WORD_BREAK_RE.sub(r"\1\2", "no break") == "no break"
    assert WORD_BREAK_RE.sub(r"\1\2", "a-\nb") == "ab"


def test_normalize_stream_text_strips_markers_and_repairs_hyphen_breaks():
    text = "**Analyzing** code-\nbase details.[Thought: true]"
    assert normalize_stream_text(text) == "**Analyzing** codebase details."


def test_clean_stream_response_separates_thinking_from_answer():
    parts = [
        "**Analyzing** I am looking at the code.[Thought: true]",
        "**Investigating** Checking the file structure.[Thought: true]",
        "Based on the analysis, here is the answer:\n\n- Point 1\n- Point 2",
    ]
    final_answer, reasoning_trace = clean_stream_response(parts)
    assert "Based on the analysis" in final_answer
    assert "[Thought:" not in final_answer
    assert "Analyzing" in reasoning_trace
    assert "Investigating" in reasoning_trace


def test_clean_stream_response_promotes_last_thinking_when_no_answer():
    parts = [
        "**Analyzing** Looking at code.[Thought: true]",
        "**Investigating** Checking files.[Thought: true]",
    ]
    final_answer, reasoning_trace = clean_stream_response(parts)
    assert len(final_answer) > 0
    assert "Investigating" in final_answer
    assert len(reasoning_trace) > 0
    assert "Analyzing" in reasoning_trace


def test_clean_stream_response_handles_empty_input():
    final_answer, reasoning_trace = clean_stream_response([])
    assert final_answer == ""
    assert reasoning_trace == ""


def test_clean_stream_response_short_interjection_classification():
    parts = [
        "Planning Next Steps",  # short interjection starting with Thinking verb
        "Here is the detailed result:\n\n```python\nprint('hello')\n```",
    ]
    final_answer, reasoning_trace = clean_stream_response(parts)
    assert "detailed result" in final_answer
    assert "Planning" in reasoning_trace


def test_clean_stream_response_realistic_word_broken():
    parts = [
        "**Analyzing Search Results** I am processing the data.[Thought: true]",
        "Based on the code-\nbase analysis:\n\n1. The module uses async.\n2. It handles retries.",
    ]
    final_answer, reasoning_trace = clean_stream_response(parts)
    assert "codebase" in final_answer
    assert "code-\nbase" not in final_answer
    assert "[Thought:" not in final_answer


def test_parse_and_summarize_includes_reasoning_trace():
    stream_stdout = (
        '{"type":"message","role":"assistant","content":"**Analyzing** ","delta":true}\n'
        '{"type":"message","role":"assistant","content":"Looking at code.[Thought: true]","delta":true}\n'
        '{"type":"tool_use","tool_name":"grep_search"}\n'
        '{"type":"message","role":"assistant","content":"Here is the answer:\\n\\n- Item 1\\n- Item 2","delta":true}\n'
        '{"type":"result","status":"success"}\n'
    )
    output = {
        "ok": True,
        "exit_code": 0,
        "stdout": stream_stdout,
        "stderr": "",
    }
    result = parse_and_summarize(output, "stream-json")
    assert result["ok"] is True
    assert "Here is the answer" in result["response"]
    assert "[Thought:" not in result["response"]
    assert "reasoning_trace" in result
    assert "Analyzing" in result["reasoning_trace"]
    assert "grep_search" in result["tools_used"]


def test_collect_assistant_response_parts_reconstructs_delta_stream():
    events = [
        {"type": "message", "role": "assistant", "content": "Hello ", "delta": True},
        {"type": "message", "role": "assistant", "content": "world", "delta": True},
        {"type": "result", "status": "success"},
    ]

    assert collect_assistant_response_parts(events) == ["Hello world"]


def test_collect_assistant_response_parts_dedupes_final_full_message_after_deltas():
    events = [
        {"type": "message", "role": "assistant", "content": "Hello ", "delta": True},
        {"type": "message", "role": "assistant", "content": "world", "delta": True},
        {"type": "message", "role": "assistant", "content": "Hello world"},
    ]

    assert collect_assistant_response_parts(events) == ["Hello world"]


def test_parse_and_summarize_handles_split_thought_marker_across_deltas():
    stream_stdout = (
        '{"type":"message","role":"assistant","content":"**Analyzing** Looking at code.[Th","delta":true}\n'
        '{"type":"message","role":"assistant","content":"ought: true]","delta":true}\n'
        '{"type":"tool_use","tool_name":"read_file"}\n'
        '{"type":"message","role":"assistant","content":"Final answer","delta":true}\n'
        '{"type":"result","status":"success"}\n'
    )

    output = {"ok": True, "exit_code": 0, "stdout": stream_stdout, "stderr": ""}
    result = parse_and_summarize(output, "stream-json")

    assert result["response"] == "Final answer"
    assert "Analyzing" in result["reasoning_trace"]
    assert "[Thought:" not in result["reasoning_trace"]


def test_parse_and_summarize_reassembles_realistic_delta_summary():
    stream_stdout = (
        '{"type":"message","role":"assistant","content":"Dưới đây là t","delta":true}\n'
        '{"type":"message","role":"assistant","content":"óm tắt cấu trúc project của repo `gemini-mcp`:\\n\\n### Root Directory","delta":true}\n'
        '{"type":"message","role":"assistant","content":"\\n- **`server.py`**: Entry point chính để khởi chạy MCP server.","delta":true}\n'
        '{"type":"result","status":"success"}\n'
    )

    output = {"ok": True, "exit_code": 0, "stdout": stream_stdout, "stderr": ""}
    result = parse_and_summarize(output, "stream-json")

    assert result["response"].startswith("Dưới đây là tóm tắt cấu trúc project")
    assert "### Root Directory" in result["response"]
    assert "server.py" in result["response"]
    assert result["reasoning_trace"] == ""


def test_parse_and_summarize_uses_last_assistant_block_as_final_answer_for_tool_runs():
    stream_stdout = (
        '{"type":"message","role":"assistant","content":"I will inspect the repository layout first.\\n\\n","delta":true}\n'
        '{"type":"tool_use","tool_name":"list_directory"}\n'
        '{"type":"message","role":"assistant","content":"I will now read the core package files.\\n\\n","delta":true}\n'
        '{"type":"tool_use","tool_name":"read_file"}\n'
        '{"type":"message","role":"assistant","content":"Final summary:\\n- `gemini_mcp/` contains the core server logic.","delta":true}\n'
        '{"type":"result","status":"success"}\n'
    )

    output = {"ok": True, "exit_code": 0, "stdout": stream_stdout, "stderr": ""}
    result = parse_and_summarize(output, "stream-json")

    assert result["response"].startswith("Final summary:")
    assert "inspect the repository layout" in result["reasoning_trace"]
    assert "read the core package files" in result["reasoning_trace"]


def test_parse_and_summarize_json_format_no_reasoning_trace():
    stdout = '{"response": "simple answer"}'
    output = {"ok": True, "exit_code": 0, "stdout": stdout, "stderr": ""}
    result = parse_and_summarize(output, "json")
    assert result["ok"] is True
    assert "reasoning_trace" not in result


def test_build_result_markdown_includes_collapsible_reasoning(tmp_path):
    store = ArtifactStore()
    run = store.create_run("analyze")
    run.status = RunStatus.COMPLETED
    run.status_message = "Done"
    run.model = "gemini-3-flash-preview"
    run.reasoning_trace = "Step 1: Looked at code\nStep 2: Found the issue"
    run.result = {"response": "The issue is in module X.", "tools_used": ["grep"]}

    md = store._build_result_markdown(run)
    assert "## Final Answer" in md
    assert "The issue is in module X." in md
    assert "Reasoning Trace" not in md

    reasoning_md = store._build_reasoning_markdown(run)
    assert "# analyze Reasoning Trace:" in reasoning_md
    assert "Step 1: Looked at code" in reasoning_md
    assert "Step 2: Found the issue" in reasoning_md


def test_build_result_markdown_no_reasoning_no_details(tmp_path):
    store = ArtifactStore()
    run = store.create_run("explore")
    run.status = RunStatus.COMPLETED
    run.status_message = "Done"
    run.result = {"response": "Clean answer only."}

    md = store._build_result_markdown(run)
    assert "## Final Answer" in md
    assert "Clean answer only." in md
    assert "Reasoning Trace" not in md


def test_build_result_markdown_empty_reasoning_no_details():
    store = ArtifactStore()
    run = store.create_run("plan")
    run.status = RunStatus.COMPLETED
    run.status_message = "Done"
    run.reasoning_trace = ""
    run.result = {"response": "Just the plan."}

    md = store._build_result_markdown(run)
    assert "Reasoning Trace" not in md
    assert "Just the plan." in md


def test_build_reasoning_markdown_empty_reasoning_returns_empty_string():
    store = ArtifactStore()
    run = store.create_run("plan")
    run.status = RunStatus.COMPLETED
    run.status_message = "Done"
    run.reasoning_trace = ""

    assert store._build_reasoning_markdown(run) == ""
