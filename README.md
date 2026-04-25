# Gemini MCP Server

A Model Context Protocol (MCP) server that provides an interface to the Gemini Expert Assistant. This server allows MCP clients to use Gemini's capabilities for code exploration, analysis, planning, and documentation.

## Installation

Ensure you have the Gemini CLI installed and configured.

```bash
uv venv
uv tool install --editable .
```

If you need the raw dependency equivalent instead of installing this package, the current requirement is:

```bash
pip install "fastmcp[tasks]>=3.2.4"
```

## Usage

### Starting the Server

You can run the server directly using Python:

```bash
python server.py
```

### Using with an MCP Client

Add the following configuration to your MCP client (e.g., Claude Desktop or other MCP-compatible IDEs):

```json
{
  "mcpServers": {
    "gemini-mcp": {
      "command": "python",
      "args": ["/path/to/gemini-mcp/server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Available Tools

The server exposes the following tools, organized by category.

#### Run Artifact Tools

Agentic tools (`explore`, `analyze`, `plan`, `document`) run as native FastMCP tasks. Persisted run artifacts are available for history and handoff.

- **`list_runs`**: List persisted Gemini run artifacts for the current working directory.

#### Gemini Model / Session Tools

- **`list_models`**: Lists available Gemini models with their tier and usage.
- **`list_sessions`**: Lists Gemini CLI sessions for the current project.

#### Chat (Fast — No Native Task Needed)

- **`chat`**: Lightweight chat with Gemini — no repo indexing, no sandbox. Auto-resumes the managed session unless `sessionMode='new'` or `sessionID` is provided.

#### Agentic Tools (Native FastMCP Tasks)

These tools execute as native FastMCP tasks. Task-aware clients can observe progress and await the final result directly. The final payload includes `runID`, `outputPath`, and `reasoningTracePath` for persisted artifacts.

- **`explore`**: Investigates the codebase, researching and mapping project structure.
- **`analyze`**: Performs technical code reviews for bugs, performance, security, and architectural issues.
- **`plan`**: Generates step-by-step implementation plans or architectural designs.
- **`document`**: Writes or updates documentation (supports `apply=true` to write changes).

### Workflow Pattern

1. Call an agentic tool (e.g., `explore`) as a native FastMCP task.
2. Observe progress updates if your client supports them.
3. Await the task result to get the full structured payload.
4. Read `outputPath` when you need the persisted Markdown artifact.

Example:

```
1. explore(prompt="Summarize the project structure") → native FastMCP task
2. task progress                                  → "Gemini initialized", "Gemini started responding", ...
3. task result                                    → {"ok": true, "response": "...", "runID": "...", "outputPath": "..."}
```

### Session Management

Most tools support seamless managed session reuse across calls, auto-resuming the managed session by default.

- Use `sessionMode="auto"` or omit `sessionMode` to reuse the current managed Gemini session for the same MCP connection and working directory.
- Use `sessionMode="new"` only when you need a fresh conversation boundary, such as switching tasks, avoiding stale context, or debugging session behavior.
- Pass `sessionID` to force a specific Gemini session explicitly.
- Agentic tool final payloads include the final active `sessionID` when Gemini reports one.

### Runtime Behavior

- Agentic tools run as native FastMCP tasks and return final structured payloads when awaited.
- `chat` runs synchronously (no native task needed — it's fast).
- Read-only agentic tools (`explore`, `analyze`, `plan`, and `document` with `apply=false`) run in `approval-mode plan` without full-process `--sandbox` so Gemini `stream-json` events can surface promptly.
- `document` with `apply=true` keeps the write/apply edit path and returns `mode: "applied"` on success.
- Use `sessionMode="new"` on any session-aware tool when you want to force a fresh managed session.

### MCP Tasks Protocol Support

Agentic tools are registered with `task=True`, enabling native MCP Tasks protocol support for task-aware clients. FastMCP task lifecycle and progress are the source of truth for live execution. Persisted run artifacts are history/handoff records only.

## Configuration

The server automatically handles isolated environments for the Gemini CLI to ensure consistency and security. It replicates a minimal subset of your Gemini configuration while disabling hooks and skills to prevent unexpected side effects during MCP execution.

## Logging

- Runtime logs are written to `/tmp/gemini-mcp.log`.
- FastMCP clients that support tool logs and progress can observe live status updates such as initialization, tool usage, response generation, retries, and completion.
