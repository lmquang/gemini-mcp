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

#### Job Management Tools

Agentic tools (`explore`, `analyze`, `plan`, `document`) run as background jobs to avoid client timeouts. Use these tools to track and retrieve results.

- **`job_status`**: Check the status of a background job. Poll until `status` is `completed`, `failed`, or `cancelled`.
- **`job_result`**: Get the result of a completed job. Only call after `job_status` shows a terminal status.
- **`list_jobs`**: List all background jobs and their statuses.
- **`cancel_job`**: Cancel a running background job.

#### Gemini Model / Session Tools

- **`list_models`**: Lists available Gemini models with their tier and usage.
- **`list_sessions`**: Lists Gemini CLI sessions for the current project.

#### Chat (Fast — No Background Job)

- **`chat`**: Lightweight chat with Gemini — no repo indexing, no sandbox. Auto-resumes the managed session unless `sessionMode='new'` or `sessionID` is provided.

#### Agentic Tools (Background Jobs)

These tools start a background Gemini job and return a `jobID` immediately. Poll `job_status` and retrieve results with `job_result`.

- **`explore`**: Investigates the codebase, researching and mapping project structure.
- **`analyze`**: Performs technical code reviews for bugs, performance, security, and architectural issues.
- **`plan`**: Generates step-by-step implementation plans or architectural designs.
- **`document`**: Writes or updates documentation (supports `apply=true` to write changes).

### Workflow Pattern

1. Call an agentic tool (e.g., `explore`) — it returns a `jobID` immediately.
2. Poll `job_status(jobID)` until `status` is `completed`, `failed`, or `cancelled`.
3. Call `job_result(jobID)` to get the full result.

Example:

```
1. explore(prompt="Summarize the project structure") → {"ok": true, "jobID": "abc123", "status": "running"}
2. job_status(jobID="abc123")                        → {"status": "running", "elapsedSeconds": 15}
3. job_status(jobID="abc123")                        → {"status": "completed", "elapsedSeconds": 32}
4. job_result(jobID="abc123")                        → {"ok": true, "response": "...", "sessionID": "..."}
```

### Session Management

Most tools support seamless managed session reuse across calls, auto-resuming the managed session by default.

- Use `sessionMode="auto"` or omit `sessionMode` to reuse the current managed Gemini session for the same MCP connection and working directory.
- Use `sessionMode="new"` only when you need a fresh conversation boundary, such as switching tasks, avoiding stale context, or debugging session behavior.
- Pass `sessionID` to force a specific Gemini session explicitly.
- Agentic job start responses may include a preview `sessionID` when an existing managed session is already known; the final active session is always returned by `job_status` and `job_result`.

### Runtime Behavior

- Agentic tools start background jobs that return immediately with a `jobID`.
- `chat` runs synchronously (no background job needed — it's fast).
- Read-only agentic tools (`explore`, `analyze`, `plan`, and `document` with `apply=false`) run in `approval-mode plan` without full-process `--sandbox` so Gemini `stream-json` events can surface promptly.
- `document` with `apply=true` keeps the write/apply edit path and returns `mode: "applied"` on success.
- Use `sessionMode="new"` on any session-aware tool when you want to force a fresh managed session.

### MCP Tasks Protocol Support

Agentic tools are registered with `task=True`, enabling native MCP Tasks protocol support for task-aware clients. When a task-aware client calls these tools, FastMCP returns a `taskId` immediately via the MCP Tasks protocol. For non-task-aware clients, the tool still returns quickly with a `jobID` and the JobManager handles background execution.

## Configuration

The server automatically handles isolated environments for the Gemini CLI to ensure consistency and security. It replicates a minimal subset of your Gemini configuration while disabling hooks and skills to prevent unexpected side effects during MCP execution.

## Logging

- Runtime logs are written to `/tmp/gemini-mcp.log`.
- FastMCP clients that support tool logs and progress can observe live status updates such as initialization, tool usage, response generation, retries, and completion.
