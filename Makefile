.PHONY: install dev run clean venv

# Installation
install: venv
	uv tool install --editable .

reinstall: venv
	uv tool install --force --editable .

venv:
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	fi

dev: venv
	@echo "Starting MCP Inspector (with hot-reload)..."
	uv run fastmcp dev inspector server.py

run: venv
	@echo "Starting Gemini MCP Server (stdio)..."
	uv run python -m gemini_mcp.server

run-http: venv
	@echo "Starting Gemini MCP Server (streamable-http)..."
	uv run python -m gemini_mcp.server --transport streamable-http

clean:
	@echo "Cleaning up..."
	rm -rf *.egg-info .pytest_cache .uv __pycache__ .venv
