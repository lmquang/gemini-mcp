# /// script
# dependencies = [
#   "fastmcp>=3.2.4",
#   "fastmcp[tasks]",
#   "aiofiles",
# ]
# ///
"""Gemini Expert Assistant MCP Server (Standalone)

MCP interface for the Gemini CLI providing code exploration, analysis, planning,
documentation, and chat capabilities through MCP-compatible clients.

Usage Example (Claude Desktop Config):
{
  "mcpServers": {
    "gemini-mcp": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
"""

import os
import logging
import sys

LOG_FILE = os.environ.get("GEMINI_LOG_FILE", "/tmp/gemini-mcp.log")
try:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=LOG_FILE,
        filemode='a'
    )
except Exception:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

from gemini_mcp.server import main

if __name__ == "__main__":
    main()
