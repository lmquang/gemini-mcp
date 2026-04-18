"""Gemini MCP server entry point."""

import argparse
import asyncio
import signal
import atexit
import logging
import sys
import os

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
logger = logging.getLogger("gemini-mcp")

from gemini_mcp.runner import active_processes


def cleanup():
    if not active_processes:
        return
    logger.info("Cleaning up %d active processes...", len(active_processes))
    for p in list(active_processes):
        try:
            p.kill()
        except Exception as e:
            logger.debug("Failed to kill process: %s", e)
    active_processes.clear()


def signal_handler(sig, frame):
    logger.info("Received signal %s, exiting...", sig)
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

from gemini_mcp.tools import mcp  # noqa: F401, E402


def main():
    parser = argparse.ArgumentParser(description="Gemini MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for streamable-http transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for streamable-http transport (default: 8000)"
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
