"""Configuration constants for Gemini MCP."""

AVAILABLE_MODELS = [
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro", "tier": "smart", "usage": "Complex reasoning, code review, architecture."},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "tier": "balanced", "usage": "Fast exploration, analysis."},
    {"id": "gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash-Lite", "tier": "cheap", "usage": "Simple docs, quick answers."},
]

MODEL_TIERS = {
    "smart": ["gemini-3.1-pro-preview"],
    "balanced": ["gemini-3-flash-preview"],
    "cheap": ["gemini-3.1-flash-lite-preview", "gemini-3-flash-preview"],
}

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_READ_TIMEOUT = 120.0
HEARTBEAT_INTERVAL_SECONDS = 10.0
