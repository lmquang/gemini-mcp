"""Exception classes for Gemini MCP."""


class GeminiError(Exception):
    """Base exception for Gemini MCP errors."""

    def __init__(self, message: str, exit_code: int | None = None, stdout: str = "", session_id: str | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.session_id = session_id


class GeminiAuthError(GeminiError):
    """Authentication failure with Gemini CLI."""


class GeminiTimeoutError(GeminiError):
    """Gemini CLI execution timed out."""


class GeminiRateLimitError(GeminiError):
    """Rate limit or capacity error from Gemini API."""
