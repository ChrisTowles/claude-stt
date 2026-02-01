"""Text improvement using Claude CLI."""

import logging
import subprocess

_logger = logging.getLogger(__name__)

IMPROVE_PROMPT = "Fix punctuation, capitalization, and grammar. Keep concise. Output only the corrected text, nothing else."


def improve_text(text: str, timeout: float = 30.0) -> str:
    """Improve transcribed text using Claude CLI.

    Args:
        text: Raw transcribed text.
        timeout: Max seconds to wait for response.

    Returns:
        Improved text, or original if improvement fails.
    """
    if not text.strip():
        return text

    try:
        result = subprocess.run(
            [
                "claude",
                "--model", "haiku",
                "--print",
                "--no-input",
                "-p", f"{IMPROVE_PROMPT}\n\nText: {text}",
            ],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            improved = result.stdout.strip()
            if improved:
                _logger.debug("Text improved: %r -> %r", text, improved)
                return improved
        _logger.warning("claude CLI failed: %s", result.stderr)
    except subprocess.TimeoutExpired:
        _logger.warning("claude CLI timed out")
    except FileNotFoundError:
        _logger.warning("claude CLI not found")
    except Exception:
        _logger.warning("Text improvement failed", exc_info=True)

    return text
