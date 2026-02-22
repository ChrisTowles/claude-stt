"""Text improvement using Claude CLI."""

import difflib
import logging
import subprocess

_logger = logging.getLogger(__name__)

# ANSI color codes
_RED = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"
_STRIKETHROUGH = "\033[9m"


def _colored_diff(original: str, improved: str) -> str:
    """Generate a colored inline diff between original and improved text.

    Shows removed words in red with strikethrough, added words in green.
    """
    orig_words = original.split()
    impr_words = improved.split()

    matcher = difflib.SequenceMatcher(None, orig_words, impr_words)
    result = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            result.extend(orig_words[i1:i2])
        else:
            # "replace" emits both removed and inserted; "delete"/"insert" emit one side
            if op in ("replace", "delete"):
                result.extend(
                    f"{_RED}{_STRIKETHROUGH}{w}{_RESET}" for w in orig_words[i1:i2]
                )
            if op in ("replace", "insert"):
                result.extend(f"{_GREEN}{w}{_RESET}" for w in impr_words[j1:j2])

    return " ".join(result)


_IMPROVE_PROMPT = (
    "Fix punctuation, capitalization, and grammar. "
    "Keep concise. Output only the corrected text, nothing else."
)


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
                "-p", f"{_IMPROVE_PROMPT}\n\nText: {text}",
            ],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            _logger.warning("claude CLI failed: %s", result.stderr)
            return text

        improved = result.stdout.strip()
        if improved != text:
            _logger.info("Text improved: %s", _colored_diff(text, improved))
        return improved
    except subprocess.TimeoutExpired:
        _logger.warning("claude CLI timed out")
    except FileNotFoundError:
        _logger.warning("claude CLI not found")
    except Exception:
        _logger.warning("Text improvement failed", exc_info=True)

    return text
