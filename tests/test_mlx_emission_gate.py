"""Tests for the MLX engine's append-only emission gate.

The gating function had a regression where any draft revision
(capitalization flicker, leading wake-word, punctuation insertion)
would cause `text.startswith(_last_emitted)` to flip False and silently
freeze the engine — the user only ever saw the first stable transcript
and nothing else for the rest of the session. These tests pin the
post-fix behavior: case-insensitive prefix match, append-only emission,
preservation of previously-typed casing, and skip-without-clobber when
the model retroactively revises an already-typed prefix.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from claude_stt.engines.mlx import ParakeetMLXEngine, _case_insensitive_lcp_len


@dataclass
class _FakeResult:
    text: str


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.result = _FakeResult(text)


def _make_engine() -> tuple[ParakeetMLXEngine, list[str]]:
    emitted: list[str] = []
    eng = ParakeetMLXEngine(on_text=emitted.append)
    return eng, emitted


class CaseInsensitiveLcpTests(unittest.TestCase):
    def test_empty_inputs(self):
        self.assertEqual(_case_insensitive_lcp_len("", ""), 0)
        self.assertEqual(_case_insensitive_lcp_len("", "abc"), 0)
        self.assertEqual(_case_insensitive_lcp_len("abc", ""), 0)

    def test_full_match(self):
        self.assertEqual(_case_insensitive_lcp_len("hello", "hello"), 5)

    def test_case_insensitive(self):
        self.assertEqual(_case_insensitive_lcp_len("what", "What is"), 4)
        self.assertEqual(_case_insensitive_lcp_len("HeLLo", "hello world"), 5)

    def test_partial_match(self):
        self.assertEqual(_case_insensitive_lcp_len("hello", "help"), 3)


class EmissionGateTests(unittest.TestCase):
    def test_initial_emission_appends(self):
        eng, emitted = _make_engine()
        eng._emit_if_extending(_FakeTranscriber("hello"))
        self.assertEqual(emitted, ["hello"])
        self.assertEqual(eng._last_emitted, "hello")

    def test_strict_extension(self):
        eng, emitted = _make_engine()
        eng._last_emitted = "hello"
        eng._emit_if_extending(_FakeTranscriber("hello world"))
        self.assertEqual(emitted, ["hello world"])

    def test_case_insensitive_extension_preserves_typed_case(self):
        # Regression: this scenario silently froze the engine before the
        # gating rewrite. "what" was already typed; the model later
        # capitalized + extended; we must keep the user's typed casing
        # ("what") and only append the new tail.
        eng, emitted = _make_engine()
        eng._last_emitted = "what"
        eng._emit_if_extending(_FakeTranscriber("What is happening"))
        self.assertEqual(emitted, ["what is happening"])
        self.assertEqual(eng._last_emitted, "what is happening")

    def test_identical_text_is_skipped(self):
        eng, emitted = _make_engine()
        eng._last_emitted = "hello"
        eng._emit_if_extending(_FakeTranscriber("hello"))
        self.assertEqual(emitted, [])

    def test_case_only_difference_is_skipped(self):
        eng, emitted = _make_engine()
        eng._last_emitted = "hello"
        eng._emit_if_extending(_FakeTranscriber("Hello"))
        self.assertEqual(emitted, [])
        self.assertEqual(eng._last_emitted, "hello")

    def test_short_retroactive_revision_is_emitted(self):
        # Model swaps a recent word ("world" → "earth"). The retype
        # window is small (5 chars), so we emit the correction. The
        # daemon's prefix-diff will issue 5 backspaces and retype.
        eng, emitted = _make_engine()
        eng._last_emitted = "hello world"
        eng._emit_if_extending(_FakeTranscriber("hello earth"))
        self.assertEqual(emitted, ["hello earth"])
        self.assertEqual(eng._last_emitted, "hello earth")

    def test_long_retroactive_revision_is_skipped(self):
        # Model wants to revise something far back in the transcript
        # (more than MAX_RETYPE_CHARS chars from the tail). We drop the
        # emission rather than risk clobbering manual edits the user
        # has made in the meantime.
        eng, emitted = _make_engine()
        eng._last_emitted = "hello world this is a longer running transcript example"
        # Diverges at "this" — that's >40 chars from the tail.
        eng._emit_if_extending(_FakeTranscriber("hello world THAT was a longer running transcript example"))
        self.assertEqual(emitted, [])
        self.assertEqual(
            eng._last_emitted,
            "hello world this is a longer running transcript example",
        )

    def test_shrinking_text_is_skipped(self):
        eng, emitted = _make_engine()
        eng._last_emitted = "hello world"
        eng._emit_if_extending(_FakeTranscriber("hello"))
        self.assertEqual(emitted, [])
        self.assertEqual(eng._last_emitted, "hello world")

    def test_empty_text_is_skipped(self):
        eng, emitted = _make_engine()
        eng._last_emitted = "hello"
        eng._emit_if_extending(_FakeTranscriber(""))
        self.assertEqual(emitted, [])
        self.assertEqual(eng._last_emitted, "hello")

    def test_long_running_session_keeps_extending(self):
        # Smoke test that successive monotonic extensions all emit.
        eng, emitted = _make_engine()
        for text in ["hello", "hello world", "hello world this", "hello world this is a test"]:
            eng._emit_if_extending(_FakeTranscriber(text))
        self.assertEqual(
            emitted,
            ["hello", "hello world", "hello world this", "hello world this is a test"],
        )

    def test_callback_exception_does_not_freeze_state(self):
        def boom(_t: str) -> None:
            raise RuntimeError("downstream error")

        eng = ParakeetMLXEngine(on_text=boom)
        # The exception must be swallowed, but _last_emitted advances so
        # we don't loop endlessly re-emitting the same text.
        eng._emit_if_extending(_FakeTranscriber("hello"))
        self.assertEqual(eng._last_emitted, "hello")


if __name__ == "__main__":
    unittest.main()
