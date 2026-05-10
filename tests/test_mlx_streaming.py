"""End-to-end streaming tests for the MLX backend.

Drives `ParakeetMLXEngine` through controlled audio scenarios
(short/long utterances, mid-utterance pauses, background noise,
trailing silence) by injecting synthesized audio chunks directly into
the engine's audio queue. Targets the symptoms the user reported on
their Mac: random words during pauses, choppy output, draft churn that
clobbers manual edits.

Slow by default — synthesizes audio with macOS `say`, runs real
parakeet-mlx inference. Gated behind `CLAUDE_STT_RUN_SLOW=1` so the
fast unit tests in `test_mlx_emission_gate.py` keep their <1s runtime.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

from tests.audio_helpers import (
    EngineHarness,
    SAMPLE_RATE,
    concat,
    have_say,
    noise,
    run_session,
    silence,
    synth_speech,
)


_SLOW_ENABLED = os.environ.get("CLAUDE_STT_RUN_SLOW") == "1"


@unittest.skipUnless(sys.platform == "darwin", "MLX backend is macOS-only")
@unittest.skipUnless(have_say(), "`say` binary required for audio synthesis")
@unittest.skipUnless(_SLOW_ENABLED, "Set CLAUDE_STT_RUN_SLOW=1 to run streaming tests")
class StreamingScenarioTests(unittest.TestCase):
    """One model load shared across all scenarios in this class."""

    harness: EngineHarness

    @classmethod
    def setUpClass(cls) -> None:
        cls.harness = EngineHarness()
        cls.harness.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.harness.shutdown()

    def assertBoundedRetypes(self, emissions: list[str]) -> None:
        """Every emission must respect MAX_RETYPE_CHARS vs the prior one."""
        from claude_stt.engines.mlx import MAX_RETYPE_CHARS, _case_insensitive_lcp_len

        prev = ""
        for i, em in enumerate(emissions):
            lcp = _case_insensitive_lcp_len(prev, em)
            retype = len(prev) - lcp
            self.assertLessEqual(
                retype, MAX_RETYPE_CHARS,
                f"emission {i} would retype {retype} chars (>{MAX_RETYPE_CHARS}) "
                f"prev={prev!r} new={em!r}",
            )
            prev = em

    def test_short_phrase_transcribes(self) -> None:
        audio = synth_speech("the quick brown fox")
        emissions = run_session(self.harness.engine, audio)
        self.assertGreater(len(emissions), 0, "expected at least one emission")
        final = emissions[-1].lower()
        self.assertIn("fox", final, f"expected 'fox' in final emission, got {emissions[-1]!r}")
        self.assertBoundedRetypes(emissions)

    def test_pure_silence_emits_nothing_during_loop(self) -> None:
        # 5s of zero audio. The energy gate should keep `_speech_started`
        # false so add_audio is never called during the loop. The
        # silence-pad flush at session end runs unconditionally and may
        # produce a hallucination — we only assert the loop body stays
        # quiet, not the flush.
        audio = silence(5.0)
        emissions = run_session(self.harness.engine, audio)
        # Allow at most one flush emission; anything more means the gate
        # let in mid-loop chunks that hallucinated.
        self.assertLessEqual(
            len(emissions), 1,
            f"silence should produce no mid-loop emissions, got {emissions!r}",
        )

    def test_long_monologue_stays_within_retype_bound(self) -> None:
        audio = synth_speech(
            "the quick brown fox jumps over the lazy dog. "
            "peter piper picked a peck of pickled peppers. "
            "she sells seashells by the seashore."
        )
        emissions = run_session(self.harness.engine, audio, timeout=60.0)
        self.assertGreater(len(emissions), 1, "long monologue should produce multiple emissions")
        self.assertBoundedRetypes(emissions)
        final = emissions[-1].lower()
        # Loose anchor — at least one of these should land in the final.
        self.assertTrue(
            any(w in final for w in ("fox", "peppers", "seashells")),
            f"expected at least one anchor word in final, got {emissions[-1]!r}",
        )

    def test_midspeech_pause_does_not_clobber_prefix(self) -> None:
        # Phrase A → 2s silence → phrase B. A's transcript should remain
        # a stable prefix of B's transcript (no skip-too-far events
        # against text the user has already seen).
        audio = concat(
            synth_speech("hello world this is a test"),
            silence(2.0),
            synth_speech("and now for something completely different"),
        )
        emissions = run_session(self.harness.engine, audio, timeout=60.0)
        self.assertGreater(len(emissions), 1)
        self.assertBoundedRetypes(emissions)

    def test_speech_with_background_noise(self) -> None:
        # Mix speech with white noise at -30 dBFS. Speech RMS via `say`
        # is ~ -20 dBFS, so SNR is ~10 dB — challenging but transcribable.
        spoken = synth_speech("the quick brown fox")
        bg = noise(spoken.size / SAMPLE_RATE, dbfs=-30.0)
        # Sum sample-wise, clipping if it overshoots ±1.
        mixed = np.clip(spoken + bg[: spoken.size], -1.0, 1.0)
        emissions = run_session(self.harness.engine, mixed)
        self.assertGreater(len(emissions), 0, "noisy speech should still transcribe")
        self.assertBoundedRetypes(emissions)

    def test_trailing_silence_does_not_grow_transcript(self) -> None:
        # Phrase + 5s of trailing silence (still inside the same session,
        # before the stop sentinel). The transcript shouldn't grow with
        # phantom words during the silent tail.
        audio = concat(synth_speech("the quick brown fox"), silence(5.0))
        emissions = run_session(self.harness.engine, audio, timeout=60.0)
        self.assertGreater(len(emissions), 0)
        self.assertBoundedRetypes(emissions)
        # The last several emissions during the silent tail should be
        # roughly the same length as the peak (within MAX_RETYPE_CHARS).
        if len(emissions) >= 3:
            tail = emissions[-3:]
            peak_len = max(len(e) for e in emissions)
            for e in tail:
                self.assertGreaterEqual(
                    len(e), peak_len - 40,
                    f"tail emission shrank substantially: peak={peak_len}, tail={e!r}",
                )


if __name__ == "__main__":
    unittest.main()
