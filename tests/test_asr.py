"""Tests for the silence gate helper used by both engines."""

import unittest

import numpy as np

from claude_stt.engines._audio import rms_dbfs


class RmsDbfsTests(unittest.TestCase):
    def test_silence_is_floor(self):
        self.assertLess(rms_dbfs(np.zeros(800, dtype=np.float32)), -100.0)

    def test_full_scale_is_zero(self):
        # A full-scale square wave has RMS = 1.0 → 0 dBFS
        chunk = np.ones(800, dtype=np.float32)
        self.assertAlmostEqual(rms_dbfs(chunk), 0.0, places=4)

    def test_half_scale_is_minus_six(self):
        # RMS of 0.5 ≈ -6 dBFS
        chunk = np.full(800, 0.5, dtype=np.float32)
        self.assertAlmostEqual(rms_dbfs(chunk), -6.0, places=1)

    def test_noise_below_threshold(self):
        # Quiet white noise (-50 dBFS-ish) sits below the default gate.
        rng = np.random.default_rng(0)
        chunk = rng.normal(0.0, 0.003, size=8000).astype(np.float32)
        self.assertLess(rms_dbfs(chunk), -45.0)

    def test_speech_level_above_threshold(self):
        # Audio at typical speech RMS (~ -25 dBFS) clears the gate easily.
        rng = np.random.default_rng(1)
        chunk = rng.normal(0.0, 0.05, size=8000).astype(np.float32)
        self.assertGreater(rms_dbfs(chunk), -45.0)


if __name__ == "__main__":
    unittest.main()
