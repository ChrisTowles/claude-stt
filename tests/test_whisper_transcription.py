"""End-to-end Whisper transcription tests using real audio samples.

Downloads a LibriSpeech sample from HuggingFace and verifies Whisper
engine produces correct transcriptions.
"""

import io
import os
import tempfile
import unittest
from typing import ClassVar, Optional

import numpy as np

from claude_stt.engines.whisper import WhisperEngine, _whisper_available


def _download_sample_audio() -> tuple[np.ndarray, int]:
    """Download HuggingFace speech sample and return audio at 16kHz.

    Returns:
        Tuple of (audio array, sample_rate). Audio is float32 mono at 16kHz.
    """
    import httpx
    import soundfile as sf

    cache_path = os.path.join(tempfile.gettempdir(), "claude_stt_test_speech_16k.wav")
    if os.path.exists(cache_path):
        audio, sr = sf.read(cache_path, dtype="float32")
        return audio, sr

    # Download sample from HuggingFace CDN
    url = "https://cdn-media.huggingface.co/speech_samples/sample1.flac"
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    audio, sr = sf.read(io.BytesIO(resp.content))

    # Normalize to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Save to cache
    audio = audio.astype(np.float32)
    sf.write(cache_path, audio, sr)
    return audio, sr


@unittest.skipUnless(_whisper_available, "faster-whisper not installed")
class WhisperTranscriptionTests(unittest.TestCase):
    """Test Whisper engine produces correct transcriptions from real audio."""

    _engine: ClassVar[Optional[WhisperEngine]] = None
    _audio: ClassVar[Optional[np.ndarray]] = None
    _sample_rate: ClassVar[int] = 16000

    @classmethod
    def setUpClass(cls) -> None:
        """Load Whisper model and download sample audio."""
        cls._engine = WhisperEngine(model_name="medium")
        assert cls._engine.load_model(), "Failed to load Whisper model"
        cls._audio, cls._sample_rate = _download_sample_audio()

    def test_transcription_returns_nonempty_string(self) -> None:
        """Verify transcription returns non-empty string."""
        result = self._engine.transcribe(self._audio, self._sample_rate)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_transcription_contains_expected_words(self) -> None:
        """Verify transcription contains expected words from sample."""
        result = self._engine.transcribe(self._audio, self._sample_rate)
        lower = result.lower()
        # Sample contains: "going along slushy country roads..."
        for word in ["going", "country", "roads", "sunday"]:
            self.assertIn(word, lower)

    def test_transcription_is_not_garbled(self) -> None:
        """Verify output is real words, not random character mashing."""
        result = self._engine.transcribe(self._audio, self._sample_rate)
        # Real transcription should be mostly alpha + spaces + punctuation
        alpha_count = sum(1 for c in result if c.isalpha() or c.isspace())
        ratio = alpha_count / max(len(result), 1)
        self.assertGreater(ratio, 0.85)

    def test_silence_returns_valid_string(self) -> None:
        """Verify silence doesn't crash (may hallucinate, which is OK)."""
        silence = np.zeros(32000, dtype=np.float32)
        result = self._engine.transcribe(silence, 16000)
        self.assertIsInstance(result, str)

    def test_short_audio_does_not_crash(self) -> None:
        """Verify very short audio doesn't crash."""
        short = self._audio[:1600]  # 100ms
        result = self._engine.transcribe(short, self._sample_rate)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
