"""Qwen3-ASR STT engine using qwen-asr with vLLM streaming."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

_qwen_available = False

try:
    from qwen_asr import Qwen3ASRModel

    _qwen_available = True
except ImportError:
    pass


def _detect_device() -> str:
    """Auto-detect the best available device: CUDA -> MPS -> CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class QwenASREngine:
    """Speech-to-text engine backed by Qwen3-ASR with vLLM streaming."""

    DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or os.environ.get("CLAUDE_STT_DEVICE") or _detect_device()
        self._model = None
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        return _qwen_available

    def load_model(self) -> bool:
        if not self.is_available():
            return False
        if self._model is not None:
            return True
        try:
            self._model = Qwen3ASRModel.LLM(
                self.model_name,
                gpu_memory_utilization=0.7,
                max_model_len=4096,
            )
            self._logger.info("Loaded vLLM model on device: %s", self.device)
            return True
        except Exception:
            self._logger.exception("Failed to load Qwen ASR model")
            return False

    def init_streaming(self, chunk_size_sec: float = 2.0):
        """Initialize a streaming transcription session.

        Returns:
            ASRStreamingState object, or None on failure.
        """
        if not self.load_model():
            return None
        try:
            return self._model.init_streaming_state(
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=2,
                unfixed_token_num=5,
            )
        except Exception:
            self._logger.exception("Failed to init streaming state")
            return None

    def stream_chunk(self, audio: np.ndarray, state) -> str:
        """Feed an audio chunk to the streaming session.

        Returns:
            Current partial transcription text.
        """
        if not self._model or state is None:
            return ""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            state = self._model.streaming_transcribe(audio, state)
            return (state.text or "").strip()
        except Exception:
            self._logger.exception("Streaming transcription failed")
            return ""

    def finish_stream(self, state) -> str:
        """Finalize a streaming session and get final transcription.

        Returns:
            Final transcription text.
        """
        if not self._model or state is None:
            return ""
        try:
            state = self._model.finish_streaming_transcribe(state)
            return (state.text or "").strip()
        except Exception:
            self._logger.exception("Failed to finish streaming transcription")
            return ""

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str = "auto"
    ) -> str:
        """Batch transcribe (non-streaming fallback)."""
        if not self.load_model():
            return ""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            results = self._model.transcribe(
                audio=(audio, sample_rate),
                language="English",
            )
            if results and hasattr(results[0], "text"):
                return results[0].text.strip()
            return str(results[0]).strip() if results else ""
        except Exception:
            self._logger.exception("Qwen ASR transcription failed")
            return ""
