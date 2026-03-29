"""Qwen3-ASR STT engine using qwen-asr."""

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
    """Speech-to-text engine backed by Qwen3-ASR."""

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
            import torch

            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=self.device,
            )
            self._logger.info("Loaded model on device: %s", self.device)
            return True
        except Exception:
            self._logger.exception("Failed to load Qwen ASR model")
            return False

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str = "auto"
    ) -> str:
        if not self.load_model():
            return ""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            kwargs = {"language": "English"}

            results = self._model.transcribe(
                audio=(audio, sample_rate),
                **kwargs,
            )
            if results and hasattr(results[0], "text"):
                return results[0].text.strip()
            return str(results[0]).strip() if results else ""
        except Exception:
            self._logger.exception("Qwen ASR transcription failed")
            return ""
