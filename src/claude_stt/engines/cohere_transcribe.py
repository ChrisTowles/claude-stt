"""Cohere Transcribe STT engine using transformers."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

_cohere_available = False

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    _cohere_available = True
except ImportError:
    pass


class CohereTranscribeEngine:
    """Speech-to-text engine backed by CohereLabs/cohere-transcribe."""

    DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or os.environ.get("CLAUDE_STT_DEVICE", "cpu")
        self._model = None
        self._processor = None
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        return _cohere_available

    def load_model(self) -> bool:
        if not self.is_available():
            return False
        if self._model is not None:
            return True
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()
            return True
        except Exception:
            self._logger.exception("Failed to load Cohere Transcribe model")
            return False

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str = "auto"
    ) -> str:
        if not self.load_model():
            return ""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            lang = language if language != "auto" else "en"

            texts = self._model.transcribe(
                processor=self._processor,
                audio_arrays=[audio],
                sample_rates=[sample_rate],
                language=lang,
            )
            return texts[0].strip() if texts else ""
        except Exception:
            self._logger.exception("Cohere Transcribe transcription failed")
            return ""
