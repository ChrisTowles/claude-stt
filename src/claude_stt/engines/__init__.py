"""ASR engine factory.

Two backends share a common shape (load/start/stop, on_text/on_error
callbacks emitting the *growing* transcript so the daemon's prefix-diff
typing logic works for either):

- `nemo`  — NVIDIA Parakeet via NeMo + CUDA (Linux/Windows).
- `mlx`   — Parakeet-MLX on Apple Silicon (macOS).

`create_engine` picks the backend based on `sys.platform`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from .base import ASREngine


def create_engine(
    *,
    model_id: str,
    chunk_ms: int,
    context_seconds: float,
    silence_threshold_dbfs: float,
    silence_reset_seconds: float = 1.5,
    input_device: str | None = None,
    on_text: Callable[[str], None],
    on_error: Callable[[str], None],
) -> ASREngine:
    if sys.platform == "darwin":
        from .mlx import ParakeetMLXEngine

        return ParakeetMLXEngine(
            model_id=_resolve_model_id(model_id, default_for="darwin"),
            chunk_ms=chunk_ms,
            context_seconds=context_seconds,
            silence_threshold_dbfs=silence_threshold_dbfs,
            silence_reset_seconds=silence_reset_seconds,
            input_device=input_device,
            on_text=on_text,
            on_error=on_error,
        )

    from .nemo import ParakeetEngine

    return ParakeetEngine(
        model_id=_resolve_model_id(model_id, default_for="nemo"),
        chunk_ms=chunk_ms,
        context_seconds=context_seconds,
        silence_threshold_dbfs=silence_threshold_dbfs,
        silence_reset_seconds=silence_reset_seconds,
        input_device=input_device,
        on_text=on_text,
        on_error=on_error,
    )


_NEMO_DEFAULT = "nvidia/parakeet-tdt-0.6b-v2"
_MLX_DEFAULT = "mlx-community/parakeet-tdt-0.6b-v3"


def _resolve_model_id(configured: str, *, default_for: str) -> str:
    """Map cross-platform default model ids to per-backend equivalents.

    Users carry a single `model` config across machines, so the NVIDIA-flavored
    default needs to land on the matching MLX repo on macOS without forcing
    them to edit the file.
    """
    if default_for == "darwin" and configured.startswith("nvidia/"):
        return _MLX_DEFAULT
    if default_for == "nemo" and configured.startswith("mlx-community/"):
        return _NEMO_DEFAULT
    return configured


__all__ = ["ASREngine", "create_engine"]
