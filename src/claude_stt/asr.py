"""ASR engine entrypoint.

Re-exports the cross-platform engine factory. The actual NeMo (CUDA) and
MLX (Apple Silicon) implementations live in `claude_stt.engines`.
"""

from .engines import ASREngine, create_engine

__all__ = ["ASREngine", "create_engine"]
