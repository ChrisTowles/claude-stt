"""Common ASR engine interface.

Both backends emit the full *growing* transcript on every callback so the
daemon's prefix-diff typing logic in `daemon_service._on_text` works
identically across platforms.
"""

from __future__ import annotations

from typing import Protocol


class ASREngine(Protocol):
    def load(self) -> None: ...

    def start(self) -> bool: ...

    def stop(self) -> None: ...
