"""Shared audio helpers for the ASR engines.

Both backends drive `sounddevice.InputStream` from the same parameters and
both compute RMS level the same way. Keeping the duplication out of
`nemo.py` and `mlx.py` so the engine modules can focus on the parts that
are genuinely different (rolling raw-audio buffer vs streaming encoder
cache).
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def resolve_input_device(name: str | None) -> tuple[int | None, str]:
    """Resolve a configured input-device name to a PortAudio index + label.

    Returns `(device_index, friendly_name)`. `device_index` is `None` when
    the caller should let `sounddevice` pick the system default. The
    friendly name is suitable for logging.

    - `None`, `""`, or `"default"` (case-insensitive) -> system default.
    - Any other string is treated as a case-insensitive substring match
      against the names of input-capable devices. The first match wins.

    On any failure (PortAudio not available, no match, etc.) this falls
    back to the system default and logs a warning rather than raising,
    so a stale config entry can't take the daemon down at boot.
    """
    import sounddevice as sd

    requested = (name or "").strip()
    want_default = not requested or requested.lower() == "default"

    try:
        devices = sd.query_devices()
    except Exception:
        _logger.exception("sounddevice could not enumerate devices")
        return None, "system default (enumeration failed)"

    if want_default:
        return None, _default_input_name(sd, devices)

    needle = requested.lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        if needle in str(dev.get("name", "")).lower():
            return idx, str(dev["name"])

    available = ", ".join(
        str(d["name"]) for d in devices if d.get("max_input_channels", 0) > 0
    )
    _logger.warning(
        "input_device %r not found; falling back to system default. Available: %s",
        requested,
        available or "(none)",
    )
    return None, _default_input_name(sd, devices)


def _default_input_name(sd, devices) -> str:
    """Best-effort friendly name for whatever PortAudio will use by default."""
    try:
        default = sd.default.device
        idx = default[0] if isinstance(default, (tuple, list)) else default
        if isinstance(idx, int) and 0 <= idx < len(devices):
            return f"{devices[idx]['name']} (system default)"
        info = sd.query_devices(kind="input")
        return f"{info['name']} (system default)"
    except Exception:
        return "system default"


def rms_dbfs(chunk: np.ndarray) -> float:
    """RMS energy of a float32 chunk in dBFS (full-scale = 0 dB)."""
    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))) + 1e-12)
    return 20.0 * np.log10(rms)
