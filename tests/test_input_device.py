"""Tests for the input-device resolution helper."""

import sys
import types
import unittest
from unittest.mock import patch


def _fake_sounddevice(devices, default_index):
    """Build a minimal sounddevice-shaped stub for monkeypatching."""

    module = types.SimpleNamespace()

    def query_devices(index=None, kind=None):  # noqa: ARG001
        if index is None and kind is None:
            return devices
        if isinstance(index, int):
            return devices[index]
        # kind="input" path: return the first device with input channels.
        for dev in devices:
            if dev.get("max_input_channels", 0) > 0:
                return dev
        raise RuntimeError("no input device")

    module.query_devices = query_devices
    module.default = types.SimpleNamespace(device=(default_index, default_index))
    return module


class ResolveInputDeviceTests(unittest.TestCase):
    DEVICES = [
        {"name": "HDA Intel PCH", "max_input_channels": 0},
        {"name": "Built-in Microphone", "max_input_channels": 2},
        {"name": "HyperX QuadCast", "max_input_channels": 1},
        {"name": "Monitor of Speakers", "max_input_channels": 2},
    ]

    def _patch(self, default_index=1):
        return patch.dict(
            sys.modules,
            {"sounddevice": _fake_sounddevice(self.DEVICES, default_index)},
        )

    def test_none_returns_default(self):
        with self._patch():
            from claude_stt.engines._audio import resolve_input_device

            idx, name = resolve_input_device(None)
        self.assertIsNone(idx)
        self.assertIn("Built-in Microphone", name)

    def test_empty_string_returns_default(self):
        with self._patch():
            from claude_stt.engines._audio import resolve_input_device

            idx, name = resolve_input_device("")
        self.assertIsNone(idx)
        self.assertIn("default", name.lower())

    def test_substring_match_case_insensitive(self):
        with self._patch():
            from claude_stt.engines._audio import resolve_input_device

            idx, name = resolve_input_device("hyperx")
        self.assertEqual(idx, 2)
        self.assertEqual(name, "HyperX QuadCast")

    def test_missing_device_falls_back_to_default(self):
        with self._patch():
            from claude_stt.engines._audio import resolve_input_device

            idx, name = resolve_input_device("Some Mic That Does Not Exist")
        self.assertIsNone(idx)
        self.assertIn("Built-in Microphone", name)


if __name__ == "__main__":
    unittest.main()
