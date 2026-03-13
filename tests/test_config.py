import unittest

from claude_stt.config import Config


class ConfigTests(unittest.TestCase):
    def test_excluded_apps_default(self):
        config = Config()
        self.assertEqual(config.excluded_apps, [])

    def test_excluded_apps_from_constructor(self):
        config = Config(excluded_apps=["Claude", "Zoom"])
        self.assertEqual(config.excluded_apps, ["Claude", "Zoom"])

    def test_improve_hotkey_default(self):
        config = Config()
        self.assertEqual(config.improve_hotkey, "cmd+alt+d")

    def test_config_validation_clamps_invalid_values(self):
        config = Config(
            mode="bad",
            output_mode="wat",
            max_recording_seconds=0,
            sample_rate=8000,
        ).validate()

        self.assertEqual(config.mode, "toggle")
        self.assertEqual(config.output_mode, "auto")
        self.assertEqual(config.max_recording_seconds, 1)
        self.assertEqual(config.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
