import unittest

from claude_stt.config import Config


class ConfigTests(unittest.TestCase):
    def test_excluded_apps_default(self):
        config = Config()
        self.assertEqual(config.excluded_apps, [])

    def test_excluded_apps_from_constructor(self):
        config = Config(excluded_apps=["Claude", "Zoom"])
        self.assertEqual(config.excluded_apps, ["Claude", "Zoom"])

    def test_default_model(self):
        config = Config()
        self.assertEqual(config.model, "nvidia/parakeet-tdt-0.6b-v2")
        self.assertEqual(config.chunk_ms, 320)
        self.assertEqual(config.context_seconds, 10.0)
        self.assertEqual(config.silence_threshold_dbfs, -45.0)

    def test_silence_threshold_clamped(self):
        self.assertEqual(Config(silence_threshold_dbfs=-200).validate().silence_threshold_dbfs, -120.0)
        self.assertEqual(Config(silence_threshold_dbfs=10).validate().silence_threshold_dbfs, 0.0)

    def test_config_validation_clamps_invalid_values(self):
        config = Config(
            mode="bad",
            output_mode="wat",
            max_recording_seconds=0,
            chunk_ms=10,
            context_seconds=0.1,
        ).validate()

        self.assertEqual(config.mode, "toggle")
        self.assertEqual(config.output_mode, "auto")
        self.assertEqual(config.max_recording_seconds, 1)
        self.assertEqual(config.chunk_ms, 80)
        self.assertEqual(config.context_seconds, 1.0)

    def test_config_validation_clamps_high_values(self):
        config = Config(
            chunk_ms=5000,
            context_seconds=999,
        ).validate()
        self.assertEqual(config.chunk_ms, 2000)
        self.assertEqual(config.context_seconds, 60.0)


if __name__ == "__main__":
    unittest.main()
