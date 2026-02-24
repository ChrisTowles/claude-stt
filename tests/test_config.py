import unittest

from claude_stt.config import Config


class ConfigTests(unittest.TestCase):
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
