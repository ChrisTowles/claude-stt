"""Tests for streaming text output logic in STTDaemon._on_text."""

import unittest
from unittest.mock import patch, call

from claude_stt.daemon_service import STTDaemon
from claude_stt.config import Config


class StreamingTextTests(unittest.TestCase):
    """Test the _on_text method that handles interim/final text from the ASR engine."""

    def setUp(self):
        config = Config(hotkey="ctrl+shift+space")
        self.daemon = STTDaemon(config=config)
        self.daemon._recording = True

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_first_text_typed_in_full(self, mock_type, mock_delete):
        self.daemon._on_text("hello world")
        mock_type.assert_called_once_with("hello world")
        mock_delete.assert_not_called()
        self.assertEqual(self.daemon._typed_text, "hello world")

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_append_new_words(self, mock_type, mock_delete):
        self.daemon._on_text("hello")
        mock_type.reset_mock()

        self.daemon._on_text("hello world")
        mock_type.assert_called_once_with(" world")
        mock_delete.assert_not_called()
        self.assertEqual(self.daemon._typed_text, "hello world")

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_correction_deletes_tail_and_retypes(self, mock_type, mock_delete):
        self.daemon._on_text("I think this would be")
        mock_type.reset_mock()
        mock_delete.reset_mock()

        self.daemon._on_text("I think this will be")
        # Common prefix: "I think this w" (14 chars)
        # Delete: 21 - 14 = 7 chars ("ould be")
        # Type: "ill be" (6 chars)
        mock_delete.assert_called_once_with(7)
        mock_type.assert_called_once_with("ill be")
        self.assertEqual(self.daemon._typed_text, "I think this will be")

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_append_after_correction(self, mock_type, mock_delete):
        """After a correction, further appends should work."""
        self.daemon._on_text("I think this would")
        self.daemon._on_text("I think this will be")
        mock_type.reset_mock()
        mock_delete.reset_mock()

        self.daemon._on_text("I think this will be great")
        mock_type.assert_called_once_with(" great")
        mock_delete.assert_not_called()

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_duplicate_text_ignored(self, mock_type, mock_delete):
        self.daemon._on_text("hello")
        mock_type.reset_mock()

        self.daemon._on_text("hello")
        mock_type.assert_not_called()
        mock_delete.assert_not_called()

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_empty_text_ignored(self, mock_type, mock_delete):
        self.daemon._on_text("")
        mock_type.assert_not_called()

        self.daemon._on_text("   ")
        mock_type.assert_not_called()

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_not_recording_ignored(self, mock_type, mock_delete):
        self.daemon._recording = False
        self.daemon._on_text("hello world")
        mock_type.assert_not_called()

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_multiple_appends(self, mock_type, mock_delete):
        self.daemon._on_text("one")
        self.daemon._on_text("one two")
        self.daemon._on_text("one two three")
        self.assertEqual(mock_type.call_args_list, [
            call("one"),
            call(" two"),
            call(" three"),
        ])
        mock_delete.assert_not_called()
        self.assertEqual(self.daemon._chars_typed, 13)

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_complete_rewrite(self, mock_type, mock_delete):
        """Entirely different text should delete all and retype."""
        self.daemon._on_text("hello")
        mock_type.reset_mock()
        mock_delete.reset_mock()

        self.daemon._on_text("goodbye")
        # Common prefix: "" (0 chars) — 'h' != 'g'
        mock_delete.assert_called_once_with(5)
        mock_type.assert_called_once_with("goodbye")

    @patch("claude_stt.daemon_service.delete_chars")
    @patch("claude_stt.daemon_service.type_text_streaming")
    def test_trailing_word_correction(self, mock_type, mock_delete):
        """Chrome corrects last word while keeping prefix."""
        self.daemon._on_text("the quick brow")
        mock_type.reset_mock()
        mock_delete.reset_mock()

        self.daemon._on_text("the quick brown fox")
        # "the quick brow" vs "the quick brown fox"
        # Common prefix: "the quick brow" (14 chars)
        # This is an append case since new starts with old
        mock_type.assert_called_once_with("n fox")
        mock_delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
