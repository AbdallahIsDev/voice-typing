"""Tests for clipboard copy and safe-paste logic."""

import pytest
from unittest.mock import patch, MagicMock

from voice_typer.clipboard import ClipboardManager


class TestCopy:
    def test_copy_puts_text_on_clipboard(self, monkeypatch):
        copied = []
        monkeypatch.setattr("voice_typer.clipboard.pyperclip", MagicMock())
        import voice_typer.clipboard as mod
        mod.pyperclip = MagicMock()

        cm = ClipboardManager(paste_enabled=False)
        result = cm.copy("hello world")

        assert result is True
        mod.pyperclip.copy.assert_called_once_with("hello world")

    def test_copy_returns_false_for_empty_text(self):
        cm = ClipboardManager(paste_enabled=False)
        assert cm.copy("") is False
        assert cm.copy(None) is False

    def test_copy_returns_false_on_exception(self, monkeypatch):
        import voice_typer.clipboard as mod
        mod.pyperclip = MagicMock()
        mod.pyperclip.copy.side_effect = Exception("clipboard locked")

        cm = ClipboardManager(paste_enabled=False)
        assert cm.copy("test") is False


class TestPaste:
    @patch("voice_typer.clipboard.is_text_input_focused", return_value=True)
    def test_paste_sends_keystroke_when_focused(self, mock_focus, monkeypatch):
        import voice_typer.clipboard as mod
        mod.time = MagicMock()

        cm = ClipboardManager(paste_enabled=True)
        cm._keyboard = MagicMock()

        result = cm.paste()

        assert result is True
        cm._keyboard.press.assert_called()
        cm._keyboard.release.assert_called()

    @patch("voice_typer.clipboard.is_text_input_focused", return_value=False)
    def test_paste_skips_when_no_focus_windows(self, mock_focus, monkeypatch):
        """On Windows, paste is skipped after retry when focus stays False."""
        import voice_typer.clipboard as mod
        mod.time = MagicMock()
        monkeypatch.setattr("voice_typer.clipboard.sys.platform", "win32")
        cm = ClipboardManager(paste_enabled=True)
        cm._keyboard = MagicMock()

        result = cm.paste()

        assert result is False
        cm._keyboard.press.assert_not_called()
        assert mock_focus.call_count == 2  # initial + retry

    def test_paste_skips_when_disabled(self):
        cm = ClipboardManager(paste_enabled=False)
        cm._keyboard = MagicMock()

        result = cm.paste()

        assert result is False
        cm._keyboard.press.assert_not_called()

    @patch("voice_typer.clipboard.is_text_input_focused", return_value=None)
    def test_paste_attempts_when_focus_unknown(self, mock_focus, monkeypatch):
        import voice_typer.clipboard as mod
        mod.time = MagicMock()

        cm = ClipboardManager(paste_enabled=True)
        cm._keyboard = MagicMock()

        result = cm.paste()

        # When focus is unknown (None), we still attempt paste
        assert result is True

    @patch("voice_typer.clipboard.is_text_input_focused")
    def test_paste_retries_on_windows_when_focus_disrupted(self, mock_focus, monkeypatch):
        """On Windows, if focus returns False then True on retry, paste succeeds."""
        import voice_typer.clipboard as mod
        mod.time = MagicMock()
        monkeypatch.setattr("voice_typer.clipboard.sys.platform", "win32")
        # Simulate: first check False (disrupted), second check True (recovered)
        mock_focus.side_effect = [False, True]

        cm = ClipboardManager(paste_enabled=True)
        cm._keyboard = MagicMock()

        result = cm.paste()

        assert result is True
        cm._keyboard.press.assert_called()
        assert mock_focus.call_count == 2

    @patch("voice_typer.clipboard.is_text_input_focused")
    def test_paste_no_retry_on_non_windows(self, mock_focus, monkeypatch):
        """On non-Windows, there is no retry — paste is skipped immediately."""
        import voice_typer.clipboard as mod
        mod.time = MagicMock()
        monkeypatch.setattr("voice_typer.clipboard.sys.platform", "linux")
        # Even though the second call would return True, on non-Windows
        # we never retry, so paste is skipped.
        mock_focus.side_effect = [False, True]

        cm = ClipboardManager(paste_enabled=True)
        cm._keyboard = MagicMock()

        result = cm.paste()

        assert result is False
        cm._keyboard.press.assert_not_called()
        assert mock_focus.call_count == 1

    @patch("voice_typer.clipboard.is_text_input_focused", return_value=True)
    def test_paste_returns_false_on_keyboard_error(self, mock_focus, monkeypatch):
        import voice_typer.clipboard as mod
        mod.time = MagicMock()

        cm = ClipboardManager(paste_enabled=True)
        cm._keyboard = MagicMock()
        cm._keyboard.press.side_effect = Exception("keyboard error")

        result = cm.paste()

        assert result is False
