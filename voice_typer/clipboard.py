"""Clipboard management and safe auto-paste, cross-platform.

Key behavior:
- copy() ALWAYS puts text on the clipboard.
- paste() only sends a paste keystroke when it is safe to do so:
    * If focus detection is available (Windows): only paste when a text
      input is confirmed focused.  If not, silently skip.
    * If focus detection is NOT available (macOS / Linux): paste only
      when the caller explicitly opted in (paste_on_stop config).
- On any failure the clipboard text is preserved.
"""

import logging
import sys
import time

import pyperclip
from pynput.keyboard import Key, Controller

from voice_typer.focus import is_text_input_focused

log = logging.getLogger(__name__)


class ClipboardManager:
    """Handles copying text to clipboard and safely pasting into the focused app."""

    def __init__(self, paste_enabled: bool = True):
        self.paste_enabled = paste_enabled
        self._keyboard = Controller()

    def copy(self, text: str) -> bool:
        """Copy text to clipboard.  Returns True on success."""
        if not text:
            return False
        try:
            pyperclip.copy(text)
            log.info("Copied %d chars to clipboard", len(text))
            return True
        except Exception as e:
            log.error("Failed to copy to clipboard: %s", e)
            return False

    def paste(self) -> bool:
        """Attempt to paste into the focused field.

        Returns True if a keystroke was sent, False if paste was skipped
        (no text input focused, detection unavailable and paste disabled,
        or paste_enabled is False).

        On Windows, includes a short retry: if focus detection returns False
        on the first attempt (which can happen if focus is briefly disrupted
        by the recording-stop or tray-update event), we wait 200 ms and
        retry once.  This handles the common case where the original text
        field regains focus almost immediately.
        """
        if not self.paste_enabled:
            log.info("Paste disabled by config -- skipping keystroke")
            return False

        focused = is_text_input_focused()

        # Retry once on Windows if focus was just disrupted
        if focused is False and sys.platform == "win32":
            log.info(
                "No text input focused on first check -- "
                "retrying after 200 ms (focus may be briefly disrupted)"
            )
            time.sleep(0.2)
            focused = is_text_input_focused()

        if focused is False:
            log.info("No text input focused -- skipping paste (clipboard has the text)")
            return False
        if focused is None:
            log.info(
                "Focus detection unavailable -- attempting paste "
                "(set paste_on_stop=false to disable)"
            )
        # focused is True or None -> attempt paste

        try:
            time.sleep(0.1)  # let focused app settle
            if sys.platform == "darwin":
                self._keyboard.press(Key.cmd)
                self._keyboard.press("v")
                self._keyboard.release("v")
                self._keyboard.release(Key.cmd)
            else:
                self._keyboard.press(Key.ctrl)
                self._keyboard.press("v")
                self._keyboard.release("v")
                self._keyboard.release(Key.ctrl)

            log.info("Sent paste keystroke")
            return True
        except Exception as e:
            log.warning("Auto-paste failed (clipboard still has the text): %s", e)
            return False
