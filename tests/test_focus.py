"""Tests for focus detection module.

Tests cover:
- _class_matches helper: substring matching for class names (no mocking needed)
- API contract: is_text_input_focused returns True, False, or None
- Modern app class names are in the detection sets
- Negative cases where paste should be skipped
- Non-Windows platforms return None

Note: The full Win32 focus detection path (caret heuristic, ancestor walk)
is best verified through runtime log evidence rather than unit tests with
heavy ctypes mocking (which causes stack overflow with MagicMock).
"""

import sys
import pytest

from voice_typer.focus import (
    is_text_input_focused,
    _class_matches,
    _get_process_name,
    _WINDOWS_TEXT_CLASSES,
    _ANCESTOR_TEXT_APP_CLASSES,
    _TEXT_PROCESS_NAMES,
)


# ─── _class_matches helper ─────────────────────────────────────────────

class TestClassMatches:
    """Test the _class_matches substring matching helper."""

    def test_exact_match_edit(self):
        assert _class_matches("edit", _WINDOWS_TEXT_CLASSES) is True

    def test_substring_match_chrome_widgetwin(self):
        """'chrome_widgetwin_1' contains 'chrome_widgetwin_'."""
        assert _class_matches("chrome_widgetwin_1", _WINDOWS_TEXT_CLASSES) is True

    def test_richedit_variant(self):
        """'richedit20w' contains 'richedit'."""
        assert _class_matches("richedit20w", _WINDOWS_TEXT_CLASSES) is True

    def test_renderwidgethost_variant(self):
        """'chrome_renderwidgethosthwnd' contains 'renderwidgethost'."""
        assert _class_matches(
            "chrome_renderwidgethosthwnd", _WINDOWS_TEXT_CLASSES
        ) is True

    def test_console_window(self):
        assert _class_matches("consolewindowclass", _WINDOWS_TEXT_CLASSES) is True

    def test_internet_explorer_server(self):
        assert _class_matches("internetexplorer_server", _WINDOWS_TEXT_CLASSES) is True

    def test_scintilla(self):
        assert _class_matches("scintilla", _WINDOWS_TEXT_CLASSES) is True

    def test_tmemo(self):
        assert _class_matches("tmemo", _WINDOWS_TEXT_CLASSES) is True

    def test_conemu(self):
        assert _class_matches("conemu", _WINDOWS_TEXT_CLASSES) is True

    def test_swt_window(self):
        assert _class_matches("swt_window", _WINDOWS_TEXT_CLASSES) is True

    def test_sun_awt(self):
        """'sun_awt_canvas' contains 'sun_awt'."""
        assert _class_matches("sun_awt_canvas", _WINDOWS_TEXT_CLASSES) is True

    def test_no_match_button(self):
        assert _class_matches("button", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_empty_string(self):
        assert _class_matches("", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_desktop(self):
        """'progman' (desktop) should NOT match."""
        assert _class_matches("progman", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_taskbar(self):
        """'shell_traywnd' (taskbar) should NOT match."""
        assert _class_matches("shell_traywnd", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_static(self):
        """'static' (label) should NOT match."""
        assert _class_matches("static", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_dialog(self):
        """'#32770' (standard dialog) should NOT match."""
        assert _class_matches("#32770", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_listbox(self):
        """'listbox' should NOT match."""
        assert _class_matches("listbox", _WINDOWS_TEXT_CLASSES) is False

    def test_no_match_combobox_without_edit(self):
        """'combobox' alone should NOT match (not a text edit)."""
        assert _class_matches("combobox", _WINDOWS_TEXT_CLASSES) is False


# ─── Ancestor class set coverage ─────────────────────────────────────

class TestAncestorClassCoverage:
    """Verify that _ANCESTOR_TEXT_APP_CLASSES contains the expected entries."""

    def test_chrome_widgetwin_in_ancestor_set(self):
        assert _class_matches("chrome_widgetwin_1", _ANCESTOR_TEXT_APP_CLASSES) is True

    def test_console_in_ancestor_set(self):
        assert _class_matches("consolewindowclass", _ANCESTOR_TEXT_APP_CLASSES) is True

    def test_conemu_in_ancestor_set(self):
        assert _class_matches("conemu", _ANCESTOR_TEXT_APP_CLASSES) is True

    def test_swt_window_in_ancestor_set(self):
        assert _class_matches("swt_window", _ANCESTOR_TEXT_APP_CLASSES) is True

    def test_desktop_not_in_ancestor_set(self):
        assert _class_matches("progman", _ANCESTOR_TEXT_APP_CLASSES) is False

    def test_taskbar_not_in_ancestor_set(self):
        assert _class_matches("shell_traywnd", _ANCESTOR_TEXT_APP_CLASSES) is False


# ─── Safety: overly generic patterns should NOT be in the sets ──────────

class TestNoOverlyGenericPatterns:
    """Ensure the detection sets don't contain patterns that would match
    too broadly (false positives)."""

    def test_no_generic_input(self):
        """'input' is too generic — would match anything with 'input'."""
        assert "input" not in _WINDOWS_TEXT_CLASSES

    def test_no_generic_qt(self):
        """'qt' is too generic — would match Qt buttons, menus, etc."""
        assert "qt" not in _WINDOWS_TEXT_CLASSES

    def test_no_generic_gtk(self):
        """'gtk' is too generic."""
        assert "gtk" not in _WINDOWS_TEXT_CLASSES

    def test_no_generic_afx(self):
        """'afx' is too generic — matches MFC control bars, toolbars."""
        assert "afx" not in _WINDOWS_TEXT_CLASSES

    def test_no_generic_chromium(self):
        """'chromium' is too broad — matches Chromium popups/tooltips."""
        assert "chromium" not in _WINDOWS_TEXT_CLASSES

    def test_no_generic_xaml(self):
        """'xaml' is too broad — matches XAML buttons, grids, etc."""
        assert "xaml" not in _WINDOWS_TEXT_CLASSES


# ─── Process-name detection set coverage ────────────────────────────

class TestProcessNameCoverage:
    """Verify that _TEXT_PROCESS_NAMES contains the expected entries."""

    def test_windows_terminal_in_set(self):
        assert "windowsterminal.exe" in _TEXT_PROCESS_NAMES

    def test_warp_in_set(self):
        assert "warp.exe" in _TEXT_PROCESS_NAMES

    def test_alacritty_in_set(self):
        assert "alacritty.exe" in _TEXT_PROCESS_NAMES

    def test_wezterm_in_set(self):
        assert "wezterm-gui.exe" in _TEXT_PROCESS_NAMES

    def test_cmd_in_set(self):
        assert "cmd.exe" in _TEXT_PROCESS_NAMES

    def test_powershell_in_set(self):
        assert "powershell.exe" in _TEXT_PROCESS_NAMES

    def test_pwsh_in_set(self):
        assert "pwsh.exe" in _TEXT_PROCESS_NAMES

    def test_notepad_in_set(self):
        assert "notepad.exe" in _TEXT_PROCESS_NAMES

    def test_notepadpp_in_set(self):
        assert "notepad++.exe" in _TEXT_PROCESS_NAMES

    def test_no_generic_explorer(self):
        """explorer.exe should NOT be in the set (not primarily a text app)."""
        assert "explorer.exe" not in _TEXT_PROCESS_NAMES

    def test_no_generic_chrome(self):
        """chrome.exe should NOT be in the set — it's caught by class name."""
        assert "chrome.exe" not in _TEXT_PROCESS_NAMES

    def test_all_entries_are_lowercase_exe(self):
        """Every entry should be lowercase and end with .exe."""
        for name in _TEXT_PROCESS_NAMES:
            assert name == name.lower(), f"{name!r} is not lowercase"
            assert name.endswith(".exe"), f"{name!r} does not end with .exe"


# ─── _get_process_name on Windows ────────────────────────────────────

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
class TestGetProcessName:
    """Test the _get_process_name helper on Windows."""

    def test_returns_string_for_valid_hwnd(self):
        """For a valid hwnd, should return a non-None lowercase .exe name."""
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        # Use the desktop window as a known-valid hwnd
        hwnd = user32.GetDesktopWindow()
        result = _get_process_name(user32, kernel32, hwnd)
        # Desktop window may not have a process we can query,
        # so just check the type when it returns something
        if result is not None:
            assert isinstance(result, str)
            assert result == result.lower()
            assert result.endswith(".exe")

    def test_returns_none_for_null_hwnd(self):
        """A null hwnd should return None."""
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        result = _get_process_name(user32, kernel32, 0)
        assert result is None

    def test_returns_none_for_invalid_hwnd(self):
        """An obviously invalid hwnd should return None."""
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        result = _get_process_name(user32, kernel32, 0xDEADBEEF)
        assert result is None


# ─── API contract ─────────────────────────────────────────────────────

class TestFocusDetectionAPI:
    """Test the API contract: is_text_input_focused returns True, False, or None."""

    def test_returns_bool_or_none(self):
        """is_text_input_focused() must return bool or None."""
        result = is_text_input_focused()
        assert result is None or isinstance(result, bool)


# ─── Non-Windows platforms ────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows test")
class TestNonWindowsFocusDetection:
    def test_returns_none_on_non_windows(self):
        """On macOS/Linux, focus detection returns None (unknown)."""
        result = is_text_input_focused()
        assert result is None
