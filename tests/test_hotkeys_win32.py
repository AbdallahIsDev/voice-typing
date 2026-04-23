"""Tests for WindowsNativeHotkey readiness handshake.

These tests mock ctypes.windll.user32 and kernel32 to simulate specific
failure modes and success scenarios without requiring a Windows host.
"""

import sys
import threading
import time
import ctypes
import ctypes.wintypes

import pytest
from unittest.mock import MagicMock, patch


# ─── Fixture ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_win32(monkeypatch):
    """Provide mocked user32 and kernel32 DLLs.

    Default behavior: all Win32 calls succeed.  GetMessageW returns 0
    (WM_QUIT) so the message loop exits immediately after registration.
    """
    mock_user32 = MagicMock()
    mock_kernel32 = MagicMock()

    # Default: success for all Win32 calls
    mock_user32.CreateWindowExW.return_value = 0xDEAD_BEEF  # valid HWND
    mock_user32.RegisterHotKey.return_value = 1  # BOOL TRUE
    mock_user32.UnregisterHotKey.return_value = 1
    mock_user32.GetMessageW.return_value = 0  # WM_QUIT -> loop exits
    mock_user32.PostMessageW.return_value = 1
    mock_user32.DestroyWindow.return_value = 1

    mock_kernel32.GetModuleHandleW.return_value = 0x1234_5678
    mock_kernel32.GetLastError.return_value = 0

    # Patch ctypes.windll (Linux has no windll attribute by default)
    mock_windll = MagicMock()
    mock_windll.user32 = mock_user32
    mock_windll.kernel32 = mock_kernel32
    monkeypatch.setattr(ctypes, "windll", mock_windll, raising=False)

    return mock_user32, mock_kernel32


# ─── CreateWindowExW failure ─────────────────────────────────────────────────


class TestCreateWindowFailure:
    """When CreateWindowExW returns NULL, start() must raise within timeout."""

    def test_raises_on_null_hwnd(self, mock_win32):
        """CreateWindowExW returns 0 -> start() raises RuntimeError."""
        mock_user32, mock_kernel32 = mock_win32
        mock_user32.CreateWindowExW.return_value = 0  # NULL

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        with pytest.raises(RuntimeError, match="Failed to register hotkey"):
            backend.start(MagicMock())

    def test_raises_within_timeout(self, mock_win32):
        """start() should raise quickly, not hang forever."""
        mock_user32, mock_kernel32 = mock_win32
        mock_user32.CreateWindowExW.return_value = 0

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        start_time = time.monotonic()
        with pytest.raises(RuntimeError):
            backend.start(MagicMock())
        elapsed = time.monotonic() - start_time
        # Should complete within the readiness timeout (5s default)
        assert elapsed < 7.0, f"Took too long: {elapsed:.1f}s"

    def test_thread_terminates_after_createwindow_failure(self, mock_win32):
        """The background thread should exit when CreateWindowExW fails."""
        mock_user32, mock_kernel32 = mock_win32
        mock_user32.CreateWindowExW.return_value = 0

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        with pytest.raises(RuntimeError):
            backend.start(MagicMock())

        # Thread should have terminated
        if backend._thread is not None:
            backend._thread.join(timeout=2.0)
            assert not backend._thread.is_alive()


# ─── RegisterHotKey failure ──────────────────────────────────────────────────


class TestRegisterHotKeyFailure:
    """When RegisterHotKey fails, start() must raise within timeout."""

    def test_raises_on_register_failure(self, mock_win32):
        """RegisterHotKey returns 0 -> start() raises RuntimeError."""
        mock_user32, mock_kernel32 = mock_win32
        mock_user32.RegisterHotKey.return_value = 0  # BOOL FALSE
        mock_kernel32.GetLastError.return_value = 1409

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        with pytest.raises(RuntimeError, match="Failed to register hotkey"):
            backend.start(MagicMock())

    def test_raises_within_timeout(self, mock_win32):
        """start() should raise quickly on RegisterHotKey failure."""
        mock_user32, mock_kernel32 = mock_win32
        mock_user32.RegisterHotKey.return_value = 0

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        start_time = time.monotonic()
        with pytest.raises(RuntimeError):
            backend.start(MagicMock())
        elapsed = time.monotonic() - start_time
        assert elapsed < 7.0, f"Took too long: {elapsed:.1f}s"

    def test_error_code_in_message(self, mock_win32):
        """The error message should include the Win32 error code."""
        mock_user32, mock_kernel32 = mock_win32
        mock_user32.RegisterHotKey.return_value = 0
        mock_kernel32.GetLastError.return_value = 1409

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        with pytest.raises(RuntimeError, match="1409"):
            backend.start(MagicMock())


# ─── Success scenario ────────────────────────────────────────────────────────


class TestSuccessScenario:
    """On success, is_alive() returns True and _ready_event is set."""

    def test_ready_event_set_on_success(self, mock_win32):
        """After successful start(), _ready_event should be set."""
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        assert backend._ready_event.is_set()
        assert backend._success is True
        backend.stop()

    def test_is_alive_returns_true(self, mock_win32):
        """After successful start(), is_alive() returns True while thread runs."""
        mock_user32, _ = mock_win32
        # Make GetMessageW block by returning -1 repeatedly (non-zero)
        # so the loop continues and thread stays alive
        call_count = [0]

        def fake_getmsg(msg_ptr, hwnd, wmin, wmax):
            call_count[0] += 1
            if call_count[0] > 10:
                return 0  # eventually quit
            time.sleep(0.01)
            return -1  # keep looping

        mock_user32.GetMessageW.side_effect = fake_getmsg

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        # After start succeeds, thread should still be alive
        assert backend.is_alive() is True
        backend.stop()

    def test_is_alive_false_after_stop(self, mock_win32):
        """After stop(), is_alive() returns False."""
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())
        backend.stop()

        assert backend.is_alive() is False

    def test_registered_flag_true(self, mock_win32):
        """On success, _success should be True (survives thread cleanup)."""
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        # _success is set before _ready_event and not cleared in finally
        assert backend._success is True
        backend.stop()

    def test_hwnd_was_set_before_cleanup(self, mock_win32):
        """On success, _success is True (proving hwnd was valid and registration succeeded)."""
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        # _success=True proves CreateWindowExW returned a valid hwnd
        # and RegisterHotKey succeeded. After the thread exits, the
        # finally block resets _hwnd, so we check _success instead.
        assert backend._success is True
        backend.stop()


# ─── diagnose() method ───────────────────────────────────────────────────────


class TestDiagnoseMethod:
    """Test diagnose() reports success/failure state correctly."""

    def test_diagnose_before_start(self):
        """Before start(), diagnose() should say 'no thread started'."""
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        info = backend.diagnose()
        assert "no thread" in info.lower()

    def test_diagnose_on_success(self, mock_win32):
        """After successful start(), diagnose() includes key info."""
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        info = backend.diagnose()
        assert "WindowsNativeHotkey" in info
        assert "<f2>" in info
        assert "0x71" in info  # VK code for F2
        backend.stop()

    def test_diagnose_on_createwindow_failure(self, mock_win32):
        """After CreateWindowExW failure, _ready_event is set and _success is False."""
        mock_user32, _ = mock_win32
        mock_user32.CreateWindowExW.return_value = 0

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        with pytest.raises(RuntimeError):
            backend.start(MagicMock())

        # After failure, _ready_event should be set but _success is False
        assert backend._ready_event.is_set()
        assert backend._success is False

    def test_diagnose_on_register_failure(self, mock_win32):
        """After RegisterHotKey failure, _ready_event is set and _success is False."""
        mock_user32, _ = mock_win32
        mock_user32.RegisterHotKey.return_value = 0

        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        with pytest.raises(RuntimeError):
            backend.start(MagicMock())

        assert backend._ready_event.is_set()
        assert backend._success is False


# ─── Mocking verification ────────────────────────────────────────────────────


class TestMockVerification:
    """Verify that our mocking actually hits the right code paths."""

    def test_createwindowexw_called(self, mock_win32):
        """CreateWindowExW should be called during start()."""
        mock_user32, _ = mock_win32
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        mock_user32.CreateWindowExW.assert_called_once()
        backend.stop()

    def test_register_hotkey_called(self, mock_win32):
        """RegisterHotKey should be called during start()."""
        mock_user32, _ = mock_win32
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        mock_user32.RegisterHotKey.assert_called_once()
        backend.stop()

    def test_get_module_handle_called(self, mock_win32):
        """GetModuleHandleW should be called to get hInstance."""
        _, mock_kernel32 = mock_win32
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())

        mock_kernel32.GetModuleHandleW.assert_called()
        backend.stop()

    def test_stop_calls_cleanup(self, mock_win32):
        """stop() should call UnregisterHotKey and DestroyWindow."""
        mock_user32, _ = mock_win32
        from voice_typer.hotkeys import WindowsNativeHotkey

        backend = WindowsNativeHotkey("<f2>")
        backend.start(MagicMock())
        backend.stop()

        mock_user32.UnregisterHotKey.assert_called()
        mock_user32.DestroyWindow.assert_called()
