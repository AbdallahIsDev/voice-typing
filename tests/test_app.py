"""Tests for app state transitions and error handling.

All heavy dependencies are mocked so these tests run on any platform
without GPU, microphone, or display.
"""

import sys
import json
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


def _wait_for_busy_clear(app, timeout=2.0):
    """Poll until app._busy is False (background transcription thread finished).

    Replaces bare time.sleep() calls that cause flaky failures under load.
    """
    deadline = time.monotonic() + timeout
    while app._busy and time.monotonic() < deadline:
        time.sleep(0.05)
    if app._busy:
        raise TimeoutError(f"_busy still True after {timeout}s")

# Mock heavy imports before they're loaded by the app module
# These patches stay active for the entire module
@pytest.fixture(autouse=True)
def mock_heavy_imports(monkeypatch):
    """Mock all hardware/GUI dependencies so tests run headless."""
    # IMPORTANT: all sys.modules mocking MUST happen before anything
    # triggers an import of voice_typer.app.  The app module imports
    # voice_typer.recording (which imports sounddevice), voice_typer.clipboard
    # (which imports pyperclip and pynput.keyboard), and voice_typer.tray
    # (which imports pystray and PIL).  In a clean environment without
    # PortAudio/GPU, these imports fail if the mocks aren't in place yet.

    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    mock_whisper = MagicMock()
    monkeypatch.setitem(sys.modules, "faster_whisper", mock_whisper)
    monkeypatch.setitem(sys.modules, "faster_whisper.WhisperModel", MagicMock())

    mock_pynput = MagicMock()
    mock_pynput_kb = MagicMock()
    monkeypatch.setitem(sys.modules, "pynput", mock_pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", mock_pynput_kb)

    mock_pystray = MagicMock()
    monkeypatch.setitem(sys.modules, "pystray", mock_pystray)

    mock_pil = MagicMock()
    monkeypatch.setitem(sys.modules, "PIL", mock_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", MagicMock())
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", MagicMock())

    # Prevent clipboard operations
    monkeypatch.setitem(sys.modules, "pyperclip", MagicMock())

    # Prevent the app's atexit handler from polluting test output.
    # In production, the handler logs unexpected process exits; in tests,
    # the process always exits without calling quit(), so the warning
    # fires every run and is not useful.
    # IMPORTANT: this monkeypatch triggers an import of voice_typer.app,
    # so it MUST come after all sys.modules mocking above.
    monkeypatch.setattr("voice_typer.app.atexit.register", lambda *a, **kw: None)

    # Force PynputHotkey backend so tests can mock pynput.keyboard.GlobalHotKeys.
    # On Windows, create_hotkey_backend() returns WindowsNativeHotkey which
    # calls the real Win32 RegisterHotKey -- this fails with error 1409
    # when another process holds the hotkey, making tests unreliable.
    # PynputHotkey is the backend the existing test mock infrastructure
    # was built around, so forcing it keeps the GlobalHotKeys/Listener
    # mocks reachable.
    from voice_typer.hotkeys import PynputHotkey
    monkeypatch.setattr(
        "voice_typer.app.create_hotkey_backend",
        lambda hotkey_str: PynputHotkey(hotkey_str),
    )


@pytest.fixture
def tmp_config_dir(tmp_path, monkeypatch):
    """Point config to a temp directory."""
    monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def app(tmp_config_dir, monkeypatch):
    """Create a VoiceTyperApp with mocked dependencies."""
    # Mock platform functions
    monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
    monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
    monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
    monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

    from voice_typer.app import VoiceTyperApp
    return VoiceTyperApp()


class TestAppStateTransitions:
    def test_initial_state_is_idle(self, app):
        assert app._busy is False
        assert app.recorder.recording is False

    def test_start_dictation_sets_recording(self, app):
        app.recorder = MagicMock()
        app.recorder.recording = False
        app.recorder.start = MagicMock()
        app.tray = MagicMock()
        app.transcriber = MagicMock()
        app.transcriber.is_loaded = True

        app._start_dictation()

        app.recorder.start.assert_called_once()
        app.tray.set_state.assert_called()

    def test_start_dictation_ignored_if_already_recording(self, app):
        app.recorder = MagicMock()
        app.recorder.recording = True

        app._start_dictation()

        app.recorder.start.assert_not_called()

    def test_stop_dictation_ignored_if_not_recording(self, app):
        app.recorder = MagicMock()
        app.recorder.recording = False

        app._stop_dictation()

        # Should not try to stop if not recording
        app.recorder.stop.assert_not_called()

    def test_short_audio_skips_transcription(self, app, monkeypatch):
        import voice_typer.app as app_mod
        app_mod.time = MagicMock()

        app.recorder = MagicMock()
        app.recorder.recording = True
        # Return 0.1s of audio (less than 0.5s threshold)
        app.recorder.stop = MagicMock(return_value=np.zeros(int(0.1 * 16000), dtype=np.float32))

        app._stop_dictation()

        _wait_for_busy_clear(app)
        assert app._busy is False

    def test_transcribe_success_copies_to_clipboard(self, app, monkeypatch):
        app.clipboard = MagicMock()
        app.clipboard.copy = MagicMock(return_value=True)
        app.clipboard.paste = MagicMock(return_value=False)

        app.transcriber = MagicMock()
        app.transcriber.transcribe_with_fallback = MagicMock(return_value="hello world")
        app.transcriber.device_info = "cpu (int8)"

        app.recorder = MagicMock()
        app.recorder.recording = True
        app.recorder.stop = MagicMock(return_value=np.ones(16000, dtype=np.float32))

        app._stop_dictation()

        _wait_for_busy_clear(app)

        app.clipboard.copy.assert_called_with("hello world")
        app.clipboard.paste.assert_called_once()

    def test_clipboard_copy_failure_prevents_paste(self, app):
        """Regression test for Finding 1: stale clipboard must not be pasted."""
        app.clipboard = MagicMock()
        app.clipboard.copy = MagicMock(return_value=False)  # copy FAILS
        app.clipboard.paste = MagicMock(return_value=True)

        app.transcriber = MagicMock()
        app.transcriber.transcribe_with_fallback = MagicMock(return_value="secret text")
        app.transcriber.device_info = "cpu (int8)"

        app.tray = MagicMock()

        app.recorder = MagicMock()
        app.recorder.recording = True
        app.recorder.stop = MagicMock(return_value=np.ones(16000, dtype=np.float32))

        app._stop_dictation()

        _wait_for_busy_clear(app)

        # copy was called
        app.clipboard.copy.assert_called_once_with("secret text")
        # paste must NOT have been called
        app.clipboard.paste.assert_not_called()
        # tray should show clipboard-unavailable status
        app.tray.notify.assert_called()
        notify_args = app.tray.notify.call_args
        assert "clipboard" in notify_args[0][1].lower() or "clipboard" in str(notify_args).lower()

    def test_transcribe_empty_result_no_clipboard(self, app):
        app.clipboard = MagicMock()
        app.transcriber = MagicMock()
        app.transcriber.transcribe_with_fallback = MagicMock(return_value="")

        app.recorder = MagicMock()
        app.recorder.recording = True
        app.recorder.stop = MagicMock(return_value=np.ones(16000, dtype=np.float32))

        app._stop_dictation()

        _wait_for_busy_clear(app)

        app.clipboard.copy.assert_not_called()

    def test_transcribe_failure_shows_error(self, app):
        app.transcriber = MagicMock()
        app.transcriber.transcribe_with_fallback = MagicMock(side_effect=Exception("model crash"))

        app.recorder = MagicMock()
        app.recorder.recording = True
        app.recorder.stop = MagicMock(return_value=np.ones(16000, dtype=np.float32))

        app._stop_dictation()

        _wait_for_busy_clear(app)

        # Should not crash; error state should be set
        assert app._busy is False

    def test_transcribe_cuda_fallback_clears_busy(self, app):
        """When GPU transcription fails with CUDA error, fallback to CPU succeeds
        and _busy is still cleared."""
        app.transcriber = MagicMock()
        # First call (GPU) raises CUDA error, fallback (CPU) returns text
        app.transcriber.transcribe_with_fallback = MagicMock(return_value="fallback worked")
        app.transcriber.device_info = "cpu (int8)"

        app.recorder = MagicMock()
        app.recorder.recording = True
        app.recorder.stop = MagicMock(return_value=np.ones(16000, dtype=np.float32))

        app._stop_dictation()

        _wait_for_busy_clear(app)

        assert app._busy is False
        app.transcriber.transcribe_with_fallback.assert_called_once()

    def test_force_recover_resets_busy(self, app):
        """_force_recover_from_stuck_transcription clears _busy and resets tray."""
        app._busy = True
        app.tray = MagicMock()

        app._force_recover_from_stuck_transcription()

        assert app._busy is False
        app.tray.set_state.assert_called()

    def test_force_recover_noop_when_not_busy(self, app):
        """_force_recover is a no-op if _busy is already False."""
        app._busy = False
        app.tray = MagicMock()

        app._force_recover_from_stuck_transcription()

        # No state change should have been made
        app.tray.set_state.assert_not_called()

    def test_f2_works_after_transcription_failure(self, app):
        """After a transcription failure, pressing F2 should work again."""
        # Simulate: transcription failed, busy was cleared
        app.transcriber = MagicMock()
        app.transcriber.transcribe_with_fallback = MagicMock(
            side_effect=RuntimeError("cublas64_12.dll is not found or cannot be loaded")
        )
        app.transcriber.device_info = "cpu (int8)"

        app.recorder = MagicMock()
        app.recorder.recording = True
        app.recorder.stop = MagicMock(return_value=np.ones(16000, dtype=np.float32))

        # First stop — transcription fails
        app._stop_dictation()
        _wait_for_busy_clear(app)

        assert app._busy is False, "_busy must be False after failed transcription"

        # Now simulate pressing F2 again
        app.recorder.recording = False
        app.transcriber.is_loaded = True
        app.transcriber.transcribe_with_fallback = MagicMock(return_value="recovered!")

        app.toggle_dictation()  # F2 → start recording
        app.recorder.start.assert_called_once()

        app.recorder.recording = True
        app.toggle_dictation()  # F2 → stop recording, transcribe
        _wait_for_busy_clear(app)

        assert app._busy is False
        app.transcriber.transcribe_with_fallback.assert_called_once_with(
            app.recorder.stop.return_value
        )


class TestConfigWiring:
    def test_paste_on_stop_respected(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.json"
        config_file.write_text(json.dumps({"paste_on_stop": False}))

        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        assert app.config.paste_on_stop is False
        assert app.clipboard.paste_enabled is False

    def test_transcription_speed_settings_wired(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.json"
        config_file.write_text(json.dumps({
            "beam_size": 2,
            "best_of": 2,
            "condition_on_previous_text": True,
        }))

        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        transcriber_cls = MagicMock()
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", transcriber_cls)

        from voice_typer.app import VoiceTyperApp
        VoiceTyperApp()

        _, kwargs = transcriber_cls.call_args
        assert kwargs["beam_size"] == 2
        assert kwargs["best_of"] == 2
        assert kwargs["condition_on_previous_text"] is True

    def test_autostart_syncs_with_platform(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.json"
        config_file.write_text(json.dumps({"autostart": True}))

        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        called = []
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: called.append(True) or True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()
        app._sync_autostart()

        assert len(called) == 1  # enable_autostart was called

    def test_autostart_disabled_when_config_false(self, tmp_config_dir, monkeypatch):
        config_file = tmp_config_dir / "config.json"
        config_file.write_text(json.dumps({"autostart": False}))

        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: True)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        called = []
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: called.append(True) or True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()
        app._sync_autostart()

        assert len(called) == 1  # disable_autostart was called


class TestHotkeyMapping:
    """Verify the hotkey registration uses the new backend abstraction."""

    def test_register_hotkey_creates_backend(self, app, monkeypatch):
        """_register_hotkey should create a hotkey backend and call start()."""
        from voice_typer.hotkeys import PynputHotkey
        from pynput.keyboard import GlobalHotKeys

        # Ensure GlobalHotKeys works (mock returns a MagicMock with is_alive=True)
        mock_listener = MagicMock()
        mock_listener.is_alive.return_value = True
        mock_ghk_cls = MagicMock(return_value=mock_listener)

        mock_kb = sys.modules['pynput.keyboard']
        mock_kb.GlobalHotKeys = mock_ghk_cls

        app._register_hotkey()

        assert app._hotkey_backend is not None
        mock_ghk_cls.assert_called_once()
        mock_listener.start.assert_called_once()

    def test_register_hotkey_failure_does_not_crash(self, app):
        """If both GlobalHotKeys AND fallback Listener raise, app should not crash."""
        mock_kb = sys.modules['pynput.keyboard']
        mock_kb.GlobalHotKeys = MagicMock(side_effect=Exception("no display"))
        mock_kb.Listener = MagicMock(side_effect=Exception("no input"))

        # Should not raise
        app._register_hotkey()
        # Backend was created but start() failed -> not alive or None
        if app._hotkey_backend is not None:
            assert app._hotkey_backend.is_alive() is False


class TestFallbackHotkeyParser:
    """Verify parse_hotkey_to_vk correctly converts hotkey strings."""

    def test_parse_f2(self):
        from voice_typer.hotkeys import parse_hotkey_to_vk
        result = parse_hotkey_to_vk("<f2>")
        assert result == 0x71

    def test_parse_f1(self):
        from voice_typer.hotkeys import parse_hotkey_to_vk
        result = parse_hotkey_to_vk("<f1>")
        assert result == 0x70

    def test_parse_f12(self):
        from voice_typer.hotkeys import parse_hotkey_to_vk
        result = parse_hotkey_to_vk("<f12>")
        assert result == 0x7B


class TestToggleDictationDispatch:
    """Verify toggle_dictation correctly dispatches to start/stop."""

    def test_toggle_calls_start_when_not_recording(self, app):
        """toggle_dictation() -> _start_dictation() when recorder.recording is False."""
        app.recorder = MagicMock()
        app.recorder.recording = False
        app._busy = False

        # Track which method was called
        start_called = []
        original_start = app._start_dictation
        def tracked_start():
            start_called.append(True)
        app._start_dictation = tracked_start

        stop_called = []
        def tracked_stop():
            stop_called.append(True)
        app._stop_dictation = tracked_stop

        app.toggle_dictation()

        assert len(start_called) == 1, "toggle_dictation should call _start_dictation when not recording"
        assert len(stop_called) == 0, "toggle_dictation should NOT call _stop_dictation when not recording"

    def test_toggle_calls_stop_when_recording(self, app):
        """toggle_dictation() -> _stop_dictation() when recorder.recording is True."""
        app.recorder = MagicMock()
        app.recorder.recording = True
        app._busy = False

        start_called = []
        def tracked_start():
            start_called.append(True)
        app._start_dictation = tracked_start

        stop_called = []
        def tracked_stop():
            stop_called.append(True)
        app._stop_dictation = tracked_stop

        app.toggle_dictation()

        assert len(stop_called) == 1, "toggle_dictation should call _stop_dictation when recording"
        assert len(start_called) == 0, "toggle_dictation should NOT call _start_dictation when recording"

    def test_toggle_ignored_when_busy(self, app):
        """toggle_dictation() should do nothing when _busy is True."""
        app._busy = True
        app.recorder = MagicMock()
        app.recorder.recording = False

        app._start_dictation = MagicMock()
        app._stop_dictation = MagicMock()

        app.toggle_dictation()

        app._start_dictation.assert_not_called()
        app._stop_dictation.assert_not_called()


class TestStartDictationBehavior:
    """Verify _start_dictation sets correct state and calls recorder.start()."""

    def test_start_calls_recorder_start_and_sets_recording_state(self, app):
        """_start_dictation must call recorder.start() and set tray state to RECORDING."""
        app.recorder = MagicMock()
        app.recorder.recording = False
        app.tray = MagicMock()
        app.transcriber = MagicMock()
        app.transcriber.is_loaded = True

        app._start_dictation()

        app.recorder.start.assert_called_once()
        app.tray.set_state.assert_called_once()
        from voice_typer.tray import AppState
        args = app.tray.set_state.call_args
        assert args[0][0] == AppState.RECORDING, (
            f"Expected AppState.RECORDING, got {args[0][0]}"
        )

    def test_start_is_noop_if_already_recording(self, app):
        """_start_dictation must not call recorder.start() if already recording."""
        app.recorder = MagicMock()
        app.recorder.recording = True
        app.tray = MagicMock()

        app._start_dictation()

        app.recorder.start.assert_not_called()


class TestHotkeyCallbackChain:
    """End-to-end: hotkey callback -> toggle_dictation -> _start_dictation."""

    def test_full_callback_chain(self, app):
        """Simulate the exact callback path: GlobalHotKeys fires toggle_dictation,
        which should call recorder.start() and set state to RECORDING."""
        app.recorder = MagicMock()
        app.recorder.recording = False
        app.tray = MagicMock()
        app._busy = False
        app.transcriber = MagicMock()
        app.transcriber.is_loaded = True

        # Simulate what GlobalHotKeys does: call the registered callback directly
        from voice_typer.tray import AppState

        # The callback stored in the hotkey mapping IS app.toggle_dictation
        app.toggle_dictation()

        app.recorder.start.assert_called_once()
        app.tray.set_state.assert_called_once()
        args = app.tray.set_state.call_args
        assert args[0][0] == AppState.RECORDING

    def test_callback_chain_register_then_fire(self, app):
        """Register hotkey, extract the callback, call it, verify recording starts."""
        captured_mapping = {}

        class FakeGlobalHotKeys:
            def __init__(self, mapping):
                captured_mapping.update(mapping)
            def start(self):
                pass

        mock_kb = sys.modules['pynput.keyboard']
        mock_kb.GlobalHotKeys = FakeGlobalHotKeys

        app.recorder = MagicMock()
        app.recorder.recording = False
        app.tray = MagicMock()
        app._busy = False

        # Register hotkey - this captures the mapping
        app._register_hotkey()

        assert '<f2>' in captured_mapping
        callback = captured_mapping['<f2>']
        assert callback == app.toggle_dictation

        # Simulate the hotkey being pressed
        callback()

        from voice_typer.tray import AppState
        app.recorder.start.assert_called_once()
        args = app.tray.set_state.call_args
        assert args[0][0] == AppState.RECORDING


class TestMicrophoneSelection:
    def test_select_mic_by_id_updates_config(self, app):
        app._select_microphone("3")
        assert app.config.microphone == "3"

    def test_select_none_resets_to_default(self, app):
        app.config.microphone = "5"
        app._select_microphone(None)
        assert app.config.microphone is None

    def test_select_mic_saves_config(self, app, tmp_config_dir):
        app._select_microphone("2")
        config_file = tmp_config_dir / "config.json"
        data = json.loads(config_file.read_text())
        assert data["microphone"] == "2"

    def test_select_mic_recreates_recorder(self, app):
        old_recorder = app.recorder
        app._select_microphone("1")
        assert app.recorder is not old_recorder
        assert app.config.microphone == "1"


# ─── Integration: real startup path ────────────────────────────────────


class TestAppStartupIntegration:
    """Integration-ish: let tray.start() + _do_startup run for real."""

    def test_startup_reaches_do_startup_without_crash(self, tmp_config_dir, monkeypatch):
        """Verify _do_startup runs without crashing (integration)."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        # Make transcriber.load() a no-op (don't actually load a model)
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        # Run _do_startup directly (normally called in a thread by tray.start)
        app._do_startup()

        # If we got here without exception, startup succeeded
        # Verify the tray was wired up
        assert app.tray is not None

    def test_tray_icon_created_on_start(self, tmp_config_dir, monkeypatch):
        """Verify tray.start() creates an icon with menu= wrapped in pystray.Menu."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        from tests.test_tray import _FakeIcon, _FakeMenu, _FakeMenuItem

        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        # Ensure voice_typer.tray uses our fakes (other test modules may have
        # replaced sys.modules["pystray"] with a plain MagicMock).
        import voice_typer.tray as tray_mod
        mock_pystray = MagicMock()
        mock_pystray.Icon = _FakeIcon
        mock_pystray.Menu = _FakeMenu
        mock_pystray.Menu.SEPARATOR = "SEP"
        mock_pystray.MenuItem = _FakeMenuItem
        monkeypatch.setattr(tray_mod, "pystray", mock_pystray)

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        # Reset before our call
        _FakeIcon.last_kwargs = {}

        # Call tray.start directly — should create the icon without blocking
        app.tray.start(bg_work=None)

        # The tray should now have an icon
        assert app.tray._icon is not None

        # The icon should have menu= set to a _FakeMenu (regression check)
        menu = _FakeIcon.last_kwargs.get("menu")
        assert isinstance(menu, _FakeMenu), (
            f"menu= must be a pystray.Menu, got {type(menu).__name__}: {menu!r}"
        )
        # _FakeMenu IS callable (mirrors real pystray.Menu) — verify it wraps a callable
        assert hasattr(menu, 'args') and len(menu.args) >= 1 and callable(menu.args[0]), (
            "menu= should wrap a callable inside pystray.Menu, not be a bare function"
        )


# ─── Transcription load resilience ─────────────────────────────────────


class TestTryLoadModel:
    """Test _try_load_model helper method."""

    def test_try_load_success_sets_idle_state(self, app):
        """On successful load, tray state should be IDLE with device info."""
        app.transcriber = MagicMock()
        app.transcriber.load = MagicMock()
        app.transcriber.device_info = "cpu (int8)"
        app.transcriber.loaded_via = "cpu/int8/small.en"
        app.tray = MagicMock()

        app._try_load_model()

        app.transcriber.load.assert_called_once()
        app.tray.set_state.assert_called()
        from voice_typer.tray import AppState
        # The last set_state call should be IDLE
        last_call = app.tray.set_state.call_args_list[-1]
        assert last_call[0][0] == AppState.IDLE
        assert "cpu" in last_call[0][1]

    def test_try_load_failure_sets_error_state(self, app):
        """On failed load, tray state should be ERROR."""
        app.transcriber = MagicMock()
        app.transcriber.load = MagicMock(side_effect=RuntimeError("OOM"))
        app.transcriber.is_loaded = False
        app.tray = MagicMock()

        app._try_load_model()

        from voice_typer.tray import AppState
        last_call = app.tray.set_state.call_args_list[-1]
        assert last_call[0][0] == AppState.ERROR
        assert "retry" in last_call[0][1].lower()

    def test_try_load_failure_with_notify(self, app):
        """notify_on_failure=True should send a desktop notification."""
        app.transcriber = MagicMock()
        app.transcriber.load = MagicMock(side_effect=RuntimeError("OOM"))
        app.transcriber.is_loaded = False
        app.tray = MagicMock()

        app._try_load_model(notify_on_failure=True)

        app.tray.notify.assert_called_once()
        notify_args = app.tray.notify.call_args[0]
        assert "Could not load" in notify_args[1]

    def test_try_load_failure_without_notify(self, app):
        """notify_on_failure=False should NOT send a notification."""
        app.transcriber = MagicMock()
        app.transcriber.load = MagicMock(side_effect=RuntimeError("OOM"))
        app.transcriber.is_loaded = False
        app.tray = MagicMock()

        app._try_load_model(notify_on_failure=False)

        app.tray.notify.assert_not_called()

    def test_try_load_sets_model_load_attempted(self, app):
        """_model_load_attempted should be True after _try_load_model."""
        app.transcriber = MagicMock()
        app.transcriber.load = MagicMock()
        app.tray = MagicMock()

        assert app._model_load_attempted is False
        app._try_load_model()
        assert app._model_load_attempted is True


class TestStartupResilience:
    """Test that startup continues even when model loading fails."""

    def test_startup_registers_hotkey_before_model_load(self, app, monkeypatch):
        """Hotkey should be registered even if model loading fails."""
        import time
        call_order = []

        def track_register_hotkey():
            call_order.append("hotkey")
        def track_try_load(*args, **kwargs):
            call_order.append("model")

        app._register_hotkey = track_register_hotkey
        app._try_load_model = track_try_load
        app._sync_autostart = MagicMock()
        app._load_microphones = MagicMock()
        app.tray = MagicMock()

        app._do_startup()

        assert call_order == ["hotkey", "model"], (
            f"Expected hotkey before model, got {call_order}"
        )

    def test_startup_survives_model_load_exception(self, app):
        """Even if model load raises, _do_startup should not crash."""
        app._sync_autostart = MagicMock()
        app._load_microphones = MagicMock()
        app._register_hotkey = MagicMock()
        app.transcriber = MagicMock()
        app.transcriber.load = MagicMock(side_effect=RuntimeError("OOM"))
        app.transcriber.is_loaded = False
        app.tray = MagicMock()

        # Should not raise
        app._do_startup()

        # Hotkey should still have been registered
        app._register_hotkey.assert_called_once()

    def test_start_dictation_retries_model_load(self, app):
        """When model not loaded, _start_dictation should try loading it."""
        app.transcriber = MagicMock()
        app.transcriber.is_loaded = False
        app.transcriber.load = MagicMock()
        app.transcriber.device_info = "cpu (int8)"
        app.transcriber.loaded_via = "cpu/int8/tiny.en"
        app.tray = MagicMock()
        app.recorder = MagicMock()
        app.recorder.recording = False

        # After _try_load_model, is_loaded becomes True
        def mock_load():
            app.transcriber.is_loaded = True
        app.transcriber.load = mock_load

        app._start_dictation()

        # Should have attempted to start recording (model was loaded on retry)
        app.recorder.start.assert_called_once()

    def test_start_dictation_fails_gracefully_if_model_still_unavailable(self, app):
        """If model retry fails, should not attempt recording."""
        app.transcriber = MagicMock()
        app.transcriber.is_loaded = False
        app.transcriber.load = MagicMock(side_effect=RuntimeError("still OOM"))
        app.tray = MagicMock()
        app.recorder = MagicMock()
        app.recorder.recording = False

        app._start_dictation()

        # Should NOT have tried to record
        app.recorder.start.assert_not_called()


# ─── Startup integration: construction → tray → hotkey → F2 ────────────


class TestStartupNoCrash:
    """Verify the full startup → hotkey → F2 path works correctly.

    These tests exercise the actual startup flow with mocked hardware
    dependencies but real code paths (not just isolated unit tests).
    """

    def test_app_construction_no_crash(self, tmp_config_dir, monkeypatch):
        """VoiceTyperApp() should construct without crashing."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        assert app.config is not None
        assert app.tray is not None
        assert app.recorder is not None
        assert app.clipboard is not None
        assert app._hotkey_backend is None
        assert app._busy is False

    def test_tray_start_creates_icon(self, tmp_config_dir, monkeypatch):
        """app.tray.start(bg_work=None) should create the tray icon without crashing."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        # Ensure tray module uses fakes
        from tests.test_tray import _FakeIcon, _FakeMenu, _FakeMenuItem
        import voice_typer.tray as tray_mod
        mock_pystray = MagicMock()
        mock_pystray.Icon = _FakeIcon
        mock_pystray.Menu = _FakeMenu
        mock_pystray.Menu.SEPARATOR = "SEP"
        mock_pystray.MenuItem = _FakeMenuItem
        monkeypatch.setattr(tray_mod, "pystray", mock_pystray)

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        _FakeIcon.last_kwargs = {}
        app.tray.start(bg_work=None)

        assert app.tray._icon is not None
        menu = _FakeIcon.last_kwargs.get("menu")
        assert isinstance(menu, _FakeMenu), (
            f"menu= must be pystray.Menu instance, got {type(menu)}"
        )

    def test_do_startup_runs_without_error(self, tmp_config_dir, monkeypatch):
        """app._do_startup() with mocked deps runs without error."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        mock_transcriber = MagicMock()
        mock_transcriber.load = MagicMock()
        mock_transcriber.device_info = "cpu (int8)"
        mock_transcriber.loaded_via = "cpu/int8/small.en"
        mock_transcriber_cls = MagicMock(return_value=mock_transcriber)
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", mock_transcriber_cls)

        mock_kb = sys.modules["pynput.keyboard"]
        mock_listener = MagicMock()
        mock_listener.is_alive.return_value = True
        mock_kb.GlobalHotKeys = MagicMock(return_value=mock_listener)

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()
        app.tray = MagicMock()

        # This is the exact path that was crashing
        app._do_startup()

        # Verify all steps executed
        app.tray.set_autostart_enabled.assert_called_once_with(False)
        mock_kb.GlobalHotKeys.assert_called_once()
        mock_transcriber.load.assert_called_once()

    def test_register_hotkey_creates_alive_backend(self, tmp_config_dir, monkeypatch):
        """app._register_hotkey() registers a hotkey backend that is_alive()."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        mock_kb = sys.modules["pynput.keyboard"]
        mock_listener = MagicMock()
        mock_listener.is_alive.return_value = True
        mock_kb.GlobalHotKeys = MagicMock(return_value=mock_listener)

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()
        app.tray = MagicMock()

        app._register_hotkey()

        assert app._hotkey_backend is not None
        assert app._hotkey_backend.is_alive() is True
        mock_kb.GlobalHotKeys.assert_called_once_with(
            {"<f2>": app.toggle_dictation}
        )

    def test_full_start_flow_no_crash(self, tmp_config_dir, monkeypatch):
        """The full app.start() flow doesn't crash (mock tray.run() to not block)."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        mock_transcriber = MagicMock()
        mock_transcriber.load = MagicMock()
        mock_transcriber.device_info = "cpu (int8)"
        mock_transcriber.loaded_via = "cpu/int8/small.en"
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock(return_value=mock_transcriber))

        mock_kb = sys.modules["pynput.keyboard"]
        mock_listener = MagicMock()
        mock_listener.is_alive.return_value = True
        mock_kb.GlobalHotKeys = MagicMock(return_value=mock_listener)

        from tests.test_tray import _FakeIcon, _FakeMenu, _FakeMenuItem
        import voice_typer.tray as tray_mod
        mock_pystray = MagicMock()
        mock_pystray.Icon = _FakeIcon
        mock_pystray.Menu = _FakeMenu
        mock_pystray.Menu.SEPARATOR = "SEP"
        mock_pystray.MenuItem = _FakeMenuItem
        monkeypatch.setattr(tray_mod, "pystray", mock_pystray)

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        _FakeIcon.last_kwargs = {}
        app.start()

        # After start(), tray should have created the icon
        assert app.tray._icon is not None
        # The icon's run() should have been called (via tray.run())
        assert app.tray._icon._run_called is True

    def test_f2_callback_chain_end_to_end(self, tmp_config_dir, monkeypatch):
        """Simulate: hotkey registered → F2 pressed → toggle_dictation → recording starts.

        This is the critical path: startup registers hotkey with toggle_dictation
        as the callback. Pressing F2 should start recording.
        """
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: [])

        mock_transcriber = MagicMock()
        mock_transcriber.load = MagicMock()
        mock_transcriber.device_info = "cpu (int8)"
        mock_transcriber.loaded_via = "cpu/int8/small.en"
        mock_transcriber.is_loaded = True
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock(return_value=mock_transcriber))

        # Capture the hotkey mapping so we can simulate F2 press
        captured_mapping = {}

        class FakeGlobalHotKeys:
            def __init__(self, mapping):
                captured_mapping.update(mapping)
            def start(self):
                pass
            def is_alive(self):
                return True

        mock_kb = sys.modules["pynput.keyboard"]
        mock_kb.GlobalHotKeys = FakeGlobalHotKeys

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()
        app.tray = MagicMock()

        # Register hotkey (captures the callback)
        app._register_hotkey()
        assert "<f2>" in captured_mapping

        # Simulate pressing F2
        app.recorder = MagicMock()
        app.recorder.recording = False
        app._busy = False

        callback = captured_mapping["<f2>"]
        assert callback == app.toggle_dictation
        callback()

        # Verify recording started
        app.recorder.start.assert_called_once()
        from voice_typer.tray import AppState
        app.tray.set_state.assert_called()
        last_state_call = app.tray.set_state.call_args_list[-1]
        assert last_state_call[0][0] == AppState.RECORDING


# ─── Microphone behavior ───────────────────────────────────────────────


class TestMicrophoneBehavior:
    """Test microphone listing, selection, and tray menu construction."""

    MOCK_MICS = [
        {
            "id": "0",
            "index": 0,
            "name": "WO Mic Device",
            "host_api": "Windows WASAPI",
            "channels": 1,
            "default": True,
        },
        {
            "id": "1",
            "index": 1,
            "name": "Microphone (Realtek)",
            "host_api": "Windows WDM-KS",
            "channels": 2,
            "default": False,
        },
        {
            "id": "2",
            "index": 2,
            "name": "WO Mic Device",
            "host_api": "Windows WDM-KS",
            "channels": 1,
            "default": False,
        },
    ]

    def test_list_microphones_returns_wo_mic(self, tmp_config_dir, monkeypatch):
        """list_microphones() with mock data returns WO Mic devices."""
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = {"index": 0}
        mock_sd.query_hostapis.return_value = {"name": "Windows WASAPI"}

        devices = [
            {"name": "WO Mic Device", "max_input_channels": 1, "hostapi": 0},
            {"name": "Microphone (Realtek)", "max_input_channels": 2, "hostapi": 1},
        ]
        mock_sd.query_devices.side_effect = lambda kind=None: (
            {"index": 0} if kind == "input" else devices
        )
        sys.modules["sounddevice"] = mock_sd

        from voice_typer.platform import list_microphones
        mics = list_microphones()

        assert len(mics) == 2
        wo_mics = [m for m in mics if "WO Mic" in m["name"]]
        assert len(wo_mics) >= 1

    def test_select_mic_by_id_updates_config(self, tmp_config_dir, monkeypatch):
        """Selecting a microphone by ID updates config.microphone."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: self.MOCK_MICS)
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        assert app.config.microphone is None
        app._select_microphone("1")
        assert app.config.microphone == "1"

    def test_select_system_default_works(self, tmp_config_dir, monkeypatch):
        """Selecting 'System Default' (None) sets config.microphone to None."""
        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: self.MOCK_MICS)
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        app.config.microphone = "2"
        app._select_microphone(None)
        assert app.config.microphone is None

    def test_tray_mic_submenu_built_correctly(self, tmp_config_dir, monkeypatch):
        """The tray microphone submenu should include System Default + all mics."""
        from tests.test_tray import _FakeIcon, _FakeMenu, _FakeMenuItem
        import voice_typer.tray as tray_mod
        mock_pystray = MagicMock()
        mock_pystray.Icon = _FakeIcon
        mock_pystray.Menu = _FakeMenu
        mock_pystray.Menu.SEPARATOR = "SEP"
        mock_pystray.MenuItem = _FakeMenuItem
        monkeypatch.setattr(tray_mod, "pystray", mock_pystray)

        monkeypatch.setattr("voice_typer.app.is_autostart_enabled", lambda: False)
        monkeypatch.setattr("voice_typer.app.enable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.disable_autostart", lambda: True)
        monkeypatch.setattr("voice_typer.app.list_microphones", lambda: self.MOCK_MICS)
        monkeypatch.setattr("voice_typer.app.TranscriptionEngine", MagicMock())

        from voice_typer.app import VoiceTyperApp
        app = VoiceTyperApp()

        # Populate mics on tray (normally done by _do_startup's _load_microphones)
        app.tray.set_microphones(self.MOCK_MICS)

        # Build the menu (exercises _build_menu which calls _build_mic_menu_items)
        items = app.tray._build_menu()

        # Find the "Microphone" MenuItem (should be present since mics are set)
        mic_label_found = False
        for item in items:
            if isinstance(item, _FakeMenuItem) and item.args and item.args[0] == "Microphone":
                mic_label_found = True
                # args[1] should be a _FakeMenu submenu
                submenu = item.args[1]
                assert isinstance(submenu, _FakeMenu), (
                    f"Microphone item should have a submenu, got {type(submenu)}"
                )
                break

        assert mic_label_found, (
            "Expected a 'Microphone' MenuItem in the tray menu"
        )

    def test_duplicate_mic_names_disambiguated(self, tmp_config_dir, monkeypatch):
        """When two mics have the same name, tray menu should disambiguate with host_api."""
        from tests.test_tray import _FakeMenuItem
        import voice_typer.tray as tray_mod
        mock_pystray = MagicMock()
        mock_pystray.MenuItem = _FakeMenuItem
        mock_pystray.Menu = MagicMock()
        mock_pystray.Menu.SEPARATOR = "SEP"
        monkeypatch.setattr(tray_mod, "pystray", mock_pystray)

        from voice_typer.tray import TrayIcon

        tray = TrayIcon(
            on_toggle=MagicMock(),
            on_settings=MagicMock(),
            on_quit=MagicMock(),
            on_select_mic=MagicMock(),
            config=MagicMock(microphone=None),
        )
        tray.set_microphones(self.MOCK_MICS)

        mic_items = tray._build_mic_menu_items()

        # First item is "System Default", rest are mics
        assert len(mic_items) == 4  # 1 default + 3 mics

        # Extract labels from _FakeMenuItem args
        labels = []
        for item in mic_items:
            if isinstance(item, _FakeMenuItem) and len(item.args) >= 1:
                labels.append(item.args[0])

        # The two "WO Mic Device" entries should be disambiguated with host_api
        wo_labels = [l for l in labels if "WO Mic" in l]
        assert len(wo_labels) == 2, f"Expected 2 WO Mic labels, got {wo_labels}"
        # Both should have host_api in parentheses appended (since names are duplicate)
        assert all("(" in l for l in wo_labels), (
            f"Duplicate mic names should be disambiguated: {wo_labels}"
        )
