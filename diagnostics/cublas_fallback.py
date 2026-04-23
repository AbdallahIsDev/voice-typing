"""Runtime integration test: proves the cublas64_12.dll failure path is handled.

Simulates the exact scenario the user reported:
1. Model loaded on GPU (CUDA)
2. Recording starts (F2)
3. Recording stops (F2)
4. Transcription fails with "cublas64_12.dll is not found or cannot be loaded"
5. App must NOT stay stuck in busy=True / Transcribing...
6. Pressing F2 again must work

This test does NOT need a display or real GPU — it mocks the model but
exercises the real code paths.
"""

import sys
import time
import threading
import numpy as np
from unittest.mock import MagicMock, patch

# ─── Mock heavy imports before any voice_typer import ───

mock_sd = MagicMock()
mock_sd.query_devices.return_value = []
sys.modules["sounddevice"] = mock_sd

mock_fw = MagicMock()
sys.modules["faster_whisper"] = mock_fw
sys.modules["faster_whisper.WhisperModel"] = MagicMock()

mock_pynput = MagicMock()
mock_pynput_kb = MagicMock()
sys.modules["pynput"] = mock_pynput
sys.modules["pynput.keyboard"] = mock_pynput_kb

mock_pystray = MagicMock()
sys.modules["pystray"] = mock_pystray

mock_pil = MagicMock()
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PIL.ImageDraw"] = MagicMock()

sys.modules["pyperclip"] = MagicMock()


def test_cublas_failure_then_f2_recovery():
    """PROOF: cublas error → fallback → busy clears → F2 works again."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())

    with patch("voice_typer.config._config_dir", return_value=tmp), \
         patch("voice_typer.app.is_autostart_enabled", return_value=False), \
         patch("voice_typer.app.enable_autostart"), \
         patch("voice_typer.app.disable_autostart"), \
         patch("voice_typer.app.list_microphones", return_value=[]):

        from voice_typer.app import VoiceTyperApp
        from voice_typer.tray import AppState

        app = VoiceTyperApp()
        app.tray = MagicMock()

        # ── Step 1: Simulate model loaded on GPU ──
        app.transcriber._device = "cuda"
        app.transcriber._compute_type = "float16"
        mock_model = MagicMock()
        app.transcriber._model = mock_model
        assert app.transcriber.is_loaded

        # ── Step 2: F2 starts recording ──
        app.recorder = MagicMock()
        app.recorder.recording = False
        app._busy = False
        app.toggle_dictation()
        app.recorder.start.assert_called_once()
        print("  [OK] F2 started recording")

        # ── Step 3: F2 stops recording ──
        app.recorder.recording = True
        audio = np.ones(16000, dtype=np.float32)  # 1s of audio
        app.recorder.stop = MagicMock(return_value=audio)

        # Simulate cuBLAS failure during GPU transcription
        mock_model.transcribe.side_effect = RuntimeError(
            "Library cublas64_12.dll is not found or cannot be loaded"
        )

        # Set up CPU fallback model
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = (
            [MagicMock(text="hello from CPU fallback")],
            MagicMock(),
        )
        mock_fw.WhisperModel.return_value = cpu_model

        # F2 → stop recording → triggers transcription
        app.toggle_dictation()

        # Wait for transcribe thread
        deadline = time.time() + 5
        while app._busy and time.time() < deadline:
            time.sleep(0.05)

        # ── Step 4: Verify NOT stuck ──
        assert not app._busy, (
            f"FAIL: app._busy is still True after transcription fallback! "
            f"The app is stuck."
        )
        print("  [OK] busy=False after transcription (not stuck)")

        # ── Step 5: Verify tray state recovered ──
        set_state_calls = app.tray.set_state.call_args_list
        last_state = set_state_calls[-1]
        assert last_state[0][0] == AppState.IDLE, (
            f"FAIL: tray state is {last_state[0][0]}, expected IDLE"
        )
        print(f"  [OK] tray state recovered to IDLE: '{last_state[0][1]}'")

        # ── Step 6: Verify fallback happened ──
        assert app.transcriber._device == "cpu", (
            f"FAIL: device is {app.transcriber._device}, expected cpu after fallback"
        )
        print("  [OK] model fell back to CPU")

        # ── Step 7: F2 works again after failure ──
        app.recorder.reset_mock()
        app.recorder.recording = False
        app.toggle_dictation()
        app.recorder.start.assert_called_once()
        print("  [OK] F2 works again after transcription failure")

        print("\n  ✓ ALL CHECKS PASSED — cublas failure path is handled correctly")


def test_hard_crash_watchdog_recovery():
    """PROOF: if transcribe thread dies hard, watchdog recovers."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())

    with patch("voice_typer.config._config_dir", return_value=tmp), \
         patch("voice_typer.app.is_autostart_enabled", return_value=False), \
         patch("voice_typer.app.enable_autostart"), \
         patch("voice_typer.app.disable_autostart"), \
         patch("voice_typer.app.list_microphones", return_value=[]):

        from voice_typer.app import VoiceTyperApp
        from voice_typer.tray import AppState

        app = VoiceTyperApp()
        app.tray = MagicMock()

        # Simulate stuck _busy = True
        app._busy = True

        # Fire the recovery directly (simulates watchdog)
        app._force_recover_from_stuck_transcription()

        assert not app._busy, "FAIL: _busy still True after force recovery"
        print("  [OK] force recovery clears _busy")

        set_state_calls = app.tray.set_state.call_args_list
        last = set_state_calls[-1]
        assert last[0][0] == AppState.IDLE
        print(f"  [OK] tray reset to IDLE: '{last[0][1]}'")

        # Verify F2 works after recovery
        app.recorder = MagicMock()
        app.recorder.recording = False
        app.transcriber = MagicMock()
        app.transcriber.is_loaded = True
        app.toggle_dictation()
        app.recorder.start.assert_called_once()
        print("  [OK] F2 works after watchdog recovery")

        print("\n  ✓ ALL CHECKS PASSED — watchdog recovery works")


def test_transcribe_with_fallback_real():
    """PROOF: transcribe_with_fallback detects cuBLAS error and retries on CPU."""
    from voice_typer.transcription import TranscriptionEngine

    engine = TranscriptionEngine(model_size="small.en", device="cuda")
    engine._device = "cuda"
    engine._compute_type = "float16"
    mock_model = MagicMock()
    engine._model = mock_model

    # GPU transcription fails with cuBLAS error
    mock_model.transcribe.side_effect = RuntimeError(
        "Library cublas64_12.dll is not found or cannot be loaded"
    )

    # Set up CPU model for fallback
    cpu_model = MagicMock()
    cpu_model.transcribe.return_value = (
        [MagicMock(text="CPU fallback text")],
        MagicMock(),
    )
    mock_fw.WhisperModel.return_value = cpu_model

    audio = np.ones(16000, dtype=np.float32)
    result = engine.transcribe_with_fallback(audio)

    assert result == "CPU fallback text", f"Expected fallback text, got: {result}"
    assert engine._device == "cpu", f"Expected cpu, got: {engine._device}"
    print("  [OK] transcribe_with_fallback correctly fell back to CPU")

    # Verify non-GPU errors do NOT trigger fallback
    engine2 = TranscriptionEngine(model_size="small.en", device="cuda")
    engine2._device = "cuda"
    mock_model2 = MagicMock()
    engine2._model = mock_model2
    mock_model2.transcribe.side_effect = ValueError("bad audio format")

    try:
        engine2.transcribe_with_fallback(audio)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # expected
    print("  [OK] non-GPU errors propagate without fallback")

    # Verify CPU model errors do NOT trigger fallback
    engine3 = TranscriptionEngine(model_size="small.en", device="cpu")
    engine3._device = "cpu"
    mock_model3 = MagicMock()
    engine3._model = mock_model3
    mock_model3.transcribe.side_effect = RuntimeError("some error")

    try:
        engine3.transcribe_with_fallback(audio)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass  # expected
    print("  [OK] CPU model errors propagate without fallback")

    print("\n  ✓ ALL CHECKS PASSED — transcribe_with_fallback works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNTIME INTEGRATION TEST: cublas64_12.dll failure path")
    print("=" * 60)

    print("\n[Test 1] cuBLAS failure → F2 recovery cycle")
    test_cublas_failure_then_f2_recovery()

    print("\n[Test 2] Hard crash watchdog recovery")
    test_hard_crash_watchdog_recovery()

    print("\n[Test 3] transcribe_with_fallback behavior")
    test_transcribe_with_fallback_real()

    print("\n" + "=" * 60)
    print("ALL RUNTIME TESTS PASSED")
    print("=" * 60)
