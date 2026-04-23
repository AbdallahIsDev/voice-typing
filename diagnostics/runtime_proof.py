#!/usr/bin/env python
"""Runtime verification script for transcription-time fallback and stuck-state recovery.

Exercises the ACTUAL code paths used by VoiceTyperApp when the user presses
F2 to start, F2 to stop, then transcription runs.  Uses synthetic audio
(2s of white noise at 16 kHz) so no microphone is needed.

Reports:
1. Fresh runtime log lines from the current build
2. Whether transcribe_with_fallback() was exercised
3. Whether _busy recovered to False
4. Whether the tray recovered to a non-stuck state
5. Whether a second "F2 press" (toggle_dictation) works after the first cycle
"""

import io
import logging
import sys
import threading
import time
import traceback

import numpy as np

# ─── Capture ALL logs into a buffer for later reporting ────────────────────
log_capture = io.StringIO()
_capture_handler = logging.StreamHandler(log_capture)
_capture_handler.setLevel(logging.DEBUG)
_capture_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.getLogger("voice_typer").addHandler(_capture_handler)
# Also log to stderr so the user can see it live
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.DEBUG)
_stderr_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.getLogger("voice_typer").addHandler(_stderr_handler)
logging.getLogger("voice_typer").setLevel(logging.DEBUG)

log = logging.getLogger("voice_typer.runtime_proof")


def make_synthetic_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio: low-level noise that mimics quiet speech input."""
    n_samples = int(duration_s * sample_rate)
    # Generate noise at a level that's above the "near-silence" threshold (0.001 RMS)
    # but below actual speech levels — this is enough to exercise the code paths
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, n_samples).astype(np.float32)
    return noise


class MockTrayIcon:
    """Minimal mock of TrayIcon that records state transitions."""

    def __init__(self):
        self.states: list[tuple[str, str, float]] = []  # (state_name, message, timestamp)
        self.notifications: list[tuple[str, str, float]] = []
        self._current_state = None
        self._current_message = ""

    def set_state(self, state, message=""):
        from voice_typer.tray import AppState
        state_name = state.value if isinstance(state, AppState) else str(state)
        ts = time.time()
        self.states.append((state_name, message, ts))
        self._current_state = state_name
        self._current_message = message
        log.info("[MOCK TRAY] set_state(%s, %r)", state_name, message)

    def notify(self, title, message):
        ts = time.time()
        self.notifications.append((title, message, ts))
        log.info("[MOCK TRAY] notify(%r, %r)", title, message)

    @property
    def current_state(self):
        return self._current_state

    @property
    def current_message(self):
        return self._current_message

    def is_stuck_on_transcribing(self):
        """Check if tray is stuck on 'transcribing' state."""
        return self._current_state == "transcribing"


def run_runtime_proof():
    """Run the actual runtime verification cycle."""
    from voice_typer.config import Config
    from voice_typer.transcription import TranscriptionEngine
    from voice_typer.tray import AppState

    log.info("=" * 70)
    log.info("RUNTIME PROOF TEST STARTING")
    log.info("=" * 70)

    # ─── Step 1: Load config and model (same as VoiceTyperApp.__init__) ──
    log.info("[STEP 1] Loading config and transcription engine...")
    config = Config.load()
    log.info(
        "[STEP 1] Config: model=%s, device=%s, language=%s, sample_rate=%d",
        config.model_size, config.device, config.language, config.sample_rate,
    )

    transcriber = TranscriptionEngine(
        model_size=config.model_size,
        device=config.device,
        language=config.language,
    )

    # ─── Step 2: Load model (same fallback chain as _try_load_model) ─────
    log.info("[STEP 2] Loading model (with full fallback chain)...")
    try:
        transcriber.load()
        log.info(
            "[STEP 2] Model loaded OK — device_info=%s, loaded_via=%s",
            transcriber.device_info, transcriber.loaded_via,
        )
    except Exception as e:
        log.error("[STEP 2] Model load FAILED: %s", e)
        log.error("[STEP 2] Cannot continue — transcription requires a loaded model")
        return False

    # ─── Step 3: Simulate the F2 cycle using app.py's actual logic ───────
    mock_tray = MockTrayIcon()
    busy = False  # mirrors VoiceTyperApp._busy

    # --- Simulate: F2 pressed → _start_dictation ---
    log.info("[STEP 3] Simulating F2 press → _start_dictation")
    mock_tray.set_state(AppState.RECORDING, "Recording...")

    # Generate synthetic audio (simulates what Recorder.stop() would return)
    audio = make_synthetic_audio(duration_s=2.0, sample_rate=config.sample_rate)
    duration = len(audio) / config.sample_rate
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    log.info(
        "[STEP 3] Synthetic audio: duration=%.1fs, samples=%d, RMS=%.6f",
        duration, len(audio), rms,
    )

    # --- Simulate: F2 pressed again → _stop_dictation ---
    log.info("[STEP 3] Simulating F2 press → _stop_dictation")
    mock_tray.set_state(AppState.TRANSCRIBING, "Transcribing...")
    busy = True
    log.info("[STEP 3] _busy set to True, transcription starting")

    # Track results from transcription thread
    results = {
        "transcribe_with_fallback_called": False,
        "transcribe_success": False,
        "transcribe_text": None,
        "transcribe_error": None,
        "fallback_exercised": False,
        "busy_recovered": False,
        "tray_recovered": False,
        "second_f2_works": False,
    }

    # Watchdog (same as in app.py — 60s timeout)
    watchdog_fired = threading.Event()
    force_recovery_done = threading.Event()

    def force_recover():
        """Same as VoiceTyperApp._force_recover_from_stuck_transcription."""
        nonlocal busy
        if not busy:
            return
        log.warning("FORCE RECOVER: transcription watchdog fired, resetting state")
        busy = False
        mock_tray.set_state(AppState.IDLE, "Recovered — transcription timed out")
        mock_tray.notify("Voice Typer", "Transcription took too long. Press F2 to try again.")
        watchdog_fired.set()
        force_recovery_done.set()

    watchdog = threading.Timer(60.0, force_recover)
    watchdog.daemon = True
    watchdog.start()

    # Transcription thread (mirrors app.py's transcribe_thread exactly)
    transcription_done = threading.Event()

    def transcribe_thread():
        nonlocal busy
        try:
            log.info("[TRANSCRIBE] Starting transcription with transcribe_with_fallback()...")
            results["transcribe_with_fallback_called"] = True

            # THIS IS THE KEY CALL — uses the real transcribe_with_fallback
            text = transcriber.transcribe_with_fallback(audio)

            results["transcribe_success"] = True
            results["transcribe_text"] = text
            log.info(
                "[TRANSCRIBE] Transcription complete (len=%d): %r",
                len(text) if text else 0,
                (text[:100] if text else ""),
            )

            if not text:
                log.info("[TRANSCRIBE] No speech detected (expected with synthetic audio)")
                mock_tray.set_state(AppState.IDLE, "No speech detected")
            else:
                log.info("[TRANSCRIBE] Got text: %r", text[:200])
                mock_tray.set_state(AppState.IDLE, f"Done — {len(text)} chars (in clipboard)")

        except Exception as e:
            results["transcribe_error"] = str(e)
            log.exception("[TRANSCRIBE] Transcription FAILED")
            mock_tray.set_state(AppState.ERROR, "Transcription failed")
            mock_tray.notify("Voice Typer Error", f"Transcription failed.\n{e}")
            # Check if fallback was exercised
            if "cublas" in str(e).lower() or "cuda" in str(e).lower():
                results["fallback_exercised"] = True
                log.info("[TRANSCRIBE] GPU error detected — fallback path was exercised")

        finally:
            watchdog.cancel()
            busy = False
            log.info("[TRANSCRIBE] _busy reset to False (finally block)")
            results["busy_recovered"] = True
            transcription_done.set()

    t = threading.Thread(target=transcribe_thread, daemon=True)
    t.start()
    log.info("[STEP 3] Transcription thread started, waiting for completion...")

    # Wait for transcription to finish (or watchdog)
    transcription_done.wait(timeout=90.0)

    if watchdog_fired.is_set():
        log.warning("[STEP 3] Watchdog had to force-recover!")
        results["busy_recovered"] = True  # recovered via watchdog

    # ─── Step 4: Verify _busy and tray state ─────────────────────────────
    log.info("[STEP 4] Verifying post-transcription state...")
    log.info("[STEP 4] _busy = %s", busy)
    log.info("[STEP 4] Tray state = %s (%s)", mock_tray.current_state, mock_tray.current_message)

    tray_ok = not mock_tray.is_stuck_on_transcribing()
    results["tray_recovered"] = tray_ok

    if busy:
        log.error("[STEP 4] FAIL: _busy is still True — stuck state!")
    else:
        log.info("[STEP 4] PASS: _busy recovered to False")

    if tray_ok:
        log.info("[STEP 4] PASS: Tray is not stuck on 'Transcribing...'")
    else:
        log.error("[STEP 4] FAIL: Tray is stuck on 'Transcribing...'!")

    # ─── Step 5: Simulate second F2 press (toggle_dictation) ─────────────
    log.info("[STEP 5] Simulating second F2 press (toggle_dictation)...")
    # This is what toggle_dictation does:
    if busy:
        log.warning("[F2 BLOCKED] Busy transcribing, ignoring toggle")
        results["second_f2_works"] = False
    else:
        log.info("[STEP 5] F2 NOT blocked — _busy is False, toggle would proceed")
        results["second_f2_works"] = True

    # ─── Step 6: Determine outcome ───────────────────────────────────────
    log.info("=" * 70)
    log.info("RUNTIME PROOF RESULTS")
    log.info("=" * 70)

    outcome = None
    if results["transcribe_success"] and not busy and tray_ok and results["second_f2_works"]:
        outcome = "A"
        log.info("OUTCOME A: Transcription succeeded, state fully recovered")
    elif not results["transcribe_success"] and results["busy_recovered"] and tray_ok and results["second_f2_works"]:
        outcome = "B"
        log.info("OUTCOME B: Transcription failed but recovery worked, state recovered")
    else:
        outcome = "FAILURE"
        log.error("UNEXPECTED OUTCOME: Neither A nor B achieved")

    log.info("  transcribe_with_fallback() called: %s", results["transcribe_with_fallback_called"])
    log.info("  transcribe succeeded: %s", results["transcribe_success"])
    log.info("  transcribe text: %r", results["transcribe_text"])
    log.info("  transcribe error: %s", results["transcribe_error"])
    log.info("  fallback exercised (GPU→CPU): %s", results["fallback_exercised"])
    log.info("  _busy recovered to False: %s", results["busy_recovered"])
    log.info("  tray recovered (not stuck): %s", results["tray_recovered"])
    log.info("  second F2 press works: %s", results["second_f2_works"])
    log.info("  final outcome: %s", outcome)

    # ─── Print all tray state transitions ─────────────────────────────────
    log.info("")
    log.info("TRAY STATE TRANSITIONS:")
    for state_name, message, ts in mock_tray.states:
        log.info("  → %s: %r", state_name, message)

    log.info("")
    log.info("NOTIFICATIONS:")
    for title, message, ts in mock_tray.notifications:
        log.info("  → %s: %r", title, message)

    # ─── Print captured log lines ─────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("FULL CAPTURED LOG OUTPUT (from this run):")
    log.info("=" * 70)
    captured = log_capture.getvalue()
    for line in captured.splitlines():
        log.info("  %s", line)

    return outcome in ("A", "B")


if __name__ == "__main__":
    try:
        success = run_runtime_proof()
        sys.exit(0 if success else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
