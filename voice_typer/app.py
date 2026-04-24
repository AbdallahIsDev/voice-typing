"""Main application orchestrator."""

import atexit
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from typing import Optional

from voice_typer.config import Config, _config_dir
from voice_typer.recording import Recorder
from voice_typer.transcription import TranscriptionEngine
from voice_typer.streaming import StreamingConfig, StreamingTranscriptionSession
from voice_typer.clipboard import ClipboardManager
from voice_typer.settings import SettingsController, SettingsWindow
from voice_typer.tray import TrayIcon, AppState
from voice_typer.platform import (
    enable_autostart,
    disable_autostart,
    is_autostart_enabled,
    list_microphones,
)
from voice_typer.hotkeys import create_hotkey_backend, HotkeyBackend

log = logging.getLogger("voice_typer")

# Module-level list of devnull file objects opened by _setup_logging()
# for pythonw.exe (where sys.stderr/stdout/stdin are None).
# Closed explicitly in VoiceTyperApp.quit() for clean shutdown.
_devnull_files: list = []


def _setup_logging():
    """Configure logging to file (not console, since we run as tray app)."""
    # Under pythonw.exe (e.g. Windows autostart), sys.stderr/stdout/stdin
    # are None.  Redirect them to devnull immediately so any accidental
    # writes don't crash the process.
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")
        _devnull_files.append(sys.stderr)
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
        _devnull_files.append(sys.stdout)
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
        _devnull_files.append(sys.stdin)

    config_dir = _config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    log_file = config_dir / "voice-typer.log"

    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=2,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger("voice_typer")
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    # Also log to stderr when running interactively (debugging)
    if sys.stderr.isatty():
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        root.addHandler(stream)


class VoiceTyperApp:
    """The main application."""

    def __init__(self):
        self.config = Config.load()
        self.config.device = "cuda"
        self.config.paste_on_stop = True
        self.config.streaming_transcription = True
        self.recorder = Recorder(self.config)
        self.transcriber = TranscriptionEngine(
            model_size=self.config.model_size,
            device=self.config.device,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )
        self.clipboard = ClipboardManager(paste_enabled=True)
        self.tray = TrayIcon(
            on_toggle=self.toggle_dictation,
            on_settings=self.show_settings,
            on_quit=self.quit,
            on_toggle_autostart=self._toggle_autostart,
            on_select_mic=self._select_microphone,
            on_select_hotkey=self._restart_hotkey,
            on_select_model=self._change_model,
            on_toggle_notifications=self._set_notifications,
            config=self.config,
        )

        self._hotkey_backend: Optional[HotkeyBackend] = None
        self._streaming_session: Optional[StreamingTranscriptionSession] = None
        self._transcription_thread: Optional[threading.Thread] = None
        self._microphones: list[dict] = []
        self._busy = False  # True during transcription
        self._model_load_attempted = False  # True after first load() call
        self._shutting_down = False  # True once quit() starts

    # ─── Startup ───────────────────────────────────────────────────────

    def start(self):
        """Initialize and run the application.

        The tray icon is created and its event loop runs on the main thread
        (required by macOS / Linux).  Model loading, hotkey registration,
        etc. happen in a background thread managed by the tray.
        """
        log.info(
            "Voice Typer starting -- model=%s, hotkey=%s, mic=%s, sample_rate=%s",
            self.config.model_size, self.config.hotkey,
            self.config.microphone or "default", self.config.sample_rate,
        )

        # Wire notifications
        self.tray.set_notifications_enabled(self.config.show_notifications)

        # Queue "Loading" state before the event loop starts
        self.tray.set_state(AppState.LOADING, "Starting...")

        # Create the icon and start background work (non-blocking)
        self.tray.start(bg_work=self._do_startup)

        # Register signal handlers on the main thread (safe before run())
        def signal_handler(sig, frame):
            log.info("[SIGNAL] Received signal %d, quitting", sig)
            self.quit()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # On Windows: install a console control handler so that closing the
        # terminal window (CTRL_CLOSE_EVENT) or logging off/shutting down
        # triggers a graceful shutdown instead of silently killing the process.
        self._install_win32_console_handler()

        # Register atexit handler to log any unexpected process exit
        atexit.register(self._atexit_log)

        # Enter pystray event loop — MUST be on the main thread
        log.info("Entering tray event loop on main thread")
        self.tray.run()

    def _do_startup(self):
        """Background work: sync autostart, load mics, load model, register hotkey."""
        log.info("[STARTUP] _do_startup begin")

        # 1. Sync autostart config with platform
        log.info("[STARTUP] Step 1: sync autostart")
        self._sync_autostart()
        self.tray.set_autostart_enabled(is_autostart_enabled())

        # 2. Enumerate microphones for the tray menu
        log.info("[STARTUP] Step 2: load microphones")
        self._load_microphones()

        # 3. Register hotkey BEFORE model load so F2 works even if model fails
        log.info("[STARTUP] Step 3: register hotkey")
        self._register_hotkey()

        # Warm expensive audio helpers off the F2 stop path while the model loads.
        threading.Thread(
            target=self.recorder.warm_up_resampler,
            name="ResamplerWarmup",
            daemon=True,
        ).start()

        # 4. Load the Whisper model (may fail — hotkey already registered)
        log.info("[STARTUP] Step 4: load model")
        self._try_load_model(notify_on_failure=True)

        log.info("[STARTUP] _do_startup complete")

    def _sync_autostart(self):
        """Ensure config.autostart matches the actual platform autostart state."""
        try:
            actual = is_autostart_enabled()
            if self.config.autostart and not actual:
                log.info("Config says autostart=true but it is disabled -- enabling")
                enable_autostart()
            elif not self.config.autostart and actual:
                log.info("Config says autostart=false but it is enabled -- disabling")
                disable_autostart()
        except Exception as e:
            log.warning("Autostart sync failed: %s", e)

    def _load_microphones(self):
        """Enumerate microphones and update the tray menu."""
        try:
            mics = list_microphones()
            self._microphones = mics
            self.tray.set_microphones(mics)
            log.info("Found %d microphone(s)", len(mics))
        except Exception as e:
            log.warning("Could not enumerate microphones: %s", e)

    def _try_load_model(self, notify_on_failure: bool = False):
        """Attempt to load the transcription model.

        Runs the full fallback chain (preferred device -> CPU int8 ->
        tiny.en -> float32).  If loading succeeds, sets tray to IDLE with
        device info.  If it fails, sets ERROR state with a friendly message
        and optionally sends a desktop notification.

        The hotkey (F2) remains active regardless so the user can retry.
        """
        self._model_load_attempted = True
        try:
            log.info("[MODEL] Loading model (size=%s, device=%s)...",
                     self.config.model_size, self.config.device)
            self.tray.set_state(AppState.LOADING, "Loading model...")
            self.transcriber.load()
            self.tray.set_state(
                AppState.IDLE, f"Ready — {self.transcriber.device_info}"
            )
            log.info("[MODEL] Loaded successfully via %s", self.transcriber.loaded_via)
        except Exception as e:
            log.exception("[MODEL] Load FAILED")
            self.tray.set_state(
                AppState.ERROR, "Model failed to load — press F2 to retry"
            )
            if notify_on_failure:
                self.tray.notify(
                    "Voice Typer",
                    f"Could not load the speech model.\n{e}\n\n"
                    "The app will keep running. Press F2 to retry loading.",
                )

    # ─── Hotkey ────────────────────────────────────────────────────────

    def _register_hotkey(self):
        """Register global hotkey using the platform-appropriate backend."""
        hotkey_str = self.config.hotkey
        log.info("[HOTKEY] Registering: %r -> toggle_dictation", hotkey_str)

        try:
            self._hotkey_backend = create_hotkey_backend(hotkey_str)
            log.info("[HOTKEY] Backend created: %s", type(self._hotkey_backend).__name__)
            self._hotkey_backend.start(self.toggle_dictation)
            log.info(
                "[HOTKEY] Registration OK (alive=%s, backend=%s)",
                self._hotkey_backend.is_alive(),
                type(self._hotkey_backend).__name__,
            )
        except Exception:
            log.exception("[HOTKEY] Registration FAILED")
            self.tray.notify(
                "Voice Typer",
                "Hotkey registration failed. Use the tray menu to toggle dictation.",
            )

    # ─── Dictation ─────────────────────────────────────────────────────

    def toggle_dictation(self):
        """Toggle recording on/off."""
        log.info(
            "[HOTKEY FIRED] toggle_dictation called "
            "(recording=%s, busy=%s, model_loaded=%s, thread=%s)",
            self.recorder.recording, self._busy,
            self.transcriber.is_loaded, threading.current_thread().name,
        )
        if self._busy:
            log.warning("[F2 BLOCKED] Busy transcribing, ignoring toggle")
            return

        if self.recorder.recording:
            self._stop_dictation()
        else:
            self._start_dictation()

    def _start_dictation(self):
        """Start a recording session."""
        if self.recorder.recording:
            log.info("[DICTATION] _start_dictation: already recording, no-op")
            return

        # Guard: refuse to record if the model never loaded
        if not self.transcriber.is_loaded:
            log.warning("[DICTATION] Model not loaded, attempting reload")
            self._try_load_model(notify_on_failure=True)
            if not self.transcriber.is_loaded:
                log.error("[DICTATION] Model reload failed, cannot record")
                threading.Timer(3.0, lambda: self.tray.set_state(
                    AppState.ERROR, "Model failed to load — press F2 to retry"
                )).start()
                return

        log.info("[DICTATION] Starting recording...")
        try:
            self.recorder.start()
            self._start_streaming_session_if_enabled()
            self.tray.set_state(AppState.RECORDING, "Recording...")
            log.info("[DICTATION] Recording started OK")
        except Exception as e:
            log.exception("[DICTATION] Failed to start recording: %s", e)
            self._cancel_streaming_session()
            self.tray.set_state(AppState.ERROR, "Recording failed")
            self.tray.notify(
                "Voice Typer",
                f"Could not start recording.\n{e}\n\n"
                "Check voice-typer.log for traceback.",
            )
            threading.Timer(3.0, lambda: self.tray.set_state(AppState.IDLE)).start()

    def _stop_dictation(self):
        """Stop recording and transcribe in background."""
        if not self.recorder.recording:
            log.info("[DICTATION] _stop_dictation: not recording, no-op")
            return

        log.info("[DICTATION] Stopping recording...")
        self._busy = True

        try:
            audio = self.recorder.stop()
        except Exception as e:
            log.exception("[DICTATION] Failed to stop recording")
            self._cancel_streaming_session()
            self.tray.set_state(AppState.ERROR, "Stop failed")
            self.tray.notify("Voice Typer", f"Could not stop recording.\n{e}")
            self._busy = False
            threading.Timer(3.0, lambda: self.tray.set_state(AppState.IDLE)).start()
            return

        # Audio has already been resampled to config.sample_rate by Recorder.stop()
        duration = len(audio) / self.config.sample_rate if len(audio) > 0 else 0
        # Capture RMS before starting transcription thread (race-safe)
        recorded_rms = self.recorder.last_rms
        log.info("[DICTATION] Recording stopped -- %.1fs of audio, _busy=True", duration)

        if duration < 0.5:
            log.info("[DICTATION] Audio too short, skipping transcription")
            self._cancel_streaming_session()
            self.tray.set_state(AppState.IDLE, "Too short — ignored")
            self._busy = False
            threading.Timer(2.0, lambda: self.tray.set_state(AppState.IDLE)).start()
            return

        log.info("[DICTATION] Starting transcription thread...")
        self.tray.set_state(AppState.TRANSCRIBING, "Transcribing...")

        # Safety watchdog: if transcription hangs for >60s, force-recover.
        # This prevents the app from being permanently stuck if a C-level
        # crash or deadlock kills the thread before the finally block runs.
        watchdog = threading.Timer(
            60.0,
            lambda: self._force_recover_from_stuck_transcription(),
        )
        watchdog.daemon = True
        watchdog.start()

        def transcribe_thread():
            try:
                log.info("[TRANSCRIBE] Starting transcription...")
                session = self._streaming_session
                if session is not None:
                    log.info("[STREAMING] Finalizing streaming transcript")
                    text = session.finalize(audio)
                    self._streaming_session = None
                else:
                    text = self.transcriber.transcribe_with_fallback(audio)
                log.info("[TRANSCRIBE] Transcription complete (len=%d)", len(text) if text else 0)

                if not text:
                    log.info("[TRANSCRIBE] No speech detected")
                    # If audio was near-silence, warn the user about mic issues
                    if recorded_rms < 0.005:
                        self.tray.set_state(
                            AppState.IDLE,
                            "No speech — check microphone",
                        )
                        self.tray.notify(
                            "Voice Typer",
                            "No speech was detected and audio was near-silence.\n"
                            "Your microphone may not be capturing audio.\n"
                            "Check that the correct mic is selected and is active.",
                        )
                    else:
                        self.tray.set_state(AppState.IDLE, "No speech detected")
                    self._busy = False
                    threading.Timer(2.0, lambda: self.tray.set_state(AppState.IDLE)).start()
                    return

                log.info("Transcription: %s...", text[:100])

                # Copy to clipboard — only attempt paste if copy succeeded.
                # Otherwise we risk pasting stale clipboard contents.
                if not self.clipboard.copy(text):
                    log.error("Clipboard copy failed -- not attempting paste")
                    self.tray.set_state(AppState.IDLE, "Done — clipboard unavailable")
                    self.tray.notify(
                        "Voice Typer",
                        "Transcription complete, but clipboard was unavailable.\n"
                        "Text was not pasted. Check the log for details.",
                    )
                    self._busy = False
                    threading.Timer(
                        3.0,
                        lambda: self.tray.set_state(
                            AppState.IDLE,
                            f"Ready — {self.transcriber.device_info}",
                        ),
                    ).start()
                    return

                # Attempt safe paste (only if paste_on_stop AND a text input is focused)
                pasted = False
                if self.config.paste_on_stop:
                    pasted = self.clipboard.paste()

                if pasted:
                    status = f"Done — {len(text)} chars (pasted)"
                else:
                    status = f"Done — {len(text)} chars (in clipboard)"

                self.tray.set_state(AppState.IDLE, status)
                self.tray.notify("Voice Typer", f"Transcribed {len(text)} characters")

                # Reset to plain "Ready" after a few seconds
                threading.Timer(
                    3.0,
                    lambda: self.tray.set_state(
                        AppState.IDLE,
                        f"Ready — {self.transcriber.device_info}",
                    ),
                ).start()

            except Exception as e:
                log.exception("[TRANSCRIBE] Transcription FAILED")
                self.tray.set_state(AppState.ERROR, "Transcription failed")
                self.tray.notify("Voice Typer Error", f"Transcription failed.\n{e}")
                threading.Timer(3.0, lambda: self.tray.set_state(AppState.IDLE)).start()

            finally:
                watchdog.cancel()
                if self._streaming_session is not None and not self.recorder.recording:
                    self._streaming_session = None
                self._busy = False
                self._transcription_thread = None
                log.info("[TRANSCRIBE] _busy reset to False")

        self._transcription_thread = threading.Thread(
            target=transcribe_thread,
            name="Transcription",
            daemon=True,
        )
        self._transcription_thread.start()

    def _streaming_enabled(self) -> bool:
        """Return whether hidden streaming should run for the next recording."""
        if os.environ.get("VOICE_TYPER_STREAMING") == "0":
            return False
        return True

    def _streaming_config(self) -> StreamingConfig:
        return StreamingConfig(
            enabled=True,
            chunk_seconds=self.config.streaming_chunk_seconds,
            step_seconds=self.config.streaming_step_seconds,
            left_overlap_seconds=self.config.streaming_left_overlap_seconds,
            right_guard_seconds=self.config.streaming_right_guard_seconds,
            min_first_chunk_seconds=self.config.streaming_min_first_chunk_seconds,
            silence_threshold=self.config.streaming_silence_threshold,
        )

    def _start_streaming_session_if_enabled(self):
        """Start hidden streaming work for the active recording if enabled."""
        self._streaming_session = None
        if not self._streaming_enabled():
            return

        try:
            session = StreamingTranscriptionSession(
                recorder=self.recorder,
                transcriber=self.transcriber,
                config=self._streaming_config(),
                sample_rate=self.config.sample_rate,
            )
            session.start()
            self._streaming_session = session
            log.info("[STREAMING] Hidden streaming session started")
        except Exception as e:
            log.exception("[STREAMING] Failed to start streaming session: %s", e)
            self._streaming_session = None

    def _cancel_streaming_session(self):
        """Cancel any active hidden streaming session."""
        session = self._streaming_session
        self._streaming_session = None
        if session is not None:
            try:
                session.cancel()
            except Exception:
                log.exception("[STREAMING] Failed to cancel streaming session")

    def _force_recover_from_stuck_transcription(self):
        """Safety net: recover from stuck transcription state.

        Called by the watchdog timer if transcription takes >60s or if
        the transcribe thread dies before the finally block runs.
        """
        if not self._busy:
            return  # Already recovered, nothing to do
        if (
            self._transcription_thread is not None
            and self._transcription_thread.is_alive()
        ):
            log.warning(
                "Transcription watchdog fired, but worker is still alive; "
                "leaving app busy to avoid overlapping model calls"
            )
            self.tray.set_state(AppState.TRANSCRIBING, "Still transcribing...")
            self.tray.notify(
                "Voice Typer",
                "Transcription is still running.\n"
                "Long recordings or CPU fallback can take extra time.",
            )
            return

        log.warning("FORCE RECOVER: transcription watchdog fired, resetting state")
        self._busy = False
        self.tray.set_state(AppState.IDLE, "Recovered — transcription timed out")
        self.tray.notify(
            "Voice Typer",
            "Transcription took too long and was cancelled.\n"
            "Press F2 to try again.",
        )
        threading.Timer(5.0, lambda: self.tray.set_state(AppState.IDLE)).start()

    # ─── Settings / Microphone ─────────────────────────────────────────

    def _toggle_autostart(self):
        """Toggle autostart on/off from the tray menu."""
        try:
            if is_autostart_enabled():
                disable_autostart()
                self.config.autostart = False
            else:
                enable_autostart()
                self.config.autostart = True
            self.config.save()
            self.tray.set_autostart_enabled(self.config.autostart)
            log.info("Autostart set to %s", self.config.autostart)
        except Exception as e:
            log.exception("Failed to toggle autostart")
            self.tray.notify("Voice Typer", f"Could not change autostart setting.\n{e}")

    def _set_autostart(self, enabled: bool):
        """Set autostart from the advanced settings window."""
        try:
            if enabled:
                enable_autostart()
            else:
                disable_autostart()
            self.config.autostart = enabled
            self.config.save()
            self.tray.set_autostart_enabled(enabled)
            log.info("Autostart set to %s", enabled)
        except Exception as e:
            log.exception("Failed to set autostart")
            self.tray.notify("Voice Typer", f"Could not change autostart setting.\n{e}")

    def _set_notifications(self, enabled: bool):
        """Set notification behavior from the settings window."""
        self.config.show_notifications = enabled
        self.config.save()
        self.tray.set_notifications_enabled(enabled)
        log.info("Notifications set to %s", enabled)

    def _select_microphone(self, mic_name: str | None):
        """Handle microphone selection from tray menu."""
        self.config.microphone = mic_name
        self.config.save()
        label = mic_name if mic_name else "System Default"

        if self.recorder.recording:
            log.info("Microphone changed to %s; applying after active recording", label)
            self.tray.notify("Voice Typer", f"Microphone next recording: {label}")
            return

        self.recorder = Recorder(self.config)  # re-create with new mic
        log.info("Microphone changed to: %s", label)
        self.tray.notify("Voice Typer", f"Microphone: {label}")

    def show_settings(self):
        """Open the native settings window."""
        controller = SettingsController(
            self.config,
            on_hotkey_changed=self._restart_hotkey,
            on_model_changed=self._change_model,
            on_microphone_changed=self._select_microphone,
            on_autostart_changed=self._set_autostart,
            on_notifications_changed=self._set_notifications,
        )
        SettingsWindow(
            controller,
            microphones=self._microphones,
            on_open_config=self._open_config_file,
        ).show()

    def _open_config_file(self):
        """Open raw settings file for troubleshooting."""
        config_file = self.config.config_dir / "config.json"
        if not config_file.exists():
            self.config.save()

        import subprocess
        try:
            if sys.platform == "win32":
                subprocess.Popen(["notepad", str(config_file)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(config_file)])
            else:
                subprocess.Popen(["xdg-open", str(config_file)])
        except Exception as e:
            log.warning("Could not open editor: %s", e)
            self.tray.notify("Voice Typer", f"Config file:\n{config_file}")

    def _restart_hotkey(self, hotkey: str):
        """Re-register the global hotkey after settings change."""
        self.config.hotkey = hotkey
        self.config.save()
        if self._hotkey_backend:
            try:
                self._hotkey_backend.stop()
            except Exception:
                log.exception("[HOTKEY] Failed to stop previous backend")
            self._hotkey_backend = None
        self._register_hotkey()

    def _change_model(self, model_size: str):
        """Apply a model change for future dictation sessions."""
        self.config.model_size = model_size
        self.config.save()
        if self.recorder.recording or self._busy:
            log.info("Model changed to %s; applying after active work", model_size)
            return
        try:
            self.transcriber.unload()
        except Exception:
            log.exception("[MODEL] Failed to unload previous model")
        self.transcriber = TranscriptionEngine(
            model_size=self.config.model_size,
            device="cuda",
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )
        self._model_load_attempted = False
        self.tray.set_state(AppState.IDLE, "Model changed — press F2 to load")

    # ─── Shutdown ──────────────────────────────────────────────────────

    def quit(self):
        """Shut down the application cleanly.

        Safe to call from any thread.  When called from the main thread
        (tray menu, signal handler), sys.exit(0) terminates the process.
        When called from a background thread, tray.stop() wakes the
        main pystray loop which then returns, and the process exits
        naturally from main().
        """
        if self._shutting_down:
            log.info("quit() already in progress, ignoring duplicate call")
            return

        is_main = threading.current_thread() is threading.main_thread()
        log.info("Shutting down (quit() called from thread=%s, is_main=%s)",
                 threading.current_thread().name, is_main)
        self._shutting_down = True

        self._cancel_streaming_session()

        if self.recorder.recording:
            self.recorder.discard()

        if self._hotkey_backend:
            self._hotkey_backend.stop()

        self.tray.stop()
        log.info("Shutdown complete, exiting")

        # Close devnull streams that were opened for pythonw.exe in
        # _setup_logging().  Not strictly required (OS closes at exit),
        # but cleaner than relying on process teardown.
        for f in _devnull_files:
            try:
                f.close()
            except Exception:
                pass
        _devnull_files.clear()

        if is_main:
            sys.exit(0)
        # From non-main thread: tray.stop() wakes the pystray loop,
        # the main thread will exit naturally.  Do NOT call sys.exit()
        # from a non-main thread — it only kills that thread, not the
        # process.

    def _atexit_log(self):
        """Log when the process exits, even if quit() was not called."""
        if not self._shutting_down:
            log.warning("[ATEXIT] Process exiting without quit() -- "
                        "likely killed externally (console close, task manager, etc.)")

    def _install_win32_console_handler(self):
        """On Windows, install a console control handler to survive console closure.

        Without this, closing the terminal window that launched the app sends
        CTRL_CLOSE_EVENT which terminates the process immediately — the tray
        icon vanishes and no 'Shutting down' log appears.

        The handler intercepts CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, and
        CTRL_SHUTDOWN_EVENT and calls quit() for a clean shutdown.
        It returns TRUE for these events so the OS does not kill the process.
        """
        if sys.platform != "win32":
            return

        try:
            import ctypes
            from ctypes import wintypes

            # Console event type constants
            CTRL_C_EVENT = 0
            CTRL_BREAK_EVENT = 1
            CTRL_CLOSE_EVENT = 2
            CTRL_LOGOFF_EVENT = 5
            CTRL_SHUTDOWN_EVENT = 6

            # HandlerRoutine callback type
            HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            # Keep references to prevent GC and for use in the handler callback
            self._console_handler = HANDLER_ROUTINE(self._win32_console_handler)
            self._kernel32 = ctypes.windll.kernel32
            kernel32 = self._kernel32
            kernel32.SetConsoleCtrlHandler.argtypes = [HANDLER_ROUTINE, wintypes.BOOL]
            kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL

            # Set argtypes for FreeConsole so the callback can call it safely
            kernel32.FreeConsole.argtypes = []
            kernel32.FreeConsole.restype = wintypes.BOOL

            result = kernel32.SetConsoleCtrlHandler(self._console_handler, True)
            if result:
                log.info("[WIN32] Console control handler installed")
            else:
                log.warning("[WIN32] SetConsoleCtrlHandler failed")
        except Exception:
            log.exception("[WIN32] Failed to install console control handler")

    def _win32_console_handler(self, ctrl_type):
        """Callback for Windows console control events.

        CTRL_CLOSE_EVENT (user closes the console window):
          Return TRUE to prevent the OS from killing the process.  The app
          keeps running in the tray.  Also call FreeConsole() to fully
          detach from the now-destroyed console so future writes to stderr
          don't fail.

        CTRL_C_EVENT / CTRL_BREAK_EVENT (user presses Ctrl+C):
          Graceful shutdown — the user explicitly asked to stop.

        CTRL_LOGOFF_EVENT / CTRL_SHUTDOWN_EVENT:
          Graceful shutdown — the system is going down.
        """
        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6

        if ctrl_type == CTRL_CLOSE_EVENT:
            log.info(
                "[WIN32] Console window closing -- "
                "keeping process alive (tray app survives)"
            )
            # Detach from the dying console so stderr/stdout don't fail.
            # After FreeConsole, stderr/stdout file descriptors are invalid,
            # so redirect them to devnull to prevent OSError on writes.
            # Store the devnull file object to prevent GC from closing it.
            try:
                self._kernel32.FreeConsole()
                self._devnull = open(os.devnull, 'w')
                sys.stdout = self._devnull
                sys.stderr = self._devnull
                log.info("[WIN32] Detached from console (FreeConsole)")
            except Exception:
                log.warning("[WIN32] FreeConsole() failed")
            return True  # TRUE = handled; don't kill the process

        if ctrl_type in (CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
            log.info("[WIN32] System event %d received, shutting down", ctrl_type)
            threading.Thread(target=self.quit, daemon=False).start()
            return True

        if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT):
            log.info("[WIN32] Ctrl+C received, shutting down")
            threading.Thread(target=self.quit, daemon=False).start()
            return True

        return False  # Let the next handler deal with it


def main():
    """Entry point."""
    _setup_logging()

    try:
        app = VoiceTyperApp()
    except Exception as e:
        log.exception("Fatal error during initialization")
        # Defensive: _setup_logging replaces None with devnull, but guard
        # in case it fails before the replacement happens.
        if sys.stderr is not None:
            print(f"Voice Typer failed to start: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        app.start()  # blocks on main thread (tray event loop)
    except Exception as e:
        log.exception("Fatal error")
        if sys.stderr is not None:
            print(f"Voice Typer crashed: {e}", file=sys.stderr)
        sys.exit(1)
