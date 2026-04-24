"""Whisper transcription engine using faster-whisper."""

import logging
import os
import site
import sys
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_WHISPER_SAMPLE_RATE = 16000  # Whisper always expects 16kHz input
_nvidia_dll_path_handles: list[object] = []
_nvidia_dll_paths_configured = False


def _configure_nvidia_dll_paths():
    """Expose NVIDIA wheel DLL directories to the Windows loader."""
    global _nvidia_dll_paths_configured
    if _nvidia_dll_paths_configured or sys.platform != "win32":
        return

    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(user_site)
    except Exception:
        pass

    candidate_parts = [
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cudnn", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
    ]
    existing_paths = os.environ.get("PATH", "").split(os.pathsep)
    new_paths: list[str] = []
    for root in roots:
        for parts in candidate_parts:
            path = os.path.join(root, *parts)
            if not os.path.isdir(path):
                continue
            if not any(name.lower().endswith(".dll") for name in os.listdir(path)):
                continue
            if path not in existing_paths and path not in new_paths:
                new_paths.append(path)
            add_dll_directory = getattr(os, "add_dll_directory", None)
            if add_dll_directory is not None:
                handle = add_dll_directory(path)
                if handle is not None:
                    _nvidia_dll_path_handles.append(handle)

    if new_paths:
        os.environ["PATH"] = os.pathsep.join(new_paths + existing_paths)
        log.info("[CUDA] Added NVIDIA wheel DLL paths: %s", new_paths)
    _nvidia_dll_paths_configured = True


class TranscriptionEngine:
    """Wraps faster-whisper model loading and transcription."""

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "auto",
        language: str = "en",
        beam_size: int = 1,
        best_of: int = 1,
        condition_on_previous_text: bool = False,
    ):
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.condition_on_previous_text = condition_on_previous_text
        self._model = None
        self._lock = threading.RLock()
        self._device, self._compute_type = self._resolve_device(device)

    def _resolve_device(self, device: str) -> tuple[str, str]:
        """Auto-detect best device and compute type."""
        if device == "cpu":
            return "cpu", "int8"

        # Try CUDA
        if device in ("auto", "cuda"):
            try:
                _configure_nvidia_dll_paths()
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    log.info("Using CUDA device for transcription")
                    return "cuda", "float16"
            except Exception:
                pass

            if device == "cuda":
                log.warning("CUDA requested but not available, falling back to CPU")

        log.info("Using CPU for transcription")
        return "cpu", "int8"

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._model is not None

    @property
    def device_info(self) -> str:
        return f"{self._device} ({self._compute_type})"

    @property
    def loaded_via(self) -> str:
        """Return a description of the device/model combo that was successfully loaded."""
        return f"{self._device}/{self._compute_type}/{self.model_size}"

    def load(self):
        """Load the Whisper model. Downloads on first run.

        Fallback chain:
          1. Configured device (e.g. CUDA/float16)
          2. CPU / int8 with original model size
          3. CPU / int8 with tiny.en
          4. CPU / float32 with tiny.en (last resort — avoids MKL int8 path)

        Stores which path succeeded via loaded_via property.
        """
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self):
        if self._model is not None:
            return

        _configure_nvidia_dll_paths()
        from faster_whisper import WhisperModel

        # Build fallback chain
        chain: list[tuple[str, str, str]] = []  # (device, compute_type, model_size)
        chain.append((self._device, self._compute_type, self.model_size))
        if self._device != "cpu" or self._compute_type != "int8":
            chain.append(("cpu", "int8", self.model_size))
        if self.model_size != "tiny.en":
            chain.append(("cpu", "int8", "tiny.en"))
        # Last resort: float32 avoids MKL int8 memory allocation entirely
        chain.append(("cpu", "float32", "tiny.en"))

        last_error = None
        for device, compute_type, model_size in chain:
            try:
                log.info(
                    "Loading Whisper model '%s' on %s (%s)...",
                    model_size, device, compute_type,
                )
                self._model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
                # Update stored device info on success
                self._device = device
                self._compute_type = compute_type
                self.model_size = model_size
                log.info("Model loaded via %s", self.loaded_via)
                return
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Model load failed on %s (%s) model=%s: %s",
                    device, compute_type, model_size, exc,
                )
                self._model = None

        # All paths exhausted
        raise RuntimeError(
            f"Failed to load Whisper model on any device/model. "
            f"Last error: {last_error}"
        ) from last_error

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array. Returns cleaned text string."""
        with self._lock:
            return self._transcribe_unlocked(audio)

    def _transcribe_unlocked(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(audio) == 0:
            return ""

        # Log audio statistics for diagnostics
        duration = len(audio) / _WHISPER_SAMPLE_RATE
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        peak = float(np.max(np.abs(audio)))
        silence_pct = float(np.sum(np.abs(audio) < 0.001) / audio.size * 100)
        log.info(
            "[TRANSCRIBE] Input audio: samples=%d, duration=%.1fs, "
            "RMS=%.6f, peak=%.6f, silence_pct=%.1f%%",
            len(audio), duration, rms, peak, silence_pct,
        )
        if rms < 0.001:
            log.warning(
                "[TRANSCRIBE] Near-silence input (RMS=%.6f). "
                "Speech detection is unlikely.",
                rms,
            )

        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            without_timestamps=True,
        )

        # Collect segments and log VAD info
        text_parts = []
        for seg in segments:
            if seg.text.strip():
                text_parts.append(seg.text.strip())
                log.debug(
                    "[TRANSCRIBE] Segment: [%.1fs - %.1fs] %s",
                    seg.start or 0.0, seg.end or 0.0, seg.text.strip(),
                )

        log.info(
            "[TRANSCRIBE] VAD result: language=%s (prob=%.2f), segments=%d",
            info.language, info.language_probability, len(text_parts),
        )

        result = " ".join(text_parts).strip()
        if result:
            log.info("[TRANSCRIBE] Result: %s", result[:200])
        else:
            log.info(
                "[TRANSCRIBE] No speech detected (RMS=%.6f, silence=%.1f%%)",
                rms, silence_pct,
            )
        return result

    def transcribe_with_fallback(self, audio: np.ndarray) -> str:
        """Transcribe with automatic CPU fallback on GPU runtime errors.

        If the first attempt fails with a CUDA/cuBLAS/runtime error and
        the model was loaded on GPU, reload on CPU and retry once.
        """
        with self._lock:
            return self._transcribe_with_fallback_unlocked(audio)

    def _transcribe_with_fallback_unlocked(self, audio: np.ndarray) -> str:
        try:
            return self._transcribe_unlocked(audio)
        except Exception as first_err:
            if not self._is_gpu_runtime_error(first_err):
                raise

            log.warning(
                "GPU transcription failed (%s), falling back to CPU",
                first_err,
            )
            # Tear down GPU model, reload on CPU
            self._model = None
            self._device = "cpu"
            self._compute_type = "int8"
            self._load_unlocked()  # reloads with the new CPU config
            return self._transcribe_unlocked(audio)

    def transcribe_words(self, audio: np.ndarray, offset_seconds: float = 0.0):
        """Transcribe audio array into word timings with a global offset."""
        with self._lock:
            return self._transcribe_words_with_fallback_unlocked(audio, offset_seconds)

    def _transcribe_words_with_fallback_unlocked(
        self,
        audio: np.ndarray,
        offset_seconds: float,
    ):
        try:
            return self._transcribe_words_unlocked(audio, offset_seconds)
        except Exception as first_err:
            if not self._is_gpu_runtime_error(first_err):
                raise

            log.warning(
                "GPU timestamped transcription failed (%s), falling back to CPU",
                first_err,
            )
            self._model = None
            self._device = "cpu"
            self._compute_type = "int8"
            self._load_unlocked()
            return self._transcribe_words_unlocked(audio, offset_seconds)

    def _transcribe_words_unlocked(self, audio: np.ndarray, offset_seconds: float):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(audio) == 0:
            return []

        from voice_typer.streaming import WordTiming

        segments, _info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=True,
            without_timestamps=False,
        )

        words = []
        for seg in segments:
            for word in getattr(seg, "words", None) or []:
                text = (word.word or "").strip()
                if not text:
                    continue
                start = (word.start or 0.0) + offset_seconds
                end = (word.end or word.start or 0.0) + offset_seconds
                words.append(
                    WordTiming(
                        word=text,
                        start_seconds=start,
                        end_seconds=end,
                    )
                )
        return words

    def _is_gpu_runtime_error(self, exc: Exception) -> bool:
        error_str = str(exc).lower()
        return (
            self._device != "cpu"
            and any(kw in error_str for kw in [
                "cublas", "cuda", "cudnn", "gpu",
                "not found or cannot be loaded",
            ])
        )

    def unload(self):
        """Free model memory."""
        with self._lock:
            self._model = None
