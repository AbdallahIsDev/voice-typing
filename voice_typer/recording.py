"""Session-based audio recording."""

import logging
import math
import threading
import time
from typing import Optional, List, Any

import numpy as np
import sounddevice as sd

from voice_typer.config import Config

log = logging.getLogger(__name__)

_resample_poly = None
_resample_poly_error: Exception | None = None
_resample_poly_lock = threading.Lock()


def _get_resample_poly():
    """Load scipy's resampler once so imports do not happen on F2 stop."""
    global _resample_poly, _resample_poly_error
    if _resample_poly is not None:
        return _resample_poly
    if _resample_poly_error is not None:
        raise _resample_poly_error

    with _resample_poly_lock:
        if _resample_poly is not None:
            return _resample_poly
        if _resample_poly_error is not None:
            raise _resample_poly_error
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            _resample_poly_error = exc
            raise
        _resample_poly = resample_poly
        return _resample_poly


class Recorder:
    """Records audio from microphone into a buffer. Session-based: start, accumulate, stop, get data."""

    def __init__(self, config: Config):
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._effective_sr: int = config.sample_rate
        self._last_rms: float = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def last_rms(self) -> float:
        """RMS level of the most recently captured audio (0.0 if never recorded)."""
        return self._last_rms

    def warm_up_resampler(self):
        """Import and initialize the high-quality resampler before recording stops."""
        try:
            resample_poly = _get_resample_poly()
            resample_poly(np.zeros(32, dtype=np.float32), 160, 441)
            log.info("[RECORDING] Resampler warmed up")
        except ImportError:
            log.warning("[RECORDING] scipy not available, will use linear interp resampling")
        except Exception as e:
            log.warning("[RECORDING] Resampler warm-up failed: %s", e)

    def _resolve_device(self):
        """Resolve config.microphone to a sounddevice device specifier.

        config.microphone is a string device index (from list_microphones)
        or None for system default.  We convert to int for unambiguous
        selection by sounddevice.
        """
        mic = self.config.microphone
        if mic is None:
            return None
        try:
            return int(mic)
        except (ValueError, TypeError):
            # Legacy: if someone put a device name string, pass it through
            return mic

    def _host_api_name(self, host_api_index: int) -> str:
        try:
            return sd.query_hostapis(host_api_index)["name"]
        except Exception:
            return ""

    def _device_index(self, fallback_index: int, device_info: dict) -> int:
        try:
            return int(device_info.get("index", fallback_index))
        except Exception:
            return fallback_index

    def _same_physical_microphone_candidates(self, device: Any) -> list[Any]:
        """Return equivalent input device IDs to try if the selected one fails."""
        candidates = [device]
        if not isinstance(device, int):
            return candidates

        try:
            selected = sd.query_devices(device)
            selected_name = selected.get("name", "").strip().lower()
            all_devices = list(sd.query_devices())
        except Exception as e:
            log.debug("[RECORDING] Could not build microphone fallback list: %s", e)
            return candidates

        if not selected_name:
            return candidates

        alternates = []
        for fallback_index, info in enumerate(all_devices):
            index = self._device_index(fallback_index, info)
            if index == device:
                continue
            if info.get("max_input_channels", 0) <= 0:
                continue
            if info.get("name", "").strip().lower() != selected_name:
                continue
            host_name = self._host_api_name(info.get("hostapi", 0))
            alternates.append((self._fallback_host_rank(host_name), index))

        alternates.sort()
        seen = set()
        ordered = []
        for candidate in candidates + [index for _, index in alternates]:
            marker = str(candidate)
            if marker in seen:
                continue
            ordered.append(candidate)
            seen.add(marker)
        return ordered

    def _fallback_host_rank(self, host_name: str) -> int:
        lower = host_name.lower()
        if lower == "mme":
            return 0
        if "wasapi" in lower:
            return 1
        if "wdm-ks" in lower:
            return 2
        if "directsound" in lower:
            return 3
        return 4

    def _resolve_effective_sample_rate(self, device: Optional[int]) -> tuple[int, Optional[dict]]:
        """Determine the effective sample rate and device info for the given device.

        Returns (effective_sr, dev_info_dict) where dev_info_dict has
        'name', 'host_api_name', 'native_rate' keys, or None if query failed.

        Strategy: always record at the device's native sample rate when it
        differs from the Whisper target rate (16kHz), and resample afterwards
        with scipy.  This avoids relying on PortAudio's internal resampling
        (which can introduce artifacts, especially via MME on Windows) and
        ensures WASAPI devices that reject non-native rates work correctly.

        Only uses the requested 16kHz rate directly when the device's native
        rate IS 16000 Hz.
        """
        target_sr = self.config.sample_rate  # 16000 for Whisper
        dev_info_extra = None
        try:
            # device=None means system default; query_devices(None) returns
            # a list of ALL devices, so we must use kind='input' instead.
            if device is None:
                dev_info = sd.query_devices(kind="input")
            else:
                dev_info = sd.query_devices(device)
            native_rate = int(dev_info["default_samplerate"])
            host_api_name = ""
            try:
                host_api_idx = dev_info.get("hostapi", 0)
                host_api_name = sd.query_hostapis(host_api_idx)["name"]
            except Exception:
                pass
            dev_info_extra = {
                "name": dev_info["name"],
                "host_api_name": host_api_name,
                "native_rate": native_rate,
            }
            log.info(
                "[RECORDING] Device query: name=%s, host_api=%s, "
                "native_rate=%d, target_rate=%d",
                dev_info["name"], host_api_name, native_rate, target_sr,
            )

            # If the device's native rate matches the target, use it directly.
            # Otherwise, always record at native rate and resample afterwards.
            # This avoids PortAudio's internal resampling (which can produce
            # lower-quality audio via MME) and ensures WASAPI devices that
            # reject non-native rates (e.g. 16kHz on a 48kHz WASAPI device)
            # work correctly.
            if native_rate == target_sr:
                log.info(
                    "[RECORDING] Native rate matches target, using %d Hz directly",
                    target_sr,
                )
                return target_sr, dev_info_extra
            else:
                log.info(
                    "[RECORDING] Native rate %d differs from target %d, "
                    "will record at native rate and resample",
                    native_rate, target_sr,
                )
                return native_rate, dev_info_extra
        except Exception as e:
            log.warning("[RECORDING] Could not query device info: %s", e)
            return target_sr, dev_info_extra

    def start(self):
        """Start recording audio."""
        if self._recording:
            return

        self._buffer.clear()

        device = self._resolve_device()
        candidates = self._same_physical_microphone_candidates(device)

        def callback(indata, frames, time_info, status):
            with self._lock:
                self._buffer.append(indata.copy())

        last_error = None
        selected_device = None
        effective_sr = self.config.sample_rate

        for candidate in candidates:
            candidate_sr, dev_info_extra = self._resolve_effective_sample_rate(candidate)

            if dev_info_extra:
                log.info(
                    "[RECORDING] Using device: [%s] %s | host_api=%s | "
                    "native_rate=%d | effective_rate=%d",
                    candidate if candidate is not None else "default",
                    dev_info_extra["name"],
                    dev_info_extra["host_api_name"],
                    dev_info_extra["native_rate"],
                    candidate_sr,
                )

            stream = None
            try:
                stream = sd.InputStream(
                    samplerate=candidate_sr,
                    channels=1,
                    dtype=np.float32,
                    device=candidate,
                    callback=callback,
                )
                stream.start()
            except Exception as e:
                last_error = e
                log.warning(
                    "[RECORDING] Failed to open input device [%s]: %s",
                    candidate if candidate is not None else "default",
                    e,
                )
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                self._stream = None
                self._recording = False
                continue

            self._stream = stream
            self._effective_sr = candidate_sr
            selected_device = candidate
            effective_sr = candidate_sr
            break

        if self._stream is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No input device could be opened")

        if selected_device != device and isinstance(selected_device, int):
            log.info(
                "[RECORDING] Selected microphone [%s] failed; using matching device [%s]",
                device,
                selected_device,
            )
            self.config.microphone = str(selected_device)
            try:
                self.config.save()
            except Exception as e:
                log.debug("[RECORDING] Could not persist microphone fallback: %s", e)

        self._recording = True

        target_sr = self.config.sample_rate
        if effective_sr != target_sr and _resample_poly is None and _resample_poly_error is None:
            threading.Thread(
                target=self.warm_up_resampler,
                name="ResamplerWarmup",
                daemon=True,
            ).start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the complete audio array."""
        if not self._recording:
            return np.array([], dtype=np.float32)

        stop_started = time.perf_counter()
        self._recording = False

        stream_started = time.perf_counter()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        stream_ms = (time.perf_counter() - stream_started) * 1000

        concat_started = time.perf_counter()
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._buffer, axis=0).reshape(-1)
            self._buffer.clear()
        concat_ms = (time.perf_counter() - concat_started) * 1000

        # Log audio statistics for diagnostics
        stats_started = time.perf_counter()
        effective_sr = self._effective_sr
        duration = len(audio) / effective_sr if len(audio) > 0 else 0
        if len(audio) > 0:
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            peak = float(np.max(np.abs(audio)))
            silence_pct = float(np.sum(np.abs(audio) < 0.001) / audio.size * 100)
            self._last_rms = rms
            log.info(
                "[RECORDING] Audio captured: duration=%.1fs, effective_sr=%d, "
                "samples=%d, RMS=%.6f, peak=%.6f, silence_pct=%.1f%%",
                duration, effective_sr, len(audio), rms, peak, silence_pct,
            )
            if rms < 0.001:
                log.warning(
                    "[RECORDING] Near-silence detected! (RMS=%.6f) "
                    "Microphone may not be capturing audio.",
                    rms,
                )
        else:
            self._last_rms = 0.0
            log.warning("[RECORDING] No audio data captured!")
        stats_ms = (time.perf_counter() - stats_started) * 1000

        resample_started = time.perf_counter()
        audio = self._prepare_audio(audio, effective_sr)
        resample_ms = (time.perf_counter() - resample_started) * 1000

        total_ms = (time.perf_counter() - stop_started) * 1000
        log.info(
            "[RECORDING] Stop timing: stream=%.1fms, concat=%.1fms, "
            "stats=%.1fms, resample=%.1fms, total=%.1fms",
            stream_ms, concat_ms, stats_ms, resample_ms, total_ms,
        )

        return audio

    def snapshot(self) -> np.ndarray:
        """Return current recorded audio without clearing the active buffer."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._buffer, axis=0).reshape(-1)
            effective_sr = self._effective_sr

        return self._prepare_audio(audio, effective_sr)

    def _prepare_audio(self, audio: np.ndarray, effective_sr: int) -> np.ndarray:
        """Convert captured audio to the configured sample rate."""
        target_sr = self.config.sample_rate  # 16000 for Whisper
        if effective_sr != target_sr and len(audio) > 0:
            orig_len = len(audio)
            resampled = False
            try:
                resample_poly = _get_resample_poly()
                gcd = math.gcd(effective_sr, target_sr)
                up = target_sr // gcd
                down = effective_sr // gcd
                audio = resample_poly(audio, up, down).astype(np.float32)
                log.info(
                    "[RECORDING] Resampled %d Hz -> %d Hz (%d -> %d samples)",
                    effective_sr, target_sr, orig_len, len(audio),
                )
                resampled = True
            except ImportError:
                log.warning(
                    "[RECORDING] scipy not available, using linear interp resampling"
                )
            except Exception as e:
                log.error("[RECORDING] scipy resample_poly failed: %s", e)

            if not resampled:
                # Fallback: simple linear interpolation resampling
                try:
                    ratio = target_sr / effective_sr
                    new_len = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_len)
                    audio = np.interp(
                        indices, np.arange(len(audio)), audio,
                    ).astype(np.float32)
                    log.info(
                        "[RECORDING] Resampled (linear interp) %d Hz -> %d Hz "
                        "(%d -> %d samples)",
                        effective_sr, target_sr, orig_len, len(audio),
                    )
                    resampled = True
                except Exception as e:
                    log.error(
                        "[RECORDING] All resampling failed: %s. "
                        "Audio at %d Hz cannot be used by Whisper.",
                        e, effective_sr,
                    )

            if not resampled:
                raise RuntimeError(
                    f"Cannot resample audio from {effective_sr} Hz to "
                    f"{target_sr} Hz. Check scipy installation and audio format."
                )

        return audio

    def discard(self):
        """Discard current recording without processing."""
        self._recording = False
        self._effective_sr = self.config.sample_rate
        self._last_rms = 0.0
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._buffer.clear()
