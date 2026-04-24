"""Core helpers for hidden streaming transcription."""

from dataclasses import dataclass, field
import logging
import math
import threading
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingConfig:
    """Timing and safety settings for streaming transcription."""

    enabled: bool = False
    chunk_seconds: float = 12.0
    step_seconds: float = 5.0
    left_overlap_seconds: float = 2.0
    right_guard_seconds: float = 1.0
    min_first_chunk_seconds: float = 6.0
    silence_threshold: float = 0.003


@dataclass(frozen=True)
class WordTiming:
    """One timestamped word in global recording time."""

    word: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, eq=False)
class AudioWindow:
    """A slice of 16 kHz mono audio and its global time bounds."""

    audio: np.ndarray
    start_seconds: float
    end_seconds: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AudioWindow):
            return NotImplemented
        return (
            np.array_equal(self.audio, other.audio)
            and self.start_seconds == other.start_seconds
            and self.end_seconds == other.end_seconds
        )


@dataclass
class AudioWindowPlanner:
    """Plan overlapping audio windows as recording audio grows."""

    config: StreamingConfig = field(default_factory=StreamingConfig)
    _last_window_end_seconds: float | None = None

    def next_window(self, audio: np.ndarray, sample_rate: int) -> AudioWindow | None:
        duration_seconds = len(audio) / sample_rate
        if self._last_window_end_seconds is None:
            if duration_seconds < self.config.min_first_chunk_seconds:
                return None
            requested_start_seconds = 0.0
            requested_end_seconds = min(duration_seconds, self.config.chunk_seconds)
        else:
            requested_end_seconds = (
                self._last_window_end_seconds + self.config.step_seconds
            )
            if duration_seconds < requested_end_seconds:
                return None
            requested_end_seconds = min(duration_seconds, requested_end_seconds)
            requested_start_seconds = max(
                0.0,
                self._last_window_end_seconds - self.config.left_overlap_seconds,
            )

        end_seconds = self._choose_boundary(
            audio=audio,
            sample_rate=sample_rate,
            requested_start_seconds=requested_start_seconds,
            requested_end_seconds=requested_end_seconds,
        )
        start_sample = int(round(requested_start_seconds * sample_rate))
        end_sample = int(round(end_seconds * sample_rate))
        window = AudioWindow(
            audio=audio[start_sample:end_sample].copy(),
            start_seconds=requested_start_seconds,
            end_seconds=end_seconds,
        )
        self._last_window_end_seconds = end_seconds
        return window

    def _choose_boundary(
        self,
        audio: np.ndarray,
        sample_rate: int,
        requested_start_seconds: float,
        requested_end_seconds: float,
    ) -> float:
        search_seconds = min(1.0, requested_end_seconds - requested_start_seconds)
        if search_seconds <= 0:
            return requested_end_seconds

        search_start = int(round((requested_end_seconds - search_seconds) * sample_rate))
        search_end = int(round(requested_end_seconds * sample_rate))
        search = audio[search_start:search_end]
        if len(search) == 0:
            return requested_end_seconds

        frame_size = max(1, int(0.05 * sample_rate))
        best_rms = float("inf")
        best_index = None
        for index in range(0, len(search), frame_size):
            frame = search[index : index + frame_size]
            if len(frame) == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            if rms < best_rms:
                best_rms = rms
                best_index = index + len(frame) // 2

        if best_index is None or best_rms > self.config.silence_threshold:
            return requested_end_seconds
        return (search_start + best_index) / sample_rate


@dataclass
class StreamingTextAssembler:
    """Commit timestamped words only after they are outside the unsafe tail."""

    _words: list[str] = field(default_factory=list)
    _seen_timestamps: set[tuple[float, float]] = field(default_factory=set)
    last_committed_time: float = 0.0

    @property
    def committed_text(self) -> str:
        return " ".join(self._words)

    def add_window(
        self,
        window: AudioWindow,
        words: Iterable[WordTiming],
        right_guard_seconds: float,
    ) -> str:
        return self.add_words(
            words,
            commit_horizon_seconds=window.end_seconds - right_guard_seconds,
        )

    def add_words(
        self,
        words: Iterable[WordTiming],
        commit_horizon_seconds: float,
    ) -> str:
        committed: list[str] = []
        for word in words:
            if word.end_seconds > commit_horizon_seconds:
                continue
            timestamp_key = (
                round(word.start_seconds, 3),
                round(word.end_seconds, 3),
            )
            if timestamp_key in self._seen_timestamps:
                continue
            if word.end_seconds <= self.last_committed_time:
                continue

            text = word.word.strip()
            if not text:
                continue
            self._seen_timestamps.add(timestamp_key)
            self._words.append(text)
            committed.append(text)
            self.last_committed_time = max(
                self.last_committed_time,
                word.end_seconds,
            )
        return " ".join(committed)


class StreamingTranscriptionSession:
    """Hidden streaming worker for one recording session."""

    def __init__(
        self,
        recorder,
        transcriber,
        config: StreamingConfig,
        sample_rate: int,
        poll_interval_seconds: float = 0.25,
    ):
        self.recorder = recorder
        self.transcriber = transcriber
        self.config = config
        self.sample_rate = sample_rate
        self.poll_interval_seconds = poll_interval_seconds
        self.planner = AudioWindowPlanner(config)
        self.assembler = StreamingTextAssembler()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fallback_required = False
        self._lock = threading.Lock()

    @property
    def confirmed_text(self) -> str:
        return self.assembler.committed_text

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start the background streaming worker."""
        if self.is_running:
            return
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="StreamingTranscription",
            daemon=True,
        )
        self._thread.start()

    def cancel(self):
        """Stop background streaming work and wait briefly for the worker."""
        self._cancel_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def process_available_audio_once(self) -> bool:
        """Process one planned window if enough audio is available."""
        if self._fallback_required:
            return False

        try:
            audio = self.recorder.snapshot()
            window = self.planner.next_window(audio, self.sample_rate)
            if window is None:
                return False

            words = self.transcriber.transcribe_words(
                window.audio,
                offset_seconds=window.start_seconds,
            )
            self._validate_words(words)
            with self._lock:
                self.assembler.add_window(
                    window,
                    words,
                    right_guard_seconds=self.config.right_guard_seconds,
                )
            return True
        except Exception as exc:
            log.exception("[STREAMING] Chunk transcription failed: %s", exc)
            self._fallback_required = True
            return False

    def finalize(self, full_audio: np.ndarray) -> str:
        """Return final transcript, using batch fallback if streaming is unsafe."""
        self.cancel()
        if self._fallback_required:
            return self.transcriber.transcribe_with_fallback(full_audio)

        try:
            tail_start_seconds = max(
                0.0,
                self.assembler.last_committed_time
                - self.config.left_overlap_seconds,
            )
            start_sample = min(
                len(full_audio),
                int(round(tail_start_seconds * self.sample_rate)),
            )
            tail_audio = full_audio[start_sample:]
            words = self.transcriber.transcribe_words(
                tail_audio,
                offset_seconds=tail_start_seconds,
            )
            self._validate_words(words)
            with self._lock:
                self.assembler.add_words(words, commit_horizon_seconds=math.inf)
                return self.assembler.committed_text
        except Exception as exc:
            log.exception("[STREAMING] Final tail merge failed: %s", exc)
            return self.transcriber.transcribe_with_fallback(full_audio)

    def _run(self):
        while not self._cancel_event.is_set():
            self.process_available_audio_once()
            self._cancel_event.wait(self.poll_interval_seconds)

    def _validate_words(self, words: Iterable[WordTiming]):
        for word in words:
            if not isinstance(word.word, str):
                raise TypeError("word text must be a string")
            if word.start_seconds is None or word.end_seconds is None:
                raise TypeError("word timestamps are required")
