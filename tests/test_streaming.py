"""Tests for streaming transcription planning and text assembly."""

import numpy as np

from voice_typer.streaming import (
    AudioWindow,
    AudioWindowPlanner,
    StreamingConfig,
    StreamingTextAssembler,
    WordTiming,
)


SAMPLE_RATE = 16000


def audio_seconds(seconds: float, amplitude: float = 0.01) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), amplitude, dtype=np.float32)


def test_streaming_config_defaults_are_disabled_and_conservative():
    config = StreamingConfig()

    assert config.enabled is False
    assert config.chunk_seconds == 12.0
    assert config.step_seconds == 5.0
    assert config.left_overlap_seconds == 2.0
    assert config.right_guard_seconds == 1.0
    assert config.min_first_chunk_seconds == 6.0
    assert config.silence_threshold == 0.003


def test_audio_window_planner_waits_until_min_first_chunk():
    planner = AudioWindowPlanner(
        StreamingConfig(min_first_chunk_seconds=4.0, chunk_seconds=10.0)
    )

    assert planner.next_window(audio_seconds(3.9), SAMPLE_RATE) is None

    window = planner.next_window(audio_seconds(4.0), SAMPLE_RATE)

    assert window is not None
    assert window.start_seconds == 0.0
    assert window.end_seconds == 4.0
    assert len(window.audio) == 4 * SAMPLE_RATE


def test_audio_window_planner_creates_overlapping_windows():
    planner = AudioWindowPlanner(
        StreamingConfig(
            min_first_chunk_seconds=4.0,
            chunk_seconds=8.0,
            step_seconds=5.0,
            left_overlap_seconds=2.0,
        )
    )

    first = planner.next_window(audio_seconds(4.0), SAMPLE_RATE)
    assert first == AudioWindow(
        audio=audio_seconds(4.0),
        start_seconds=0.0,
        end_seconds=4.0,
    )

    assert planner.next_window(audio_seconds(8.9), SAMPLE_RATE) is None

    second = planner.next_window(audio_seconds(9.0), SAMPLE_RATE)

    assert second is not None
    assert second.start_seconds == 2.0
    assert second.end_seconds == 9.0
    assert len(second.audio) == 7 * SAMPLE_RATE


def test_audio_window_planner_prefers_silence_near_requested_boundary():
    config = StreamingConfig(
        min_first_chunk_seconds=5.0,
        chunk_seconds=5.0,
        silence_threshold=0.003,
    )
    audio = audio_seconds(5.0, amplitude=0.02)
    audio[int(4.5 * SAMPLE_RATE) : int(4.6 * SAMPLE_RATE)] = 0.0
    planner = AudioWindowPlanner(config)

    window = planner.next_window(audio, SAMPLE_RATE)

    assert window is not None
    assert 4.49 <= window.end_seconds <= 4.61
    assert len(window.audio) == int(window.end_seconds * SAMPLE_RATE)


def test_streaming_text_assembler_commits_words_before_right_guard():
    assembler = StreamingTextAssembler()
    window = AudioWindow(audio=audio_seconds(5.0), start_seconds=0.0, end_seconds=5.0)
    words = [
        WordTiming("hello", start_seconds=0.2, end_seconds=0.7),
        WordTiming("world", start_seconds=4.4, end_seconds=4.8),
    ]

    committed = assembler.add_window(window, words, right_guard_seconds=1.0)

    assert committed == "hello"
    assert assembler.committed_text == "hello"
    assert assembler.last_committed_time == 0.7


def test_streaming_text_assembler_avoids_duplicate_overlap_by_timestamp():
    assembler = StreamingTextAssembler()
    first = [
        WordTiming("turn", start_seconds=0.0, end_seconds=0.3),
        WordTiming("left", start_seconds=0.4, end_seconds=0.8),
    ]
    second = [
        WordTiming("left", start_seconds=0.4, end_seconds=0.8),
        WordTiming("now", start_seconds=1.0, end_seconds=1.3),
    ]

    assembler.add_words(first, commit_horizon_seconds=2.0)
    committed = assembler.add_words(second, commit_horizon_seconds=2.0)

    assert committed == "now"
    assert assembler.committed_text == "turn left now"


def test_streaming_text_assembler_preserves_repeated_words_at_later_timestamps():
    assembler = StreamingTextAssembler()

    committed = assembler.add_words(
        [
            WordTiming("yes", start_seconds=0.0, end_seconds=0.2),
            WordTiming("yes", start_seconds=0.5, end_seconds=0.7),
        ],
        commit_horizon_seconds=1.0,
    )

    assert committed == "yes yes"
    assert assembler.committed_text == "yes yes"
