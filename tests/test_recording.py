"""Tests for recording module — device resolution."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock sounddevice at module level
@pytest.fixture(autouse=True)
def mock_sounddevice(monkeypatch):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

import sys


class TestResolveDevice:
    def test_none_config_returns_none(self):
        from voice_typer.recording import Recorder
        config = MagicMock()
        config.microphone = None
        config.sample_rate = 16000
        r = Recorder(config)
        assert r._resolve_device() is None

    def test_string_index_converts_to_int(self):
        from voice_typer.recording import Recorder
        config = MagicMock()
        config.microphone = "7"
        config.sample_rate = 16000
        r = Recorder(config)
        assert r._resolve_device() == 7

    def test_legacy_name_string_passes_through(self):
        """If someone put a device name (not numeric), pass it as-is."""
        from voice_typer.recording import Recorder
        config = MagicMock()
        config.microphone = "Blue Yeti"
        config.sample_rate = 16000
        r = Recorder(config)
        assert r._resolve_device() == "Blue Yeti"


class TestStopAudioPrep:
    def test_stop_concatenates_chunks_to_1d_and_clears_buffer(self, monkeypatch):
        from voice_typer.recording import Recorder

        config = MagicMock(sample_rate=16000, microphone=None)
        r = Recorder(config)
        r._recording = True
        r._effective_sr = 16000
        r._stream = MagicMock()
        r._buffer = [
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[3.0]], dtype=np.float32),
        ]

        audio = r.stop()

        np.testing.assert_array_equal(audio, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert r._buffer == []
        assert r._stream is None

    def test_stop_resamples_when_effective_rate_differs(self, monkeypatch):
        from voice_typer.recording import Recorder

        calls = []

        def fake_resample_poly(audio, up, down):
            calls.append((up, down))
            return np.array([0.25, 0.5], dtype=np.float32)

        monkeypatch.setattr("voice_typer.recording._get_resample_poly", lambda: fake_resample_poly)

        config = MagicMock(sample_rate=16000, microphone=None)
        r = Recorder(config)
        r._recording = True
        r._effective_sr = 48000
        r._stream = MagicMock()
        r._buffer = [np.ones((6, 1), dtype=np.float32)]

        audio = r.stop()

        np.testing.assert_array_equal(audio, np.array([0.25, 0.5], dtype=np.float32))
        assert calls == [(1, 3)]

    def test_stop_skips_resample_when_rate_matches_target(self, monkeypatch):
        from voice_typer.recording import Recorder

        get_resampler = MagicMock()
        monkeypatch.setattr("voice_typer.recording._get_resample_poly", get_resampler)

        config = MagicMock(sample_rate=16000, microphone=None)
        r = Recorder(config)
        r._recording = True
        r._effective_sr = 16000
        r._stream = MagicMock()
        r._buffer = [np.ones((4, 1), dtype=np.float32)]

        r.stop()

        get_resampler.assert_not_called()
