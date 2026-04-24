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

    def test_start_failure_resets_recording_state(self, monkeypatch):
        from voice_typer.recording import Recorder
        import voice_typer.recording as recording_mod

        class FailingStream:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("device failed")

            def close(self):
                pass

        monkeypatch.setattr(recording_mod.sd, "InputStream", FailingStream)

        config = MagicMock(sample_rate=16000, microphone=None)
        r = Recorder(config)

        with pytest.raises(RuntimeError, match="device failed"):
            r.start()

        assert r.recording is False
        assert r._stream is None

    def test_start_falls_back_to_same_microphone_on_another_host_api(self, monkeypatch):
        from voice_typer.recording import Recorder
        import voice_typer.recording as recording_mod

        devices = [
            {"index": 0, "name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "default_samplerate": 44100, "hostapi": 0},
            {"index": 1, "name": "Microphone (WO Mic Device)", "max_input_channels": 1, "default_samplerate": 44100, "hostapi": 0},
            {"index": 8, "name": "Primary Sound Capture Driver", "max_input_channels": 2, "default_samplerate": 44100, "hostapi": 1},
            {"index": 9, "name": "Microphone (WO Mic Device)", "max_input_channels": 1, "default_samplerate": 44100, "hostapi": 1},
        ]
        host_apis = {
            0: {"name": "MME", "default_input_device": 1},
            1: {"name": "Windows DirectSound", "default_input_device": 8},
        }

        def query_devices(device=None, kind=None):
            if kind == "input":
                return devices[1]
            if device is None:
                return devices
            return next(dev for dev in devices if dev["index"] == device)

        monkeypatch.setattr(recording_mod.sd, "query_devices", query_devices)
        monkeypatch.setattr(recording_mod.sd, "query_hostapis", lambda idx=None: host_apis[idx])

        opened_devices = []

        class FallbackStream:
            def __init__(self, *args, **kwargs):
                opened_devices.append(kwargs["device"])
                if kwargs["device"] == 9:
                    raise RuntimeError("DirectSound error")
                self.closed = False
                self.started = False

            def start(self):
                self.started = True

            def close(self):
                self.closed = True

        monkeypatch.setattr(recording_mod.sd, "InputStream", FallbackStream)

        config = MagicMock(sample_rate=16000, microphone="9")
        r = Recorder(config)

        r.start()

        assert opened_devices == [9, 1]
        assert r.recording is True
        assert r._stream is not None
        assert config.microphone == "1"
        config.save.assert_called_once()

    def test_snapshot_returns_audio_without_clearing_buffer(self):
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

        snapshot = r.snapshot()

        np.testing.assert_array_equal(snapshot, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert len(r._buffer) == 2

        stopped = r.stop()
        np.testing.assert_array_equal(stopped, snapshot)

    def test_snapshot_returns_empty_float32_when_no_buffer_exists(self):
        from voice_typer.recording import Recorder

        config = MagicMock(sample_rate=16000, microphone=None)
        r = Recorder(config)
        r._recording = True
        r._effective_sr = 16000
        r._buffer = []

        snapshot = r.snapshot()

        assert snapshot.dtype == np.float32
        assert snapshot.size == 0

    def test_snapshot_uses_same_resampling_path_as_stop(self, monkeypatch):
        from voice_typer.recording import Recorder

        calls = []

        def fake_resample_poly(audio, up, down):
            calls.append((audio.copy(), up, down))
            return np.array([0.25, 0.5], dtype=np.float32)

        monkeypatch.setattr("voice_typer.recording._get_resample_poly", lambda: fake_resample_poly)

        config = MagicMock(sample_rate=16000, microphone=None)
        r = Recorder(config)
        r._recording = True
        r._effective_sr = 48000
        r._buffer = [np.ones((6, 1), dtype=np.float32)]

        snapshot = r.snapshot()

        np.testing.assert_array_equal(snapshot, np.array([0.25, 0.5], dtype=np.float32))
        assert len(r._buffer) == 1
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0][0], np.ones(6, dtype=np.float32))
        assert calls[0][1:] == (1, 3)
