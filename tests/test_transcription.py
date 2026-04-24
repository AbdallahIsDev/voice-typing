"""Tests for TranscriptionEngine fallback chain.

All faster_whisper imports are mocked so these tests run on any platform
without GPU or model downloads.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call


@pytest.fixture(autouse=True)
def mock_faster_whisper(monkeypatch):
    """Mock faster_whisper so no real model is loaded."""
    mock_fw = MagicMock()
    monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
    monkeypatch.setitem(sys.modules, "faster_whisper.WhisperModel", MagicMock())


class TestFallbackChain:
    """Verify the fallback chain tries the right device/compute combinations."""

    def test_chain_includes_float32_last_resort(self):
        """The fallback chain must include CPU/float32/tiny.en as the last resort."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="auto")
        # Force all WhisperModel calls to fail
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")
        mod_obj.WhisperModel.side_effect = RuntimeError("mkl_malloc: failed to allocate memory")

        with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
            engine.load()

        # Verify the last call was with float32
        calls = mod_obj.WhisperModel.call_args_list
        assert len(calls) >= 1
        last_call = calls[-1]
        args, kwargs = last_call[0], last_call[1] if len(last_call) > 1 else {}
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "float32"
        assert args[0] == "tiny.en"

    def test_chain_tries_preferred_device_first(self):
        """First attempt should use the configured device/compute type."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cuda")
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")
        mod_obj.WhisperModel.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            engine.load()

        first_call = mod_obj.WhisperModel.call_args_list[0]
        args, kwargs = first_call[0], first_call[1] if len(first_call) > 1 else {}
        assert kwargs["device"] == "cuda"
        assert kwargs["compute_type"] == "float16"

    def test_chain_falls_through_to_cpu_int8(self):
        """After CUDA fails, should try CPU/int8 with same model."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cuda")
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")
        mod_obj.WhisperModel.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            engine.load()

        calls = mod_obj.WhisperModel.call_args_list
        # Second call should be CPU/int8/small.en
        second = calls[1]
        args, kwargs = second[0], second[1] if len(second) > 1 else {}
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "int8"
        assert args[0] == "small.en"

    def test_chain_tries_tiny_en_before_float32(self):
        """Should try CPU/int8/tiny.en before CPU/float32/tiny.en."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="medium.en", device="cuda")
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")
        mod_obj.WhisperModel.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            engine.load()

        calls = mod_obj.WhisperModel.call_args_list
        # Should have 4 calls: cuda/float16/medium.en, cpu/int8/medium.en,
        # cpu/int8/tiny.en, cpu/float32/tiny.en
        assert len(calls) == 4
        third = calls[2]
        args, kwargs = third[0], third[1] if len(third) > 1 else {}
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "int8"
        assert args[0] == "tiny.en"

    def test_succeeds_on_fallback(self):
        """If preferred fails but fallback succeeds, load() should succeed."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cuda")
        mock_model = MagicMock()
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")

        # First call (CUDA) fails, second call (CPU/int8) succeeds
        mod_obj.WhisperModel.side_effect = [
            RuntimeError("CUDA OOM"),
            mock_model,
        ]

        engine.load()

        assert engine.is_loaded
        assert engine._device == "cpu"
        assert engine._compute_type == "int8"
        assert engine.model_size == "small.en"

    def test_float32_success_updates_device_info(self):
        """If only float32 succeeds, device_info should reflect that."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cuda")
        mock_model = MagicMock()
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")

        # All fail except the last (float32/tiny.en)
        mod_obj.WhisperModel.side_effect = [
            RuntimeError("CUDA fail"),
            RuntimeError("int8 fail"),
            RuntimeError("int8 tiny fail"),
            mock_model,  # float32 succeeds
        ]

        engine.load()

        assert engine.is_loaded
        assert engine._device == "cpu"
        assert engine._compute_type == "float32"
        assert engine.model_size == "tiny.en"
        assert "float32" in engine.device_info

    def test_loaded_via_reflects_actual_path(self):
        """loaded_via should show which fallback path was used."""
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="auto")
        mock_model = MagicMock()
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")

        # Force resolution to CPU (auto with no CUDA)
        engine._device = "cpu"
        engine._compute_type = "int8"

        # First try (cpu/int8/small.en) succeeds
        mod_obj.WhisperModel.return_value = mock_model

        engine.load()

        assert engine.loaded_via == "cpu/int8/small.en"


class TestLoadIdempotent:
    """Calling load() twice should not re-download or re-initialize."""

    def test_second_load_is_noop(self):
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cpu")
        mock_model = MagicMock()
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")
        mod_obj.WhisperModel.return_value = mock_model

        engine.load()
        first_call_count = mod_obj.WhisperModel.call_count

        engine.load()
        assert mod_obj.WhisperModel.call_count == first_call_count


class TestTranscribeWithFallback:
    """Verify transcribe_with_fallback handles GPU errors and retries on CPU."""

    def _make_engine_with_model(self):
        from voice_typer.transcription import TranscriptionEngine
        engine = TranscriptionEngine(model_size="small.en", device="cuda")
        engine._device = "cuda"
        engine._compute_type = "float16"
        mock_model = MagicMock()
        engine._model = mock_model
        return engine, mock_model

    def test_successful_transcribe_no_fallback(self):
        """If transcribe succeeds, fallback is not triggered."""
        import numpy as np
        engine, mock_model = self._make_engine_with_model()
        mock_model.transcribe.return_value = ([MagicMock(text="hello")], MagicMock())

        result = engine.transcribe_with_fallback(np.zeros(16000, dtype=np.float32))
        assert result == "hello"
        # Model should still be the same (no reload)
        assert engine._model is mock_model
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["beam_size"] == 1
        assert kwargs["best_of"] == 1
        assert kwargs["temperature"] == 0.0
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["without_timestamps"] is True

    def test_custom_decode_settings_are_passed_to_model(self):
        """Configurable decode settings should reach faster-whisper."""
        import numpy as np
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(
            model_size="small.en",
            device="cuda",
            beam_size=3,
            best_of=2,
            condition_on_previous_text=True,
        )
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([MagicMock(text="custom")], MagicMock())
        engine._model = mock_model

        result = engine.transcribe(np.zeros(16000, dtype=np.float32))

        assert result == "custom"
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["beam_size"] == 3
        assert kwargs["best_of"] == 2
        assert kwargs["condition_on_previous_text"] is True
        assert kwargs["without_timestamps"] is True

    def test_gpu_error_triggers_cpu_fallback(self):
        """GPU cuBLAS error triggers model reload on CPU and retry."""
        import numpy as np
        import voice_typer.transcription as mod
        mod_obj = sys.modules.get("faster_whisper")

        engine, mock_model = self._make_engine_with_model()
        # First call (GPU) raises CUDA error
        mock_model.transcribe.side_effect = RuntimeError(
            "Library cublas64_12.dll is not found or cannot be loaded"
        )

        # After fallback, load() creates a new CPU model
        cpu_model = MagicMock()
        cpu_model.transcribe.return_value = (
            [MagicMock(text="fallback text")], MagicMock()
        )
        mod_obj.WhisperModel.return_value = cpu_model

        result = engine.transcribe_with_fallback(np.zeros(16000, dtype=np.float32))
        assert result == "fallback text"
        assert engine._device == "cpu"
        assert engine._compute_type == "int8"

    def test_transcribe_words_gpu_error_triggers_cpu_fallback(self):
        """Timestamped streaming transcription should use the same GPU fallback."""
        import numpy as np
        import voice_typer.transcription as mod
        from voice_typer.streaming import WordTiming
        mod_obj = sys.modules.get("faster_whisper")

        engine, mock_model = self._make_engine_with_model()
        mock_model.transcribe.side_effect = RuntimeError(
            "Library cublas64_12.dll is not found or cannot be loaded"
        )

        cpu_model = MagicMock()
        segment = MagicMock()
        segment.words = [MagicMock(word=" fixed", start=0.1, end=0.4)]
        cpu_model.transcribe.return_value = ([segment], MagicMock())
        mod_obj.WhisperModel.return_value = cpu_model

        result = engine.transcribe_words(np.zeros(16000, dtype=np.float32))

        assert result == [WordTiming("fixed", start_seconds=0.1, end_seconds=0.4)]
        assert engine._device == "cpu"
        assert engine._compute_type == "int8"

    def test_non_gpu_error_propagates(self):
        """Non-GPU errors (e.g., ValueError) should NOT trigger fallback."""
        import numpy as np
        engine, mock_model = self._make_engine_with_model()
        mock_model.transcribe.side_effect = ValueError("bad audio format")

        with pytest.raises(ValueError, match="bad audio format"):
            engine.transcribe_with_fallback(np.zeros(16000, dtype=np.float32))

    def test_cpu_model_error_no_fallback(self):
        """If already on CPU and transcription fails, no fallback attempted."""
        import numpy as np
        engine, mock_model = self._make_engine_with_model()
        engine._device = "cpu"  # already on CPU
        mock_model.transcribe.side_effect = RuntimeError("some runtime error")

        with pytest.raises(RuntimeError, match="some runtime error"):
            engine.transcribe_with_fallback(np.zeros(16000, dtype=np.float32))

    def test_empty_audio_returns_empty(self):
        """Empty audio should return empty string without calling model."""
        import numpy as np
        engine, mock_model = self._make_engine_with_model()
        result = engine.transcribe_with_fallback(np.array([], dtype=np.float32))
        assert result == ""
        mock_model.transcribe.assert_not_called()


class TestTranscribeWords:
    def test_transcribe_words_passes_timestamp_options_and_applies_offset(self):
        import numpy as np

        from voice_typer.streaming import WordTiming
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cpu")
        mock_model = MagicMock()
        segment = MagicMock()
        segment.words = [
            MagicMock(word=" hello", start=0.25, end=0.75),
            MagicMock(word="world", start=0.8, end=1.2),
        ]
        mock_model.transcribe.return_value = ([segment], MagicMock())
        engine._model = mock_model

        words = engine.transcribe_words(
            np.zeros(16000, dtype=np.float32),
            offset_seconds=5.0,
        )

        assert words == [
            WordTiming(word="hello", start_seconds=5.25, end_seconds=5.75),
            WordTiming(word="world", start_seconds=5.8, end_seconds=6.2),
        ]
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["word_timestamps"] is True
        assert kwargs["without_timestamps"] is False

    def test_transcribe_words_empty_audio_returns_empty_without_calling_model(self):
        import numpy as np
        from voice_typer.transcription import TranscriptionEngine

        engine = TranscriptionEngine(model_size="small.en", device="cpu")
        mock_model = MagicMock()
        engine._model = mock_model

        assert engine.transcribe_words(np.array([], dtype=np.float32)) == []
        mock_model.transcribe.assert_not_called()

    def test_model_operations_are_guarded_by_engine_lock(self, monkeypatch):
        import numpy as np
        from voice_typer.transcription import TranscriptionEngine

        class TrackingLock:
            def __init__(self):
                self.depth = 0
                self.entries = 0

            def __enter__(self):
                self.depth += 1
                self.entries += 1

            def __exit__(self, exc_type, exc, tb):
                self.depth -= 1

        engine = TranscriptionEngine(model_size="small.en", device="cpu")
        lock = TrackingLock()
        engine._lock = lock

        mock_model = MagicMock()

        def transcribe(*args, **kwargs):
            assert lock.depth == 1
            return ([MagicMock(text="locked")], MagicMock())

        mock_model.transcribe.side_effect = transcribe
        engine._model = mock_model

        assert engine.transcribe_with_fallback(np.zeros(16000, dtype=np.float32)) == "locked"
        engine.unload()

        assert lock.entries >= 2
        assert engine._model is None
