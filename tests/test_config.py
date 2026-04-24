"""Tests for config load/save and field behavior."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from voice_typer.config import Config, _config_dir


class TestConfigDefaults:
    def test_default_values(self):
        c = Config()
        assert c.hotkey == "<f2>"
        assert c.sample_rate == 16000
        assert c.microphone is None
        assert c.model_size == "small.en"
        assert c.language == "en"
        assert c.device == "cuda"
        assert c.beam_size == 1
        assert c.best_of == 1
        assert c.condition_on_previous_text is False
        assert c.streaming_transcription is True
        assert c.streaming_chunk_seconds == 12.0
        assert c.streaming_step_seconds == 5.0
        assert c.streaming_left_overlap_seconds == 2.0
        assert c.streaming_right_guard_seconds == 1.0
        assert c.streaming_min_first_chunk_seconds == 6.0
        assert c.streaming_silence_threshold == 0.003
        assert c.autostart is True
        assert c.paste_on_stop is True
        assert c.show_notifications is True


class TestConfigLoadSave:
    def test_save_creates_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        c = Config(hotkey="<f3>", autostart=True)
        c.save()

        config_file = tmp_path / "config.json"
        assert config_file.exists()

        data = json.loads(config_file.read_text())
        assert data["hotkey"] == "<f3>"
        assert data["autostart"] is True

    def test_load_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        c = Config.load()
        assert c.hotkey == "<f2>"
        assert c.autostart is True

    def test_load_reads_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "hotkey": "<f9>",
            "microphone": "WO Mic",
            "autostart": True,
            "paste_on_stop": False,
            "show_notifications": False,
        }))

        c = Config.load()
        assert c.hotkey == "<f9>"
        assert c.microphone == "WO Mic"
        assert c.autostart is True
        assert c.paste_on_stop is True
        assert c.show_notifications is False

    def test_load_normalizes_legacy_streaming_and_paste_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "streaming_transcription": False,
            "paste_on_stop": False,
            "device": "cpu",
        }))

        c = Config.load()
        assert c.streaming_transcription is True
        assert c.paste_on_stop is True
        assert c.device == "cuda"

    @pytest.mark.parametrize("legacy_model", ["tiny.en", "large-v3", "base.en", "unsupported"])
    def test_load_normalizes_legacy_or_unsupported_model_to_small_en(
        self, tmp_path, monkeypatch, legacy_model
    ):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"model_size": legacy_model}))

        c = Config.load()
        assert c.model_size == "small.en"

    def test_load_keeps_medium_en_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"model_size": "medium.en"}))

        c = Config.load()
        assert c.model_size == "medium.en"

    def test_load_ignores_unknown_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "hotkey": "<f5>",
            "bogus_key": "should be ignored",
        }))

        c = Config.load()
        assert c.hotkey == "<f5>"
        assert not hasattr(c, "bogus_key")

    def test_load_returns_defaults_on_corrupt_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        config_file = tmp_path / "config.json"
        config_file.write_text("NOT VALID JSON {{{")

        c = Config.load()
        assert c.hotkey == "<f2>"  # defaults

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_typer.config._config_dir", lambda: tmp_path)
        c1 = Config(
            hotkey="<f7>",
            microphone="Blue Yeti",
            model_size="medium.en",
            device="cpu",
            beam_size=3,
            best_of=2,
            condition_on_previous_text=True,
            streaming_transcription=True,
            streaming_chunk_seconds=10.0,
            streaming_step_seconds=4.0,
            streaming_left_overlap_seconds=1.5,
            streaming_right_guard_seconds=0.75,
            streaming_min_first_chunk_seconds=5.0,
            streaming_silence_threshold=0.001,
            autostart=True,
            paste_on_stop=False,
            show_notifications=False,
        )
        c1.save()
        c2 = Config.load()
        assert c1.hotkey == c2.hotkey
        assert c1.microphone == c2.microphone
        assert c1.model_size == c2.model_size
        assert c2.device == "cuda"
        assert c1.beam_size == c2.beam_size
        assert c1.best_of == c2.best_of
        assert c1.condition_on_previous_text == c2.condition_on_previous_text
        assert c2.streaming_transcription is True
        assert c1.streaming_chunk_seconds == c2.streaming_chunk_seconds
        assert c1.streaming_step_seconds == c2.streaming_step_seconds
        assert c1.streaming_left_overlap_seconds == c2.streaming_left_overlap_seconds
        assert c1.streaming_right_guard_seconds == c2.streaming_right_guard_seconds
        assert c1.streaming_min_first_chunk_seconds == c2.streaming_min_first_chunk_seconds
        assert c1.streaming_silence_threshold == c2.streaming_silence_threshold
        assert c1.autostart == c2.autostart
        assert c2.paste_on_stop is True
        assert c1.show_notifications == c2.show_notifications
