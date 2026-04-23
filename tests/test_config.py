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
        assert c.device == "auto"
        assert c.beam_size == 1
        assert c.best_of == 1
        assert c.condition_on_previous_text is False
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
        assert c.paste_on_stop is False
        assert c.show_notifications is False

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
            autostart=True,
            paste_on_stop=False,
            show_notifications=False,
        )
        c1.save()
        c2 = Config.load()
        assert c1.hotkey == c2.hotkey
        assert c1.microphone == c2.microphone
        assert c1.model_size == c2.model_size
        assert c1.device == c2.device
        assert c1.beam_size == c2.beam_size
        assert c1.best_of == c2.best_of
        assert c1.condition_on_previous_text == c2.condition_on_previous_text
        assert c1.autostart == c2.autostart
        assert c1.paste_on_stop == c2.paste_on_stop
        assert c1.show_notifications == c2.show_notifications
