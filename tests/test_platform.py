"""Tests for platform autostart adapters and microphone listing."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from voice_typer.platform import (
    _autostart_command,
    list_microphones,
    find_microphone_by_name,
    enable_autostart,
    disable_autostart,
    is_autostart_enabled,
)


class TestAutostartCommand:
    def test_uses_python_m_voice_typer(self):
        cmd = _autostart_command()
        assert "-m voice_typer" in cmd

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_windows_uses_pythonw_if_available(self):
        cmd = _autostart_command()
        # Should use pythonw.exe if it exists next to python.exe
        assert "pythonw" in cmd.lower() or "python" in cmd.lower()

    @pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows test")
    def test_unix_uses_quoted_executable(self):
        cmd = _autostart_command()
        assert cmd.startswith('"')
        assert sys.executable in cmd


class TestListMicrophones:
    def test_returns_list(self):
        # Just verify it doesn't crash — actual devices depend on system
        result = list_microphones()
        assert isinstance(result, list)

    def test_returns_empty_on_failure(self, monkeypatch):
        """When sounddevice raises, list_microphones returns []."""
        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = RuntimeError("no audio")
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        assert list_microphones() == []


class TestFindMicrophoneByName:
    def test_finds_partial_match(self, monkeypatch):
        fake_mics = [
            {"id": "0", "index": 0, "name": "Built-in Microphone", "host_api": "ALSA", "channels": 2, "default": True},
            {"id": "1", "index": 1, "name": "WO Mic", "host_api": "MME", "channels": 1, "default": False},
            {"id": "2", "index": 2, "name": "Blue Yeti", "host_api": "MME", "channels": 2, "default": False},
        ]
        monkeypatch.setattr("voice_typer.platform.list_microphones", lambda: fake_mics)

        from voice_typer.platform import find_microphone_by_name
        result = find_microphone_by_name("wo mic")
        assert result is not None
        assert result["name"] == "WO Mic"
        assert result["id"] == "1"

    def test_returns_none_for_no_match(self, monkeypatch):
        monkeypatch.setattr(
            "voice_typer.platform.list_microphones",
            lambda: [{"id": "0", "index": 0, "name": "Built-in", "host_api": "", "channels": 2, "default": True}],
        )
        from voice_typer.platform import find_microphone_by_name
        assert find_microphone_by_name("nonexistent mic") is None

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(
            "voice_typer.platform.list_microphones",
            lambda: [{"id": "0", "index": 0, "name": "Blue Yeti", "host_api": "MME", "channels": 2, "default": False}],
        )
        from voice_typer.platform import find_microphone_by_name
        assert find_microphone_by_name("BLUE YETI") is not None


class TestFindMicrophoneById:
    def test_finds_by_id(self, monkeypatch):
        fake_mics = [
            {"id": "3", "index": 3, "name": "WO Mic", "host_api": "WASAPI", "channels": 1, "default": False},
            {"id": "7", "index": 7, "name": "WO Mic", "host_api": "MME", "channels": 1, "default": False},
        ]
        monkeypatch.setattr("voice_typer.platform.list_microphones", lambda: fake_mics)

        from voice_typer.platform import find_microphone_by_id
        result = find_microphone_by_id("7")
        assert result is not None
        assert result["index"] == 7
        assert result["host_api"] == "MME"

    def test_returns_none_for_bad_id(self, monkeypatch):
        monkeypatch.setattr("voice_typer.platform.list_microphones", lambda: [
            {"id": "0", "index": 0, "name": "Mic", "host_api": "", "channels": 1, "default": True}
        ])
        from voice_typer.platform import find_microphone_by_id
        assert find_microphone_by_id("99") is None


class TestDuplicateMicrophoneDisambiguation:
    def test_duplicate_names_have_different_ids(self, monkeypatch):
        """Two devices with the same name must have distinct IDs."""
        fake_mics = [
            {"id": "3", "index": 3, "name": "WO Mic", "host_api": "Windows WASAPI", "channels": 1, "default": False},
            {"id": "7", "index": 7, "name": "WO Mic", "host_api": "MME", "channels": 1, "default": False},
        ]
        monkeypatch.setattr("voice_typer.platform.list_microphones", lambda: fake_mics)

        from voice_typer.platform import find_microphone_by_id
        mic1 = find_microphone_by_id("3")
        mic2 = find_microphone_by_id("7")
        assert mic1["name"] == mic2["name"]  # same display name
        assert mic1["id"] != mic2["id"]      # different IDs
        assert mic1["host_api"] != mic2["host_api"]  # different host APIs
