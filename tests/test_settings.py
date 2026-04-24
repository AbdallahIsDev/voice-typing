"""Headless tests for settings helpers and controller behavior."""

from unittest.mock import MagicMock

import pytest

from voice_typer.config import Config
from voice_typer.settings import (
    ALLOWED_MODELS,
    SettingsController,
    display_hotkey,
    format_function_hotkey,
)


def test_allowed_models_are_small_and_medium_english():
    assert ALLOWED_MODELS == ("small.en", "medium.en")


@pytest.mark.parametrize(
    ("raw", "display"),
    [
        ("<f2>", "F2"),
        ("<F12>", "F12"),
    ],
)
def test_display_hotkey_formats_function_keys(raw, display):
    assert display_hotkey(raw) == display


@pytest.mark.parametrize("number", range(1, 13))
def test_format_function_hotkey_accepts_f1_through_f12(number):
    assert format_function_hotkey(f"F{number}") == f"<f{number}>"


@pytest.mark.parametrize("value", ["F0", "F13", "A", "<f2>", "", "Ctrl+F2"])
def test_format_function_hotkey_rejects_unsupported_values(value):
    with pytest.raises(ValueError):
        format_function_hotkey(value)


def test_settings_controller_applies_config_values_and_callbacks():
    config = Config(
        hotkey="<f2>",
        model_size="small.en",
        microphone=None,
        autostart=False,
        show_notifications=True,
        paste_on_stop=False,
        streaming_transcription=False,
    )
    config.save = MagicMock()
    callbacks = {
        "hotkey": MagicMock(),
        "model": MagicMock(),
        "microphone": MagicMock(),
        "autostart": MagicMock(),
        "notifications": MagicMock(),
    }

    controller = SettingsController(
        config,
        on_hotkey_changed=callbacks["hotkey"],
        on_model_changed=callbacks["model"],
        on_microphone_changed=callbacks["microphone"],
        on_autostart_changed=callbacks["autostart"],
        on_notifications_changed=callbacks["notifications"],
    )

    controller.apply(
        hotkey="<f3>",
        model_size="medium.en",
        microphone="mic-1",
        autostart=True,
        show_notifications=False,
    )

    assert config.hotkey == "<f3>"
    assert config.model_size == "medium.en"
    assert config.microphone == "mic-1"
    assert config.autostart is True
    assert config.show_notifications is False
    assert config.paste_on_stop is True
    assert config.streaming_transcription is True
    config.save.assert_called_once_with()
    callbacks["hotkey"].assert_called_once_with("<f3>")
    callbacks["model"].assert_called_once_with("medium.en")
    callbacks["microphone"].assert_called_once_with("mic-1")
    callbacks["autostart"].assert_called_once_with(True)
    callbacks["notifications"].assert_called_once_with(False)


def test_settings_controller_rejects_invalid_model():
    controller = SettingsController(Config())

    with pytest.raises(ValueError):
        controller.apply(
            hotkey="<f2>",
            model_size="tiny.en",
            microphone=None,
            autostart=True,
            show_notifications=True,
        )
