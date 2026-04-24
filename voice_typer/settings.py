"""Settings helpers, controller, and native tkinter window."""

from __future__ import annotations

import re
from typing import Callable, Optional


ALLOWED_MODELS = ("small.en", "medium.en")

_FUNCTION_KEY_RE = re.compile(r"^F([1-9]|1[0-2])$")
_HOTKEY_RE = re.compile(r"^<f([1-9]|1[0-2])>$", re.IGNORECASE)


def display_hotkey(value: str) -> str:
    """Return a user-facing label for a stored function-key hotkey."""
    match = _HOTKEY_RE.match(value.strip())
    if not match:
        return value
    return f"F{match.group(1)}"


def format_function_hotkey(key_name: str) -> str:
    """Return pynput-style hotkey text for F1-F12."""
    key = key_name.strip().upper()
    match = _FUNCTION_KEY_RE.match(key)
    if not match:
        raise ValueError(f"Unsupported hotkey: {key_name}")
    return f"<f{match.group(1)}>"


class SettingsController:
    """Apply settings changes to Config and notify app callbacks."""

    def __init__(
        self,
        config,
        on_hotkey_changed: Optional[Callable[[str], None]] = None,
        on_model_changed: Optional[Callable[[str], None]] = None,
        on_microphone_changed: Optional[Callable[[Optional[str]], None]] = None,
        on_autostart_changed: Optional[Callable[[bool], None]] = None,
        on_notifications_changed: Optional[Callable[[bool], None]] = None,
    ):
        self.config = config
        self.on_hotkey_changed = on_hotkey_changed
        self.on_model_changed = on_model_changed
        self.on_microphone_changed = on_microphone_changed
        self.on_autostart_changed = on_autostart_changed
        self.on_notifications_changed = on_notifications_changed

    def apply(
        self,
        *,
        hotkey: str,
        model_size: str,
        microphone: Optional[str],
        autostart: bool,
        show_notifications: bool,
    ):
        if model_size not in ALLOWED_MODELS:
            raise ValueError(f"Unsupported model: {model_size}")
        if not _HOTKEY_RE.match(hotkey.strip()):
            raise ValueError(f"Unsupported hotkey: {hotkey}")

        hotkey = hotkey.strip().lower()
        changes = {
            "hotkey": self.config.hotkey != hotkey,
            "model_size": self.config.model_size != model_size,
            "microphone": self.config.microphone != microphone,
            "autostart": self.config.autostart is not bool(autostart),
            "show_notifications": (
                self.config.show_notifications is not bool(show_notifications)
            ),
        }

        self.config.hotkey = hotkey
        self.config.model_size = model_size
        self.config.microphone = microphone
        self.config.autostart = bool(autostart)
        self.config.show_notifications = bool(show_notifications)
        self.config.paste_on_stop = True
        self.config.streaming_transcription = True
        self.config.save()

        if changes["hotkey"] and self.on_hotkey_changed:
            self.on_hotkey_changed(hotkey)
        if changes["model_size"] and self.on_model_changed:
            self.on_model_changed(model_size)
        if changes["microphone"] and self.on_microphone_changed:
            self.on_microphone_changed(microphone)
        if changes["autostart"] and self.on_autostart_changed:
            self.on_autostart_changed(bool(autostart))
        if changes["show_notifications"] and self.on_notifications_changed:
            self.on_notifications_changed(bool(show_notifications))


class SettingsWindow:
    """Small native settings window.

    The window is intentionally thin: widgets collect values and delegate all
    validation and side effects to SettingsController.
    """

    def __init__(self, controller: SettingsController, microphones=None, parent=None):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.controller = controller
        self.microphones = microphones or []
        self._messagebox = messagebox

        self.root = tk.Toplevel(parent) if parent is not None else tk.Tk()
        self.root.title("Voice Typer Settings")
        self.root.resizable(False, False)

        self.hotkey_var = tk.StringVar(
            self.root, value=display_hotkey(controller.config.hotkey)
        )
        self.model_var = tk.StringVar(self.root, value=controller.config.model_size)
        self.microphone_var = tk.StringVar(
            self.root, value=self._microphone_label(controller.config.microphone)
        )
        self.autostart_var = tk.BooleanVar(
            self.root, value=bool(controller.config.autostart)
        )
        self.notifications_var = tk.BooleanVar(
            self.root, value=bool(controller.config.show_notifications)
        )

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Hotkey").grid(row=0, column=0, sticky="w", pady=4)
        hotkeys = [f"F{number}" for number in range(1, 13)]
        ttk.Combobox(
            frame,
            textvariable=self.hotkey_var,
            values=hotkeys,
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Model").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.model_var,
            values=ALLOWED_MODELS,
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Microphone").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.microphone_var,
            values=self._microphone_labels(),
            state="readonly",
            width=30,
        ).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(
            frame,
            text="Start on login",
            variable=self.autostart_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Checkbutton(
            frame,
            text="Show notifications",
            variable=self.notifications_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=self.root.destroy).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(buttons, text="Save", command=self._save).grid(row=0, column=1)

    def show(self):
        self.root.deiconify()
        self.root.lift()
        return self.root

    def _microphone_labels(self):
        return ["System Default"] + [mic["name"] for mic in self.microphones]

    def _microphone_label(self, microphone):
        if microphone is None:
            return "System Default"
        for mic in self.microphones:
            if mic.get("id") == microphone:
                return mic.get("name", microphone)
        return microphone

    def _microphone_id(self):
        selected = self.microphone_var.get()
        if selected == "System Default":
            return None
        for mic in self.microphones:
            if mic.get("name") == selected:
                return mic.get("id")
        return selected

    def _save(self):
        try:
            self.controller.apply(
                hotkey=format_function_hotkey(self.hotkey_var.get()),
                model_size=self.model_var.get(),
                microphone=self._microphone_id(),
                autostart=self.autostart_var.get(),
                show_notifications=self.notifications_var.get(),
            )
        except ValueError as exc:
            self._messagebox.showerror("Voice Typer Settings", str(exc))
            return
        self.root.destroy()
