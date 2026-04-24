"""System tray icon using pystray, with dynamic state and menu.

Threading model:
- ``start()`` creates the icon and launches background work (model loading,
  hotkey registration, etc.) in a daemon thread.  It does NOT block.
- ``run()`` blocks the **main** thread with ``pystray.Icon.run()``.  This is
  required on macOS (NSApplication) and Linux (Gtk.main).  Call it from the
  main thread after ``start()``.
- State updates (icon, title, notifications) from the background thread are
  dispatched safely by pystray on every platform.
- Before ``run()`` starts, state / notification calls are queued and flushed
  once the event loop is live.
"""

import logging
import threading
from enum import Enum
from typing import Optional, Callable

import pystray
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)


class AppState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    LOADING = "loading"
    ERROR = "error"


class TrayIcon:
    """Cross-platform system tray icon with dynamic menu and state indication."""

    def __init__(
        self,
        on_toggle: Callable,
        on_settings: Callable,
        on_quit: Callable,
        on_toggle_autostart: Optional[Callable] = None,
        on_select_mic: Optional[Callable[[Optional[str]], None]] = None,
        config=None,
    ):
        self.on_toggle = on_toggle
        self.on_settings = on_settings
        self.on_quit = on_quit
        self.on_toggle_autostart = on_toggle_autostart
        self.on_select_mic = on_select_mic
        self._config = config  # reference to live Config object

        self._icon: Optional[pystray.Icon] = None
        self._state = AppState.IDLE
        self._message = ""
        self._notifications_enabled = True
        self._microphones: list[dict] = []  # populated by app
        self._autostart_enabled = False

        # Pre-run state queue — flushed once pystray event loop is live
        self._pending_states: list[tuple[AppState, str]] = []
        self._pending_notifications: list[tuple[str, str]] = []
        self._bg_work_fn: Optional[Callable] = None
        self._bg_thread: Optional[threading.Thread] = None

    # ─── Public API ─────────────────────────────────────────────────────

    @property
    def state(self) -> AppState:
        return self._state

    def set_state(self, state: AppState, message: str = ""):
        """Update tray icon state and tooltip.

        If the event loop is not yet running the update is queued and applied
        once ``run()`` starts.
        """
        self._state = state
        self._message = message
        if self._icon:
            self._apply_state(state, message)
        else:
            self._pending_states.append((state, message))

    def set_microphones(self, mics: list[dict]):
        """Update the cached microphone list."""
        self._microphones = mics

    def set_autostart_enabled(self, enabled: bool):
        """Update the cached autostart state."""
        self._autostart_enabled = enabled

    def set_notifications_enabled(self, enabled: bool):
        self._notifications_enabled = enabled

    def start(self, bg_work: Optional[Callable] = None):
        """Create the tray icon and start background work.

        Does **not** block — call ``run()`` afterwards to enter the main loop.
        *bg_work* is called in a daemon thread; it should do model loading,
        hotkey registration, etc. and update state via ``set_state()``.
        """
        self._bg_work_fn = bg_work

        # Build menu — always wrap in pystray.Menu to prevent
        # "argument after * must be an iterable" TypeError
        menu = pystray.Menu(self._build_menu)

        try:
            self._icon = pystray.Icon(
                name="voice-typer",
                icon=_make_icon(AppState.IDLE),
                title="Voice Typer",
                menu=menu,
            )
        except TypeError as e:
            raise RuntimeError(
                f"Failed to create tray icon (pystray Menu construction error): {e}"
            ) from e

        # Start background work immediately so it runs in parallel with the
        # (not-yet-started) event loop.
        if self._bg_work_fn:
            self._bg_thread = threading.Thread(target=self._bg_work_fn, daemon=True)
            self._bg_thread.start()

        log.info("Tray icon created, background work started")

    def run(self):
        """Block the main thread with pystray's event loop.

        MUST be called from the main thread on macOS / Linux.
        Flushes any pending state / notifications that were queued before
        the event loop was live.
        """
        if self._icon is None:
            raise RuntimeError("call start() before run()")

        # Flush queued state while the icon exists
        for state, msg in self._pending_states:
            self._apply_state(state, msg)
        self._pending_states.clear()

        for title, message in self._pending_notifications:
            self._do_notify(title, message)
        self._pending_notifications.clear()

        log.info("Tray event loop starting (main thread)")
        self._icon.run()  # blocks until stop() is called

    def stop(self):
        """Stop the tray icon and exit the event loop."""
        if self._icon:
            self._icon.stop()
            self._icon = None
        log.info("Tray icon stopped")

    def notify(self, title: str, message: str):
        """Show a notification if notifications are enabled.

        If the event loop is not yet running the notification is queued.
        """
        if not self._notifications_enabled:
            return
        if self._icon:
            self._do_notify(title, message)
        else:
            self._pending_notifications.append((title, message))

    # ─── Internals ──────────────────────────────────────────────────────

    def _apply_state(self, state: AppState, message: str):
        """Apply state to the live icon (safe from any thread)."""
        if not self._icon:
            return
        self._icon.icon = _make_icon(state)
        title = "Voice Typer"
        if message:
            title += f" — {message}"
        elif state != AppState.IDLE:
            title += f" — {state.value}"
        self._icon.title = title

    def _do_notify(self, title: str, message: str):
        """Send a notification through the icon."""
        try:
            self._icon.notify(message, title)
        except Exception as e:
            log.warning("Notification failed: %s", e)

    def _build_menu(self):
        """Build the tray menu dynamically on each right-click."""
        items = []
        hotkey = self._display_hotkey()

        # Toggle dictation
        items.append(
            pystray.MenuItem(
                f"Toggle Dictation ({hotkey})",
                self._wrap(self.on_toggle),
                default=True,
            )
        )

        items.append(pystray.Menu.SEPARATOR)

        items.append(
            pystray.MenuItem(
                f"Hotkey: {hotkey}",
                None,
                enabled=False,
            )
        )

        # Microphone submenu
        if self.on_select_mic and self._microphones:
            mic_items = self._build_mic_menu_items()
            items.append(
                pystray.MenuItem(
                    "Microphone",
                    pystray.Menu(*mic_items),
                )
            )

        items.append(pystray.Menu.SEPARATOR)

        # Settings
        items.append(
            pystray.MenuItem("Settings...", self._wrap(self.on_settings))
        )

        items.append(pystray.Menu.SEPARATOR)

        # Quit
        items.append(pystray.MenuItem("Quit", self._wrap(self.on_quit)))

        return tuple(items)

    def _display_hotkey(self) -> str:
        """Return the configured hotkey in a user-facing form."""
        hotkey = getattr(self._config, "hotkey", "<f2>") or "<f2>"
        return hotkey.strip("<>").upper()

    def _build_mic_menu_items(self):
        """Build microphone radio items with duplicate-name disambiguation."""
        mic_items = []
        current = self._config.microphone if self._config else None

        # "System Default" option
        mic_items.append(
            pystray.MenuItem(
                "System Default",
                self._wrap(lambda: self.on_select_mic(None)),
                checked=lambda item: current is None,
                radio=True,
            )
        )

        # Detect duplicate names for disambiguation
        names_seen: dict[str, int] = {}
        for mic in self._microphones:
            n = mic["name"]
            names_seen[n] = names_seen.get(n, 0) + 1
        has_duplicates = any(c > 1 for c in names_seen.values())

        for mic in self._microphones:
            name = mic["name"]
            mic_id = mic["id"]
            display = name

            # Disambiguate duplicate names with host API
            if has_duplicates and names_seen.get(name, 0) > 1:
                host_api = mic.get("host_api", "")
                if host_api:
                    display = f"{name} ({host_api})"

            if len(display) > 50:
                display = display[:47] + "..."

            mic_items.append(
                pystray.MenuItem(
                    display,
                    self._wrap(lambda mid=mic_id: self.on_select_mic(mid)),
                    checked=lambda item, mid=mic_id: current == mid,
                    radio=True,
                )
            )

        return mic_items

    @staticmethod
    def _wrap(fn):
        """Wrap callback so pystray doesn't break on extra args."""
        def wrapper(icon, item):
            fn()
        return wrapper


def _make_icon(state: AppState, size: int = 64) -> Image.Image:
    """Generate a colored microphone icon based on state."""
    colors = {
        AppState.IDLE: (120, 120, 120, 255),
        AppState.RECORDING: (235, 64, 52, 255),
        AppState.TRANSCRIBING: (52, 152, 219, 255),
        AppState.LOADING: (243, 156, 18, 255),  # yellow/orange
        AppState.ERROR: (231, 76, 60, 255),
    }
    color = colors.get(state, (120, 120, 120, 255))

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2

    # Microphone body (rounded rect)
    mic_w, mic_h = size // 5, size // 3
    draw.rounded_rectangle(
        [cx - mic_w, cy - mic_h, cx + mic_w, cy + mic_h // 3],
        radius=mic_w // 2,
        fill=color,
    )

    # Stand arc
    stand_radius = size // 3
    draw.arc(
        [cx - stand_radius, cy - stand_radius + mic_h // 4, cx + stand_radius, cy + stand_radius],
        start=0, end=180,
        fill=color, width=max(2, size // 20),
    )

    # Base line
    base_y = cy + stand_radius
    draw.line(
        [cx - stand_radius // 2, base_y, cx + stand_radius // 2, base_y],
        fill=color, width=max(2, size // 20),
    )

    return img
