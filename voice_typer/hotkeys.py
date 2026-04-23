"""Hotkey backend abstraction.

Provides platform-aware hotkey listening with two implementations:

- PynputHotkey: Uses pynput.keyboard.GlobalHotKeys (cross-platform).
- WindowsNativeHotkey: Uses Win32 RegisterHotKey via ctypes (Windows only).

The factory function ``create_hotkey_backend`` picks the best available backend.

All backends share a common interface:
    - start(callback) -> None
    - stop() -> None
    - is_alive() -> bool
    - diagnose() -> str
"""

import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

log = logging.getLogger("voice_typer")


# ─── Base class ──────────────────────────────────────────────────────────────


class HotkeyBackend(ABC):
    """Abstract base for hotkey backends."""

    @abstractmethod
    def start(self, callback: Callable[[], None]) -> None:
        """Start listening for the hotkey. Calls *callback* when pressed."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release resources."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return True if the listener thread is running."""

    @abstractmethod
    def diagnose(self) -> str:
        """Return a human-readable diagnostic string."""


# ─── Pynput backend ──────────────────────────────────────────────────────────


class PynputHotkey(HotkeyBackend):
    """Hotkey backend using pynput.keyboard.GlobalHotKeys.

    Falls back to a regular ``Listener`` with manual key matching if
    ``GlobalHotKeys`` fails (common on some Windows / WSL setups).
    """

    def __init__(self, hotkey_str: str):
        self.hotkey_str = hotkey_str
        self._listener = None
        self._fallback = False

    def start(self, callback: Callable[[], None]) -> None:
        from pynput.keyboard import GlobalHotKeys, Listener, Key, KeyCode

        log.info(
            "Registering hotkey via pynput: %r -> callback", self.hotkey_str
        )

        try:
            self._listener = GlobalHotKeys(
                {self.hotkey_str: callback}
            )
            self._listener.start()
            time.sleep(0.5)
            alive = self._listener.is_alive()
            log.info(
                "Pynput GlobalHotKeys started (alive=%s, daemon=%s)",
                alive,
                getattr(self._listener, "daemon", "?"),
            )
            if not alive:
                log.error(
                    "GlobalHotKeys thread died immediately; "
                    "falling back to manual Listener"
                )
                self._stop_listener()
                self._start_fallback(callback, Listener, Key, KeyCode)
        except Exception:
            log.exception("GlobalHotKeys failed; trying fallback Listener")
            try:
                self._start_fallback(callback, Listener, Key, KeyCode)
            except Exception:
                log.exception("Fallback Listener also failed")

    # --- internal helpers ---------------------------------------------------

    def _start_fallback(self, callback, Listener, Key, KeyCode) -> None:
        target = _parse_hotkey_to_pynput(self.hotkey_str, Key, KeyCode)
        if target is None:
            raise RuntimeError(
                f"Cannot parse hotkey {self.hotkey_str!r} for fallback"
            )

        def on_press(key):
            if key == target:
                log.info("[HOTKEY FALLBACK] Matched key: %s", key)
                callback()

        self._listener = Listener(on_press=on_press)
        self._listener.start()
        time.sleep(0.5)
        self._fallback = True
        log.info(
            "Fallback listener started, watching for %s (alive=%s)",
            target,
            self._listener.is_alive(),
        )

    def _stop_listener(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    # --- public interface ---------------------------------------------------

    def stop(self) -> None:
        if self._listener is not None:
            log.info("Stopping pynput hotkey listener")
            self._stop_listener()

    def is_alive(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def diagnose(self) -> str:
        if self._listener is None:
            return "PynputHotkey: no listener registered"
        alive = self._listener.is_alive()
        daemon = getattr(self._listener, "daemon", "?")
        name = getattr(self._listener, "name", "?")
        mode = "fallback" if self._fallback else "GlobalHotKeys"
        return (
            f"PynputHotkey ({mode})\n"
            f"Hotkey: {self.hotkey_str}\n"
            f"Thread name: {name}\n"
            f"Thread alive: {alive}\n"
            f"Thread daemon: {daemon}"
        )


def _parse_hotkey_to_pynput(hotkey_str, Key, KeyCode):
    """Parse '<f2>' -> pynput Key or KeyCode for fallback matching."""
    clean = hotkey_str.strip("<>").lower()
    if hasattr(Key, clean):
        return getattr(Key, clean)
    if clean.startswith("f") and clean[1:].isdigit():
        fnum = int(clean[1:])
        if 1 <= fnum <= 24:
            return KeyCode.from_vk(0x6F + fnum)
    if len(clean) == 1:
        return KeyCode.from_char(clean)
    return None


# ─── Windows native backend ──────────────────────────────────────────────────

# Win32 constants
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_MOD_NOREPEAT = 0x4000
_HWND_MESSAGE = -3  # HWND_MESSAGE: creates a message-only window
_GWLP_USERDATA = -21

# Common virtual-key code mappings for function keys and printable keys.
_VK_MAP = {}


def _win32_vk(vk_name: str) -> Optional[int]:
    """Look up a VK code by name, initializing the map lazily."""
    _init_vk_map()
    return _VK_MAP.get(vk_name)


def _init_vk_map():
    """Populate _VK_MAP lazily to avoid issues at import on non-Windows."""
    if _VK_MAP:
        return
    # F1-F24
    for i in range(1, 25):
        _VK_MAP[f"f{i}"] = 0x70 + (i - 1)  # F1=0x70, F2=0x71, ...
    # Digits 0-9
    for c in "0123456789":
        _VK_MAP[c] = ord(c)
    # Letters a-z
    for c in "abcdefghijklmnopqrstuvwxyz":
        _VK_MAP[c] = ord(c.upper())


def parse_hotkey_to_vk(hotkey_str: str) -> Optional[int]:
    """Convert a hotkey string like '<f2>' to a Win32 virtual-key code.

    Returns None if the key cannot be parsed.
    """
    _init_vk_map()
    clean = hotkey_str.strip("<>").lower()
    return _VK_MAP.get(clean)


class WindowsNativeHotkey(HotkeyBackend):
    """Hotkey backend using Win32 RegisterHotKey via ctypes.

    Creates a hidden message-only window (HWND_MESSAGE) and runs a message
    loop in a daemon thread.  Falls back to GetAsyncKeyState polling if
    RegisterHotKey succeeds but WM_HOTKEY is never delivered (can happen
    on some Windows configurations with HWND_MESSAGE windows).
    """

    def __init__(self, hotkey_str: str):
        self.hotkey_str = hotkey_str
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # signalled when registration completes
        self._hotkey_id = 1  # arbitrary ID for RegisterHotKey
        self._registered = False
        self._user32 = None
        self._kernel32 = None
        self._success = False  # True only when both CreateWindowExW AND RegisterHotKey succeed
        self._vk: Optional[int] = None
        self._hwnd = None  # message-only window handle
        self._using_polling = False  # True if falling back to GetAsyncKeyState

    def start(self, callback: Callable[[], None]) -> None:
        import ctypes
        import ctypes.wintypes

        self._vk = _win32_vk(self.hotkey_str.strip("<>").lower())
        if self._vk is None:
            raise ValueError(
                f"Cannot parse hotkey {self.hotkey_str!r} to a VK code"
            )

        self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self._kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        self._stop_event.clear()
        self._ready_event.clear()
        self._success = False
        self._last_error = None  # captured GetLastError() on failure

        # ── Set proper argtypes BEFORE any Win32 call ──
        # Without these, ctypes defaults to c_int which truncates 64-bit pointers.
        from ctypes.wintypes import (
            BOOL, DWORD, HINSTANCE, HMENU, HWND, INT, LPCWSTR, LPARAM, LPMSG,
            MSG, UINT, WPARAM,
        )

        # HWND CreateWindowExW(...)
        self._user32.CreateWindowExW.argtypes = [
            DWORD, LPCWSTR, LPCWSTR, DWORD,
            INT, INT, INT, INT,
            HWND, HMENU, HINSTANCE, ctypes.c_void_p,
        ]
        self._user32.CreateWindowExW.restype = HWND

        # BOOL RegisterHotKey(HWND, int, UINT, UINT)
        self._user32.RegisterHotKey.argtypes = [HWND, INT, UINT, UINT]
        self._user32.RegisterHotKey.restype = BOOL

        # BOOL UnregisterHotKey(HWND, int)
        self._user32.UnregisterHotKey.argtypes = [HWND, INT]
        self._user32.UnregisterHotKey.restype = BOOL

        # BOOL GetMessageW(LPMSG, HWND, UINT, UINT)
        self._user32.GetMessageW.argtypes = [LPMSG, HWND, UINT, UINT]
        self._user32.GetMessageW.restype = BOOL  # actually int, but BOOL works for 0/-1 check

        # BOOL PostMessageW(HWND, UINT, WPARAM, LPARAM)
        self._user32.PostMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
        self._user32.PostMessageW.restype = BOOL

        # BOOL PostThreadMessageW(DWORD threadId, UINT msg, WPARAM, LPARAM)
        self._user32.PostThreadMessageW.argtypes = [DWORD, UINT, WPARAM, LPARAM]
        self._user32.PostThreadMessageW.restype = BOOL

        # BOOL TranslateMessage(const MSG*)
        self._user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
        self._user32.TranslateMessage.restype = BOOL

        # LRESULT DispatchMessageW(const MSG*)
        self._user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
        self._user32.DispatchMessageW.restype = LPARAM

        # BOOL DestroyWindow(HWND)
        self._user32.DestroyWindow.argtypes = [HWND]
        self._user32.DestroyWindow.restype = BOOL

        # DWORD GetLastError(void)
        self._kernel32.GetLastError.argtypes = []
        self._kernel32.GetLastError.restype = DWORD

        # HANDLE GetModuleHandleW(LPCWSTR)
        self._kernel32.GetModuleHandleW.argtypes = [LPCWSTR]
        self._kernel32.GetModuleHandleW.restype = HINSTANCE

        def run():
            """Message loop thread: creates HWND_MESSAGE, registers hotkey, runs GetMessage."""
            try:
                # Create a message-only window to receive WM_HOTKEY
                self._hwnd = self._user32.CreateWindowExW(
                    0,                     # dwExStyle
                    "STATIC",              # lpClassName (simple class)
                    None,                  # lpWindowName
                    0,                     # dwStyle
                    0, 0, 0, 0,            # x, y, width, height
                    _HWND_MESSAGE,         # hWndParent = HWND_MESSAGE
                    None,                  # hMenu
                    self._kernel32.GetModuleHandleW(None),  # hInstance
                    None,                  # lpParam
                )
                if not self._hwnd:
                    err = self._kernel32.GetLastError()
                    self._last_error = err
                    log.error(
                        "CreateWindowExW failed (hwnd is NULL), GetLastError=%d (0x%X)",
                        err, err,
                    )
                    self._ready_event.set()
                    return
                log.info("Created message-only window hwnd=0x%X", self._hwnd)

                # Register the hotkey.  IMPORTANT: pass NULL (0) as hWnd.
                # RegisterHotKey(NULL, ...) binds the hotkey to the calling
                # thread so WM_HOTKEY is posted to the thread message queue.
                # Using an HWND (even HWND_MESSAGE) causes GetMessageW to
                # silently miss the message on some Windows configurations.
                result = self._user32.RegisterHotKey(
                    0, self._hotkey_id, _MOD_NOREPEAT, self._vk
                )
                if not result:
                    err = self._kernel32.GetLastError()
                    self._last_error = err
                    log.error(
                        "RegisterHotKey failed for VK=0x%X, GetLastError=%d (0x%X)",
                        self._vk,
                        err, err,
                    )
                    self._ready_event.set()
                    return
                self._registered = True
                self._success = True
                log.info(
                    "RegisterHotKey succeeded: hotkey=%s vk=0x%X id=%d",
                    self.hotkey_str,
                    self._vk,
                    self._hotkey_id,
                )
                self._ready_event.set()

                # ── Hotkey detection ──
                # Use GetAsyncKeyState polling for reliable hotkey detection.
                # RegisterHotKey + GetMessageW does not reliably deliver WM_HOTKEY
                # on all Windows configurations (especially with HWND_MESSAGE windows).
                # Polling at 30Hz uses negligible CPU and works universally.
                log.info("Starting hotkey detection via GetAsyncKeyState polling")
                self._using_polling = True
                self._run_polling_loop(callback)

            except Exception:
                log.exception("Windows hotkey thread error")
            finally:
                # Cleanup
                if self._registered:
                    self._user32.UnregisterHotKey(0, self._hotkey_id)
                    self._registered = False
                    log.info("UnregisterHotKey done")
                if self._hwnd:
                    self._user32.DestroyWindow(self._hwnd)
                    self._hwnd = None

        # Also set GetAsyncKeyState argtypes for the polling fallback
        self._user32.GetAsyncKeyState.argtypes = [INT]
        self._user32.GetAsyncKeyState.restype = ctypes.c_short

        # Set Sleep argtypes
        self._kernel32.Sleep.argtypes = [DWORD]
        self._kernel32.Sleep.restype = None

        self._thread = threading.Thread(target=run, daemon=True, name="WinHotkey")
        self._thread.start()

        # Wait for the registration thread to signal readiness (or timeout)
        if not self._ready_event.wait(timeout=5.0):
            self._last_error = -1
            raise RuntimeError(
                f"Timed out waiting for hotkey registration of {self.hotkey_str!r}"
            )
        if not self._success:
            err = self._last_error
            raise RuntimeError(
                f"Failed to register hotkey {self.hotkey_str!r} "
                f"(Win32 error {err}, 0x{(err if err and err >= 0 else 0):X})"
            )

    def _run_polling_loop(self, callback):
        """GetAsyncKeyState polling fallback for hotkey detection.

        Polls the key state every 33ms (~30Hz).  Detects key-down
        transitions by checking the high bit of GetAsyncKeyState.
        """
        import ctypes
        vk = self._vk
        was_pressed = False
        log.info("Polling loop started for VK=0x%X", vk)
        while not self._stop_event.is_set():
            state = self._user32.GetAsyncKeyState(vk)
            is_pressed = bool(state & 0x8000)  # high bit = currently down
            if is_pressed and not was_pressed:
                log.info("[HOTKEY FIRED] GetAsyncKeyState detected key-down")
                callback()
            was_pressed = is_pressed
            self._kernel32.Sleep(33)  # ~30Hz polling rate

    def stop(self) -> None:
        log.info("Stopping Windows native hotkey listener")
        self._stop_event.set()
        if self._user32 is not None and self._thread is not None:
            thread_id = self._thread.ident
            if thread_id is not None:
                self._user32.PostThreadMessageW(thread_id, _WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def diagnose(self) -> str:
        if self._thread is None:
            return "WindowsNativeHotkey: no thread started"
        mode = "polling" if self._using_polling else "message-loop"
        return (
            "WindowsNativeHotkey\n"
            f"Hotkey: {self.hotkey_str}\n"
            f"VK: 0x{self._vk:X} ({self._vk})\n"
            f"Mode: {mode}\n"
            f"Thread name: {self._thread.name}\n"
            f"Thread alive: {self._thread.is_alive()}\n"
            f"Registered: {self._registered}"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────


def create_hotkey_backend(hotkey_str: str) -> HotkeyBackend:
    """Create the best hotkey backend for the current platform.

    - On Windows: returns ``WindowsNativeHotkey``.
    - Elsewhere: returns ``PynputHotkey``.
    """
    if sys.platform == "win32":
        log.info("Platform is win32 -> using WindowsNativeHotkey")
        return WindowsNativeHotkey(hotkey_str)

    log.info("Platform is %s -> using PynputHotkey", sys.platform)
    return PynputHotkey(hotkey_str)
