"""Platform-aware text input focus detection.

On Windows, uses multiple heuristics to detect whether the focused
control accepts text input:

1. **Class-name matching** — checks the focused window's class name
   against an expanded set of known text-input class names.
2. **Caret detection** — if Windows reports a visible caret (blinking
   cursor), that is a strong signal the window is a text input.
3. **Ancestor walking** — for Chromium/Electron apps, the focused
   hwnd is a low-level renderer child; the parent window's class name
   reveals the app (Chrome_WidgetWin_1, etc.).
4. **Process-name detection** — some apps (notably Windows Terminal)
   use a completely generic window class name ("Window Class") that
   cannot be matched by class name alone.  Checking the owning
   process name (e.g. WindowsTerminal.exe) resolves this.

On macOS and Linux there is no reliable way to do this without
elevated permissions, so detection returns None (unknown).
"""

import ctypes
import logging
import sys

log = logging.getLogger(__name__)


def is_text_input_focused() -> bool | None:
    """Return True if a text input is focused, False if not, None if unknown."""
    if sys.platform == "win32":
        return _windows_is_text_focused()
    # macOS / Linux — no reliable detection available
    return None


# ─── Windows ───────────────────────────────────────────────────────────

# Editable control class names — checked via case-insensitive substring
# match against GetClassNameW output.
#
# Expanded to cover modern frameworks:
#   - Chrome / Edge / Chromium-Electron apps
#   - Windows Terminal / ConEmu / ConsoleWindowClass (cmd/powershell)
#   - Visual Studio, VS Code, IntelliJ
#   - Windows UI frameworks (XAML, WinUI, UWP)
#   - Qt / GTK text widgets on Windows
#   - Legacy: Edit, RichEdit, Scintilla, Delphi, IE
_WINDOWS_TEXT_CLASSES: set[str] = {
    # ── Legacy Win32 edit controls ──
    "edit",
    "richedit",
    "scintilla",
    "tmemo",                    # Delphi
    "internetexplorer_server",  # IE / WebView editable areas
    # ── Chromium / Electron ──
    "chrome_widgetwin_",        # Chrome, Edge, Electron (VS Code, Slack, etc.)
    "renderwidgethost",         # Chromium renderer sub-window
    # ── Consoles / terminals ──
    "consolewindowclass",       # cmd.exe, powershell.exe
    "cascadia_hosting_window_class",  # Windows Terminal
    "conemu",                   # ConEmu terminal
    # ── IDEs / editors ──
    "swt_window",               # Eclipse / IntelliJ (SWT)
    "sun_awt",                  # Java AWT text components
}

# Application-level class names that are *known* to contain text inputs
# in their subtree.  When we see these on an ancestor, we consider the
# focused descendant to be a text input.
_ANCESTOR_TEXT_APP_CLASSES: set[str] = {
    "chrome_widgetwin_",        # Chrome, Edge, Electron
    "cascadia_hosting_window_class",  # Windows Terminal
    "consolewindowclass",       # cmd.exe, powershell.exe
    "conemu",                   # ConEmu
    "swt_window",               # Eclipse / IntelliJ
}

# Maximum ancestor levels to walk when checking parent windows.
_MAX_ANCESTOR_DEPTH = 8

# Process names (lowercased, exe extension included) whose main window
# is always a text-input context.  These are terminal emulators, shells,
# and text-focused tools where the user's primary interaction is typing.
#
# Scope rule: only include apps where the PRIMARY interaction is text
# input.  Multi-purpose apps (Chrome, VS Code, Slack) are excluded
# because they have non-text UI surfaces (settings dialogs, toolbars,
# etc.) where paste would be inappropriate.  For those, the class-name
# and caret heuristics provide more precise detection.
#
# Defense-in-depth: some apps here (cmd.exe, powershell.exe) are also
# caught by class-name/ancestor heuristics; the process-name check
# provides a fallback in case the focused child window has a generic
# class name.
_TEXT_PROCESS_NAMES: set[str] = {
    # ── Terminal emulators ──
    "windowsterminal.exe",       # Windows Terminal (defense-in-depth: also caught by cascadia_hosting_window_class)
    "warp.exe",                  # Warp Terminal (class="Window Class")
    "alacritty.exe",             # Alacritty (class="AlacrittyWindow")
    "wezterm-gui.exe",           # WezTerm
    "conemu64.exe",             # ConEmu
    "conemu.exe",               # ConEmu (32-bit)
    "cmd.exe",                  # Command Prompt (defense-in-depth: also caught by consolewindowclass)
    "powershell.exe",           # Windows PowerShell (defense-in-depth: also caught by consolewindowclass)
    "pwsh.exe",                 # PowerShell Core (defense-in-depth: also caught by consolewindowclass)
    # ── Text editors (common ones with generic class names) ──
    "notepad.exe",              # Notepad (usually Edit class, but just in case)
    "notepad++.exe",            # Notepad++
}


# ─── Win32 ctypes argtypes/restype (64-bit safety) ────────────────────
# Set these once at module level to avoid 64-bit handle truncation.
# ctypes defaults return types to c_int (32-bit), which truncates
# 64-bit HANDLEs on Win64.
if sys.platform == "win32":
    import ctypes as _ctypes
    from ctypes import wintypes as _wintypes

    _user32 = _ctypes.windll.user32
    _kernel32 = _ctypes.windll.kernel32

    _user32.GetAncestor.argtypes = [_wintypes.HWND, _wintypes.UINT]
    _user32.GetAncestor.restype = _wintypes.HWND
    _user32.GetClassNameW.argtypes = [_wintypes.HWND, _wintypes.LPWSTR, _ctypes.c_int]
    _user32.GetClassNameW.restype = _ctypes.c_int
    _user32.GetGUIThreadInfo.argtypes = [_wintypes.DWORD, _ctypes.c_void_p]
    _user32.GetGUIThreadInfo.restype = _wintypes.BOOL
    _user32.GetWindowThreadProcessId.argtypes = [
        _wintypes.HWND, _ctypes.POINTER(_wintypes.DWORD),
    ]
    _user32.GetWindowThreadProcessId.restype = _wintypes.DWORD

    _kernel32.OpenProcess.argtypes = [
        _wintypes.DWORD, _wintypes.BOOL, _wintypes.DWORD,
    ]
    _kernel32.OpenProcess.restype = _wintypes.HANDLE
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        _wintypes.HANDLE, _wintypes.DWORD,
        _wintypes.LPWSTR, _ctypes.POINTER(_wintypes.DWORD),
    ]
    _kernel32.QueryFullProcessImageNameW.restype = _wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [_wintypes.HANDLE]
    _kernel32.CloseHandle.restype = _wintypes.BOOL

    # GUITHREADINFO struct — defined once at module level to avoid
    # per-call overhead.
    class _GUITHREADINFO(_ctypes.Structure):
        _fields_ = [
            ("cbSize", _wintypes.DWORD),
            ("flags", _wintypes.DWORD),
            ("hwndActive", _wintypes.HWND),
            ("hwndFocus", _wintypes.HWND),
            ("hwndCapture", _wintypes.HWND),
            ("hwndMenuOwner", _wintypes.HWND),
            ("hwndMoveSize", _wintypes.HWND),
            ("hwndCaret", _wintypes.HWND),
            ("rcCaret", _wintypes.RECT),
        ]


def _windows_is_text_focused() -> bool | None:
    """Use Win32 GetGUIThreadInfo + heuristics to detect text input focus.

    Strategy (all must pass for False; any positive signal → True):
    1. Class-name check on the focused hwnd itself.
    2. Caret-based heuristic: if Windows reports a visible caret with a
       non-zero bounding rect, the focused window almost certainly accepts
       text input.
    3. Ancestor walk: for Chromium/Electron, focus lands on a renderer
       child whose class name is generic.  If an ancestor matches a known
       app class, treat as text input.
    4. Process-name check: if the focused window's owning process is a
       known text-input app (e.g. WindowsTerminal.exe), treat as text
       input.  This catches apps with completely generic class names.
    """
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32   # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        gui_info = _GUITHREADINFO()
        gui_info.cbSize = ctypes.sizeof(_GUITHREADINFO)

        if not user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)):
            log.debug("[FOCUS] GetGUIThreadInfo returned False")
            return None

        hwnd = gui_info.hwndFocus
        if not hwnd:
            log.debug("[FOCUS] hwndFocus is NULL")
            return None

        # ── Heuristic 1: Class-name check on focused hwnd ──────────
        class_name = _get_class_name(user32, hwnd)
        log.debug("[FOCUS] hwndFocus=%#x, class=%r", hwnd, class_name)

        if _class_matches(class_name, _WINDOWS_TEXT_CLASSES):
            log.info(
                "[FOCUS] Text input detected (class match: %r)",
                class_name,
            )
            return True

        # ── Heuristic 2: Caret-based detection ────────────────────
        # If the system reports a caret hwnd with a non-degenerate
        # bounding rect, the focused window is almost certainly
        # accepting text input (the blinking cursor is there).
        hwnd_caret = gui_info.hwndCaret
        if hwnd_caret:
            caret_rect = gui_info.rcCaret
            # A non-zero width or height means the caret is visible
            if caret_rect.right != 0 or caret_rect.bottom != 0:
                log.info(
                    "[FOCUS] Text input detected (caret at hwnd=%#x)",
                    hwnd_caret,
                )
                return True

        # ── Heuristic 3: Ancestor walk ────────────────────────────
        # Chromium/Electron puts focus on "Chrome_RenderWidgetHostHWND"
        # or a generic child -- but the top-level window class reveals
        # the app.  Walk ancestors looking for a known app class.
        ancestor = user32.GetAncestor(hwnd, 1)  # GA_PARENT=1 -> walk one level up
        depth = 0
        while ancestor and depth < _MAX_ANCESTOR_DEPTH:
            ancestor_class = _get_class_name(user32, ancestor)
            log.debug(
                "[FOCUS] Ancestor depth=%d hwnd=%#x class=%r",
                depth, ancestor, ancestor_class,
            )
            if _class_matches(ancestor_class, _ANCESTOR_TEXT_APP_CLASSES):
                log.info(
                    "[FOCUS] Text input detected (ancestor match: %r at depth=%d)",
                    ancestor_class, depth,
                )
                return True
            ancestor = user32.GetAncestor(ancestor, 1)
            depth += 1

        # ── Heuristic 4: Process-name detection ───────────────
        # Some apps (notably Windows Terminal / Warp) use a completely
        # generic class name ("Window Class") that cannot be matched by
        # class name alone.  If the owning process is a known text-input
        # app, treat the focused window as a text input.
        process_name = _get_process_name(user32, kernel32, hwnd)
        if process_name:
            if process_name in _TEXT_PROCESS_NAMES:
                log.info(
                    "[FOCUS] Text input detected (process match: %r for class=%r)",
                    process_name, class_name,
                )
                return True

        log.info(
            "[FOCUS] No text input detected (class=%r, process=%r)",
            class_name, process_name,
        )
        return False

    except Exception:
        log.debug("Focus detection unavailable", exc_info=True)
        return None


def _get_class_name(user32, hwnd: int) -> str:
    """Get the window class name, returned lowercase."""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value.lower()


def _get_process_name(user32, kernel32, hwnd: int) -> str | None:
    """Get the owning process's executable name (lowercased).

    Uses GetWindowThreadProcessId → OpenProcess → QueryFullProcessImageNameW
    to resolve the full path, then extracts just the filename.
    Returns None if any step fails (e.g. insufficient access rights).

    argtypes/restype are set at module level for 64-bit safety.
    """
    try:
        from ctypes import wintypes

        # Get PID from hwnd
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return None

        # Open process with QUERY_LIMITED_INFORMATION (most permissive
        # access that doesn't require admin rights)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h_process = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if not h_process:
            log.debug("[FOCUS] OpenProcess failed for pid=%d", pid.value)
            return None

        try:
            # Query full process image name
            size = wintypes.DWORD(512)
            buf = ctypes.create_unicode_buffer(512)
            # 0 = WIN32 path format
            if not kernel32.QueryFullProcessImageNameW(
                h_process, 0, buf, ctypes.byref(size)
            ):
                log.debug(
                    "[FOCUS] QueryFullProcessImageNameW failed for pid=%d",
                    pid.value,
                )
                return None

            # Extract just the filename (lowercased)
            full_path = buf.value
            filename = full_path.rsplit("\\", 1)[-1].lower()
            return filename
        finally:
            kernel32.CloseHandle(h_process)

    except Exception:
        log.debug("[FOCUS] Process name lookup failed", exc_info=True)
        return None


def _class_matches(class_name: str, known_classes: set[str]) -> bool:
    """Check if *class_name* contains any entry from *known_classes*.

    Uses substring matching (case-insensitive, since class_name is
    already lowered).  For example, "chrome_renderwidgethosthwnd"
    contains "renderwidgethost".
    """
    return any(pattern in class_name for pattern in known_classes)
