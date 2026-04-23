"""Platform-specific adapters: autostart, microphone listing."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SYSTEM = sys.platform  # "win32", "darwin", "linux"


# ─── Microphone helpers ────────────────────────────────────────────────

def list_microphones() -> list[dict]:
    """Return available input devices with stable identifiers.

    Each dict:
        {
            "id": str,          # stable identifier (device index as string)
            "index": int,       # sounddevice device index
            "name": str,        # display name
            "host_api": str,    # host API name (e.g. "Windows WASAPI")
            "channels": int,    # max input channels
            "default": bool,    # True if system default input device
        }
    Returns empty list on failure.
    """
    try:
        import sounddevice as sd
        default_input = sd.query_devices(kind="input")
        default_index = default_input["index"] if default_input else -1
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                host_api = ""
                try:
                    host_api_idx = dev.get("hostapi", 0)
                    host_api = sd.query_hostapis(host_api_idx)["name"]
                except Exception:
                    pass
                devices.append({
                    "id": str(i),
                    "index": i,
                    "name": dev["name"],
                    "host_api": host_api,
                    "channels": dev["max_input_channels"],
                    "default": i == default_index,
                })
        return devices
    except Exception:
        log.debug("Could not enumerate microphones", exc_info=True)
        return []


def find_microphone_by_name(partial_name: str) -> Optional[dict]:
    """Find a microphone whose name contains *partial_name* (case-insensitive)."""
    lower = partial_name.lower()
    for mic in list_microphones():
        if lower in mic["name"].lower():
            return mic
    return None


def find_microphone_by_id(mic_id: str) -> Optional[dict]:
    """Find a microphone by its stable ID (device index string)."""
    for mic in list_microphones():
        if mic["id"] == mic_id:
            return mic
    return None


# ─── Autostart ─────────────────────────────────────────────────────────

def _autostart_command() -> str:
    """Build the command that the autostart entry should run.

    Uses the currently-running Python interpreter with ``-m voice_typer``.
    This requires the package to be installed (``pip install .``).

    On Windows, prefers pythonw.exe (no console window) when available.
    pythonw startup has been verified via Start-Process, but actual
    reboot/login-startup has not been manually tested yet.
    """
    if sys.platform == "win32":
        pythonw = Path(sys.executable).parent / "pythonw.exe"
        if pythonw.exists():
            return f'"{pythonw}" -m voice_typer'
        return f'"{sys.executable}" -m voice_typer'
    return f'"{sys.executable}" -m voice_typer'


def get_autostart_dir() -> Path:
    if SYSTEM == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    elif SYSTEM == "darwin":
        return Path.home() / "Library" / "LaunchAgents"
    else:
        return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart"


def enable_autostart() -> bool:
    try:
        if SYSTEM == "win32":
            return _enable_autostart_windows()
        elif SYSTEM == "darwin":
            return _enable_autostart_macos()
        else:
            return _enable_autostart_linux()
    except Exception as e:
        log.error("Failed to enable autostart: %s", e)
        return False


def disable_autostart() -> bool:
    try:
        if SYSTEM == "win32":
            return _disable_autostart_windows()
        elif SYSTEM == "darwin":
            return _disable_autostart_macos()
        else:
            return _disable_autostart_linux()
    except Exception as e:
        log.error("Failed to disable autostart: %s", e)
        return False


def is_autostart_enabled() -> bool:
    try:
        if SYSTEM == "win32":
            return _is_autostart_windows()
        elif SYSTEM == "darwin":
            return _is_autostart_macos()
        else:
            return _is_autostart_linux()
    except Exception:
        return False


# ─── Windows ───────────────────────────────────────────────────────────

def _enable_autostart_windows() -> bool:
    import winreg
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0, winreg.KEY_SET_VALUE,
    )
    cmd = _autostart_command()
    winreg.SetValueEx(key, "VoiceTyper", 0, winreg.REG_SZ, cmd)
    winreg.CloseKey(key)
    log.info("Autostart enabled (Windows): %s", cmd)
    return True


def _disable_autostart_windows() -> bool:
    import winreg
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0, winreg.KEY_SET_VALUE,
    )
    try:
        winreg.DeleteValue(key, "VoiceTyper")
    except FileNotFoundError:
        pass
    winreg.CloseKey(key)
    log.info("Autostart disabled (Windows)")
    return True


def _is_autostart_windows() -> bool:
    import winreg
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_READ,
        )
        try:
            val, _ = winreg.QueryValueEx(key, "VoiceTyper")
            return bool(val)
        except FileNotFoundError:
            return False
        finally:
            winreg.CloseKey(key)
    except FileNotFoundError:
        return False


# ─── macOS ─────────────────────────────────────────────────────────────

def _enable_autostart_macos() -> bool:
    plist_dir = get_autostart_dir()
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.voicetyper.plist"
    cmd = _autostart_command()

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voicetyper</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>voice_typer</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>"""
    plist_path.write_text(plist_content)
    log.info("Autostart enabled (macOS): %s", plist_path)
    return True


def _disable_autostart_macos() -> bool:
    plist_path = get_autostart_dir() / "com.voicetyper.plist"
    if plist_path.exists():
        plist_path.unlink()
    log.info("Autostart disabled (macOS)")
    return True


def _is_autostart_macos() -> bool:
    return (get_autostart_dir() / "com.voicetyper.plist").exists()


# ─── Linux ─────────────────────────────────────────────────────────────

def _enable_autostart_linux() -> bool:
    autostart_dir = get_autostart_dir()
    autostart_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = autostart_dir / "voice-typer.desktop"

    desktop_content = f"""[Desktop Entry]
Type=Application
Name=Voice Typer
Comment=Background voice-to-text utility
Exec={sys.executable} -m voice_typer
Icon=audio-input-microphone
Hidden=false
NoDisplay=true
X-GNOME-Autostart-enabled=true
"""
    desktop_path.write_text(desktop_content)
    log.info("Autostart enabled (Linux): %s", desktop_path)
    return True


def _disable_autostart_linux() -> bool:
    desktop_path = get_autostart_dir() / "voice-typer.desktop"
    if desktop_path.exists():
        desktop_path.unlink()
    log.info("Autostart disabled (Linux)")
    return True


def _is_autostart_linux() -> bool:
    return (get_autostart_dir() / "voice-typer.desktop").exists()
