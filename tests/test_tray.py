"""Tests for the tray threading model.

The tray must support:
- start() is non-blocking and kicks off background work
- run() blocks the main thread (pystray event loop)
- State / notifications queued before run() are flushed once run() starts
- menu= passed to pystray.Icon must be a pystray.Menu instance, NOT a bare callable

Suppresses ResourceWarning / unraisable destructor warnings from pystray
Icon objects that never entered a real event loop.
"""

import gc
import sys
import time
import warnings
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock


class _FakeMenu:
    """Lightweight stand-in for pystray.Menu that records construction args.

    Mirrors real pystray.Menu behavior: if a callable is passed, it is stored
    and can be invoked with zero positional args to materialize menu items.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # Store the callable (like real pystray.Menu does)
        self._callable = args[0] if args and callable(args[0]) else None

    def __call__(self):
        """Materialize menu items by invoking the stored callable with zero args."""
        if self._callable is not None:
            return self._callable()
        return self.args

    SEPARATOR = "SEP"


class _FakeMenuItem:
    """Lightweight stand-in for pystray.MenuItem."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeIcon:
    """Record how pystray.Icon was constructed so we can assert on kwargs."""
    last_kwargs = {}

    def __init__(self, **kwargs):
        _FakeIcon.last_kwargs = kwargs
        self.menu = kwargs.get("menu")
        self.icon = kwargs.get("icon")
        self.title = kwargs.get("title", "")
        self._run_called = False

    def run(self):
        self._run_called = True

    def stop(self):
        pass

    def notify(self, *a, **kw):
        pass


# Mock heavy imports
@pytest.fixture(autouse=True)
def mock_heavy_imports(monkeypatch):
    mock_pystray = MagicMock()
    mock_pystray.Icon = _FakeIcon
    mock_pystray.Menu = _FakeMenu
    mock_pystray.Menu.SEPARATOR = "SEP"
    mock_pystray.MenuItem = _FakeMenuItem
    monkeypatch.setitem(sys.modules, "pystray", mock_pystray)

    # Also patch the attributes on the already-imported tray module so that
    # even if other test modules replaced sys.modules["pystray"] first,
    # tray.py still uses our fakes.
    import voice_typer.tray as tray_mod
    monkeypatch.setattr(tray_mod, "pystray", mock_pystray)

    mock_pil = MagicMock()
    monkeypatch.setitem(sys.modules, "PIL", mock_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", MagicMock())
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", MagicMock())


@pytest.fixture
def tray():
    from voice_typer.tray import TrayIcon
    _FakeIcon.last_kwargs = {}
    t = TrayIcon(
        on_toggle=MagicMock(),
        on_settings=MagicMock(),
        on_quit=MagicMock(),
    )
    yield t
    # Suppress unraisable destructor warnings from pystray Icon objects
    # that were created but never entered a real event loop.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pytest.PytestUnraisableExceptionWarning)
        del t


# ─── Regression: menu= must be a pystray.Menu instance ──────────────────

class TestMenuIsPystrayMenuInstance:
    """Regression test: menu= must be a pystray.Menu, NOT a bare callable."""

    def test_menu_is_fake_menu_instance(self, tray):
        """After start(), the menu kwarg passed to pystray.Icon must be a _FakeMenu."""
        tray.start(bg_work=None)
        menu = _FakeIcon.last_kwargs.get("menu")
        assert isinstance(menu, _FakeMenu), (
            f"menu= must be a pystray.Menu instance, got {type(menu).__name__}: {menu!r}"
        )

    def test_menu_is_not_raw_callable(self, tray):
        """menu= must be a pystray.Menu instance, not a bare function/callable."""
        tray.start(bg_work=None)
        menu = _FakeIcon.last_kwargs.get("menu")
        # _FakeMenu (and real pystray.Menu) is itself callable, but it should
        # be a Menu *instance*, not a bare function reference.
        assert isinstance(menu, _FakeMenu), (
            "menu= should be a pystray.Menu instance, not a bare callable"
        )

    def test_menu_callable_is_passed_to_menu_constructor(self, tray):
        """The callable (build_menu) should be an arg to _FakeMenu, not to _FakeIcon."""
        tray.start(bg_work=None)
        menu = _FakeIcon.last_kwargs.get("menu")
        assert isinstance(menu, _FakeMenu)
        assert len(menu.args) >= 1
        assert callable(menu.args[0]), "pystray.Menu should receive a callable as its argument"


# ─── Threading model ────────────────────────────────────────────────────

class TestTrayStartIsNonBlocking:
    def test_start_returns_without_blocking(self, tray):
        """start() must return immediately — it must NOT call run()."""
        bg_called = []
        def bg_work():
            bg_called.append(True)

        tray.start(bg_work=bg_work)

        # The tray's own icon should not have had run() called yet
        assert not tray._icon._run_called
        # Background work should have started
        time.sleep(0.1)
        assert len(bg_called) == 1

    def test_start_without_bg_work_does_not_crash(self, tray):
        tray.start(bg_work=None)
        assert tray._icon is not None


class TestTrayRunBlocksMainThread:
    def test_run_calls_icon_run(self, tray):
        """run() must call pystray.Icon.run() (which blocks)."""
        tray.start(bg_work=None)
        assert not tray._icon._run_called

        tray.run()
        assert tray._icon._run_called


class TestTrayPendingState:
    def test_state_before_run_is_queued(self, tray):
        """set_state() before run() must queue the state."""
        from voice_typer.tray import AppState
        tray.set_state(AppState.LOADING, "Loading model...")

        # Icon doesn't exist yet — state should be pending
        assert len(tray._pending_states) == 1
        assert tray._pending_states[0] == (AppState.LOADING, "Loading model...")

    def test_pending_state_flushed_on_run(self, tray):
        """After run(), queued states are applied to the live icon."""
        from voice_typer.tray import AppState
        tray.set_state(AppState.LOADING, "Starting...")

        tray.start(bg_work=None)
        tray.run()

        # Pending states should be flushed
        assert len(tray._pending_states) == 0
        # Icon state should reflect the flushed state
        assert tray._state == AppState.LOADING

    def test_notification_before_run_is_queued(self, tray):
        tray.notify("Title", "Message")
        assert len(tray._pending_notifications) == 1

    def test_pending_notification_flushed_on_run(self, tray):
        tray.notify("Test", "Hello")
        tray.start(bg_work=None)
        tray.run()

        assert len(tray._pending_notifications) == 0


# ─── Regression: menu callable signature ────────────────────────────────

class TestMenuCallableSignature:
    """Regression: the menu-generator callable must accept zero positional args."""

    def test_menu_callable_takes_zero_positional_args(self, tray):
        """Calling _FakeMenu's stored callable with zero args must not raise TypeError.

        Real pystray.Menu invokes its callable with no positional arguments
        each time the tray menu is opened.  If _build_menu accidentally
        requires arguments, this test catches it.
        """
        tray.start(bg_work=None)
        menu = _FakeIcon.last_kwargs.get("menu")
        assert isinstance(menu, _FakeMenu), "menu must be a _FakeMenu instance"
        # This mirrors how pystray materializes the menu
        result = menu()
        # Should return items without raising TypeError
        assert result is not None

    def test_menu_materialization_works(self, tray):
        """After start(), materializing the menu callable returns a tuple of items."""
        tray.start(bg_work=None)
        menu = _FakeIcon.last_kwargs.get("menu")
        assert isinstance(menu, _FakeMenu)

        items = menu()
        assert isinstance(items, tuple), f"Expected tuple, got {type(items)}"
        assert len(items) > 0, "Menu should have at least one item"
        # Each item should be a _FakeMenuItem or the SEPARATOR sentinel
        for item in items:
            assert isinstance(item, (_FakeMenuItem, str)), (
                f"Unexpected menu item type: {type(item)}"
            )


# ─── Settings UX: simplified tray menu ─────────────────────────────────

class TestSettingsUxTrayMenu:
    def _menu_labels(self, tray):
        tray.start(bg_work=None)
        return [
            item.args[0]
            for item in _FakeIcon.last_kwargs["menu"]()
            if isinstance(item, _FakeMenuItem)
        ]

    def test_main_menu_does_not_include_start_on_login(self):
        from voice_typer.tray import TrayIcon

        tray = TrayIcon(
            on_toggle=MagicMock(),
            on_settings=MagicMock(),
            on_quit=MagicMock(),
            on_toggle_autostart=MagicMock(),
        )

        labels = self._menu_labels(tray)

        assert "Start on Login" not in labels

    def test_toggle_label_includes_current_hotkey(self):
        from voice_typer.tray import TrayIcon

        tray = TrayIcon(
            on_toggle=MagicMock(),
            on_settings=MagicMock(),
            on_quit=MagicMock(),
            config=SimpleNamespace(hotkey="<f9>"),
        )

        labels = self._menu_labels(tray)

        assert "Toggle Dictation (F9)" in labels

    def test_menu_includes_disabled_hotkey_info_item(self, tray):
        tray._config = SimpleNamespace(hotkey="<f2>")

        tray.start(bg_work=None)
        hotkey_item = next(
            item
            for item in _FakeIcon.last_kwargs["menu"]()
            if isinstance(item, _FakeMenuItem) and item.args[0] == "Hotkey: F2"
        )

        assert hotkey_item.kwargs["enabled"] is False

    def test_settings_label_uses_ellipsis(self, tray):
        labels = self._menu_labels(tray)

        assert "Settings..." in labels
        assert "Settings" not in labels

    def test_microphone_submenu_remains_when_mics_are_present(self, tray):
        tray.on_select_mic = MagicMock()
        tray._config = SimpleNamespace(microphone=None, hotkey="<f2>")
        tray.set_microphones([
            {"id": "mic-1", "name": "Built-in Mic", "host_api": "WASAPI"},
        ])

        tray.start(bg_work=None)
        mic_item = next(
            item
            for item in _FakeIcon.last_kwargs["menu"]()
            if isinstance(item, _FakeMenuItem) and item.args[0] == "Microphone"
        )

        assert isinstance(mic_item.args[1], _FakeMenu)


# ─── Integration: full start + run cycle ────────────────────────────────

class TestFullStartRunCycle:
    """Integration test: start() + run() together must not crash."""

    def test_full_start_run_cycle_no_crash(self, tray):
        """Calling start() followed by run() should complete without any exception."""
        try:
            tray.start(bg_work=None)
            tray.run()
        except Exception as exc:
            pytest.fail(
                f"start() + run() cycle raised unexpectedly: {exc}"
            )


# ─── REAL pystray regression tests ─────────────────────────────────────

def _has_display():
    """Check if a display server is available for pystray."""
    import os
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    # Check for xvfb-run
    import shutil
    if shutil.which("xvfb-run"):
        return True
    return False


_skip_no_display = pytest.mark.skipif(
    not _has_display(),
    reason="No display server available (need DISPLAY, WAYLAND_DISPLAY, or xvfb-run)",
)


@_skip_no_display
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestRealPystrayIntegration:
    """Regression tests using REAL pystray (not faked).

    These tests exercise the actual pystray.Menu and pystray.Icon classes
    to catch the exact TypeError that occurred when a bare callable was
    passed to Icon(menu=...) instead of Icon(menu=pystray.Menu(...)).

    They restore the real pystray module (bypassing the autouse mock fixture).
    """

    @pytest.fixture(autouse=True)
    def _restore_real_pystray(self):
        """Undo the autouse mock_heavy_imports fixture for this class."""
        import importlib
        # Remove the mock from sys.modules so import loads the real module
        if "pystray" in sys.modules:
            del sys.modules["pystray"]
        if "pystray._base" in sys.modules:
            del sys.modules["pystray._base"]
        if "pystray._xorg" in sys.modules:
            del sys.modules["pystray._xorg"]
        # Force reimport
        import pystray as real_pystray
        importlib.reload(real_pystray)
        self.pystray = real_pystray
        yield
        # Force garbage collection of pystray Icon objects so their
        # destructors fire here (within the module-level filterwarnings
        # scope) rather than at arbitrary later GC points.
        gc.collect()
        # After test, the autouse fixture will re-mock on next test

    def test_real_menu_with_callable_works(self):
        """pystray.Menu(some_callable) should construct without error."""
        pystray = self.pystray
        from PIL import Image

        def build_items():
            return (
                pystray.MenuItem("Test", lambda icon, item: None),
            )

        menu = pystray.Menu(build_items)
        # Menu should be callable (pystray.Menu is callable)
        assert callable(menu)

    def test_real_menu_with_bound_method_works(self):
        """pystray.Menu(bound_method) should construct without error.

        This is the exact pattern used in tray.py: pystray.Menu(self._build_menu)
        where _build_menu is a bound method of TrayIcon.
        """
        pystray = self.pystray

        class _Dummy:
            def build_menu(self):
                return (
                    pystray.MenuItem("Item", lambda icon, item: None),
                )

        obj = _Dummy()
        menu = pystray.Menu(obj.build_menu)
        assert callable(menu)

    def test_real_icon_with_menu_instance_works(self):
        """pystray.Icon(menu=pystray.Menu(callable)) should succeed."""
        pystray = self.pystray
        from PIL import Image

        def build_items():
            return (
                pystray.MenuItem("Test", lambda icon, item: None),
            )

        menu = pystray.Menu(build_items)
        img = Image.new("RGBA", (16, 16), "red")
        icon = pystray.Icon("test-regression", icon=img, menu=menu)
        assert icon is not None
        icon.stop()

    def test_bare_callable_to_icon_raises_typeerror(self):
        """pystray.Icon(menu=some_callable) MUST raise TypeError.

        This is the exact bug that was present: passing a bare callable
        directly to Icon's menu= parameter instead of wrapping it in
        pystray.Menu() first.
        """
        pystray = self.pystray
        from PIL import Image

        def build_items():
            return (
                pystray.MenuItem("Test", lambda icon, item: None),
            )

        img = Image.new("RGBA", (16, 16), "red")
        with pytest.raises(TypeError, match="argument after \\* must be an iterable"):
            pystray.Icon("test-bare", icon=img, menu=build_items)

    def test_menu_star_callable_raises_typeerror(self):
        """pystray.Menu(*bare_callable) raises TypeError — the exact crash.

        If someone accidentally wrote pystray.Menu(*self._build_menu) instead
        of pystray.Menu(self._build_menu), this is the error they'd get.
        """
        pystray = self.pystray

        def build_items():
            return (
                pystray.MenuItem("Test", lambda icon, item: None),
            )

        with pytest.raises(TypeError, match="argument after \\* must be an iterable"):
            pystray.Menu(*build_items)

    def test_tray_code_uses_menu_wrapper(self):
        """Verify tray.py explicitly wraps the callable in pystray.Menu().

        This is a code-level regression check: the source must contain
        'pystray.Menu(' before 'pystray.Icon(' to prevent the bare-callable bug.
        """
        import inspect
        from voice_typer.tray import TrayIcon

        source = inspect.getsource(TrayIcon.start)
        # The start() method must create a pystray.Menu before passing to Icon
        assert "pystray.Menu(" in source, (
            "TrayIcon.start() must wrap the menu callable in pystray.Menu()"
        )
        # And it must pass the Menu instance to Icon
        assert "pystray.Icon(" in source, (
            "TrayIcon.start() must call pystray.Icon()"
        )

    def test_real_tray_icon_construction_via_tray_class(self):
        """End-to-end: TrayIcon.start() should create a real pystray.Icon without TypeError.

        This exercises the actual TrayIcon class with real pystray (not faked)
        and verifies the defensive fix in tray.py line 107 works.
        """
        pystray = self.pystray

        # Temporarily patch the tray module to use real pystray
        import voice_typer.tray as tray_mod
        old_pystray = tray_mod.pystray
        tray_mod.pystray = pystray
        try:
            from voice_typer.tray import TrayIcon
            from unittest.mock import MagicMock

            tray = TrayIcon(
                on_toggle=MagicMock(),
                on_settings=MagicMock(),
                on_quit=MagicMock(),
            )
            # This should NOT raise TypeError — the fix wraps _build_menu in pystray.Menu()
            tray.start(bg_work=None)
            assert tray._icon is not None
            tray.stop()
        finally:
            tray_mod.pystray = old_pystray
