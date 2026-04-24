# Settings UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw JSON settings as the primary UX with a native settings window, simplify tray options, and make streaming/CUDA-first behavior the fixed default.

**Architecture:** Keep `Config` as the persistence layer. Add a small settings controller/window module that updates config through app callbacks. Tray remains pystray-based and delegates settings/hotkey/microphone actions to `VoiceTyperApp`.

**Tech Stack:** Python, tkinter, pystray, pytest, unittest.mock.

---

## File Structure

- Create `voice_typer/settings.py`: settings data model/controller helpers and tkinter window.
- Create `tests/test_settings.py`: headless unit tests for allowed values and controller behavior.
- Modify `voice_typer/config.py`: approved defaults and normalization.
- Modify `tests/test_config.py`: defaults and legacy normalization tests.
- Modify `voice_typer/tray.py`: simplified tray menu and hotkey labels.
- Modify `tests/test_tray.py`: tray menu expectations.
- Modify `voice_typer/app.py`: settings window integration, immediate hotkey re-registration, fixed streaming/paste behavior, model update handling.
- Modify `tests/test_app.py`: settings callback integration and removed autostart menu behavior.

### Task 1: Config Normalization

**Files:**
- Modify: `voice_typer/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Cover defaults:

- `Config().streaming_transcription is True`
- `Config().paste_on_stop is True`
- `Config().device == "cuda"`
- `Config().model_size == "small.en"`

Cover legacy loading:

- config file with `"streaming_transcription": false` loads as `True`
- config file with `"paste_on_stop": false` loads as `True`
- config file with `"model_size": "tiny.en"` loads as `"small.en"`
- config file with unsupported model loads as `"small.en"`
- config file with `"model_size": "medium.en"` remains `"medium.en"`

Run: `pytest tests/test_config.py -q`

Expected: fails before implementation.

- [ ] **Step 2: Implement normalization**

Add allowed user models in config and normalize loaded config before constructing `Config`.

Run: `pytest tests/test_config.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add voice_typer/config.py tests/test_config.py
git commit -m "feat: normalize fast default config"
```

### Task 2: Tray Menu Simplification

**Files:**
- Modify: `voice_typer/tray.py`
- Modify: `tests/test_tray.py`

- [ ] **Step 1: Write failing tests**

Cover:

- main tray menu does not include `Start on Login`
- toggle item label includes current hotkey
- menu includes a disabled/info item `Hotkey: F2`
- settings label is `Settings...`
- microphone submenu still exists when mics are present

Run: `pytest tests/test_tray.py -q`

Expected: fails before implementation.

- [ ] **Step 2: Implement tray changes**

Remove the main-menu autostart item and build hotkey labels from config.

Run: `pytest tests/test_tray.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add voice_typer/tray.py tests/test_tray.py
git commit -m "feat: simplify tray settings menu"
```

### Task 3: Settings Controller And Window

**Files:**
- Create: `voice_typer/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write failing controller tests**

Cover:

- allowed model values are `small.en` and `medium.en`
- hotkey display converts `<f2>` to `F2`
- hotkey capture formatter accepts `F1` through `F12`
- settings controller saves model/hotkey/microphone/advanced values
- settings controller forces `paste_on_stop=True` and `streaming_transcription=True`

Run: `pytest tests/test_settings.py -q`

Expected: fails before implementation.

- [ ] **Step 2: Implement settings module**

Add:

- `ALLOWED_MODELS = ("small.en", "medium.en")`
- `display_hotkey(value: str) -> str`
- `format_function_hotkey(key_name: str) -> str`
- `SettingsController`
- `SettingsWindow` using tkinter

Run: `pytest tests/test_settings.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add voice_typer/settings.py tests/test_settings.py
git commit -m "feat: add native settings controller"
```

### Task 4: App Integration

**Files:**
- Modify: `voice_typer/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing app tests**

Cover:

- `show_settings()` opens `SettingsWindow` instead of notepad
- saving settings updates config and calls hotkey re-registration
- changing model recreates transcriber for next use
- app constructs clipboard with paste enabled regardless of config file value
- app constructs streaming config enabled regardless of config file value

Run: `pytest tests/test_app.py -q`

Expected: fails before implementation.

- [ ] **Step 2: Implement app wiring**

Integrate `SettingsWindow`, add `_apply_settings`, `_restart_hotkey`, and enforce fixed streaming/paste behavior in app construction.

Run: `pytest tests/test_app.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add voice_typer/app.py tests/test_app.py
git commit -m "feat: wire native settings window"
```

### Task 5: Verification And Restart

**Files:**
- Any small fixes from test failures.

- [ ] **Step 1: Run targeted verification**

```bash
pytest tests/test_config.py tests/test_tray.py tests/test_settings.py tests/test_app.py -q
```

Expected: pass.

- [ ] **Step 2: Run full verification**

```bash
pytest -q
```

Expected: pass.

- [ ] **Step 3: Restart tray process**

Stop current `pythonw.exe -m voice_typer`, start it from this branch, and check the latest log for clean startup.

- [ ] **Step 4: Push**

```bash
git push
```
