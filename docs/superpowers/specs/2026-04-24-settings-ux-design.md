# Settings UX Design

## Goal

Replace raw JSON editing as the main settings experience with a small native settings window and simplify the product around the fastest normal path.

## Approved Product Rules

- Hidden streaming transcription is the only normal algorithm.
- The old full-recording batch path remains only as an invisible emergency fallback when streaming cannot produce safe text.
- CUDA is the default and only user-facing device policy. If CUDA fails, the app silently falls back to CPU.
- The user should not choose `auto`, `cuda`, or `cpu`.
- Models shown to users are only `small.en` and `medium.en`.
- `tiny.en` is hidden from settings. It may remain as an internal last-resort model fallback only if the model loader needs it to keep the app alive.
- Paste after transcription is always enabled and not configurable.
- Streaming is always enabled and not configurable.
- Start on login stays default-on. The main tray menu should not expose it. It can be disabled in Advanced settings.

## Tray Menu

The tray menu should be simple:

- `Toggle Dictation (<current hotkey>)`
- `Hotkey: <current hotkey>`
- `Microphone`
- `Settings...`
- `Quit`

The microphone submenu remains available because it is quick and useful from the tray. Duplicate microphone names should keep showing host API labels so WO Mic variants are distinguishable.

## Settings Window

Use a lightweight native `tkinter` window because it ships with Python and keeps the tray utility dependency-light.

Main settings:

- Hotkey row with current value and a `Change...` button.
- Microphone dropdown with friendly labels and selected value.
- Model dropdown with `small.en` and `medium.en`.

Advanced settings:

- Start on login checkbox.
- Show notifications checkbox.
- Open raw config file button for troubleshooting.

Buttons:

- Save
- Cancel

Saving settings should update the config file and apply changes where possible:

- Hotkey changes re-register the global hotkey immediately.
- Microphone changes apply to the next recording if a recording is active, otherwise recreate the recorder immediately.
- Model changes unload/recreate the transcription engine for future sessions and reload on the next start attempt if necessary.
- Start on login toggles the platform autostart state.
- Notification changes update the tray notification flag immediately.

## Hotkey Capture

The hotkey capture flow should be conservative:

- Click `Change...`.
- A small capture dialog says it is waiting for a key.
- The next supported key becomes the hotkey.
- Support common single-key function keys first: `F1` through `F12`.
- Support `Esc` to cancel capture.

This keeps the first implementation safe and avoids broad global shortcut parsing bugs. The app already uses strings like `<f2>`, so the captured key should save as `<f3>`, `<f4>`, etc.

## Config Defaults And Loading

The `Config` dataclass should move toward the approved defaults:

- `device = "cuda"`
- `paste_on_stop = True`
- `autostart = True`
- `streaming_transcription = True`
- `model_size = "small.en"`

Loading older config files should normalize invalid or hidden user-facing values:

- `model_size = "tiny.en"` becomes `small.en`.
- unsupported model values become `small.en`.
- `device` values are ignored by the settings UI; runtime behavior remains CUDA-first with CPU fallback.
- `paste_on_stop` is forced to `True`.
- `streaming_transcription` is forced to `True`.

## Testing

Add tests for:

- config normalization of hidden/legacy values
- tray menu no longer exposes Start on Login
- tray menu shows current hotkey in toggle and hotkey label
- settings controller saves allowed model values only
- settings controller re-registers hotkeys
- settings controller keeps paste and streaming forced on
- hotkey formatting for `F1` through `F12`

Avoid GUI event-loop tests for visual layout. Unit-test the controller and tray menu construction so tests remain headless.
