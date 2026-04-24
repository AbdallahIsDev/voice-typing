# Streaming Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hidden streaming transcription so long recordings do most transcription work before the second F2 press while preserving final-only clipboard/paste behavior.

**Architecture:** Keep the current batch path as fallback. Add a focused `voice_typer.streaming` module for window planning, timestamped word assembly, and streaming session orchestration. Extend recorder snapshots, timestamped transcription, and app lifecycle wiring with tests around every fallback boundary.

**Tech Stack:** Python, NumPy, sounddevice, faster-whisper, pytest, unittest.mock.

---

## File Structure

- Create `voice_typer/streaming.py`: streaming config, audio window planner, text assembler, and session worker.
- Create `tests/test_streaming.py`: core planner, assembler, and streaming session unit tests.
- Modify `voice_typer/config.py`: streaming config fields.
- Modify `tests/test_config.py`: defaults and round-trip coverage for streaming config.
- Modify `voice_typer/recording.py`: snapshot and shared resampling helper.
- Modify `tests/test_recording.py`: snapshot/resampling behavior.
- Modify `voice_typer/transcription.py`: model lock and timestamped word transcription.
- Modify `tests/test_transcription.py`: lock/timestamp behavior.
- Modify `voice_typer/app.py`: start/finalize/cancel streaming session and share final text handling.
- Modify `tests/test_app.py`: final-only paste, fallback, kill switch, and cancellation behavior.

### Task 1: Config And Streaming Core

**Files:**
- Create: `voice_typer/streaming.py`
- Create: `tests/test_streaming.py`
- Modify: `voice_typer/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing config and core streaming tests**

Cover:

- streaming config defaults are present and disabled by default
- `AudioWindowPlanner` waits until enough audio exists
- `AudioWindowPlanner` creates overlap windows
- planner prefers silence near boundary
- `StreamingTextAssembler` commits words before right guard
- assembler avoids duplicate overlap by timestamp
- assembler preserves repeated words at later timestamps

Run: `pytest tests/test_config.py tests/test_streaming.py -q`

Expected: `tests/test_streaming.py` import/attribute failures and config default failures.

- [ ] **Step 2: Implement config fields and core streaming classes**

Implement:

- `StreamingConfig`
- `WordTiming`
- `AudioWindow`
- `AudioWindowPlanner`
- `StreamingTextAssembler`

Run: `pytest tests/test_config.py tests/test_streaming.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add voice_typer/streaming.py voice_typer/config.py tests/test_streaming.py tests/test_config.py
git commit -m "feat: add streaming core"
```

### Task 2: Recorder Snapshot And Transcription Timestamps

**Files:**
- Modify: `voice_typer/recording.py`
- Modify: `tests/test_recording.py`
- Modify: `voice_typer/transcription.py`
- Modify: `tests/test_transcription.py`

- [ ] **Step 1: Write failing tests**

Cover:

- `Recorder.snapshot()` returns 16 kHz audio while recording without clearing the buffer
- snapshot returns empty array when no buffer exists
- snapshot uses the same resampling path as stop
- `TranscriptionEngine.transcribe_words()` passes `word_timestamps=True` and `without_timestamps=False`
- timestamped transcription returns `WordTiming` values with offset applied
- model calls are guarded by an engine lock

Run: `pytest tests/test_recording.py tests/test_transcription.py -q`

Expected: missing method failures.

- [ ] **Step 2: Implement minimal recorder and transcriber changes**

Implement:

- shared `_prepare_audio(audio, effective_sr)` helper in recorder
- `Recorder.snapshot()`
- `threading.RLock` in `TranscriptionEngine`
- private unlocked transcription helpers where needed
- `TranscriptionEngine.transcribe_words(audio, offset_seconds=0.0)`

Run: `pytest tests/test_recording.py tests/test_transcription.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add voice_typer/recording.py voice_typer/transcription.py tests/test_recording.py tests/test_transcription.py
git commit -m "feat: support streaming audio snapshots"
```

### Task 3: Streaming Session

**Files:**
- Modify: `voice_typer/streaming.py`
- Modify: `tests/test_streaming.py`

- [ ] **Step 1: Write failing session tests**

Cover:

- session starts a worker and can cancel cleanly
- session finalizes only the uncommitted tail
- chunk failures mark the session for batch fallback
- final merge failures call batch fallback
- no provisional text is returned before finalization

Run: `pytest tests/test_streaming.py -q`

Expected: missing `StreamingTranscriptionSession` behavior failures.

- [ ] **Step 2: Implement session worker and fallback behavior**

Implement one background worker with:

- one model call at a time
- snapshot polling
- planner-created windows
- timestamped word assembly
- fallback flag on any streaming error
- `finalize(full_audio)` returning final text or batch fallback text
- `cancel()` for shutdown/discard paths

Run: `pytest tests/test_streaming.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add voice_typer/streaming.py tests/test_streaming.py
git commit -m "feat: add streaming session finalization"
```

### Task 4: App Integration

**Files:**
- Modify: `voice_typer/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing app integration tests**

Cover:

- first F2 starts a streaming session when config enables it
- second F2 stops recording even while streaming is active
- streaming partials never call clipboard copy or paste
- final text copies and pastes once
- streaming session failure falls back to existing batch path
- `VOICE_TYPER_STREAMING=0` forces batch path
- quit cancels active streaming session
- microphone selection during recording does not replace active recorder

Run: `pytest tests/test_app.py -q`

Expected: streaming behavior failures.

- [ ] **Step 2: Implement app wiring**

Implement:

- `_streaming_session` field
- `_streaming_enabled()` helper with env kill switch
- session creation after recorder start
- finalization path in `_stop_dictation`
- shared `_handle_transcription_result(text, recorded_rms)` method
- cancel in `quit()`
- protect microphone changes during active recording

Run: `pytest tests/test_app.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add voice_typer/app.py tests/test_app.py
git commit -m "feat: wire hidden streaming transcription"
```

### Task 5: Verification And Rollout

**Files:**
- Any small fixes from full-suite failures.

- [ ] **Step 1: Run targeted verification**

Run:

```bash
pytest tests/test_streaming.py tests/test_app.py tests/test_recording.py tests/test_transcription.py tests/test_config.py -q
```

Expected: pass.

- [ ] **Step 2: Run full verification**

Run:

```bash
pytest -q
```

Expected: pass.

- [ ] **Step 3: Commit fixes if needed**

If verification required changes:

```bash
git add <changed-files>
git commit -m "test: stabilize streaming transcription"
```

- [ ] **Step 4: Report rollout instructions**

Report:

- branch and commits
- tests run and result
- how to enable streaming in config
- how to disable streaming with `VOICE_TYPER_STREAMING=0`
