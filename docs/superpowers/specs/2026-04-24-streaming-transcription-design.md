# Streaming Transcription Design

## Goal

Make long dictation feel much faster without changing the user's Whisper model or changing the visible F2 workflow.

The app should start recording on the first F2 press. While the user speaks, it may transcribe safe overlapping chunks in the background. It must not paste partial text into the focused input field. On the second F2 press, it stops recording, finalizes only the unconfirmed tail, copies the final text to the clipboard, and pastes once through the existing clipboard/paste path.

## Current Baseline

The current app records the full session, then transcribes the full audio after the second F2 press. That is reliable, but long recordings have a large stop-time delay because no transcription work starts until the user is finished speaking.

The current optimized batch settings should stay in place:

- `beam_size=1`
- `best_of=1`
- `temperature=0.0`
- `condition_on_previous_text=False`
- `without_timestamps=True` for ordinary final batch transcription
- current GPU-to-CPU fallback behavior

## References Reviewed

Buzz is the most useful reference. Its live recording path separates audio capture from transcription work, uses bounded buffering, cuts chunks near silence when possible, transcribes overlapping chunks, and hides unconfirmed overlap text until later chunks confirm it. The implementation should adopt those ideas, not the Buzz GUI architecture.

References:

- https://github.com/chidiwilliams/buzz?tab=readme-ov-file
- https://chidiwilliams.github.io/buzz/docs/usage/live_recording
- https://github.com/chidiwilliams/buzz/blob/main/buzz/transcriber/recording_transcriber.py
- https://github.com/chidiwilliams/buzz/blob/main/buzz/widgets/recording_transcriber_widget.py

`whisper-typing` is less useful for this goal because it retranscribes the full growing buffer repeatedly for preview. That gives visible preview text but wastes CPU/GPU work and can become worse as recordings get long. It does have a simple lock-protected audio snapshot pattern and event-based thread cancellation that are worth mirroring.

References:

- https://github.com/rpfilomeno/whisper-typing
- https://github.com/rpfilomeno/whisper-typing/blob/main/src/whisper_typing/app_controller.py
- https://github.com/rpfilomeno/whisper-typing/blob/main/src/whisper_typing/audio_capture.py

`C:\Users\11\Apps\turbo-whisper` is also batch-oriented. It has useful tray-app patterns and a clean final copy/paste flow, but it does not provide a streaming transcription algorithm to reuse. Its HTTP transcription backend and PyAudio recorder should not replace this app's local `faster-whisper` and `sounddevice` stack.

## User Experience

The F2 behavior remains:

1. First F2 starts recording.
2. The tray shows recording.
3. Background streaming work may run invisibly.
4. Second F2 stops recording.
5. The app finalizes the transcript.
6. The app copies the final transcript to the clipboard.
7. If `paste_on_stop` is enabled and the focused target is still safe, the app pastes once.

No partial text is inserted into the focused input field while the user is speaking. This avoids text flicker, cursor jumps, duplicated words, and accidental edits in the active app.

## Architecture

Add a new module, `voice_typer.streaming`, with small focused units:

- `StreamingConfig`: chunk/window timing, overlap, silence threshold, and safety settings.
- `AudioWindow`: one slice of 16 kHz audio plus its global start/end times.
- `AudioWindowPlanner`: decides when enough audio exists for another chunk and prefers silence-adjacent boundaries.
- `StreamingTextAssembler`: accepts timestamped words, commits only words that are outside the unsafe overlap/tail region, and tracks `last_committed_time`.
- `StreamingTranscriptionSession`: owns one background worker for the active recording session. It snapshots audio, plans chunks, transcribes them one at a time, stores confirmed text internally, and finalizes the unconfirmed tail when recording stops.

Existing modules get narrow additions:

- `voice_typer.recording.Recorder` adds a snapshot method that returns the current audio without stopping recording. It also shares the existing resampling code so snapshots and final audio are both 16 kHz.
- `voice_typer.transcription.TranscriptionEngine` adds a timestamped-word transcription method for streaming chunks, plus a lock so only one model call or model reload happens at a time.
- `voice_typer.app.VoiceTyperApp` starts/stops the streaming session around the existing recorder lifecycle and reuses the existing final clipboard/paste behavior.
- `voice_typer.config.Config` adds streaming settings with conservative defaults.

## Chunking Strategy

The streaming session should not use tiny 1-second chunks. Too-small chunks increase model overhead and make boundary errors more likely. The planned defaults are:

- first chunk after at least 6 seconds of audio
- chunk window around 12 seconds
- new chunk every 5 seconds
- 2 seconds of left overlap
- 1 second of right guard that remains unconfirmed until a later chunk or final stop
- bounded in-flight work: one transcription call at a time

When possible, the planner should cut near low audio energy in the last part of the planned chunk. This reduces the chance of splitting a word or phrase. If no good silence boundary exists, it should still make progress using the time boundary.

## Text Assembly

Streaming chunk transcription should request word timestamps. The assembler commits only words whose global end time is safely before the chunk's right guard. Words inside the unsafe tail remain provisional.

On final stop, the session transcribes only the tail starting slightly before `last_committed_time`. It drops any final words that overlap already committed time, appends the remaining words, and returns the final text.

This timestamp-first design avoids the weakest form of chunking: blindly concatenating chunk text. It also handles repeated phrases better than simple string deduplication because a repeated word at a new timestamp remains valid.

## Fallback Behavior

The existing batch path remains the safety net. The app should fall back to `transcribe_with_fallback(full_audio)` when:

- streaming setup fails
- a streaming chunk throws an unexpected exception
- timestamped words are unavailable or malformed
- the background worker falls too far behind
- final tail merge fails
- the env kill switch disables streaming

Fallback must preserve current reliability: copy/paste once, no partial paste, and `_busy` must always reset.

## Configuration And Rollout

Add config fields:

- `streaming_transcription: bool = False`
- `streaming_chunk_seconds: float = 12.0`
- `streaming_step_seconds: float = 5.0`
- `streaming_left_overlap_seconds: float = 2.0`
- `streaming_right_guard_seconds: float = 1.0`
- `streaming_min_first_chunk_seconds: float = 6.0`
- `streaming_silence_threshold: float = 0.003`

Defaulting `streaming_transcription` to `False` keeps the GitHub backup and batch behavior stable. After implementation and tests pass, the user's local config can opt in by setting it to `true`. This gives a rollback switch without code changes.

Also support `VOICE_TYPER_STREAMING=0` as an emergency kill switch even when config enables streaming.

## Concurrency Rules

Recording must stay responsive. The audio callback only copies audio into the recorder buffer. Snapshotting, resampling, chunk planning, and transcription all happen outside the callback.

Only one model call may run at a time. `TranscriptionEngine` owns a lock around model load, fallback reload, batch transcription, and timestamped streaming transcription. This prevents chunk transcription from racing with final fallback or GPU-to-CPU reload.

`VoiceTyperApp` should keep `_busy=False` while recording so the second F2 press can stop the recording. It sets `_busy=True` only after stop begins and finalization starts.

Quit and microphone changes must cancel or discard the active streaming session. Microphone changes should apply to the next session, not mutate the active recorder.

## Testing

Add focused unit tests before implementation:

- chunk planner waits until enough audio exists
- chunk planner produces overlapping windows
- chunk planner prefers silence near the requested boundary
- assembler commits only words before the right guard
- assembler does not duplicate overlap words
- assembler preserves legitimate repeated words at later timestamps
- streaming session finalizes only the uncommitted tail
- streaming session falls back to full batch transcription on chunk failure
- app starts a streaming session on first F2 when enabled
- app still stops on second F2 while background streaming is active
- partial streaming results never call clipboard copy or paste
- final text copies and pastes once through the existing path
- empty final text behaves like current no-speech behavior
- shutdown cancels an active streaming session
- env kill switch forces existing batch behavior

Run at least:

- `pytest tests/test_streaming.py -q`
- `pytest tests/test_app.py tests/test_recording.py tests/test_transcription.py -q`
- full `pytest -q`

## Success Criteria

For short recordings, behavior should match the current app. For long recordings, stop-time delay should be much shorter because most of the transcript is already confirmed before the second F2 press.

Reliability is more important than maximum eagerness. The app should never paste provisional text and should never lose already recorded audio. If streaming cannot prove it has a safe final transcript, it must use the current full-session batch transcription path.
