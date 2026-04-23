# Diagnostic Scripts

These are **ad-hoc interactive scripts**, not automated tests. They are excluded
from the pytest suite (`tests/`) because they require real hardware (GPU,
microphone, display) or print diagnostic output instead of making assertions.

| Script | Purpose |
|--------|---------|
| `diagnose_f2.py` | Traces the full F2 → recording → transcription path with mocked hardware |
| `runtime_proof.py` | End-to-end runtime verification of transcription fallback and stuck-state recovery |
| `cublas_fallback.py` | Proves the cuBLAS DLL failure path is handled correctly |

Run individually:

```bash
python diagnostics/diagnose_f2.py
python diagnostics/runtime_proof.py
python diagnostics/cublas_fallback.py
```

All unique test coverage from these scripts is already captured in the
automated test suite under `tests/`.
