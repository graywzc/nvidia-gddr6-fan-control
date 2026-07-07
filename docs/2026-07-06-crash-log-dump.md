# Crash-evidence log dump before watchdog revive

## Context

On 2026-07-06 the vLLM container on aipc1 died from simultaneous Xid 31 GPU MMU
faults. The watchdog revived it correctly, but the revive path recreates the
container, which destroys its `docker logs` stream — the Python traceback and
in-flight request were lost. Only `dmesg` evidence survived. Recovery must not
destroy forensics: before the first revive attempt of a crash episode, dump the
tail of the crashed container's logs to a file on disk.

## Implementation (all in `aipc_observer.py` + `tests/test_observer.py`)

### 1. Constants (next to the existing watchdog constants ~line 76)

```python
CRASH_DUMP_DIR = "/var/log/aipc-observer-crash-dumps"
CRASH_DUMP_TAIL = 500   # lines of docker logs to capture
CRASH_DUMP_KEEP = 20    # newest dump files retained; older pruned
```

### 2. `dump_crash_logs(...)` — module-level function near `classify_crash`

```python
def dump_crash_logs(container_name, buffered_logs, reason=None, model=None,
                    dump_dir=CRASH_DUMP_DIR):
    """Persist crash evidence to a file before the revive destroys it.
    Returns the path written, or None. Must never raise."""
```

Behavior:

- `os.makedirs(dump_dir, exist_ok=True)`.
- Filename: `{YYYYmmdd-HHMMSS}-{container_name or 'unknown'}.log`.
- **Primary source**: `docker logs --tail CRASH_DUMP_TAIL <container>` via
  `subprocess.run([...], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
  text=True, timeout=30)`. Merging stderr into stdout is REQUIRED — vLLM and
  llama.cpp write their logs to stderr (mirror `tail_docker_logs`, which uses
  `stderr=subprocess.STDOUT` for the same reason). Do NOT use the `_run` helper
  (it returns stdout only and raises on nonzero exit).
- **Fallback**: if `container_name` is falsy, the command fails, times out, or
  produces no output, write `buffered_logs` (the observer's in-memory
  `state.docker_logs` ring buffer) instead.
- File starts with a short header: ISO timestamp, container name, reason,
  model, and which source was used (`docker-logs` or `ring-buffer`), then the
  log lines.
- **Retention**: after writing, list `*.log` in `dump_dir` sorted by name and
  delete the oldest beyond `CRASH_DUMP_KEEP`.
- Any exception anywhere → return `None` (best-effort; a dump failure must
  never block the revive).

### 3. Wire into the watchdog state machine

- Add `_dump_crash_logs_now()` next to `_classify_crash_now()`:
  `return dump_crash_logs(state.container_name, list(state.docker_logs), reason=..., model=...)`
  (reason/model may be passed by the caller; keep the signature simple).
- `WatchdogState.tick(...)`: add an injectable `dump=None` parameter
  (default `_dump_crash_logs_now`), consistent with `classify`/`revive`/`notify`.
- In the crash-confirmed block (`if self.armed:` — where `classify()` runs and
  the "down" alert fires), call `dump()` **once per crash episode**, right
  after `classify()`. Store the result on `self.last_dump`, reset it in
  `_reset()`, and:
  - `audit("watchdog", f"crash logs dumped: {path}")` when a path came back
    (no audit line when dump returned None).
  - include `"last_dump"` in `summary()`.
  - pass `dump=path` through to `notify("down", ...)`, and in
    `_watchdog_notify` append `Logs: {path}.` to the "down" message when set.
- Revive attempts 2 and 3 must NOT dump again (they'd capture the new booting
  container, not the crash). Do not change any revive/backoff/grace behavior.

### 4. Tests (`tests/test_observer.py`, follow existing conventions)

Extend `WatchdogStateTests` (see `test_healthy_then_crash_alerts_and_revives_once`
for the injectable-tick pattern):

- dump is called exactly once per crash episode, even across 3 revive attempts;
  called again in a *new* episode after recovery.
- dump is NOT called: before first health, during `loading`, when
  `control_busy`, or after a deliberate stop.
- `summary()` exposes `last_dump`; "down" notification includes the path.

New `DumpCrashLogsTests` using `tempfile.TemporaryDirectory`:

- happy path: fake `subprocess.run` (patch) returning merged output → file
  written, header says `docker-logs`, content present, returns path.
- fallback: docker command raises / returns empty / container_name None →
  ring-buffer lines written, header says `ring-buffer`.
- retention: with `CRASH_DUMP_KEEP` exceeded, oldest files pruned.
- never raises: unwritable `dump_dir` → returns `None`.

## Constraints for the implementing agent

- You can edit files but CANNOT run shell commands — do not try; the reviewer
  runs the tests.
- Keep changes minimal and localized; match the file's existing comment
  density and docstring style.
- Do not touch the dashboard JS/HTML.
