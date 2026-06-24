# Plan: Issue #106 — Stop model button + prevent watchdog revive

**Date:** 2026-06-24
**Branch:** `qwen/issue-106-stop-model`
**Status:** proposed (not yet implemented)

## Goal

On the observer variant page, replace the "running" label for the active model
with a **"Stop" button**. Clicking it gracefully brings down the model container.
The watchdog must **not** auto-revive a deliberately stopped model.

## Requirements

1. **UI change:** The variant row for the running model shows a "Stop" button
   instead of the text "running".
2. **Backend endpoint:** The existing `POST /observer/api/stop` endpoint already
   exists and works (`stop_model()` at line 1860). It calls `switch.sh --down`
   or `docker compose down`. No endpoint change is needed.
3. **Watchdog suppression:** When the user deliberately stops the model, the
   watchdog must not treat the resulting "down" state as a crash and must not
   attempt to revive it.
4. **UI recovery:** After a deliberate stop, the variant list should show the
   variant as no longer running (back to "start" button).
5. **Audit trail:** The stop action is already audit-logged by `stop_model()`.

## Design

### 3a. Backend: mark deliberate stop in watchdog state

Add a `deliberately_stopped` flag to `WatchdogState`. When set, the watchdog
skips all crash detection and revive logic until the model comes back up
(either via a manual start/switch or external means).

**Changes to `WatchdogState`:**

- Add `self.deliberately_stopped = False` in `__init__`.
- Add a new method `mark_deliberately_stopped()` that sets the flag.
- In `tick()`, if `self.deliberately_stopped` is True and status is "down",
  return immediately (no alarm, no revive).
- In `tick()`, if `self.deliberately_stopped` is True and status becomes
  "ready", clear the flag and reset normally (the model is back).
- Include `deliberately_stopped` in the `summary()` output so the dashboard
  can surface it.

**Changes to `stop_model()`:**

- After successfully stopping the container, call
  `_watchdog.mark_deliberately_stopped()`.

### 3b. Frontend: replace "running" label with "Stop" button

In the variant list rendering (around line 3973), the action for the running
variant is currently:

```javascript
let action=k===runKey?'<span class="good">running</span>':...
```

Change it to:

```javascript
let action=k===runKey?'<button class="btn" onclick="doStop()">Stop</button>':...
```

The `doStop()` function already exists (line 3956) and posts to
`POST /observer/api/stop`. The existing confirm dialog and in-flight-warning
are sufficient.

After the stop completes, the SSE update will refresh the variant list with the
new state (no container → "start" button reappears).

### 3c. Catalog card: reflect "stopped" state

The catalog card (around line 4003) currently shows "Running variant" when
`runKey` is set. After a deliberate stop, `d.container` is falsy, so this
card naturally disappears — no change needed.

## Files to modify

| File | Change |
|------|--------|
| `aipc_observer.py` | Add `deliberately_stopped` to `WatchdogState`; add `mark_deliberately_stopped()`; update `tick()`; update `summary()`; call `_watchdog.mark_deliberately_stopped()` in `stop_model()` on success; change "running" label to "Stop" button in JS |
| `tests/test_observer.py` | Add tests for deliberate stop suppression (see below) |

## Tests to add

Add to `tests/test_observer.py` in the `WatchdogStateTests` class:

1. **`test_deliberately_stopped_suppresses_alarm`** — Set `seen_healthy=True`,
   then call `mark_deliberately_stopped()`, then tick "down" past grace.
   Verify: no "down" event, no revive, `deliberately_stopped` remains True.

2. **`test_deliberately_stopped_clears_on_recovery`** — After
   `mark_deliberately_stopped()`, tick "ready". Verify: flag cleared,
   watchdog re-armed.

3. **`test_deliberately_stopped_in_summary`** — Verify `summary()` includes
   `deliberately_stopped: true` after the flag is set.

4. **`test_stop_model_sets_watchdog_flag`** — Mock `runner` to succeed,
   verify `_watchdog.deliberately_stopped` is True after the call.

## Verification commands

```bash
# Run all observer tests
python -m pytest tests/test_observer.py -v

# Run just the new watchdog tests
python -m pytest tests/test_observer.py -v -k "deliberately_stopped"
```

## Agent communication rules

- **Before starting:** Read this plan and comment `▶ In progress` under the
  section you are implementing.
- **After finishing:** Comment `✓ Done` under your section and note any
  deviations from the plan.
- **If you discover a needed change:** Update this plan, commit it, and note
  the change before proceeding.
- **Do not modify** files outside the scope listed above without updating this
  plan first.

## Risks and uncertainties

1. **`stop_model()` runs synchronously in the handler** — the HTTP response
   returns before the container is fully down. The watchdog sees "down" almost
   immediately after. The `mark_deliberately_stopped()` call must happen
   *before* the container actually stops (i.e., before calling `runner()`),
   not after, to avoid a race where the watchdog fires between the stop and
   the flag being set. **Decision:** set the flag *before* calling `runner()`
   in `stop_model()`.

2. **The `doStop()` JS function already exists** but is only wired to the
   standalone stop button (not the variant list). Verify it works in the
   variant-list context — the confirm dialog and `ctlPost` call are
   already correct.

3. **External revive:** If someone manually starts the model (e.g., via SSH),
   the flag clears on the next "ready" tick — correct behavior.

## Progress

- [x] Plan authored and committed
- [x] Backend: `WatchdogState` changes (`mark_deliberately_stopped`, tick guard, summary) — ✓ Done
- [x] Backend: `stop_model()` calls `_watchdog.mark_deliberately_stopped()` before stopping — ✓ Done; review revision clears flag if stop runner raises
- [x] Frontend: replace "running" label with "Stop" button in variant list — ✓ Done
- [x] Tests: add deliberate-stop suppression tests — ✓ Done; review revision covers stop-failure clearing
- [x] Verification: all available tests pass — ✓ Done

## Completion Notes

- Implemented deliberate-stop watchdog state. `tick()` now clears the flag on
  `ready` and suppresses down-state crash detection/revive while the flag is set.
- `stop_model()` now checks for a running container before either stop path,
  marks the watchdog deliberate-stop flag before invoking the stop command, and
  leaves the flag unset for no-op stops.
- The variant-list running row now renders a `Stop` button that calls the
  existing `doStop()` function.
- Added watchdog and stop-model tests, including an ordering assertion that the
  flag is set before the stop runner is called.
- Deviation: `python` and `pytest` were unavailable in this worktree shell
  (`python`: command not found; `python3 -m pytest`: no module named `pytest`).
  Ran `python3 -m unittest tests.test_observer -v` and
  `python3 -m py_compile aipc_observer.py tests/test_observer.py` instead; both
  passed.

## Review Revision Notes

- Finding: `stop_model()` set `_watchdog.deliberately_stopped` before invoking
  the stop runner, but a runner exception left the flag set even though the stop
  failed.
- Resolution: both stop paths now keep the pre-runner flag set for race
  protection, clear it in `except`, and re-raise the original exception.
- Added tests for failed `switch.sh --down` and failed docker compose `down`
  paths. Each test verifies the flag is set before the runner raises and cleared
  afterward.
- Verification: `python3 -m unittest tests.test_observer -v`,
  `python3 -m py_compile aipc_observer.py tests/test_observer.py`, and
  `git diff --check origin/main...HEAD` passed.
