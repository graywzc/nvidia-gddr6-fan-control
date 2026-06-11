# Plan: model control plane — run, observe, update club-3090 from one place

**Date:** 2026-06-11
**Status:** proposed (not yet implemented)

## Goal

Today the model server is launched by hand on each host
(`PORT=8020 bash scripts/switch.sh beellama/dflash` in
`/home/graywzc/projects/club-3090`). The goal is to use this app as the
single place to **run** the model (club-3090 scripts + additional insight
flags), **observe** it (existing `/observer` dashboard), and **update**
club-3090 (git pull, since the repo advances quickly and we want to keep
tracing it).

## Does the design make sense?

Yes, with one framing clarification: the control logic should live in the
**Linux-side daemon** (`fan_control.py`), not in the macOS app itself. The
macOS menubar app is a thin HTTP client over Tailscale; the daemon already
runs on each GPU host next to docker and the club-3090 checkout, already has
an HTTP server, and already hosts the observer. So:

```
macOS menubar app / browser  ──HTTP (Tailscale)──►  fan_control.py on aipc/aipc1
                                                      ├─ fan control (existing)
                                                      ├─ observer (existing)
                                                      └─ model control plane (new)
                                                           ├─ club-3090 git status / pull
                                                           └─ launch/restart model via
                                                              club-3090 scripts + extra flags
```

Architecture principle: **club-3090 stays pristine** (a tracked upstream
clone, never locally modified), and this app layers additional flags on top
at launch time.

## Flag-injection mechanism (key technical choice)

Nearly every flag we want is settable via `LLAMA_ARG_*` environment
variables (confirmed in the binary's `--help`):

- `--metrics` → `LLAMA_ARG_ENDPOINT_METRICS=1`
- `-lv 4` → `LLAMA_ARG_LOG_VERBOSITY=4`
- `--props` → `LLAMA_ARG_ENDPOINT_PROPS=1`
- `--log-timestamps` → `LLAMA_ARG_LOG_TIMESTAMPS=1`
- `--cache-ram 8192` → `LLAMA_ARG_CACHE_RAM=8192`

So the cleanest injection path is a small **compose override file** (owned
by this app, stored outside club-3090) that only adds `environment:`
entries, applied as `docker compose -f dflash.yml -f override.yml up -d`.
No club-3090 file is ever edited.

**Discovery task (first implementation step):** read
`scripts/switch.sh` / `scripts/lib` to find the supported extension seam —
extra `-f` override, `COMPOSE_FILE` env, or `.env` passthrough. If switch.sh
has no seam, the daemon replicates only its final `docker compose` invocation
(the scripts print/centralize it) while still using switch.sh for everything
else (preflight, variant resolution).

## Phases

### Phase 1 — read-only (low risk, immediate value)

- `GET /model/info` on the daemon:
  - running container image, full command-line flags (`docker inspect`),
    resolved compose variant;
  - club-3090 version: `git describe` / HEAD commit, **behind-count** vs
    `origin/main` (via `git fetch` on a timer), and the new `CHANGELOG.md`
    section between HEAD and origin/main — this is the "keep tracing the
    repo" feature in passive form.
- Surface in the observer dashboard (new panel) and menubar app (model
  name + "club-3090: N commits behind").

### Phase 2 — control (guarded)

- `POST /model/update` — `git -C ~/projects/club-3090 pull --ff-only`,
  run **as user graywzc** (daemon runs as root; a root `git pull` would
  leave root-owned files in the user's repo — use `sudo -u graywzc` or
  `runuser`). Returns old→new commit range and CHANGELOG delta.
- `POST /model/restart` — body selects a **flag preset**, e.g.:
  - `baseline` — club-3090 verbatim;
  - `insight` — `+ metrics, props, -lv 4, log-timestamps`;
  - `insight+cache` — insight + `cache-ram 8192`.
  Implemented as: write override yml → re-run compose with both files.
- Guardrails:
  - explicit confirmation in the UI, never automatic;
  - **single-flight lock** per host (no concurrent restarts/pulls);
  - the daemon already knows in-flight requests via the observer — refuse
    (or warn) restart while a request is decoding;
  - append-only audit log of every control action;
  - control endpoints POST-only; Tailscale binding remains the auth
    boundary (consistent with the existing API design).

### Phase 3 — consume the new insight in the observer

- Poll `/metrics` → queue depth (deferred requests), server-side token
  counters; derive queue time / TTFT.
- Parse debug-verbosity logs → request bodies → hermes `session_id`
  grouping (the earlier grouping investigation's passive path).
- New dashboard panels: queue, cache-hit (now meaningful with cache-ram
  re-enabled), adaptive-DM activity, reasoning-loop-guard interventions.

### macOS app changes (small, last)

- Menubar: model name, queue depth, club-3090 behind-count.
- Buttons → "Update club-3090…" / "Restart model with preset…" with
  confirmation dialogs, calling the daemon endpoints.

## Alternatives considered

- **macOS app drives everything over ssh** — rejected: duplicates logic in
  every client, no audit trail, no synergy with observer state (e.g.
  "refuse restart while decoding"), and the daemon is already the
  host-side agent.
- **Editing club-3090 composes in place** — rejected: makes `git pull`
  conflict-prone and breaks the "trace upstream cleanly" goal.

## Open questions

- Exact switch.sh extension seam (Phase 2 discovery task).
- Whether the beellama fork logs request bodies at `-lv 5` like upstream
  (determines if hermes grouping needs a proxy at all).
- Preset persistence across host reboots: the override yml + a recorded
  "last preset" make `restart unless-stopped` keep the flags; document this.

## Related

- [2026-06-11-beellama-observability-opportunities.md](2026-06-11-beellama-observability-opportunities.md)
  — the flag/endpoint survey the presets are built from.
