# llama.cpp request messages via `--log-prompts-dir`

## Why

On b9967 the request/response body stdout logging is compiled out
(`// srv->set_logger(log_server_request)`, upstream's "logs: reduce" campaign),
so the observer's Recent Requests card shows no prompt for any llama.cpp model.
Reverting the engine pin to b9570 (where the logger is enabled) would work but is
a downgrade — it forfeits b9967's validated +4 thinking-quality and re-adds the
log spam upstream removed.

Upstream's *own* replacement is `--log-prompts-dir` (llama.cpp #22031, 2026-06-09,
hardened #… 2026-07-08): the server writes one rendered-prompt file per request
to a directory. It's opt-in, file-based, and actively maintained — the sanctioned
way to capture prompts on modern builds. So we adopt it and stay on b9967.

## Design (no new thread — hook the existing correlation point)

`--log-prompts-dir` writes the prompt file when the request ARRIVES, before the
slot picks it up. The observer already correlates request→row at the
`launch_slot_` line. So we consume the file there, in the existing log-tail flow:

    request arrives → llama.cpp writes HOST_PROMPT_DIR/<id>  (before launch)
    "launch_slot_: task N" hits the docker log
      → RequestTracker consumes the OLDEST unconsumed prompt file,
        parses it, attaches to the row, deletes it

`-np 1` (every llama.cpp compose here) serializes requests, so oldest-file == the
launching request. FIFO by (mtime, name). No background thread, no poll race.

## Pieces

1. **Capability** `prompt_logging` → `--log-prompts-dir CONTAINER_PROMPT_DIR`,
   added to `MODE_CAPABILITIES["debug"]`. Rides the existing debug preset — no new
   operator knob. `resolve_preset` drops it gracefully on a build whose --help
   lacks the flag.
2. **Mount** — `boot_model_once` (llama.cpp branch) sets
   `preset_volumes=[HOST_PROMPT_DIR:CONTAINER_PROMPT_DIR]` when the capability
   survives, and resets HOST_PROMPT_DIR first (stale files must not mis-attach to
   the next model). Observer runs as root, so it creates + reads the root-owned
   files the container writes.
3. **Parse — reuse** — the file is rendered ChatML; route through the existing
   `_parse_chatml_messages` → `request_detail_metadata` (the "meaningful only"
   policy) → `request_group_metadata`. Session summary / user turns / tool
   previews all come for free.
4. **Prune** — delete-on-consume; reset-on-boot; an orphan cap
   (`PROMPT_DIR_ORPHAN_KEEP`) drops the oldest excess if a request errored before
   its `launch_slot_` (file written, never consumed).

## Scope / non-goals

- llama.cpp only. vLLM keeps `--enable-log-requests` (untouched).
- Stays on b9967 — no engine change, no club-3090 change. Deploys via the normal
  observer pipeline (merge → deploy-linux.yml).
- Images: the policy already stubs `[image]`; the file read is size-capped.
- Edge: if a prompt file isn't flushed when launch fires, that row gets no
  summary and the file is consumed by the next launch — rare, self-correcting.
