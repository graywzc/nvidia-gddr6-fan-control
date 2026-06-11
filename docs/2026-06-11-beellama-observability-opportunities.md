# beellama.cpp / club-3090 observability opportunities

**Date:** 2026-06-11
**Scope:** survey of every lever available for getting more insight into the LLM
inference stack running on aipc/aipc1, gathered by inspecting the live container,
the server binary's `--help` (~718 lines), and the club-3090 repo on aipc.
Performance may be sacrificed for insight — that is the stated goal.

## What is running

- **Engine:** `beellama.cpp` (Anbeeld's llama.cpp fork), image
  `ghcr.io/noonghunna/beellama-cpp:multiarch-v0.3.0-efe856397`
  (club-3090's multi-arch build of Anbeeld's official image).
- **Compose:** `models/qwen3.6-27b/beellama/compose/single/beellama-q5ks-dflash/dflash.yml`
  in `/home/graywzc/projects/club-3090` (launched via
  `PORT=8020 bash scripts/switch.sh beellama/dflash`).
- **Config:** Qwen3.6-27B Q5_K_S + DFlash IQ4_XS drafter, 102 400 ctx,
  single slot (`-np 1`), unified KV, q5_0/q4_1 KV quant, flash-attn on,
  reasoning off, **`--cache-ram 0`**.

## Two discoveries that explain earlier measurements

1. **`--cache-ram 0` disables the host-RAM prompt cache entirely.**
   This is almost certainly why the observer's log replay showed ~75 % of
   requests hitting `forcing full prompt re-processing due to lack of cache
   data` ("cache defeated"). It also disables `--cache-idle-slots`, which
   requires cache-ram. Re-enabling (default `8192` MiB, or `-1` unlimited)
   is both a likely perf win for multi-turn hermes work and a prerequisite
   for meaningful prompt-cache stats.
2. **`-np 1` means all concurrent requests queue.** Intentional per the
   compose comments (DFlash is single-slot by default; extra slots divide a
   compute-bound GPU), but it makes queue time a real, large quantity —
   measurable via `--metrics` below.

## Tier 1 — flag flips (restart-only, no code changes)

| Flag | What it unlocks | Env-var form |
|---|---|---|
| `--metrics` | Prometheus `/metrics`: **deferred (queued) requests**, processing count, cumulative prompt/generated tokens & seconds, KV usage. Fills the queue-time/TTFT gap server-side. | `LLAMA_ARG_ENDPOINT_METRICS=1` |
| `-lv 4` (trace) / `-lv 5` (debug) | Per-batch scheduling, slot decisions; upstream logs **full request/response JSON** at debug — would let the observer read a `user`/session field from logs and group hermes requests passively. Verify the fork kept this. | `LLAMA_ARG_LOG_VERBOSITY=4` |
| `--props` | `POST /props` — flip global properties at runtime and observe the effect. | `LLAMA_ARG_ENDPOINT_PROPS=1` |
| `--perf` | Internal libllama performance timings. | — |
| `--slot-save-path DIR` | Snapshot a slot's KV cache to disk for inspection. | — |
| `--log-timestamps --log-prefix --log-file F` | Precise native timestamps / durable logs instead of relying on docker's. | `LLAMA_ARG_LOG_TIMESTAMPS=1` etc. |
| `--cache-ram 8192` (or `-1`) | Re-enables prompt cache (see discovery #1). | `LLAMA_ARG_CACHE_RAM=8192` |

Recommended first move: `--metrics --props -lv 4 --log-timestamps` in one
restart, then check (a) `/metrics` queue depth, (b) whether debug logs include
request bodies (solves hermes session grouping without a proxy).

## Tier 2 — fork-specific signals worth surfacing in the observer

- **Adaptive draft-max controller** (`--spec-dm-*`): profit controller with
  EWMA stats, probe cycles, off-dwell — its raise/lower decisions appear in
  logs at higher verbosity.
- **Reasoning loop guard** (`--reasoning-loop-guard force-close`, active per
  `/slots` params): intervention events are detectable → good health stat.
- **Context checkpoints** (`-ctxcp`, 32/slot, min step 256) and context-shift
  events — partially parsed already; trace verbosity shows more.
- `/slots` already exposes `speculative`, `n_prompt_tokens_cache` /
  `_processed` splits — mostly consumed by the observer already.

## Tier 3 — infrastructure (more effort, deepest insight)

- **Thin reverse-proxy in front of :8020** — full request/response bodies,
  true arrival time (real queue time + TTFT), exact session grouping.
  Try Tier-1 verbose logging first; it may cover 80 % of this.
- **tcpdump/pcap on port 8020** — traffic is plaintext HTTP; capture request
  bodies *without restarting or proxying*. Zero inference-path risk; good
  for one-off inspection.
- **Build the fork with custom log lines** — club-3090 already publishes its
  own multiarch build (`ghcr.io/noonghunna/beellama-cpp`), so a build
  pipeline exists. Best pure-learning tool.
- **GPU-level:** `nvidia-smi dmon -s pucvmet`, DCGM exporter, or
  `nsys profile` of the container for kernel-level traces.

## Already in club-3090, untapped

- Diagnostics/bench suite: `scripts/diagnose-profile.sh`, `soak-test.sh`,
  `bench-agentic.sh`, `verify-full.sh`, `health.sh`, `BENCHMARKS.md`.
- Docs worth reading for the learning goal: `docs/KV_MATH.md`,
  `docs/CLIFFS.md`, `docs/LOOP.md`, `docs/INFERENCE_ENGINES.md`.
- `tools/residency-instrument/` — vLLM-only monkey-patching harness, but a
  good template for observational-only instrumentation.

## Known observer gaps these would close

- **Queue/wait time & TTFT:** logs have no request-arrival timestamp; `/metrics`
  deferred-request gauge (Tier 1) or a proxy (Tier 3) provides it.
- **Hermes request grouping:** hermes already has `session_id` + `extra_body`;
  debug request-body logging (Tier 1) or a proxy (Tier 3) surfaces it.
- **Prompt-cache hit stats:** currently meaningless because the cache is
  disabled (`--cache-ram 0`).
