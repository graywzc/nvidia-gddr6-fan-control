# vLLM cache visibility + polish bundle

Date: 2026-07-15
Branch: `claude/debug-recent-request`

## Motivation

Diagnosing the token-bomb conversation on aipc1 (PT≈187k, TTFT≈195 s,
prefix-cache hit 1.7%) required manual Prometheus counter math. The
dashboard should surface prompt-caching behavior directly, per request
and over time. Two smaller gaps found in the same investigation ride
along.

## Features

### 1. Per-request cache hit % for vLLM rows

vLLM exposes no per-request cache data, but its cumulative counters
(`prefix_cache_queries_total` / `hits_total`) tick when a request is
*scheduled*. The observer scrapes /metrics every 2 s, so the counter
delta between consecutive scrapes belongs to whichever request(s)
entered prefill in that window.

- `summarize_vllm_metrics` gains `prefix_cache_queries_delta` /
  `prefix_cache_hits_delta` (0 when no prev scrape).
- New `ObserverState.attribute_vllm_cache(delta_q, delta_h)`:
  - candidates = active vLLM rows (`request_id` set) without
    `cache_hit_pct`.
  - Exactly one candidate → assign `cache_hit_pct = 100·Δh/Δq`,
    `cached_tokens = Δh`.
  - Several candidates → assign only if Δq matches exactly one
    candidate's `prompt_tokens` within 15%; else skip (never guess).
- Attribution mutates the shared row dict, so the value survives onto
  the completed row. UI: the existing Cache column just lights up.

### 2. Windowed hit rate: Server Metrics row + timeline series

- `vllm_timeline_sample` carries `cq`/`ch` (per-scrape deltas).
- Server Metrics card: "Prefix cache hit (5m)" row — sum of deltas over
  samples newer than 300 s, shown beside the cumulative figure.
- vLLM activity timeline: rolling hit-% series from the same samples.

### 3. Derived P t/s on vLLM rows

For vLLM, TTFT ≈ prefill time (arrival → first streaming delta). Once
the first delta lands and PT is known (DEBUG details line), set
`prompt_tps = prompt_tokens / (ttft_ms/1000)` — on the live row and
carried to completion. Non-streaming requests have no delta timing and
keep "-". Caveat accepted: TTFT includes queue wait, so queued
requests understate the rate.

### 4. Stop polling /slots and /props under vLLM

`poll_slots` (and its `detect_n_ctx` call) target llama.cpp-only
endpoints; under vLLM they 404 twice every 2 s in the access log —
the last remaining log noise after #127. Gate each loop iteration on
`infer_engine(state.model_info) != "vllm"`, re-evaluated per tick so
engine switches self-correct.

## Non-goals

- Live P t/s during prefill (vLLM exposes no per-request progress;
  KV-growth attribution lies under concurrency).
- Cache attribution under concurrent admission (shows "-" instead).

## Verification

TDD per feature; full suite; live deploy gated on review, then
re-check on aipc1: Cache column fills on next Hermes round, 404s stop,
timeline shows hit-rate series.
