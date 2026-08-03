# Recent Requests card: restore request/response detail for llama.cpp (b9967)

## Problem

On the observer's **Recent Requests** card, llama.cpp-family models (Deckard-40B,
Tess, HauhauCS, gemma-12b, vibethinker) render metric-only rows: Status/Time/PT/
tokens/TTFT populate, but the per-request detail modal shows *"No request body
captured"* and *"No response body captured yet"* — even under the **debug** preset.

vLLM models are unaffected (`VllmLogTracker` parses `--enable-log-requests`).

## Root cause

`RequestTracker` (the llama.cpp parser) targets an **older llama.cpp log format**.
The engine pin was bumped b9246 → **b9967** on 2026-07-11 (`engines/llama-cpp-mainline.yml`),
and the request/response body log shape changed. The stale matchers:

- `RE_REQUEST_BODY  = r"\brequest:\s*(\{.*\})\s*$"`
- `RE_RESPONSE_BODY = r"\bresponse:\s*(\{.*\})\s*$"`
- `RE_ACCESS        = r"done request:\s+..."`

match **nothing** in b9967 output. Metric lines (`slot print_timing`, `launch_slot_`)
still match, which is why rows appear with numbers but no bodies.

### What b9967 actually logs (verified on the live Deckard container)

At `--log-verbosity 5` (what the debug preset sets via the `trace_logging`
capability):

1. **Streaming responses** — one line per token delta:
   ```
   D srv operator(): http: streamed chunk: data: {"choices":[{"finish_reason":null,
   "index":0,"delta":{"content":"——"}}],"created":...,"id":"chatcmpl-<rid>","model":
   "deckard-40b","object":"chat.completion.chunk","timings":{"cache_n":601,
   "prompt_n":779,"prompt_ms":1456.4,"prompt_per_second":534.9,"predicted_n":695,
   "predicted_ms":22569.5,"predicted_per_second":30.79,"draft_n":714,
   "draft_n_accepted":338}}
   ```
   - `choices[0].delta.content` — the token text (accumulate for output)
   - `choices[0].finish_reason` — null until the terminal chunk (stop/length)
   - `timings` — cumulative; the **last** chunk carries final PT/CT/TPS/cache.
     Richer than today's parse: real `cache_n` (prefix-cache hit), separate
     prompt/predicted TPS, MTP `draft_n`/`draft_n_accepted` acceptance.

2. **Request messages** — NOT logged at verbosity 5 in any form. Confirmed with
   unique-marker probes (streaming and non-streaming): the request body text
   never appears. The llama.cpp source (`server-http.cpp::log_server_request`)
   emits `SRV_DBG("request:  %s")` / `SRV_DBG("response: %s")`, but these did not
   surface at `--log-verbosity 5` — they require `-v` / `--log-verbose`
   (verbosity = infinity). **Format under `-v` is still UNCONFIRMED** and needs a
   controlled capture (see Phase 2).

## Correlation note

Streamed-chunk lines carry the OpenAI `id` (`chatcmpl-…`), NOT the llama.cpp
integer task id that `RequestTracker` keys rows by (`launch_slot_: task N`). The
two identifiers are not linked in the logs. Deckard (and the dual llama.cpp
composes) run `-np 1` — a single decode slot, enforced by the compose ("do NOT
raise it"). So with one in-flight request at a time, streamed chunks attach
unambiguously to the single active row. Multi-slot llama.cpp (np>1) would need a
slot id on the chunk line, which b9967 does not emit — documented as a known
limitation; those composes are not in use.

## Plan

### Phase 1 — response output + timings (fully specified, no restart)
1. Add `RE_STREAM_CHUNK = r"http: streamed chunk: data:\s*(\{.*\})\s*$"`.
2. In `RequestTracker.process_line`, on a chunk:
   - parse JSON; pull `choices[0].delta.content`, `finish_reason`, `timings`.
   - attach to the single active request (np=1 assumption; if >1 active, skip and
     log once — never mis-attribute).
   - accumulate content into `response_output` (bounded by
     `VLLM_OUTPUT_PREVIEW_MAX` for parity).
   - on non-null `finish_reason`, set final PT/CT from `timings`
     (`prompt_n`/`predicted_n`), TPS (`prompt_per_second`/`predicted_per_second`),
     cache% (`cache_n`/`prompt_n`), MTP accept (`draft_n_accepted`/`draft_n`),
     finish reason, and finalize.
3. Keep the legacy `RE_RESPONSE_BODY` matcher (older builds / non-streaming) as a
   fallback — additive, not a replacement.

### Phase 2 — request messages (needs one controlled `-v` capture)
1. Safely capture `-v` output: disable the observer watchdog for the window,
   recreate Deckard with `-v` via a temp compose override, fire tagged probes,
   record the exact `request:  {…}` line shape, restore.
2. Add `-v` to the llama.cpp `trace_logging` capability (or a new capability) so
   debug mode emits request bodies.
3. Parse the request JSON `messages[]` → reuse the vLLM path's
   `_parse_chatml_messages` / `request_group_metadata` for label + grouping.

### Phase 3 — regression test
Unit test feeding captured b9967 lines through `RequestTracker`, asserting the
row gets `response_output`, PT/CT, cache%, and (Phase 2) messages + label. Guards
the next engine-pin bump from silently breaking this again.

## Out of scope / decisions
- No deploy to aipc1 until reviewed (user rule). Land on `feat/llamacpp-request-detail`.
- Frontend already engine-agnostic — no card/JS changes needed for Phase 1.
- Debug preset stays opt-in; a separate change could make it Deckard's default.
