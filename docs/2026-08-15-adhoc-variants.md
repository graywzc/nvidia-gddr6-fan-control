# Ad-hoc variants — serve a model club-3090 doesn't catalog yet

## Why

Qwen released [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on 2026-08-14.
club-3090 doesn't catalog it (the registry stops at `qwen3.6-*`, `gemma-4-*`,
`qwen3-omni-30b-a3b`), and we don't own that repo — `noonghunna/club-3090` is
upstream, read-only for us. So there is no Switch button for it and, on the
current design, never will be until upstream onboards it.

But the observer doesn't actually need upstream's blessing to boot a model.
`/observer/api/switch` → `_switch_worker` → `boot_model_once(repo, key, entry, …)`
resolves everything it needs from `entry["compose_path"]`:

    docker compose -f <compose_path> config --format json   → image, service, command
    docker compose -f <compose_path> up -d                  → boot

`switch_model()` (the `scripts/switch.sh` wrapper, which *does* require a registry
slug) is not on that path. Any catalog entry with a valid `compose_path` boots
through the existing machinery unchanged.

So this is a **catalog injection** problem, not a UI-plumbing problem: give the
observer one more variant entry from a source that isn't upstream's registry.

## Design — a sidecar catalog merged into the local ref

`extract_catalog(repo, ref)` runs `git show <ref>:scripts/lib/profiles/compose_registry.py`
through an isolated subprocess and returns `{"variants": {...}, "defaults": {...}}`,
projecting exactly ten fields per entry:

    model  engine  workload  status  status_note
    max_ctx  compose_path  default_port  kv_format  tp

Ad-hoc entries are read from a sidecar JSON on the box and merged into that dict
using the same ten fields, so a merged entry is indistinguishable in shape from a
registry one and every downstream consumer (validate, boot, installed-assets,
compare) works without special-casing.

    /etc/aipc-observer-adhoc.json      ← sidecar, hand-edited, one entry per ad-hoc model

`/etc` matches the precedent already on aipc1 (`/etc/aipc-observer-vllm-logging.json`
is mounted into the model container) and survives observer redeploys, which are
`scp` + restart with no git checkout.

### Three rules the merge must follow

1. **Local ref only.** Merge into the `HEAD` catalog, never the `@{upstream}`
   extraction. The dashboard diffs local vs upstream to show what upstream now
   recommends; ad-hoc entries in the upstream side would read as drift.
2. **Never shadow.** An ad-hoc key must not overwrite a registry key. If the key
   later appears in `compose_registry.py`, the real entry wins and the observer
   logs that the ad-hoc one is shadowed — that is the promotion signal, for free.
3. **`status: "experimental"`.** `validate_switch()` already rejects anything
   outside `("production", "caveats")` unless `force` is passed, and the UI has
   that affordance for the 🧪 variants. An unvalidated ad-hoc model inherits the
   existing confirm-to-proceed guard instead of getting a bypass.

### Install is disabled for ad-hoc entries

`/observer/api/install` drives upstream's `scripts/setup.sh <model>`, which
resolves weights through their ModelProfile YAMLs. It cannot know a model that
isn't in their registry. Ad-hoc weights are staged by hand, so the Install
affordance is hidden for these rows rather than left to fail confusingly.

## First ad-hoc variant — Qwen3.8-27B FP8, dual 3090

### Why this is nearly free

Qwen3.8-27B and Qwen3.6-27B are the *same architecture*. Diffing their
`config.json`: identical `architectures` (`Qwen3_5ForConditionalGeneration`),
`model_type` (`qwen3_5`), 64 layers / 5120 hidden / 24 heads / 4 KV heads,
`full_attention_interval: 4` (→ 48 GDN + 16 full-attention layers), head_dim 256,
linear 16 K-heads / 48 V-heads × 128, conv kernel 4, vocab 248320, ctx 262144,
`mtp_num_hidden_layers: 1`, vision depth 27. The only differing field is
`transformers_version` (4.57.1 → 5.8.0.dev0).

Every field `scripts/lib/profiles/models/qwen3.6-27b.yml` records therefore
matches, so upstream's KV math projects identically and the runtime shape of the
validated `vllm/qwen-27b-dual-max` transfers as-is.

### Endpoint contract (fixed now, unchanged through promotion)

| | value |
|---|---|
| host port | **8020** (the port the 3.6 dual uses today) |
| served-model-name | **`qwen3.8-27b`** |
| container | `vllm-qwen38-27b-dual-max` |
| sidecar key | `vllm/qwen38-27b-dual-max` |

Clients keep the URL and change the `model` field once, now. When upstream
catalogs the model, the registry entry lands on the same port and name and the
handover is invisible.

### Self-contained compose

Upstream's `models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml` mounts four things
by relative path out of their tree. Ours depends on none of them:

| Upstream depends on | Ours |
|---|---|
| `../../../../../../models-cache` | `MODEL_DIR=/home/graywzc/models/hf` |
| `../../../cache/{torch_compile,triton}` | `/var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/…` |
| `../../../patches/froggeric-chat-template/…` | dropped — use 3.8's own template |
| `scripts/detect_nvlink.sh` | dropped — hardcode `--disable-custom-all-reduce` |

The NVLink note deserves a line: aipc1 *does* have NVLink (`nvidia-smi topo -m`
reports `NV4`, P2P OK), but upstream's `.env` there sets
`DISABLE_CUSTOM_ALL_REDUCE=1` and the live 3.6 container runs with
`NCCL_P2P_DISABLE=1`. We match the empirically-working live configuration rather
than second-guess it, which also removes the dependency on their detect script.
An env knob flips it for later experimentation.

Inherited unchanged, because the architecture is identical: `vllm/vllm-openai:v0.22.0`
(≥ the 0.17.0 the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) requires),
TP=2, `--max-model-len 262144`, `--kv-cache-dtype int8_per_token_head`,
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`,
`--gpu-memory-utilization 0.92`, `--max-num-seqs 2`.

Changed for 3.8: `--override-generation-config` becomes
`temperature 0.7, top_p 0.80, top_k 20` (the model card's non-thinking
recommendation; 3.6's 0.6/0.95/20 no longer applies), and
`--default-chat-template-kwargs '{"enable_thinking": false}'` is dropped for the
first boot because 3.8's thinking control is a different surface
(`enable_thinking`, `preserve_thinking`, `reasoning_effort`).

### Artifacts and where they live

| What | Where |
|---|---|
| compose | `adhoc-variants/qwen3.8-27b-dual-fp8.yml` (this repo), deployed to `/etc/aipc-observer-adhoc/` on aipc1 |
| sidecar | `/etc/aipc-observer-adhoc.json` on aipc1, `compose_path` **absolute** |
| weights | `/home/graywzc/models/hf/qwen3.8-27b-fp8` (~29 GB; 1.4 T free on `/`) |
| caches | `/var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/{torch_compile,triton}` |

`compose_path` must be absolute: `boot_model_once` runs `docker compose -f` with
`cwd` set to the club-3090 repo, and our compose is no longer inside it.

## Bring-up

1. `hf download Qwen/Qwen3.8-27B-FP8 --local-dir /home/graywzc/models/hf/qwen3.8-27b-fp8`
2. Deploy the compose + sidecar to aipc1, restart the observer
3. Click the ad-hoc row in the variant page (confirm through the experimental guard)
4. Smoke it: `MODEL=qwen3.8-27b URL=http://localhost:8020 bash scripts/verify-full.sh`
   from the club-3090 checkout — driving by endpoint, which is what
   [BRING_YOUR_OWN.md](https://github.com/noonghunna/club-3090) prescribes for
   uncataloged models

Rollback is switching back to a club-3090 variant from the same page.

## Risks

- **Block-wise FP8 on Ampere is the likeliest first-boot failure.** 3.8's FP8 is
  fine-grained block quantization, block size 128; 3.6's was per-tensor/dynamic.
  Upstream's dual-max notes rely on the Marlin W8A16 path on sm_86, which expects
  per-tensor/per-channel scales. Block-128 may fall back to a slow Triton path or
  hard-error. Fallback lane: `cyankiwi/Qwen3.8-27B-AWQ-INT4` (~17 GB).
- **`transformers_version: 5.8.0.dev0`.** vLLM dispatches on `architectures` and
  won't care, but the tokenizer, multimodal processor, and chat template are
  still transformers-owned inside the image. Processor mismatch fails loudly;
  chat-template mismatch fails quietly as broken tool calls.
- **MTP head may be absent from the FP8 repo.** The card says "trained with
  multiple steps" but lists no `mtp.safetensors`. If `--speculative-config`
  fails to load, drop the flag and re-boot.
- **No simultaneous A/B.** TP=2 takes both cards, so 3.6 vs 3.8 comparisons are
  sequential.

## Promotion

When upstream catalogs the model:

1. `ln -s /home/graywzc/models/hf/qwen3.8-27b-fp8 ~/projects/club-3090/models-cache/`
   — their `.gitignore` covers `models-cache/` and explicitly accounts for
   symlinks to out-of-tree clones, so no 29 GB copy
2. Delete the sidecar entry; the never-shadow rule means the real registry entry
   takes over even before that
3. Port 8020 and served name `qwen3.8-27b` never change

The cost of keeping their tree clean: upstream authors the in-tree compose rather
than adopting ours. Given we don't own the repo, that was always their call.
