# Fix GGUF install verify failure in the unified start flow

## Context

The unified Start/Switch button always runs club-3090's `setup.sh` before booting
("always ensure"). For **GGUF** variants (llama.cpp / ik-llama / beellama) that step
falsely fails: `setup.sh`'s verify is hardcoded to check `*.safetensors`, finds none
next to the `.gguf`, and aborts with "No *.safetensors found … download may have
failed" — even though the weights are present and intact (confirmed: the 16 GB
`Qwen3.6-27B-MTP-IQ4_KS.gguf` is on disk on aipc). Result: **no GGUF variant can
start via the unified button.** The old separate Start button used `switch.sh` (no
verify) so it worked — this is a regression the "always ensure" design exposed.

**We do not own club-3090, so the fix is observer-side**, not in `setup.sh`.
`setup.sh` already honors documented env overrides for the verify glob —
`VERIFY_GLOB_OVERRIDE` (main-model path) and `WEIGHT_VERIFY_GLOB` (weight-key path),
each defaulting to `*.safetensors`. The observer already sets `setup.sh` env
(`MODEL_DIR`, `WEIGHT_KEY`); we extend that to also set the verify glob to `*.gguf`
when installing a GGUF variant. vLLM/safetensors and SGLang variants are untouched.

## Change — `aipc_observer.py`

### 1. GGUF detection helper
Add `variant_is_gguf(entry)` that classifies a catalog variant by engine, reusing
the marker vocabulary already in `infer_engine`. Build a lowercase blob from the
entry's `engine` + `compose_path` (+ `model`) and return True when it matches a
llama.cpp-family marker: `("llamacpp", "llama.cpp", "ik-llama", "ik_llama",
"beellama")`. Catalog entries carry `engine`/`compose_path`, so the failing
`ik-llama/iq4ks-mtp` entry matches; `vllm` and `sglang` do not.

### 2. Set the verify-glob override in `install_variant_assets`
After the existing `env` is built (where `MODEL_DIR`/`WEIGHT_KEY` are set), add:

```python
if variant_is_gguf(entry):
    env["VERIFY_GLOB_OVERRIDE"] = "*.gguf"
    env["WEIGHT_VERIFY_GLOB"] = "*.gguf"
```

This makes `setup.sh` SHA-verify the `.gguf` (+ `mmproj.gguf`) instead of
nonexistent safetensors, so a present GGUF passes and a genuinely missing one still
fails correctly. Reflect it in the `audit`/`detail` string for traceability.

### Known coupling
Relies on club-3090's env-var contract (`VERIFY_GLOB_OVERRIDE` /
`WEIGHT_VERIFY_GLOB`), which we don't control — the same kind of dependency we
already have on `MODEL_DIR`/`WEIGHT_KEY`. Documented here so it's findable if
club-3090 ever renames them.

## Tests — `tests/test_observer.py`
- `variant_is_gguf`: True for entries with engine `ik-llama`/`llamacpp`/`beellama`;
  False for `vllm`/`sglang`/empty.
- `install_variant_assets` with a **GGUF** catalog entry + a `FakeRunner` (records
  `env` per call): assert the captured env has `VERIFY_GLOB_OVERRIDE == "*.gguf"`
  and `WEIGHT_VERIFY_GLOB == "*.gguf"`.
- Same with a **vLLM** entry: assert neither key is set (verify stays
  `*.safetensors`).
Reuse the existing `FakeRunner` + `SWITCH_CATALOG` fixtures already in the test file.

## Verification
1. `python3 -m unittest tests.test_observer` — all green, including the new tests.
2. Live on **aipc**: click the unified **start** on `ik-llama/iq4ks-mtp` (or POST
   `/observer/api/install {variant, retry:true}`). The Last-start card should show
   the verify step now checking `*.gguf`, pass (weights already present), proceed to
   switch/restart, and the model come up — no more "No *.safetensors found." Confirm
   a **vLLM** variant (aipc1) still installs/starts normally.
