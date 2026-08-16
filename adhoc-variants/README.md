# Ad-hoc variants

Composes for models the observer can serve that club-3090's registry does not
carry. Design: [../docs/2026-08-15-adhoc-variants.md](../docs/2026-08-15-adhoc-variants.md).

club-3090 is upstream and read-only for us — nothing here writes into that
checkout, and no path in these composes resolves inside it.

## Deploying one to aipc1

Stage the weights (~29 GB for Qwen3.8-27B FP8; `/` has 1.4 T free):

    ssh aipc1 'hf download Qwen/Qwen3.8-27B-FP8 \
      --local-dir /home/graywzc/models/hf/qwen3.8-27b-fp8'

Create the cache dirs the compose mounts:

    ssh aipc1 'sudo mkdir -p /var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/torch_compile \
      /var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/triton'

Copy the compose and the sidecar to /etc. (The deploy path below must stay
exactly as written — the sidecar's `compose_path` matches it character for
character. The `dual` subdirectory is just where this one happens to live;
it's the explicit `topology` field in the sidecar, not the path, that the
dashboard actually reads.)

    scp adhoc-variants/qwen3.8-27b-dual-fp8.yml aipc1:/tmp/
    scp adhoc-variants/aipc-observer-adhoc.example.json aipc1:/tmp/
    ssh aipc1 'sudo mkdir -p /etc/aipc-observer-adhoc/dual \
      && sudo mv /tmp/qwen3.8-27b-dual-fp8.yml /etc/aipc-observer-adhoc/dual/qwen3.8-27b-fp8.yml \
      && sudo mv /tmp/aipc-observer-adhoc.example.json /etc/aipc-observer-adhoc.json'

The observer only reads the sidecar path at startup, so a *new* sidecar file
(as above) needs an observer restart before it's picked up. *Edits* to an
already-loaded sidecar don't — they take effect on the next catalog poll by
themselves. Either way that poll runs at most every `REPO_POLL_INTERVAL`
(900 s = 15 min), so a plain restart-and-wait can take up to 15 minutes;
`POST /observer/api/update` wakes the poll immediately if you don't want to
wait. Once it's picked up, click the `ad-hoc` row on the variant page.

`weights_path` in the sidecar and the compose's `MODEL_DIR` mount are declared
independently — `boot_model_once` builds the compose environment from
`dict(os.environ)`, so if `MODEL_DIR` is set in the observer daemon's own
environment it silently overrides the mount. Make sure `MODEL_DIR` is unset
there, or that `weights_path` equals `$MODEL_DIR/<subdir>`; otherwise the
dashboard can say "staged" while vLLM can't find the model.

## Verifying

From the club-3090 checkout, driving by endpoint (the uncataloged-model path
their BRING_YOUR_OWN.md prescribes):

    MODEL=qwen3.8-27b URL=http://localhost:8020 bash scripts/verify-full.sh

## When upstream catalogs the model

1. `ln -s /home/graywzc/models/hf/qwen3.8-27b-fp8 ~/projects/club-3090/models-cache/`
   — their `.gitignore` covers `models-cache/` and accounts for symlinks, so no
   29 GB copy
2. Delete the entry from `/etc/aipc-observer-adhoc.json`

The registry entry wins over a same-keyed sidecar entry even before step 2, and
the observer logs that the sidecar entry is redundant. Port 8020 and the served
name `qwen3.8-27b` do not change, so nothing downstream notices.
