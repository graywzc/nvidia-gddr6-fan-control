# Ad-hoc Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ad-hoc row to the observer's variant page that boots Qwen3.8-27B FP8 on dual 3090, without creating or modifying a single file in the club-3090 checkout.

**Architecture:** The observer's Switch button already boots from `entry["compose_path"]` via `boot_model_once` — it does not need a club-3090 registry slug. So we read extra variant entries from a sidecar JSON on the box (`/etc/aipc-observer-adhoc.json`) and merge them into the locally-extracted catalog, never shadowing a real registry key. The compose those entries point at is self-contained and lives outside club-3090.

**Tech Stack:** Python 3 stdlib (`aipc_observer.py` is a single-file daemon, no third-party deps), `unittest` (`tests/test_observer.py`), vanilla JS embedded in the same Python file as a dashboard template string, Docker Compose v2, vLLM `v0.22.0`.

Design doc: [docs/2026-08-15-adhoc-variants.md](2026-08-15-adhoc-variants.md)

## Global Constraints

- **club-3090 is read-only.** No file may be created, modified, moved, or deleted under `~/projects/club-3090`. Reading it (the existing `git show` catalog extraction) is fine.
- **No third-party imports.** `aipc_observer.py` uses stdlib only. Do not add dependencies.
- Sidecar path is exactly `/etc/aipc-observer-adhoc.json`.
- Ad-hoc `compose_path` values MUST be absolute. `boot_model_once` runs `docker compose -f` with `cwd` set to the club-3090 repo, so a relative path would resolve inside their tree.
- Ad-hoc entries always get `status: "experimental"` regardless of what the sidecar says — a hand-edited local file must not be able to claim `production` and skip `validate_switch()`'s force-confirm guard.
- An ad-hoc key must never overwrite a registry key.
- Ad-hoc entries must never appear in the local-vs-upstream catalog diff.
- Endpoint contract for the first ad-hoc variant, fixed and unchanging through eventual promotion: host port **8020**, `--served-model-name` **`qwen3.8-27b`**, container **`vllm-qwen38-27b-dual-max`**, sidecar key **`vllm/qwen38-27b-dual-max`**.
- Run tests with: `PYTHONPATH=. python3 -m unittest tests.test_observer -v` from the repo root. (`pytest` is not installed in this environment; read every per-task `pytest tests/test_observer.py[::SomeClass]` command below as its `unittest` equivalent, e.g. `PYTHONPATH=. python3 -m unittest tests.test_observer.SomeClass -v`.)

---

### Task 1: Sidecar loader and never-shadow merge

Two pure functions with no I/O beyond reading one file. Everything else in this plan depends on their exact names and return shapes.

**Files:**
- Modify: `aipc_observer.py` (add constant near `VLLM_LOGGING_CONFIG_FILE` at line 1542; add functions immediately after `extract_catalog`, which ends at line ~1345)
- Test: `tests/test_observer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ADHOC_CATALOG_FILE: str` — `"/etc/aipc-observer-adhoc.json"`
  - `load_adhoc_variants(path: str = ADHOC_CATALOG_FILE) -> dict[str, dict]` — key → entry dict. Never raises. Each entry carries the ten registry fields plus `adhoc: True`, `weights_path: str | None`, and `topology: str`.
  - `merge_adhoc_variants(catalog: dict, adhoc: dict) -> dict` — mutates and returns `catalog`.

- [ ] **Step 1: Write the failing tests**

Add this class to `tests/test_observer.py`, after the existing `CatalogDiffTests` class (which ends around line 1042):

```python
class AdhocVariantTests(unittest.TestCase):
    def _write(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(payload if isinstance(payload, str) else json.dumps(payload))
        self.addCleanup(os.unlink, path)
        return path

    def _entry(self, **over):
        entry = {
            "model": "qwen3.8-27b", "engine": "vllm",
            "workload": "long-ctx-single", "status": "experimental",
            "status_note": "ad-hoc", "max_ctx": 262144,
            "compose_path": "/etc/aipc-observer-adhoc/q38.yml",
            "default_port": 8020, "kv_format": "int8_per_token_head", "tp": 2,
            "weights_path": "/home/graywzc/models/hf/qwen3.8-27b-fp8",
        }
        entry.update(over)
        return entry

    def test_missing_file_yields_no_variants(self):
        got = aipc_observer.load_adhoc_variants("/nonexistent/adhoc.json")
        self.assertEqual(got, {})

    def test_malformed_json_yields_no_variants(self):
        path = self._write("{not json")
        self.assertEqual(aipc_observer.load_adhoc_variants(path), {})

    def test_valid_entry_is_loaded_and_marked_adhoc(self):
        path = self._write({"variants": {"vllm/q38": self._entry()}})
        got = aipc_observer.load_adhoc_variants(path)
        self.assertIn("vllm/q38", got)
        entry = got["vllm/q38"]
        self.assertTrue(entry["adhoc"])
        self.assertEqual(entry["model"], "qwen3.8-27b")
        self.assertEqual(entry["default_port"], 8020)
        self.assertEqual(entry["compose_path"],
                         "/etc/aipc-observer-adhoc/q38.yml")
        self.assertEqual(entry["weights_path"],
                         "/home/graywzc/models/hf/qwen3.8-27b-fp8")

    def test_relative_compose_path_is_rejected(self):
        path = self._write({"variants": {
            "vllm/q38": self._entry(compose_path="models/q38/compose.yml")}})
        self.assertEqual(aipc_observer.load_adhoc_variants(path), {})

    def test_status_is_forced_to_experimental(self):
        path = self._write({"variants": {
            "vllm/q38": self._entry(status="production")}})
        got = aipc_observer.load_adhoc_variants(path)
        self.assertEqual(got["vllm/q38"]["status"], "experimental")

    def test_topology_derived_from_tp_when_absent(self):
        path = self._write({"variants": {
            "a": self._entry(tp=1), "b": self._entry(tp=2),
            "c": self._entry(tp=4)}})
        got = aipc_observer.load_adhoc_variants(path)
        self.assertEqual(got["a"]["topology"], "single")
        self.assertEqual(got["b"]["topology"], "dual")
        self.assertEqual(got["c"]["topology"], "multi4")

    def test_explicit_topology_wins_over_tp(self):
        path = self._write({"variants": {
            "a": self._entry(tp=2, topology="single")}})
        got = aipc_observer.load_adhoc_variants(path)
        self.assertEqual(got["a"]["topology"], "single")

    def test_merge_adds_new_key(self):
        catalog = {"variants": {"vllm/dual": {"status": "production"}}}
        aipc_observer.merge_adhoc_variants(
            catalog, {"vllm/q38": {"adhoc": True, "model": "qwen3.8-27b"}})
        self.assertIn("vllm/q38", catalog["variants"])
        self.assertIn("vllm/dual", catalog["variants"])

    def test_registry_key_is_never_shadowed(self):
        catalog = {"variants": {"vllm/q38": {"status": "production",
                                             "model": "from-registry"}}}
        aipc_observer.merge_adhoc_variants(
            catalog, {"vllm/q38": {"adhoc": True, "model": "from-sidecar"}})
        self.assertEqual(catalog["variants"]["vllm/q38"]["model"],
                         "from-registry")
        self.assertNotIn("adhoc", catalog["variants"]["vllm/q38"])

    def test_merge_skips_errored_catalog(self):
        catalog = {"error": "boom"}
        aipc_observer.merge_adhoc_variants(catalog, {"vllm/q38": {"adhoc": True}})
        self.assertNotIn("variants", catalog)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_observer.py::AdhocVariantTests -v`
Expected: FAIL — `AttributeError: module 'aipc_observer' has no attribute 'load_adhoc_variants'`

- [ ] **Step 3: Add the constant**

In `aipc_observer.py`, directly below the existing line 1542:

```python
VLLM_LOGGING_CONFIG_FILE = "/etc/aipc-observer-vllm-logging.json"
```

add:

```python
# Variants this box serves that club-3090's registry does not carry. Hand-edited
# on the host; /etc matches VLLM_LOGGING_CONFIG_FILE above and survives observer
# redeploys, which are scp + restart with no git checkout.
ADHOC_CATALOG_FILE = "/etc/aipc-observer-adhoc.json"
```

- [ ] **Step 4: Add the loader and merge functions**

In `aipc_observer.py`, immediately after the `extract_catalog` function (which ends with its `except Exception as e: return {"error": str(e), "ref": ref}`, around line 1345):

```python
# The exact fields _CATALOG_EXTRACT_CODE projects from the registry. Ad-hoc
# entries carry the same ten so every downstream consumer — validate_switch,
# boot_model_once, detect_installed_assets, the dashboard — treats them
# identically to registry entries.
_ADHOC_FIELDS = ("model", "engine", "workload", "status", "status_note",
                 "max_ctx", "compose_path", "default_port", "kv_format", "tp")


def _adhoc_topology(entry):
    """Explicit topology, else derived from tensor-parallel degree.

    The dashboard filters the variant list by topology. Registry entries get
    theirs from '/dual/' or '/multi4/' in the compose path; an ad-hoc compose
    lives at an arbitrary absolute path, so it must say so directly.
    """
    explicit = entry.get("topology")
    if explicit in ("single", "dual", "multi4"):
        return explicit
    try:
        tp = int(entry.get("tp") or 1)
    except (TypeError, ValueError):
        tp = 1
    if tp >= 4:
        return "multi4"
    return "dual" if tp >= 2 else "single"


def load_adhoc_variants(path=ADHOC_CATALOG_FILE):
    """Load ad-hoc variant entries from the sidecar JSON.

    Never raises: a missing, unreadable, or malformed sidecar means "no ad-hoc
    variants", because the dashboard has to keep working without one.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"WARNING: ad-hoc catalog {path} unreadable: {e}",
              file=sys.stderr)
        return {}
    variants = (data or {}).get("variants")
    if not isinstance(variants, dict):
        return {}
    out = {}
    for key, entry in variants.items():
        if not isinstance(entry, dict):
            continue
        compose_path = str(entry.get("compose_path") or "")
        # boot_model_once runs `docker compose -f` with cwd set to the
        # club-3090 repo, so a relative path would resolve inside their tree.
        if not compose_path.startswith("/"):
            print(f"WARNING: ad-hoc variant {key!r} needs an absolute "
                  f"compose_path; skipped", file=sys.stderr)
            continue
        loaded = {f: entry.get(f) for f in _ADHOC_FIELDS}
        # Forced, not copied: a hand-edited local file must not be able to
        # claim 'production' and skip validate_switch()'s force-confirm guard.
        loaded["status"] = "experimental"
        loaded["adhoc"] = True
        loaded["topology"] = _adhoc_topology(entry)
        loaded["weights_path"] = entry.get("weights_path")
        out[str(key)] = loaded
    return out


def merge_adhoc_variants(catalog, adhoc):
    """Fold ad-hoc entries into an extracted catalog, never shadowing it.

    Registry keys always win. When club-3090 later catalogs a model we have
    been serving ad-hoc, the real entry takes over and we log that the sidecar
    entry is now redundant — that log line is the promotion signal.
    """
    if not adhoc or not isinstance(catalog, dict) or "error" in catalog:
        return catalog
    variants = catalog.setdefault("variants", {})
    for key, entry in adhoc.items():
        if key in variants:
            print(f"[observer] ad-hoc variant {key!r} is now in the club-3090 "
                  f"registry — sidecar entry ignored, safe to delete",
                  flush=True)
            continue
        variants[key] = dict(entry)
    return catalog
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_observer.py::AdhocVariantTests -v`
Expected: PASS, 10 tests.

- [ ] **Step 6: Run the full suite for regressions**

Run: `python3 -m pytest tests/test_observer.py -v`
Expected: PASS, no failures.

- [ ] **Step 7: Commit**

```bash
git add aipc_observer.py tests/test_observer.py
git commit -m "observer: load ad-hoc variant entries from a sidecar catalog

Ad-hoc entries carry the same ten fields extract_catalog projects from
club-3090's registry, so downstream consumers need no special-casing.
Status is forced to experimental so a hand-edited file cannot claim
production and skip the force-confirm guard, compose_path must be
absolute because boot_model_once runs with cwd in the club-3090 repo,
and topology is explicit because the dashboard's path-derived topology
filter cannot read an arbitrary absolute path.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Merge into the live catalog without polluting the upstream diff

`refresh_catalog` caches extracted catalogs by git sha. The merge must happen *outside* that cache (the sidecar is hand-edited and must take effect without HEAD moving) and the local-vs-upstream diff must use the *unmerged* catalog (ad-hoc entries are not upstream drift).

**Files:**
- Modify: `aipc_observer.py:3535-3566` (`refresh_catalog`)
- Test: `tests/test_observer.py`

**Interfaces:**
- Consumes: `load_adhoc_variants()`, `merge_adhoc_variants()` from Task 1.
- Produces: `refresh_catalog(repo, info, cache, observer_state=None, adhoc_loader=load_adhoc_variants)` — new keyword-only-in-practice parameter `adhoc_loader` so tests can inject entries without writing to `/etc`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_observer.py`, after the `AdhocVariantTests` class:

```python
class AdhocRefreshCatalogTests(unittest.TestCase):
    def setUp(self):
        self.state = aipc_observer.ObserverState()
        self.adhoc = {"vllm/q38": {
            "model": "qwen3.8-27b", "status": "experimental", "adhoc": True,
            "compose_path": "/etc/aipc-observer-adhoc/q38.yml", "tp": 2,
            "topology": "dual", "weights_path": None}}

    def _refresh(self, local, upstream=None, info=None):
        cache = {}
        calls = []

        def fake_extract(repo, ref="HEAD"):
            calls.append(ref)
            return local if ref == "HEAD" else (upstream or {"variants": {}})

        info = info or {"head": "sha1"}
        with mock.patch.object(aipc_observer, "extract_catalog", fake_extract), \
             mock.patch.object(aipc_observer, "detect_installed_assets",
                               lambda *a, **k: {}):
            aipc_observer.refresh_catalog(
                "/repo", info, cache, observer_state=self.state,
                adhoc_loader=lambda: dict(self.adhoc))
        return cache

    def test_adhoc_entry_appears_in_catalog(self):
        self._refresh({"variants": {"vllm/dual": {"status": "production"}}})
        variants = self.state.catalog["variants"]
        self.assertIn("vllm/q38", variants)
        self.assertIn("vllm/dual", variants)

    def test_adhoc_entry_is_not_written_into_the_sha_cache(self):
        cache = self._refresh(
            {"variants": {"vllm/dual": {"status": "production"}}})
        self.assertNotIn("vllm/q38", cache["sha1"]["variants"])

    def test_adhoc_entry_is_absent_from_the_upstream_diff(self):
        self._refresh(
            {"variants": {"vllm/dual": {"status": "production"}}},
            upstream={"variants": {"vllm/dual": {"status": "production"}}},
            info={"head": "sha1", "behind": 1, "upstream_sha": "sha2"},
        )
        diff = self.state.catalog_diff
        self.assertFalse(aipc_observer.catalog_has_changes(diff))

    def test_errored_catalog_is_passed_through_untouched(self):
        self._refresh({"error": "git show failed"})
        self.assertEqual(self.state.catalog.get("error"), "git show failed")
        self.assertNotIn("variants", self.state.catalog)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_observer.py::AdhocRefreshCatalogTests -v`
Expected: FAIL — `TypeError: refresh_catalog() got an unexpected keyword argument 'adhoc_loader'`

- [ ] **Step 3: Rewrite `refresh_catalog`**

Replace the body of `refresh_catalog` at `aipc_observer.py:3535-3566` with:

```python
def refresh_catalog(repo, info, cache, observer_state=None,
                    adhoc_loader=None):
    """Re-extract the catalog when HEAD or upstream moved; update the diff.

    `cache` maps sha -> extracted catalog so the subprocess only runs when a
    ref actually changes, not on every poll.

    Ad-hoc variants are merged on top of the cached extraction on every
    refresh — never into the cached object, because the sidecar is hand-edited
    and has to take effect without HEAD moving. The diff against upstream uses
    the unmerged catalog: an ad-hoc entry is a local addition, not drift.
    """
    st = observer_state or state
    adhoc_loader = adhoc_loader or load_adhoc_variants
    head = info.get("head")
    if not head:
        return
    raw_local = cache.get(head)
    if raw_local is None:
        raw_local = extract_catalog(repo, "HEAD")
        if "error" not in raw_local:
            cache[head] = raw_local
    local = raw_local
    if "error" not in raw_local:
        local = dict(raw_local)
        local["variants"] = dict(raw_local.get("variants") or {})
        merge_adhoc_variants(local, adhoc_loader())
    st.set_catalog(local)
    if "error" not in local:
        st.merge_installed_assets(
            detect_installed_assets(repo, local, st.model_info)
        )
    upstream_sha = info.get("upstream_sha")
    if info.get("behind") and upstream_sha and "error" not in raw_local:
        upstream = cache.get(upstream_sha)
        if upstream is None:
            upstream = extract_catalog(repo, "@{upstream}")
            if "error" not in upstream:
                cache[upstream_sha] = upstream
        if "error" not in upstream:
            st.set_catalog_diff(diff_catalogs(raw_local, upstream))
            return
    st.set_catalog_diff({})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_observer.py::AdhocRefreshCatalogTests -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/test_observer.py -v`
Expected: PASS. If an existing `refresh_catalog` test fails, it is because it asserted on the cached object identity — update it to read `state.catalog` instead of `cache[sha]`.

- [ ] **Step 6: Commit**

```bash
git add aipc_observer.py tests/test_observer.py
git commit -m "observer: merge ad-hoc variants into the live catalog

Merged outside the sha cache so a hand-edited sidecar takes effect
without HEAD moving, and the upstream diff keeps using the unmerged
extraction so ad-hoc entries never read as registry drift.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Report ad-hoc weights as installed

`detect_installed_assets` resolves weights through `infer_variant_setup`, which derives a club-3090 `setup.sh` model/weight key from the compose path. Ad-hoc weights live outside their cache roots, so without this the row would permanently read "needs download".

**Files:**
- Modify: `aipc_observer.py:2740-2762` (`detect_installed_assets`)
- Test: `tests/test_observer.py`

**Interfaces:**
- Consumes: the `adhoc` and `weights_path` fields from Task 1.
- Produces: no signature change. Ad-hoc detections are `{"model": ..., "source": "adhoc", "path": ...}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_observer.py`, after `AdhocRefreshCatalogTests`:

```python
class AdhocInstalledAssetsTests(unittest.TestCase):
    def _catalog(self, weights_path):
        return {"variants": {"vllm/q38": {
            "model": "qwen3.8-27b", "adhoc": True, "weights_path": weights_path,
            "compose_path": "/etc/aipc-observer-adhoc/q38.yml"}}}

    def test_adhoc_weights_on_disk_report_installed(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.safetensors"), "w") as f:
                f.write("x")
            got = aipc_observer.detect_installed_assets(
                "/repo", self._catalog(d), {})
        self.assertIn("vllm/q38", got)
        self.assertEqual(got["vllm/q38"]["source"], "adhoc")

    def test_adhoc_weights_missing_report_not_installed(self):
        got = aipc_observer.detect_installed_assets(
            "/repo", self._catalog("/nonexistent/weights"), {})
        self.assertNotIn("vllm/q38", got)

    def test_adhoc_without_weights_path_is_skipped(self):
        got = aipc_observer.detect_installed_assets(
            "/repo", self._catalog(None), {})
        self.assertNotIn("vllm/q38", got)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_observer.py::AdhocInstalledAssetsTests -v`
Expected: FAIL — `test_adhoc_weights_on_disk_report_installed` fails with `KeyError`/assertion, because the ad-hoc entry is resolved through the club-3090 cache roots and never matches.

- [ ] **Step 3: Add the ad-hoc branch**

In `detect_installed_assets` at `aipc_observer.py:2740`, insert the ad-hoc branch as the first thing inside the loop, so the body becomes:

```python
def detect_installed_assets(repo, catalog, model_info=None):
    """Detect catalog variants whose model assets already exist on disk."""
    variants = (catalog or {}).get("variants") or {}
    detected = {}
    roots = list(_model_cache_roots(repo, model_info))
    for key, entry in variants.items():
        # Ad-hoc weights are staged by hand outside club-3090's cache roots,
        # so they are checked directly rather than via a setup.sh hint.
        if entry.get("adhoc"):
            path = entry.get("weights_path")
            if path and _asset_path_has_files(path):
                detected[key] = {"model": entry.get("model"),
                                 "source": "adhoc", "path": path}
            continue
        hint = infer_variant_setup(entry)
        if not hint.get("model"):
            continue
        for root in roots:
            candidates = _variant_asset_candidates(root, hint)
            matched = next((p for p in candidates if _asset_path_has_files(p)),
                           None)
            if matched:
                detail = dict(hint)
                detail.update({
                    "source": "disk",
                    "path": matched,
                })
                detected[key] = detail
                break
    return detected
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_observer.py::AdhocInstalledAssetsTests -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/test_observer.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aipc_observer.py tests/test_observer.py
git commit -m "observer: detect ad-hoc weights by explicit path

Ad-hoc weights are staged by hand outside club-3090's cache roots, so
infer_variant_setup can never resolve them; check weights_path directly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Refuse install for ad-hoc variants

`/observer/api/install` runs club-3090's `scripts/setup.sh <model>`, which resolves weights through their ModelProfile YAMLs and cannot know a model absent from their registry. A stale dashboard tab or a hand-rolled `curl` must not be able to trigger it.

**Files:**
- Modify: `aipc_observer.py:5143-5160` (the `/observer/api/install` branch, just after `variant = normalize_switch_variant(variant, state.catalog)`)
- Test: `tests/test_observer.py`

**Interfaces:**
- Consumes: the `adhoc` field from Task 1.
- Produces: `assert_installable(variant, catalog) -> dict` — returns the entry, raises `ValueError` for ad-hoc variants. The surrounding handler already maps `ValueError` to HTTP 400 and releases the control lock in its `finally`, so raising is the correct control flow here — do not call `_send_json` directly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_observer.py`, after `AdhocInstalledAssetsTests`:

```python
class AssertInstallableTests(unittest.TestCase):
    def test_registry_variant_is_installable(self):
        catalog = {"variants": {"vllm/dual": {"status": "production"}}}
        entry = aipc_observer.assert_installable("vllm/dual", catalog)
        self.assertEqual(entry["status"], "production")

    def test_adhoc_variant_is_refused(self):
        catalog = {"variants": {"vllm/q38": {"adhoc": True,
                                             "status": "experimental"}}}
        with self.assertRaises(ValueError) as ctx:
            aipc_observer.assert_installable("vllm/q38", catalog)
        self.assertIn("ad-hoc", str(ctx.exception))

    def test_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            aipc_observer.assert_installable("vllm/nope", {"variants": {}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_observer.py::AssertInstallableTests -v`
Expected: FAIL — `AttributeError: module 'aipc_observer' has no attribute 'assert_installable'`

- [ ] **Step 3: Add the guard function**

In `aipc_observer.py`, immediately after `validate_switch` (which ends at line ~2795, returning `entry`):

```python
def assert_installable(variant, catalog):
    """Reject install for variants whose weights we stage by hand.

    /observer/api/install drives club-3090's scripts/setup.sh, which resolves
    weights through their ModelProfile YAMLs. It cannot know a model that is
    not in their registry, so an ad-hoc install would fail confusingly minutes
    in. Raises ValueError, which the control handler maps to HTTP 400.
    """
    variants = (catalog or {}).get("variants") or {}
    entry = variants.get(variant)
    if entry is None:
        raise ValueError(f"unknown variant {variant!r}")
    if entry.get("adhoc"):
        raise ValueError(
            f"variant {variant!r} is ad-hoc; its weights are staged by hand, "
            f"so there is nothing to install — start it instead"
        )
    return entry
```

- [ ] **Step 4: Wire it into the install handler**

In the `/observer/api/install` branch, the existing lines read:

```python
            variant = normalize_switch_variant(variant, state.catalog)
            print(f"[observer] INSTALL: variant={variant!r} preset={preset!r} cache_ram={cache_ram} force={force} retry={retry} nvlink_mode={nvlink_mode!r}", flush=True)
            validate_switch(variant, state.catalog, force=True)
```

Insert the guard between the print and `validate_switch`:

```python
            variant = normalize_switch_variant(variant, state.catalog)
            print(f"[observer] INSTALL: variant={variant!r} preset={preset!r} cache_ram={cache_ram} force={force} retry={retry} nvlink_mode={nvlink_mode!r}", flush=True)
            assert_installable(variant, state.catalog)
            validate_switch(variant, state.catalog, force=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_observer.py::AssertInstallableTests -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest tests/test_observer.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aipc_observer.py tests/test_observer.py
git commit -m "observer: refuse install for ad-hoc variants

setup.sh cannot resolve a model absent from club-3090's registry, so an
ad-hoc install would fail confusingly minutes in. Raise ValueError, which
the control handler already maps to 400 and unlocks in its finally.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Dashboard — render the ad-hoc row and route its click to switch

Three JS changes in the dashboard template string. Without the first one the row never renders at all: `variantTopology` reads `/dual/` out of the compose path, and an ad-hoc compose at an arbitrary absolute path would be classified `single` and filtered out on a 2-GPU box.

**Files:**
- Modify: `aipc_observer.py:4663` (`variantTopology`), `:4578-4582` (`doStartOrSwitch`), `:4674-4675` (the row template inside `renderVariantListModal`)

**Interfaces:**
- Consumes: the `adhoc` and `topology` fields from Task 1.
- Produces: `doStartOrSwitch(v, status, adhoc)` — third parameter defaults falsy, so the existing 2-argument call site at line 4570 (the "Install + retry" hint button) keeps its install behavior unchanged.

This task has no unit test: `tests/test_observer.py` does not exercise the embedded JS, and adding a browser harness is out of scope. It is verified by hand in Step 5.

- [ ] **Step 1: Make `variantTopology` honor an explicit topology**

Replace line 4663:

```javascript
function variantTopology(v){let p=(v&&v.compose_path)||'';return p.indexOf('/multi4/')>=0?'multi4':(p.indexOf('/dual/')>=0?'dual':'single')}
```

with:

```javascript
function variantTopology(v){if(v&&v.topology)return v.topology;let p=(v&&v.compose_path)||'';return p.indexOf('/multi4/')>=0?'multi4':(p.indexOf('/dual/')>=0?'dual':'single')}
```

- [ ] **Step 2: Route ad-hoc clicks to switch instead of install**

Replace `doStartOrSwitch` at lines 4578-4582 with:

```javascript
function doStartOrSwitch(v,status,adhoc){let p=selectedPreset(),cache=selectedCacheRam(),nvlink=selectedNvlinkMode();let warn=lastActive?`\n⚠ ${lastActive} request(s) in flight will be killed!`:'';let exp=(status!=='production'&&status!=='caveats')?`\n⚠ status is '${status}' — will pass --force.`:'';let nvlinkNote=nvlink!=='auto'?`\nNVLINK_MODE=${nvlink}`:'';
let prep=adhoc?'Weights must already be staged on disk — nothing will be downloaded.':'This downloads any missing files, then boots — takes a few minutes.';
let confirmMsg=`Start/switch model '${v}' with mode '${p}' and cache ${cache?'on':'off'}?${nvlinkNote}\n${prep}${exp}${warn}`;
if(!confirm(confirmMsg))return;
let body={variant:v,preset:p,cache_ram:cache,force:lastActive>0||(status!=='production'&&status!=='caveats'),retry:true,nvlink_mode:nvlink!=='auto'?nvlink:null};
ctlPost(adhoc?'/observer/api/switch':'/observer/api/install',body).then(()=>{document.getElementById('ctlStatus').textContent='⏳ preparing & starting…'}).catch(e=>{document.getElementById('ctlStatus').textContent='✗ '+e.message})}
```

- [ ] **Step 3: Badge the row and pass the flag through**

In `renderVariantListModal`, line 4674 currently builds `action` as:

```javascript
let action=k===runKey?`<button class="btn switch-btn" style="padding:2px 8px;font-size:11px" onclick="doStop()"${lastBusy?' disabled':''}>Stop</button>`:`<button class="btn switch-btn" style="padding:2px 8px;font-size:11px" onclick="doStartOrSwitch('${esc(k)}','${esc(v.status)}')"${lastBusy?' disabled':''}>${running?'switch':'start'}</button>`;
```

Add an `adhoc` local just before it and pass it to the call:

```javascript
let adhoc=v.adhoc?'true':'false';let action=k===runKey?`<button class="btn switch-btn" style="padding:2px 8px;font-size:11px" onclick="doStop()"${lastBusy?' disabled':''}>Stop</button>`:`<button class="btn switch-btn" style="padding:2px 8px;font-size:11px" onclick="doStartOrSwitch('${esc(k)}','${esc(v.status)}',${adhoc})"${lastBusy?' disabled':''}>${running?'switch':'start'}</button>`;
```

Then in the row template on line 4675, the variant-name div currently reads:

```javascript
<div class="variant-name">${mark}${esc(k)}</div>
```

Replace it with:

```javascript
<div class="variant-name">${mark}${esc(k)}${v.adhoc?' <span class="hot" title="Ad-hoc: not in the club-3090 registry, served from a local sidecar compose">ad-hoc</span>':''}</div>
```

- [ ] **Step 4: Check the file still parses**

Run: `python3 -c "import aipc_observer; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/test_observer.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aipc_observer.py
git commit -m "observer: render ad-hoc variants and route their clicks to switch

variantTopology now honors an explicit topology field — without it an
ad-hoc compose at an absolute path classifies as single and is filtered
out of the dual-GPU list entirely. Ad-hoc rows carry a badge and post to
/observer/api/switch, skipping the setup.sh install path.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The Qwen3.8-27B compose, example sidecar, and deploy notes

The deployable artifacts. Self-contained: no path in this compose points into the club-3090 checkout.

**Files:**
- Create: `adhoc-variants/qwen3.8-27b-dual-fp8.yml`
- Create: `adhoc-variants/aipc-observer-adhoc.example.json`
- Create: `adhoc-variants/README.md`

**Interfaces:**
- Consumes: the sidecar schema from Task 1 (`compose_path` absolute, `weights_path`, `tp`, optional `topology`).
- Produces: nothing consumed by later tasks — this is the last one.

- [ ] **Step 1: Write the compose**

Create `adhoc-variants/qwen3.8-27b-dual-fp8.yml`:

```yaml
# ===========================================================================
# Ad-hoc variant — Qwen3.8-27B official FP8, dual RTX 3090 (TP=2)
#
#   Model:     Qwen/Qwen3.8-27B-FP8 (fine-grained block-128 e4m3)
#   Topology:  Dual 3090 (TP=2), PCIe path — custom all-reduce off
#   Drafter:   MTP n=3
#   KV:        int8_per_token_head
#   Max ctx:   262144
#   Status:    experimental — not in the club-3090 registry
#
# Deliberately self-contained: nothing here resolves into the club-3090
# checkout, which we do not own. The runtime shape is otherwise the same as
# their validated vllm/qwen-27b-dual-max, which is sound because Qwen3.8-27B
# and Qwen3.6-27B are the same architecture — identical architectures,
# model_type, layer counts, head dims, GDN/attention split, vocab, and context.
#
# Custom all-reduce is disabled unconditionally. aipc1 does have NVLink
# (nvidia-smi topo -m reports NV4), but the live club-3090 config there runs
# with DISABLE_CUSTOM_ALL_REDUCE=1 and NCCL_P2P_DISABLE=1, so we match what is
# empirically working rather than second-guess it. To experiment, delete the
# --disable-custom-all-reduce line below and the NCCL_P2P_DISABLE env var.
#
# Boot via the observer's variant page. To boot by hand:
#   PORT=8020 docker compose -f /etc/aipc-observer-adhoc/dual/qwen3.8-27b-fp8.yml up -d
# ===========================================================================
services:
  vllm-qwen38-27b-dual-max:
    image: ${VLLM_IMAGE:-vllm/vllm-openai:v0.22.0}
    container_name: "${ESTATE_CONTAINER:-vllm-qwen38-27b-dual-max}"
    restart: ${CLUB3090_RESTART:-unless-stopped}
    ports:
      - "${BIND_HOST:-0.0.0.0}:${ESTATE_PORT:-${PORT:-8020}}:8000"
    volumes:
      - ${MODEL_DIR:-/home/graywzc/models/hf}:/root/.cache/huggingface
      # torch.compile + Triton caches: first boot warms them (~60-90 s), later
      # boots reuse the graphs. Under our own state dir, not club-3090's tree.
      - /var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/torch_compile:/root/.cache/vllm/torch_compile_cache
      - /var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/triton:/root/.triton/cache
    environment:
      - NVIDIA_VISIBLE_DEVICES=${ESTATE_GPUS:-${NVIDIA_VISIBLE_DEVICES:-all}}
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
      - HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
      - TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - NCCL_CUMEM_ENABLE=0
      - NCCL_P2P_DISABLE=1
      - VLLM_NO_USAGE_STATS=1
      - VLLM_USE_FLASHINFER_SAMPLER=1
      - OMP_NUM_THREADS=1
      - PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
    shm_size: "16gb"
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    entrypoint: ["vllm", "serve"]
    command:
      - --model
      - /root/.cache/huggingface/qwen3.8-27b-fp8
      - --served-model-name
      - qwen3.8-27b
      - --quantization
      - fp8
      - --dtype
      - bfloat16
      - --tensor-parallel-size
      - "${TP:-2}"
      - --pipeline-parallel-size
      - "1"
      - --max-model-len
      - "${MAX_MODEL_LEN:-262144}"
      - --gpu-memory-utilization
      - "${GPU_MEMORY_UTILIZATION:-0.92}"
      - --max-num-seqs
      - "2"
      - --max-num-batched-tokens
      - "8192"
      - --kv-cache-dtype
      - "${KV_CACHE_DTYPE:-int8_per_token_head}"
      - --disable-custom-all-reduce
      - --trust-remote-code
      # No --chat-template override: 3.8 ships developer-role and
      # reasoning_effort support that club-3090's froggeric 3.6 template
      # predates, so its own template is the correct starting point.
      - --reasoning-parser
      - qwen3
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_coder
      - --enable-prefix-caching
      - --enable-chunked-prefill
      - --long-prefill-token-threshold
      - "2048"
      # Drop this flag and re-boot if the FP8 repo turns out not to ship the
      # MTP head — the card claims MTP training but lists no mtp.safetensors.
      - --speculative-config
      - '{"method":"mtp","num_speculative_tokens":3}'
      # The model card's non-thinking recommendation. Qwen3.6's 0.6/0.95/20
      # does not carry over.
      - --override-generation-config
      - '{"temperature":0.7,"top_p":0.80,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}'
      - --host
      - 0.0.0.0
      - --port
      - "8000"
```

- [ ] **Step 2: Write the example sidecar**

Create `adhoc-variants/aipc-observer-adhoc.example.json`:

```json
{
  "variants": {
    "vllm/qwen38-27b-dual-max": {
      "model": "qwen3.8-27b",
      "engine": "vllm",
      "workload": "long-ctx-single",
      "status_note": "Ad-hoc — not in the club-3090 registry. Qwen3.8-27B official FP8 (block-128 e4m3), TP=2, int8-PTH KV, MTP n=3 @262K. Architecture is identical to qwen3.6-27b, so upstream's dual-max runtime shape transfers unchanged.",
      "max_ctx": 262144,
      "compose_path": "/etc/aipc-observer-adhoc/dual/qwen3.8-27b-fp8.yml",
      "weights_path": "/home/graywzc/models/hf/qwen3.8-27b-fp8",
      "default_port": 8020,
      "kv_format": "int8_per_token_head",
      "tp": 2,
      "topology": "dual"
    }
  }
}
```

Note there is no `"status"` key: the loader forces `experimental` regardless, so writing one would only mislead a reader.

- [ ] **Step 3: Write the deploy notes**

Create `adhoc-variants/README.md`:

```markdown
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

    ssh aipc1 'sudo mkdir -p /var/lib/nvidia-gddr6-fan-control/adhoc/qwen3.8-27b/cache/{torch_compile,triton}'

Copy the compose to a path whose parent directory is the topology name, and
the sidecar to /etc:

    scp adhoc-variants/qwen3.8-27b-dual-fp8.yml aipc1:/tmp/
    scp adhoc-variants/aipc-observer-adhoc.example.json aipc1:/tmp/
    ssh aipc1 'sudo mkdir -p /etc/aipc-observer-adhoc/dual \
      && sudo mv /tmp/qwen3.8-27b-dual-fp8.yml /etc/aipc-observer-adhoc/dual/qwen3.8-27b-fp8.yml \
      && sudo mv /tmp/aipc-observer-adhoc.example.json /etc/aipc-observer-adhoc.json'

Restart the observer so it picks up the sidecar, then click the `ad-hoc` row on
the variant page. Sidecar edits take effect on the next catalog poll — only the
observer binary needs a restart, not the sidecar.

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
```

- [ ] **Step 4: Validate the compose parses**

Run: `docker compose -f adhoc-variants/qwen3.8-27b-dual-fp8.yml config >/dev/null && echo ok`
Expected: `ok`. If Docker is not installed on the dev machine, run it on aipc1 after copying the file there instead.

- [ ] **Step 5: Commit**

```bash
git add adhoc-variants/
git commit -m "adhoc-variants: Qwen3.8-27B FP8 dual compose, sidecar example, deploy notes

Self-contained: weights, caches, and the compose itself all live outside
the club-3090 checkout. Runtime shape is upstream's validated dual-max
minus the froggeric chat template and the NVLink detect script, with 3.8's
own sampling defaults.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Deviations from the spec

One, worth flagging at review time: the spec said the custom-all-reduce setting would get "an env knob". It is hardcoded instead. Docker Compose interpolates `${VAR}` in the compose file at config time from the *daemon's* environment, not the container's, so an env knob would have depended on the observer daemon's environment — surprising and hard to discover. A commented one-line edit is clearer. Everything else in the plan matches the spec.

## Manual verification after all tasks

Not part of any task's test cycle, because it needs the real box and a 29 GB download:

1. Deploy per `adhoc-variants/README.md`, restart the observer.
2. Open the variant page. The `vllm/qwen38-27b-dual-max` row should be present in the dual-GPU list, badged `ad-hoc`, showing `installed`, with `experimental` status.
3. Click `switch`, confirm through the experimental warning. Watch for the boot in the control status stream.
4. `curl -s http://aipc1:8020/v1/models | python3 -m json.tool` — expect `qwen3.8-27b`.
5. `MODEL=qwen3.8-27b URL=http://localhost:8020 bash scripts/verify-full.sh` from the club-3090 checkout on aipc1.
6. Confirm no new or modified files in club-3090: `git -C ~/projects/club-3090 status --short` should show only what was already there.

Watch for the three risks from the design doc during step 3: block-128 FP8 on Ampere failing or falling back to a slow Triton path, a transformers 5.x processor mismatch, and a missing MTP head (drop `--speculative-config` and re-boot).
