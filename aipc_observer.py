#!/usr/bin/env python3
"""Integrated aipc observer dashboard.

This module folds the old aipc-observer sidecar into the fan controller's HTTP
server. It monitors llama.cpp docker logs, active TCP connections, and nvidia-smi
GPU stats, then serves a small dashboard at /observer with SSE updates.
"""

import json
import hashlib
import os
import pwd
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

DEFAULT_MONITOR_PORT = 8020
# None => auto-detect the container publishing the monitor port.
DEFAULT_CONTAINER = None
# The club-3090 checkout that launches the model server; the observer reports
# its version and how far it is behind upstream. Empty/None disables this.
DEFAULT_MODEL_REPO = "/home/graywzc/projects/club-3090"
GPU_POLL_INTERVAL = 2.0
CONN_CHECK_INTERVAL = 3.0
MODEL_POLL_INTERVAL = 30.0
SLOTS_POLL_INTERVAL = 2.0
METRICS_POLL_INTERVAL = 2.0
CONTAINER_DETECT_INTERVAL = 5.0
MODEL_INFO_POLL_INTERVAL = 30.0
REPO_POLL_INTERVAL = 900.0
REQUEST_LOG_MAX = 200
# JSON is valid YAML, so the generated compose override is written as JSON.
OVERRIDE_FILE = "/tmp/aipc-observer-compose-override.yml"
AUDIT_LOG = "/var/log/aipc-observer-actions.log"
# Docker log rotation applied on every controlled restart. The container
# otherwise runs json-file with no limits, and the insight-debug preset logs
# full request bodies — a long context is a ~400 KB log line.
LOG_ROTATE_MAX_SIZE = "100m"
LOG_ROTATE_MAX_FILE = "5"

HOSTNAME = socket.gethostname().split(".")[0]


class ObserverState:
    def __init__(self):
        self.lock = threading.Lock()
        self.gpu_stats = []
        self.gpu_history = deque(maxlen=600)
        self.requests = deque(maxlen=REQUEST_LOG_MAX)
        self.active_connections = {}
        self.start_time = time.time()
        self.sse_subscribers = []
        self.model_name = None
        self.container_name = None
        self.n_ctx = 0
        self.slots = []
        self.cancelled_count = 0
        self.cache_defeated_count = 0
        self.context_shift_count = 0
        # HTTP status -> count for completion POSTs, from trace access logs.
        self.http_statuses = {}
        # Reasoning-budget deactivations other than a natural end.
        self.budget_hit_count = 0
        # task_id -> in-flight request dict, shown as live "processing" rows.
        self.active_requests = {}
        # GPU index -> real VRAM temp (°C) from the gddr6 reader, since
        # nvidia-smi reports temperature.memory as N/A on consumer cards.
        self.vram_temps = {}
        # docker-inspect view of the model container (image, flags, variant).
        self.model_info = {}
        # club-3090 checkout state (HEAD, commits behind upstream).
        self.repo_info = {}
        # Variant catalog extracted from the repo's compose registry, and the
        # recommendation diff between local HEAD and the fetched upstream.
        self.catalog = {}
        self.catalog_diff = {}
        # Latest scrape of the llama.cpp Prometheus /metrics endpoint.
        self.metrics = {}

    def add_request(self, req):
        with self.lock:
            self.requests.append(req)

    def add_gpu_stats(self, stats):
        with self.lock:
            self.gpu_stats = stats
            self.gpu_history.append({"timestamp": time.time(), "gpus": stats})

    def set_active_connections(self, conns):
        with self.lock:
            self.active_connections = conns

    def set_model(self, name):
        with self.lock:
            self.model_name = name

    def set_container(self, name):
        with self.lock:
            self.container_name = name

    def set_runtime(self, n_ctx):
        with self.lock:
            self.n_ctx = n_ctx

    def set_slots(self, slots):
        with self.lock:
            self.slots = slots

    def add_active_request(self, req):
        with self.lock:
            self.active_requests[req["task_id"]] = req

    def remove_active_request(self, task_id):
        with self.lock:
            return self.active_requests.pop(task_id, None)

    def update_active_request(self, task_id, fields):
        """Merge fields into a live row without clobbering slot enrichment."""
        with self.lock:
            req = self.active_requests.get(task_id)
            if req is not None:
                req.update(fields)

    def enrich_active_from_slots(self, slots):
        """Update in-flight rows with live decode progress; drop ghost rows.

        A slot's id_task ties it to the request the log tracker registered, so
        we can show the live decoded-token count and context %% as it runs. If
        no slot is processing a registered task and it is not brand new, we
        assume its release line was missed and drop the stale row.
        """
        now = time.time()
        processing = {s.get("id_task"): s for s in slots if s.get("is_processing")}
        with self.lock:
            for task_id in list(self.active_requests):
                req = self.active_requests[task_id]
                s = processing.get(task_id)
                if s is not None:
                    prompt = int(s.get("prompt_tokens") or 0)
                    processed = int(s.get("processed_tokens") or 0)
                    cache = int(s.get("cache_tokens") or 0)
                    decoded = int(s.get("decoded") or 0)
                    if prompt:
                        req["prompt_tokens"] = prompt
                    req["completion_tokens"] = decoded
                    req["kv_pct"] = s.get("kv_pct", 0)
                    req["cache_hit_pct"] = s.get("cache_hit_pct")
                    req["cached_tokens"] = cache
                    req["recomputed_tokens"] = processed
                    if decoded > 0 and not req.get("ttft_ms"):
                        # Poll-resolution estimate; print_timing's accurate
                        # prompt eval time overwrites this at finalize.
                        req["ttft_ms"] = (now - req.get("start_time", now)) * 1000
                    if decoded > 0:
                        # Generation has started, so the prompt is fully ingested.
                        req["phase"] = "generating"
                        req["prefill_pct"] = 100
                    elif prompt:
                        # Still ingesting the prompt: cached + recomputed / total.
                        done = min(prompt, cache + processed)
                        req["phase"] = "prefill"
                        req["prefill_pct"] = int(round(100.0 * done / prompt))
                elif now - req.get("start_time", now) > SLOTS_POLL_INTERVAL * 2:
                    del self.active_requests[task_id]

    def set_vram_temps(self, mapping):
        with self.lock:
            self.vram_temps = dict(mapping)

    def set_model_info(self, info):
        with self.lock:
            self.model_info = dict(info)

    def set_repo_info(self, info):
        with self.lock:
            self.repo_info = dict(info)

    def set_catalog(self, catalog):
        with self.lock:
            self.catalog = dict(catalog)

    def set_catalog_diff(self, diff):
        with self.lock:
            self.catalog_diff = dict(diff)

    def set_metrics(self, metrics):
        with self.lock:
            self.metrics = dict(metrics)

    def incr_cancelled(self):
        with self.lock:
            self.cancelled_count += 1

    def incr_cache_defeated(self):
        with self.lock:
            self.cache_defeated_count += 1

    def incr_http_status(self, status):
        with self.lock:
            key = str(status)
            self.http_statuses[key] = self.http_statuses.get(key, 0) + 1

    def incr_budget_hit(self):
        with self.lock:
            self.budget_hit_count += 1

    def incr_context_shift(self):
        with self.lock:
            self.context_shift_count += 1

    def notify_subscribers(self):
        for evt in list(self.sse_subscribers):
            try:
                evt.set()
            except Exception:
                self.unsubscribe_sse(evt)

    def subscribe_sse(self):
        evt = threading.Event()
        with self.lock:
            self.sse_subscribers.append(evt)
        return evt

    def unsubscribe_sse(self, evt):
        with self.lock:
            try:
                self.sse_subscribers.remove(evt)
            except ValueError:
                pass

    def snapshot(self):
        with self.lock:
            uptime = time.time() - self.start_time
            return {
                "timestamp": time.time(),
                "hostname": HOSTNAME,
                "model": self.model_name,
                "container": self.container_name,
                "n_ctx": self.n_ctx,
                "slots": list(self.slots),
                "cancelled_count": self.cancelled_count,
                "cache_defeated_count": self.cache_defeated_count,
                "context_shift_count": self.context_shift_count,
                "http_statuses": dict(self.http_statuses),
                "budget_hit_count": self.budget_hit_count,
                "model_info": dict(self.model_info),
                "repo_info": dict(self.repo_info),
                "catalog": dict(self.catalog),
                "catalog_diff": dict(self.catalog_diff),
                "metrics": dict(self.metrics),
                "uptime_seconds": uptime,
                "uptime_human": format_duration(uptime),
                "gpu_stats": list(self.gpu_stats),
                "gpu_history": list(self.gpu_history)[-100:],
                "active_connections": dict(self.active_connections),
                "active_count": len(self.active_connections),
                "requests": list(self.requests),
                "active_requests": [
                    dict(r)
                    for r in sorted(
                        self.active_requests.values(),
                        key=lambda r: r.get("start_time", 0),
                    )
                ],
            }


state = ObserverState()
_started = False
_start_lock = threading.Lock()
# Filled in by start_observer; the control endpoints need them.
_config = {"monitor_port": DEFAULT_MONITOR_PORT, "model_repo": None}
# Set to make poll_repo refresh immediately (e.g. right after an update).
_repo_wake = threading.Event()


def format_duration(seconds):
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def local_time_str(epoch):
    try:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"


def _text_from_content(content):
    """Best-effort text label from OpenAI-style message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return " ".join(content.split())
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        text = " ".join(parts)
        return " ".join(text.split())
    return " ".join(json.dumps(content, sort_keys=True).split())


def _shorten(text, limit=96):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def request_group_metadata(payload):
    """Derive passive conversation grouping from an insight-debug request body.

    Store only a short hash and a truncated first-user label, not the full
    prompt. Hermes requests from the same conversation share the initial
    system/developer context plus first user message, which is stable enough
    for grouping rows without requiring an explicit user field.
    """
    if not isinstance(payload, dict):
        return {}
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return {}
    first_user_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            first_user_idx = i
            break
    label_msg = messages[first_user_idx] if first_user_idx is not None else messages[0]
    label = _shorten(_text_from_content((label_msg or {}).get("content")))
    if not label:
        label = "conversation"
    prefix_end = first_user_idx + 1 if first_user_idx is not None else 1
    prefix = messages[:prefix_end]
    digest_source = {
        "model": payload.get("model"),
        "prefix": prefix,
    }
    raw = json.dumps(digest_source, sort_keys=True, separators=(",", ":"))
    group_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return {
        "request_group_id": group_id,
        "request_group_label": label,
        "request_message_count": len(messages),
        "request_has_tools": bool(payload.get("tools")),
        "request_has_response_format": bool(payload.get("response_format")),
        "request_stream": bool(payload.get("stream")),
    }


def detect_container(monitor_port):
    """Find the running docker container publishing the monitor port."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, ports = parts[0], parts[1]
            if f":{monitor_port}->" in ports:
                return name
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"WARNING: observer container detection error: {e}", file=sys.stderr)
    return None


def variant_from_compose_path(path):
    """Reduce a compose file path to the club-3090 variant it represents.

    .../models/qwen3.6-27b/beellama/compose/single/x/dflash.yml
    -> qwen3.6-27b/beellama/single/x/dflash
    """
    if not path:
        return None
    # compose may report several config files (override case); use the first.
    path = path.split(",")[0]
    p = path.split("models/", 1)[-1]
    p = p.replace("/compose/", "/")
    if p.endswith((".yml", ".yaml")):
        p = p.rsplit(".", 1)[0]
    return p


# Server flags worth surfacing on the dashboard (flag -> snapshot key).
_NOTABLE_FLAGS = {
    "--ctx-size": "ctx_size",
    "-c": "ctx_size",
    "-np": "parallel",
    "--parallel": "parallel",
    "--cache-ram": "cache_ram_mib",
    "-cram": "cache_ram_mib",
    "--cache-type-k": "kv_type_k",
    "--cache-type-v": "kv_type_v",
    "--spec-type": "spec_type",
    "--flash-attn": "flash_attn",
    "--reasoning": "reasoning",
}


def summarize_command(cmd):
    """Extract the notable value-taking flags from the server command line."""
    flags = {}
    for i, tok in enumerate(cmd or []):
        key = _NOTABLE_FLAGS.get(tok)
        if key and i + 1 < len(cmd):
            flags[key] = cmd[i + 1]
    return flags


def _entrypoint_argv(entrypoint):
    if not entrypoint:
        return []
    if isinstance(entrypoint, str):
        try:
            return shlex.split(entrypoint)
        except ValueError:
            return [entrypoint]
    return [str(tok) for tok in entrypoint if str(tok)]


def parse_help_flags(help_text):
    """Parse a server --help screen into flag -> help-text entries.

    The descriptions come from the running binary, not from a local glossary.
    The parser intentionally accepts the common "options column, then help"
    layout used by llama.cpp and many related CLIs.
    """
    entries = {}
    current = None
    for raw in (help_text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            current = None
            continue
        flags = re.findall(r"(?<![\w-])-{1,2}[A-Za-z][A-Za-z0-9_-]*", stripped)
        if flags:
            # Pick the wide padding before the prose column, not the short
            # spacing between aliases such as "-h,    --help".
            description = ""
            for m in reversed(list(re.finditer(r"\s{2,}", stripped))):
                if m.start() >= 8:
                    description = stripped[m.end():].strip()
                    break
            aliases = sorted(set(flags), key=flags.index)
            for flag in aliases:
                entries[flag] = {
                    "aliases": aliases,
                    "description": description,
                }
            current = aliases
        elif current and (line.startswith(" ") or line.startswith("\t")):
            extra = stripped
            for flag in current:
                existing = entries[flag].get("description") or ""
                entries[flag]["description"] = (existing + " " + extra).strip()
    return entries


def command_guide(command, help_index):
    """Pair live argv flags with descriptions extracted from --help."""
    guide = []
    argv = list(command or [])
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("-"):
            i += 1
            continue
        value = None
        if "=" in tok and tok.startswith("--"):
            flag, value = tok.split("=", 1)
        else:
            flag = tok
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                value = argv[i + 1]
                i += 1
        help_entry = (help_index or {}).get(flag)
        row = {
            "flag": flag,
            "value": value,
            "known": help_entry is not None,
        }
        if help_entry:
            row.update(help_entry)
        guide.append(row)
        i += 1
    return guide


def inspect_container_help(name, entrypoint):
    """Run the live container's server --help and parse flag descriptions."""
    argv = _entrypoint_argv(entrypoint)
    if not name or not argv:
        return None
    try:
        result = subprocess.run(
            ["docker", "exec", name, *argv, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return {"error": str(e)}
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode != 0:
        return {"error": output.strip()[-400:] or "server --help failed"}
    index = parse_help_flags(output)
    return {
        "source": " ".join([*argv, "--help"]),
        "flag_count": len(index),
        "flags": index,
    }


def inspect_container(name):
    """docker-inspect the model container for image, flags, and compose origin.

    Also captures the runtime facts a controlled re-launch needs to reproduce
    the boot environment: compose service/working dir, the /models mount
    source, the published host port, and the assigned GPU ids.
    """
    try:
        result = subprocess.run(
            [
                "docker", "inspect", name, "--format",
                '{"image":{{json .Config.Image}},"cmd":{{json .Config.Cmd}},'
                '"entrypoint":{{json .Config.Entrypoint}},'
                '"labels":{{json .Config.Labels}},"mounts":{{json .Mounts}},'
                '"ports":{{json .HostConfig.PortBindings}},'
                '"devices":{{json .HostConfig.DeviceRequests}}}',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.strip())
    except Exception:
        return None
    labels = data.get("labels") or {}
    compose_file = labels.get("com.docker.compose.project.config_files") or ""
    cmd = data.get("cmd") or []
    help_info = inspect_container_help(name, data.get("entrypoint"))
    help_index = (help_info or {}).get("flags") or {}
    if help_info and "flags" in help_info:
        help_info = {k: v for k, v in help_info.items() if k != "flags"}
    info = {
        "container": name,
        "image": data.get("image"),
        "entrypoint": _entrypoint_argv(data.get("entrypoint")),
        "compose_file": compose_file,
        "variant": variant_from_compose_path(compose_file),
        "command": cmd,
        "preset": infer_insight_preset(cmd),
        "flags": summarize_command(cmd),
        "help": help_info,
        "command_guide": command_guide(cmd, help_index),
        "service": labels.get("com.docker.compose.service"),
        "working_dir": labels.get("com.docker.compose.project.working_dir"),
    }
    for mount in data.get("mounts") or []:
        if mount.get("Destination") == "/models":
            info["model_dir"] = mount.get("Source")
    for bindings in (data.get("ports") or {}).values():
        for b in bindings or []:
            if b.get("HostPort"):
                info["host_port"] = b["HostPort"]
                break
    for dev in data.get("devices") or []:
        ids = dev.get("DeviceIDs")
        if ids:
            info["gpu_ids"] = ",".join(ids)
    return info


def _repo_owner_cmd(repo, cmd):
    """Prefix cmd with runuser to drop to the repo owner when running as root.

    The daemon runs as root (fan control needs it), but touching the user's
    checkout as root would leave root-owned files and miss their ssh
    identity — and the repo's code should never execute with root privileges.
    """
    try:
        if os.geteuid() == 0:
            owner = pwd.getpwuid(os.stat(repo).st_uid).pw_name
            if owner != "root":
                return ["runuser", "-u", owner, "--"] + cmd
    except OSError:
        pass
    return cmd


def repo_git(repo, *args, timeout=60):
    """Run git in the model repo, dropping to the repo owner when root."""
    cmd = _repo_owner_cmd(repo, ["git", "-C", repo, *args])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def collect_repo_info(repo, fetch=True):
    """Report the model repo's HEAD and how far it is behind its upstream."""
    info = {"path": repo, "checked_at": time.time()}
    try:
        info["branch"] = repo_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        head = repo_git(repo, "log", "-1", "--format=%h%x09%s%x09%ci")
        parts = head.split("\t")
        info["head"] = parts[0]
        info["head_subject"] = parts[1] if len(parts) > 1 else ""
        info["head_date"] = parts[2] if len(parts) > 2 else ""
    except Exception as e:
        info["error"] = str(e)
        return info
    try:
        if fetch:
            repo_git(repo, "fetch", "--quiet", timeout=120)
        info["behind"] = int(repo_git(repo, "rev-list", "--count", "HEAD..@{upstream}"))
        info["upstream_sha"] = repo_git(repo, "rev-parse", "--short", "@{upstream}")
        if info["behind"]:
            subjects = repo_git(
                repo, "log", "--format=%h %s", "-n", "15", "HEAD..@{upstream}"
            )
            info["upstream_commits"] = subjects.splitlines()
    except Exception as e:
        info["fetch_error"] = str(e)
    return info


# club-3090's machine-readable variant catalog: a pure-data module (no
# imports), which is what makes ref-based extraction via `git show` possible.
REGISTRY_MODULE_PATH = "scripts/lib/profiles/compose_registry.py"

# Runs in an isolated subprocess with the registry module source on stdin.
# exec'ing repo code is confined to that process, never the daemon, and
# _repo_owner_cmd drops it to the repo owner when the daemon is root.
_CATALOG_EXTRACT_CODE = """
import json, sys, types
mod = types.ModuleType("compose_registry")
exec(compile(sys.stdin.read(), "compose_registry.py", "exec"), mod.__dict__)
fields = ("model", "engine", "workload", "status", "status_note", "max_ctx",
          "compose_path", "default_port", "kv_format", "tp")
variants = {}
for key, val in getattr(mod, "COMPOSE_REGISTRY", {}).items():
    if not isinstance(val, dict):
        val = getattr(val, "__dict__", {})
    variants[str(key)] = {f: val.get(f) for f in fields}
defaults = {}
for key, val in getattr(mod, "DEFAULTS", {}).items():
    k = "/".join(str(p) for p in key) if isinstance(key, tuple) else str(key)
    defaults[k] = val
print(json.dumps({"variants": variants, "defaults": defaults}))
"""


def extract_catalog(repo, ref="HEAD"):
    """Load the compose registry at a git ref without checking it out.

    Returns {"variants": {...}, "defaults": {...}} or {"error": ...}; never
    raises. Extracting at @{upstream} is what lets the dashboard show what
    upstream now recommends before any pull happens.
    """
    try:
        src = repo_git(repo, "show", f"{ref}:{REGISTRY_MODULE_PATH}")
        cmd = _repo_owner_cmd(
            repo, [sys.executable or "python3", "-c", _CATALOG_EXTRACT_CODE]
        )
        result = subprocess.run(
            cmd, input=src, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()[-300:] or "registry extraction failed"
            )
        catalog = json.loads(result.stdout)
        catalog["ref"] = ref
        return catalog
    except Exception as e:
        return {"error": str(e), "ref": ref}


# Registry fields whose upstream change amounts to a changed recommendation.
_CATALOG_DIFF_FIELDS = ("status", "max_ctx", "workload", "compose_path")


def diff_catalogs(local, upstream):
    """Summarize recommendation changes between two extracted catalogs."""
    lv = local.get("variants") or {}
    uv = upstream.get("variants") or {}
    diff = {
        "added": sorted(set(uv) - set(lv)),
        "removed": sorted(set(lv) - set(uv)),
        "changed": [],
    }
    for key in sorted(set(lv) & set(uv)):
        fields = {}
        for f in _CATALOG_DIFF_FIELDS:
            if lv[key].get(f) != uv[key].get(f):
                fields[f] = [lv[key].get(f), uv[key].get(f)]
        if fields:
            diff["changed"].append({"key": key, "fields": fields})
    ld = local.get("defaults") or {}
    ud = upstream.get("defaults") or {}
    defaults = {
        k: [ld.get(k), ud.get(k)]
        for k in sorted(set(ld) | set(ud))
        if ld.get(k) != ud.get(k)
    }
    if defaults:
        diff["default_changes"] = defaults
    return diff


def catalog_has_changes(diff):
    return bool(
        diff.get("added") or diff.get("removed") or diff.get("changed")
        or diff.get("default_changes")
    )


# --- model control plane (phase 2b) ----------------------------------------

# Insight presets are command-line flag tweaks, not env vars: llama.cpp CLI
# args take precedence over LLAMA_ARG_* env, and flags like --cache-ram are
# already present in the compose command, so env could never override them.
# (flag, None) appends a switch if absent; (flag, value) replaces or appends.
INSIGHT_PRESETS = {
    "baseline": [],
    "insight": [
        ("--metrics", None),
        ("--props", None),
        ("--log-verbosity", "4"),
        ("--log-timestamps", None),
    ],
    "insight-cache": [
        ("--metrics", None),
        ("--props", None),
        ("--log-verbosity", "4"),
        ("--log-timestamps", None),
        ("--cache-ram", "8192"),
    ],
    # Debug verbosity logs full request/response bodies — the passive path to
    # grouping agent requests by session. Very chatty; use for experiments.
    "insight-debug": [
        ("--metrics", None),
        ("--props", None),
        ("--log-verbosity", "5"),
        ("--log-timestamps", None),
        ("--cache-ram", "8192"),
    ],
}

_control_lock = threading.Lock()


def audit(action, detail, path=AUDIT_LOG):
    line = f"{datetime.now().isoformat(timespec='seconds')} {action}: {detail}"
    print(f"Observer control: {line}", file=sys.stderr)
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def apply_preset_to_command(cmd, tweaks):
    """Apply preset flag tweaks to a server argv: replace values, add flags."""
    argv = list(cmd)
    for flag, value in tweaks:
        i = next((j for j, tok in enumerate(argv) if normalize_preset_flag(tok) == flag),
                 None)
        if i is not None:
            if value is not None and i + 1 < len(argv):
                argv[i + 1] = value
        else:
            argv.append(flag)
            if value is not None:
                argv.append(value)
    return argv


PRESET_FLAG_ALIASES = {
    "-lv": "--log-verbosity",
}


def normalize_preset_flag(flag):
    return PRESET_FLAG_ALIASES.get(flag, flag)


def _command_option_map(cmd):
    options = {}
    argv = list(cmd or [])
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if not tok.startswith("-"):
            i += 1
            continue
        if tok.startswith("--") and "=" in tok:
            flag, value = tok.split("=", 1)
            options[normalize_preset_flag(flag)] = value
        elif i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
            options[normalize_preset_flag(tok)] = str(argv[i + 1])
            i += 1
        else:
            options[normalize_preset_flag(tok)] = None
        i += 1
    return options


def preset_option_map(tweaks):
    options = {normalize_preset_flag(flag): value for flag, value in tweaks}
    return options


def infer_insight_preset(cmd):
    """Infer which observer-managed insight preset the live argv matches."""
    options = _command_option_map(cmd)
    managed_flags = {
        flag for tweaks in INSIGHT_PRESETS.values() for flag, _ in tweaks
    }
    live_managed = {
        flag: value for flag, value in options.items() if flag in managed_flags
    }
    if live_managed.get("--cache-ram") == "0":
        live_managed.pop("--cache-ram")
    for name in ("insight-debug", "insight-cache", "insight"):
        if live_managed == preset_option_map(INSIGHT_PRESETS[name]):
            return name
    if not live_managed:
        return "baseline"
    return "custom"


def build_compose_override(service, argv=None, image=None):
    svc = {
        # Cap docker log growth; club-3090 composes set no logging limits.
        "logging": {
            "driver": "json-file",
            "options": {
                "max-size": LOG_ROTATE_MAX_SIZE,
                "max-file": LOG_ROTATE_MAX_FILE,
            },
        },
    }
    if argv is not None:
        svc["command"] = argv
    if image:
        svc["image"] = image
    return {"services": {service: svc}}


def check_restart_allowed(observer_state, force=False):
    active = len(observer_state.active_requests)
    if active and not force:
        raise RuntimeError(
            f"{active} request(s) currently in flight; pass force to restart anyway"
        )


def _run(cmd, env=None, cwd=None, timeout=600, input_text=None):
    result = subprocess.run(
        cmd, env=env, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, input=input_text,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout).strip()[-400:] or "command failed"
        )
    return result.stdout


def _compose_env(model_info):
    """Reproduce the compose substitution env the container was booted with.

    Substitution vars (PORT, MODEL_DIR, …) only exist at compose-up time;
    the running container's mounts/ports/devices are where their effective
    values survive, so read them back from the inspect data.
    """
    env = dict(os.environ)
    if model_info.get("host_port"):
        env["PORT"] = env["ESTATE_PORT"] = str(model_info["host_port"])
    if model_info.get("model_dir"):
        env["MODEL_DIR"] = model_info["model_dir"]
    if model_info.get("container"):
        env["ESTATE_CONTAINER"] = model_info["container"]
    if model_info.get("gpu_ids"):
        env["CUDA_VISIBLE_DEVICES"] = env["ESTATE_GPUS"] = model_info["gpu_ids"]
    return env


def compose_baseline_command(compose_file, service, env, cwd, runner=_run):
    """Resolve the variant's own (preset-free) command via compose config."""
    out = runner(
        ["docker", "compose", "-f", compose_file, "config", "--format", "json"],
        env=env, cwd=cwd, timeout=60,
    )
    services = (json.loads(out).get("services") or {})
    command = (services.get(service) or {}).get("command")
    if not command:
        raise RuntimeError(f"service {service!r} has no command in {compose_file}")
    if isinstance(command, str):
        import shlex

        command = shlex.split(command)
    return [str(tok) for tok in command]


def restart_model(preset_name, model_info=None, runner=_run,
                  override_path=OVERRIDE_FILE):
    """Recreate the model container with an insight preset applied.

    Presets always build on the compose file's own command (resolved via
    `docker compose config`), never the running container's — so switching
    insight -> baseline sheds flags instead of accumulating them.
    """
    tweaks = INSIGHT_PRESETS.get(preset_name)
    if tweaks is None:
        raise ValueError(f"unknown preset {preset_name!r}; "
                         f"known: {', '.join(INSIGHT_PRESETS)}")
    mi = model_info if model_info is not None else state.model_info
    missing = [k for k in ("compose_file", "service", "working_dir", "command")
               if not mi.get(k)]
    if missing:
        raise RuntimeError(f"model info incomplete ({', '.join(missing)}); "
                           "is the model container running?")
    compose_file = mi["compose_file"].split(",")[0]
    env = _compose_env(mi)
    cwd = mi["working_dir"]
    argv = None
    if preset_name != "baseline":
        baseline = compose_baseline_command(compose_file, mi["service"], env,
                                            cwd, runner=runner)
        argv = apply_preset_to_command(baseline, tweaks)
    # Always pin the running image: launchers may have injected a different
    # image (e.g. BEELLAMA_IMAGE) than the compose file's fallback, and a
    # restart must never silently switch images.
    with open(override_path, "w") as f:
        json.dump(
            build_compose_override(mi["service"], argv, image=mi.get("image")),
            f, indent=1,
        )
    os.chmod(override_path, 0o644)
    audit("restart", f"preset={preset_name} variant={mi.get('variant')}")
    runner(
        ["docker", "compose", "-f", compose_file, "-f", override_path,
         "up", "-d", "--remove-orphans"],
        env=env, cwd=cwd,
    )
    return {"restarted": True, "preset": preset_name,
            "variant": mi.get("variant")}


def update_repo(repo):
    """Fast-forward the club-3090 checkout to its upstream."""
    before = collect_repo_info(repo, fetch=True)
    if "error" in before:
        raise RuntimeError(before["error"])
    if not before.get("behind"):
        return {"updated": False, "head": before.get("head"),
                "detail": "already up to date"}
    audit("update", f"{before['head']} -> {before.get('upstream_sha')} "
                    f"({before['behind']} commits)")
    repo_git(repo, "pull", "--ff-only", timeout=300)
    after = collect_repo_info(repo, fetch=False)
    return {
        "updated": True,
        "from": before.get("head"),
        "to": after.get("head"),
        "commits": before.get("upstream_commits", []),
    }


def fetch_json(url, timeout=5):
    """GET a JSON endpoint, returning the parsed body or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def fetch_text(url, timeout=5):
    """GET a text endpoint, returning the body or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return None


def parse_prometheus(text):
    """Parse Prometheus exposition text into {metric_name: float}.

    llama.cpp's metrics carry no labels, but tolerate them by stripping any
    {...} suffix from the name. Unparseable lines are skipped.
    """
    values = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            values[name] = float(parts[1])
        except ValueError:
            continue
    return values


# llamacpp:* metric -> snapshot key. requests_deferred is the queue depth the
# logs can never provide (they only show a task once compute starts).
_METRIC_KEYS = {
    "llamacpp:requests_deferred": "queued",
    "llamacpp:requests_processing": "processing",
    "llamacpp:prompt_tokens_seconds": "prompt_tps_avg",
    "llamacpp:predicted_tokens_seconds": "gen_tps_avg",
    "llamacpp:prompt_tokens_total": "prompt_tokens_total",
    "llamacpp:tokens_predicted_total": "gen_tokens_total",
    "llamacpp:prompt_seconds_total": "prompt_seconds_total",
    "llamacpp:tokens_predicted_seconds_total": "gen_seconds_total",
    "llamacpp:n_decode_total": "decode_calls_total",
    "llamacpp:n_busy_slots_per_decode": "busy_slots_per_decode",
    "llamacpp:kv_cache_usage_ratio": "kv_cache_usage_ratio",
}


def summarize_metrics(values):
    """Map raw Prometheus values to the dashboard's metric snapshot."""
    out = {"available": True, "scraped_at": time.time()}
    for raw_name, key in _METRIC_KEYS.items():
        if raw_name in values:
            out[key] = values[raw_name]
    for key in ("queued", "processing", "prompt_tokens_total", "gen_tokens_total",
                "decode_calls_total"):
        if key in out:
            out[key] = int(out[key])
    return out


def poll_metrics(monitor_port):
    """Scrape the llama.cpp /metrics endpoint (enabled by the insight presets).

    When the server runs without --metrics the endpoint 404s; report
    available=False so the dashboard can hint at the preset instead of
    showing stale numbers.
    """
    url = f"http://127.0.0.1:{monitor_port}/metrics"
    last_queue_state = None
    while True:
        text = fetch_text(url)
        values = parse_prometheus(text) if text else {}
        if values:
            metrics = summarize_metrics(values)
            state.set_metrics(metrics)
            queue_state = (metrics.get("queued"), metrics.get("processing"))
            if queue_state != last_queue_state:
                last_queue_state = queue_state
                state.notify_subscribers()
        else:
            if state.metrics.get("available"):
                state.set_metrics({"available": False})
                state.notify_subscribers()
            elif not state.metrics:
                state.set_metrics({"available": False})
        time.sleep(METRICS_POLL_INTERVAL)


def detect_model(monitor_port):
    """Query the llama.cpp frontend for the model it is currently serving."""
    data = fetch_json(f"http://127.0.0.1:{monitor_port}/v1/models")
    if not data:
        return None
    models = data.get("data") or []
    if not models:
        return None
    model_id = models[0].get("id") or ""
    # llama.cpp often reports a filesystem path; show just the model file/name.
    return model_id.rsplit("/", 1)[-1] or None


def detect_n_ctx(monitor_port):
    """Read the context window size from the llama.cpp /props endpoint."""
    props = fetch_json(f"http://127.0.0.1:{monitor_port}/props")
    if not props:
        return 0
    gen = props.get("default_generation_settings") or {}
    return int(gen.get("n_ctx") or props.get("n_ctx") or 0)


def summarize_slots(raw, default_n_ctx):
    """Reduce a raw /slots payload to the cache/context fields we display."""
    slots = []
    for s in raw:
        prompt = int(s.get("n_prompt_tokens") or 0)
        cache = int(s.get("n_prompt_tokens_cache") or 0)
        processed = int(s.get("n_prompt_tokens_processed") or 0)
        nt = s.get("next_token") or []
        decoded = int((nt[0] or {}).get("n_decoded") or 0) if nt else 0
        n_ctx = int(s.get("n_ctx") or default_n_ctx or 0)
        kv_used = prompt + decoded
        considered = cache + processed
        slots.append({
            "id": s.get("id", 0),
            "is_processing": bool(s.get("is_processing")),
            "id_task": s.get("id_task", -1),
            "n_ctx": n_ctx,
            "decoded": decoded,
            "kv_used": kv_used,
            "kv_pct": round(100.0 * kv_used / n_ctx, 1) if n_ctx else 0,
            "prompt_tokens": prompt,
            "cache_tokens": cache,
            "processed_tokens": processed,
            # Fraction of the prompt served from cache rather than recomputed.
            "cache_hit_pct": round(100.0 * cache / considered, 1) if considered else None,
        })
    return slots


def poll_model(monitor_port):
    while True:
        model = detect_model(monitor_port)
        if model:
            state.set_model(model)
        time.sleep(MODEL_POLL_INTERVAL)


def poll_slots(monitor_port):
    base = f"http://127.0.0.1:{monitor_port}"
    while True:
        if not state.n_ctx:
            n_ctx = detect_n_ctx(monitor_port)
            if n_ctx:
                state.set_runtime(n_ctx)
        raw = fetch_json(f"{base}/slots")
        if isinstance(raw, list):
            slots = summarize_slots(raw, state.n_ctx)
            state.set_slots(slots)
            state.enrich_active_from_slots(slots)
            state.notify_subscribers()
        time.sleep(SLOTS_POLL_INTERVAL)


def poll_model_info(monitor_port):
    last = None
    while True:
        name = state.container_name or detect_container(monitor_port)
        if name:
            info = inspect_container(name)
            if info and info != last:
                state.set_model_info(info)
                state.notify_subscribers()
                last = info
        time.sleep(MODEL_INFO_POLL_INTERVAL)


def refresh_catalog(repo, info, cache, observer_state=None):
    """Re-extract the catalog when HEAD or upstream moved; update the diff.

    `cache` maps sha -> extracted catalog so the subprocess only runs when a
    ref actually changes, not on every poll.
    """
    st = observer_state or state
    head = info.get("head")
    if not head:
        return
    local = cache.get(head)
    if local is None:
        local = extract_catalog(repo, "HEAD")
        if "error" not in local:
            cache[head] = local
    st.set_catalog(local)
    upstream_sha = info.get("upstream_sha")
    if info.get("behind") and upstream_sha and "error" not in local:
        upstream = cache.get(upstream_sha)
        if upstream is None:
            upstream = extract_catalog(repo, "@{upstream}")
            if "error" not in upstream:
                cache[upstream_sha] = upstream
        if "error" not in upstream:
            st.set_catalog_diff(diff_catalogs(local, upstream))
            return
    st.set_catalog_diff({})


def poll_repo(repo):
    catalog_cache = {}
    while True:
        info = collect_repo_info(repo)
        state.set_repo_info(info)
        try:
            refresh_catalog(repo, info, catalog_cache)
        except Exception as e:
            print(f"WARNING: observer catalog refresh error: {e}", file=sys.stderr)
        state.notify_subscribers()
        _repo_wake.wait(timeout=REPO_POLL_INTERVAL)
        _repo_wake.clear()


def overlay_vram_temps(gpus, vram_temps):
    """Replace each GPU's nvidia-smi mem temp with the gddr6 reading when known."""
    for g in gpus:
        real = vram_temps.get(g["index"])
        if real is not None:
            g["mem_temp_c"] = float(real)
    return gpus


def poll_gpu_stats():
    while True:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,temperature.memory,"
                    "utilization.gpu,utilization.memory,memory.used,memory.total,"
                    "fan.speed,clocks.gr,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 12:
                        continue
                    mem_temp_str = parts[3].strip()
                    mem_temp = (
                        safe_float(mem_temp_str)
                        if mem_temp_str and mem_temp_str.upper() != "N/A"
                        else -1
                    )
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "temp_c": safe_float(parts[2]),
                        "mem_temp_c": mem_temp,
                        "gpu_util_pct": safe_float(parts[4]),
                        "mem_util_pct": safe_float(parts[5]),
                        "mem_used_mib": safe_float(parts[6]),
                        "mem_total_mib": safe_float(parts[7]),
                        "fan_pct": safe_float(parts[8]),
                        "clock_mhz": safe_float(parts[9]),
                        "power_w": safe_float(parts[10]),
                        "power_limit_w": safe_float(parts[11]),
                    })
                # Overlay the gddr6-derived VRAM temps over nvidia-smi's N/A.
                with state.lock:
                    vram_temps = dict(state.vram_temps)
                overlay_vram_temps(gpus, vram_temps)
                if gpus:
                    state.add_gpu_stats(gpus)
                    state.notify_subscribers()
        except FileNotFoundError:
            print("WARNING: nvidia-smi not found; observer GPU polling disabled.",
                  file=sys.stderr)
            break
        except subprocess.TimeoutExpired:
            print("WARNING: observer nvidia-smi timed out.", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: observer GPU poll error: {e}", file=sys.stderr)
        time.sleep(GPU_POLL_INTERVAL)


def monitor_connections(monitor_port):
    while True:
        try:
            result = subprocess.run(
                ["ss", "-tnp", f"state established '( dport = :{monitor_port} )'"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                conns = {}
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    local = parts[3]
                    remote = parts[4]
                    if remote.startswith("["):
                        body, port = remote[1:].rsplit("]", 1)
                        peer_ip, peer_port = body, port.lstrip(":")
                    else:
                        peer_ip, peer_port = remote.rsplit(":", 1)
                    key = f"{peer_ip}:{peer_port}"
                    conns[key] = {
                        "peer_ip": peer_ip,
                        "peer_port": peer_port,
                        "local_addr": local,
                        "seen_at": time.time(),
                    }
                state.set_active_connections(conns)
                if conns:
                    state.notify_subscribers()
        except Exception as e:
            print(f"WARNING: observer connection monitor error: {e}", file=sys.stderr)
        time.sleep(CONN_CHECK_INTERVAL)


RE_LAUNCH_SLOT = re.compile(
    r"I\s+slot\s+launch_slot_:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+processing task"
)
RE_PRINT_TIMING = re.compile(
    r"I\s+slot\s+print_timing:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|"
)
RE_RELEASE = re.compile(
    r"I\s+slot\s+release:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+stop processing:\s+n_tokens\s*=\s*(\d+)"
    r"(?:,\s*truncated\s*=\s*(\d+))?"
)
RE_DRAFT = re.compile(
    r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)"
)
RE_FULL_REPROC = re.compile(r"task\s+(\d+)\s+\|\s+forcing full prompt re-processing")
RE_CANCEL = re.compile(r"cancel task,\s+id_task\s*=\s*(\d+)")
RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=?\s+([\d.]+)\s*ms\s*/\s+(\d+)\s+tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s+tokens per second\)"
)
RE_EVAL_TIME = re.compile(
    r"eval time\s*=?\s+([\d.]+)\s*ms\s*/\s+(\d+)\s+tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s+tokens per second\)"
)
RE_TOTAL_TIME = re.compile(r"total time\s*=?\s+([\d.]+)\s*ms\s*/\s+(\d+)\s+tokens")
RE_DECODED = re.compile(r"n_decoded\s*=\s+(\d+),\s*tg\s*=\s+([\d.]+)\s*t/s")
RE_CTX_SHIFT = re.compile(
    r"slot context shift,\s*n_keep\s*=\s*(\d+),\s*n_left\s*=\s*(\d+),\s*n_discard\s*=\s*(\d+)"
)
RE_ROUTE_WARM = re.compile(r"selected slot by LCP similarity,\s*sim_best\s*=\s*([\d.]+)")
RE_ROUTE_LRU = re.compile(r"selected slot by LRU")
# Trace-level (-lv 4) lines from the insight presets:
RE_ACCESS = re.compile(r"done request:\s+(\w+)\s+(\S+)\s+\S+\s+(\d{3})")
RE_REQUEST_BODY = re.compile(r"\brequest:\s*(\{.*\})\s*$")
RE_ADAPTIVE_DM = re.compile(r"adaptive dm:\s+(\w+)\s*=\s*([\d.-]+)\s+n_max\s*=\s*(\d+)")
RE_GRAPHS_REUSED = re.compile(r"graphs reused\s*=\s*(\d+)")
RE_NEW_PROMPT = re.compile(r"new prompt,.*task\.n_tokens\s*=\s*(\d+)")
RE_CACHED_TOKENS = re.compile(r"cached n_tokens\s*=\s*(\d+),\s*memory_seq_rm")
RE_BUDGET_OFF = re.compile(r"reasoning-budget:\s+deactivated\s+\(([^)]+)\)")
RE_PREFILL_PROGRESS = re.compile(
    r"prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+),"
    r"\s*t\s*=\s*([\d.]+)\s*s\s*/\s*([\d.]+)\s+tokens per second"
)
RE_SLOT_ID = re.compile(r"\bid\s+(\d+)\s+\|")
RE_TASK_ID = re.compile(r"\btask\s+(\d+)\b")


class RequestTracker:
    def __init__(self, observer_state=None):
        self.state = observer_state or state
        self.active = {}
        self.task_counter = 0
        self.current_timing_task_id = None
        # slot_id -> ("warm"|"cold", similarity); the selection line is logged
        # before launch_slot_ and carries the previous task id, so it is keyed
        # by slot and consumed by the next launch on that slot.
        self.pending_routes = {}
        self.pending_request_meta = deque(maxlen=128)

    def process_line(self, line):
        m = RE_REQUEST_BODY.search(line)
        if m:
            try:
                meta = request_group_metadata(json.loads(m.group(1)))
            except json.JSONDecodeError:
                meta = {}
            if meta:
                self.pending_request_meta.append(meta)
            return

        m = RE_ACCESS.search(line)
        if m:
            method, path, status = m.group(1), m.group(2), int(m.group(3))
            # Only completion POSTs; the observer's own GET polling is noise.
            if method == "POST" and not path.startswith("/props"):
                self.state.incr_http_status(status)
            return

        m = RE_BUDGET_OFF.search(line)
        if m:
            if m.group(1).strip() != "natural end":
                self.state.incr_budget_hit()
            return

        m = RE_ROUTE_WARM.search(line)
        if m:
            sid = RE_SLOT_ID.search(line)
            if sid:
                self.pending_routes[int(sid.group(1))] = ("warm", safe_float(m.group(1)))
            return

        if RE_ROUTE_LRU.search(line):
            sid = RE_SLOT_ID.search(line)
            if sid:
                self.pending_routes[int(sid.group(1))] = ("cold", None)
            return

        m = RE_LAUNCH_SLOT.search(line)
        if m:
            slot_id, task_id = int(m.group(1)), int(m.group(2))
            route = self.pending_routes.pop(slot_id, None)
            meta = self.pending_request_meta.popleft() if self.pending_request_meta else {}
            now = time.time()
            self.active[task_id] = {
                "id": self.task_counter,
                "task_id": task_id,
                "slot_id": slot_id,
                "start_time": now,
                "start_time_str": local_time_str(now),
                "status": "processing",
                "path": "/v1/chat/completions",
                "model": self.state.model_name or "unknown",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_eval_ms": 0,
                "prompt_tps": 0,
                "eval_ms": 0,
                "gen_tps": 0,
                "total_ms": 0,
                "ttft_ms": 0,
                "cache_hit_pct": None,
                "kv_pct": 0,
                "cached_tokens": 0,
                "recomputed_tokens": 0,
                "truncated": False,
                "finish_reason": None,
                "draft_acceptance": None,
                "draft_accepted": 0,
                "draft_generated": 0,
                "cache_defeated": False,
                "context_shifts": 0,
                "slot_route": route[0] if route else None,
                "route_similarity": route[1] if route else None,
                "phase": "starting",
                "prefill_pct": 0,
                **meta,
            }
            self.task_counter += 1
            self.state.add_active_request(dict(self.active[task_id]))
            self.state.notify_subscribers()
            return

        m = RE_PRINT_TIMING.search(line)
        if m:
            task_id = int(m.group(2))
            self.current_timing_task_id = task_id if task_id in self.active else None
            line = line[m.end():]

        req = self._current_request_for_line(line)
        m = RE_PROMPT_EVAL.search(line)
        if m and req is not None:
            req["prompt_eval_ms"] = safe_float(m.group(1))
            req["prompt_tokens"] = int(m.group(2))
            req["prompt_tps"] = safe_float(m.group(4))
            # Time to first token: the prompt must be fully ingested first.
            req["ttft_ms"] = req["prompt_eval_ms"]
            return

        m = RE_EVAL_TIME.search(line)
        if m and req is not None:
            req["eval_ms"] = safe_float(m.group(1))
            req["completion_tokens"] = int(m.group(2))
            req["gen_tps"] = safe_float(m.group(4))
            return

        m = RE_TOTAL_TIME.search(line)
        if m and req is not None:
            req["total_ms"] = safe_float(m.group(1))
            req["total_tokens"] = int(m.group(2))
            return

        m = RE_DECODED.search(line)
        if m and req is not None:
            req["completion_tokens"] = int(m.group(1))
            req["gen_tps"] = safe_float(m.group(2))
            return

        m = RE_CTX_SHIFT.search(line)
        if m:
            self.state.incr_context_shift()
            if req is not None:
                req["context_shifts"] = req.get("context_shifts", 0) + 1
                self.state.update_active_request(
                    req["task_id"], {"context_shifts": req["context_shifts"]}
                )
                self.state.notify_subscribers()
            return

        m = RE_PREFILL_PROGRESS.search(line)
        if m and req is not None:
            req["prompt_tps"] = safe_float(m.group(4))
            self.state.update_active_request(req["task_id"], {
                "prompt_tps": req["prompt_tps"],
                "phase": "prefill",
                "prefill_pct": int(round(100 * safe_float(m.group(2)))),
            })
            self.state.notify_subscribers()
            return

        m = RE_DRAFT.search(line)
        if m and req is not None:
            req["draft_acceptance"] = safe_float(m.group(1))
            req["draft_accepted"] = int(m.group(2))
            req["draft_generated"] = int(m.group(3))
            return

        m = RE_ADAPTIVE_DM.search(line)
        if m and req is not None:
            req["dm_controller"] = m.group(1)
            req["dm_rate"] = safe_float(m.group(2))
            req["draft_n_max"] = int(m.group(3))
            return

        m = RE_GRAPHS_REUSED.search(line)
        if m and req is not None:
            req["graphs_reused"] = int(m.group(1))
            return

        m = RE_NEW_PROMPT.search(line)
        if m and req is not None:
            # Earliest prompt-size signal: logged at ingest start, before the
            # first /slots poll catches the row.
            if not req.get("prompt_tokens"):
                req["prompt_tokens"] = int(m.group(1))
                self.state.update_active_request(
                    req["task_id"], {"prompt_tokens": req["prompt_tokens"]}
                )
                self.state.notify_subscribers()
            return

        m = RE_CACHED_TOKENS.search(line)
        if m and req is not None:
            req["cached_tokens"] = int(m.group(1))
            self.state.update_active_request(
                req["task_id"], {"cached_tokens": req["cached_tokens"]}
            )
            return

        m = RE_RELEASE.search(line)
        if m:
            task_id = int(m.group(2))
            n_tokens = int(m.group(3))
            truncated = m.group(4)
            if task_id not in self.active:
                return
            req = self.active.pop(task_id)
            req["total_tokens"] = max(req["total_tokens"], n_tokens)
            if truncated is not None:
                req["truncated"] = bool(int(truncated))
            req["finish_reason"] = "length" if req["truncated"] else "stop"
            self._finalize(task_id, req, "completed")
            return

        m = RE_FULL_REPROC.search(line)
        if m:
            self.state.incr_cache_defeated()
            req = self.active.get(int(m.group(1)))
            if req is not None:
                req["cache_defeated"] = True
            return

        m = RE_CANCEL.search(line)
        if m:
            task_id = int(m.group(1))
            self.state.incr_cancelled()
            req = self.active.pop(task_id, None)
            if req is not None:
                req["finish_reason"] = "cancelled"
                self._finalize(task_id, req, "cancelled")
            return

    def _finalize(self, task_id, req, status):
        if self.current_timing_task_id == task_id:
            self.current_timing_task_id = None
        req["status"] = status
        req["end_time"] = time.time()
        req["end_time_str"] = local_time_str(req["end_time"])
        req["elapsed_ms"] = (req["end_time"] - req["start_time"]) * 1000
        with self.state.lock:
            conns = list(self.state.active_connections.values())
        req["client_ip"] = conns[0]["peer_ip"] if conns else "unknown"
        # Carry slot-derived cache/context stats (only the live row has them)
        # onto the completed record; the tracker's accurate ttft wins over the
        # poll-resolution estimate.
        enriched = self.state.remove_active_request(task_id)
        if enriched:
            for key in ("cache_hit_pct", "kv_pct", "cached_tokens", "recomputed_tokens"):
                if enriched.get(key) is not None:
                    req[key] = enriched[key]
            if not req.get("ttft_ms") and enriched.get("ttft_ms"):
                req["ttft_ms"] = enriched["ttft_ms"]
        self.state.add_request(req)
        self.state.notify_subscribers()

    def _current_request_for_line(self, line):
        m = RE_TASK_ID.search(line)
        if m:
            return self.active.get(int(m.group(1)))
        if self.current_timing_task_id in self.active:
            return self.active[self.current_timing_task_id]
        if len(self.active) == 1:
            return next(iter(self.active.values()))
        return None


request_tracker = RequestTracker()


def tail_docker_logs(container, monitor_port):
    """Tail llama.cpp logs, auto-detecting the container when not pinned.

    Re-detects after the stream ends (container restart) so the observer keeps
    working across redeploys without a hardcoded container name.
    """
    while True:
        target = container or detect_container(monitor_port)
        if not target:
            time.sleep(CONTAINER_DETECT_INTERVAL)
            continue
        state.set_container(target)
        try:
            proc = subprocess.Popen(
                ["docker", "logs", "-f", "--tail", "100", target],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"Observer tailing docker logs for {target}", file=sys.stderr)
            for line in proc.stdout:
                try:
                    request_tracker.process_line(line)
                except Exception as e:
                    print(f"Observer log parse error: {e} ({line[:100]})",
                          file=sys.stderr)
        except FileNotFoundError:
            print("WARNING: docker not found; observer request tracking disabled.",
                  file=sys.stderr)
            return
        except Exception as e:
            print(f"WARNING: observer docker log tail error: {e}", file=sys.stderr)
        # Stream ended; drop the stale container and re-detect after a pause.
        state.set_container(None)
        time.sleep(CONTAINER_DETECT_INTERVAL)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Observer</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;--accent:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;--purple:#bc8cff}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.header,.card{background:var(--surface);border:1px solid var(--border);border-radius:8px}.header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;margin-bottom:16px}.header h1{font-size:18px;margin:0}.meta,.label{color:var(--dim)}.model{color:var(--accent);font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{padding:16px}.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 12px}.gpu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.gpu-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px}.gpu-name{font-weight:650;color:var(--accent);margin-bottom:8px}.row{display:flex;justify-content:space-between;gap:16px;padding:4px 0;font-size:13px}.value{font-variant-numeric:tabular-nums;font-weight:600}.bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden}.fill{height:100%;background:var(--accent);border-radius:3px}.fill.mem{background:var(--purple)}.fill.power{background:var(--yellow)}.fill.fan{background:var(--green)}.hot{color:var(--yellow)}.critical{color:var(--red)}.summary{display:flex;gap:24px;flex-wrap:wrap}.summary-item{text-align:center}.summary-value{font-size:28px;font-weight:750;font-variant-numeric:tabular-nums}.summary-label{font-size:11px;text-transform:uppercase;color:var(--dim);letter-spacing:.05em}.full{grid-column:1/-1}.requests{max-height:520px;overflow:auto}.request-row{display:grid;grid-template-columns:88px 150px minmax(150px,1.4fr) 60px 56px 70px 74px 78px 74px 60px 78px minmax(80px,1fr);gap:8px;align-items:center;padding:7px 8px;border-bottom:1px solid var(--border);font-size:12px}.group-label{color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.good{color:var(--green)}.request-head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700}.status{border-radius:999px;padding:2px 8px;text-align:center;font-size:10px;text-transform:uppercase;font-weight:700}.completed{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}.processing{background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.3)}.cancelled{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}.request-row.live{box-shadow:inset 3px 0 0 var(--accent)}
.btn{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px;font-family:inherit}.btn:hover{border-color:var(--accent)}.btn:disabled{opacity:.5;cursor:wait}.controls{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}.preset-pill{border:1px solid var(--border);border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700}.preset-match{color:var(--green);border-color:rgba(63,185,80,.45);background:rgba(63,185,80,.12)}.preset-diff{color:var(--yellow);border-color:rgba(210,153,34,.45);background:rgba(210,153,34,.12)}.preset-custom{color:var(--purple);border-color:rgba(188,140,255,.45);background:rgba(188,140,255,.12)}.preset-desc{line-height:1.35;max-width:360px;text-align:right}.cmd-line{font-size:11px;color:var(--dim);word-break:break-word;padding:4px 0;line-height:1.75}.cmd-token{display:inline-block;border-radius:4px;padding:0 3px;margin:1px 0}.cmd-same{color:var(--green);background:rgba(63,185,80,.12);outline:1px solid rgba(63,185,80,.25)}.cmd-change{color:var(--yellow);background:rgba(210,153,34,.13);outline:1px solid rgba(210,153,34,.32)}.cmd-remove{color:var(--red);background:rgba(248,81,73,.12);outline:1px solid rgba(248,81,73,.28);text-decoration:line-through}.cmd-add{display:inline-block;border-radius:999px;border:1px solid rgba(88,166,255,.38);background:rgba(88,166,255,.1);color:var(--accent);padding:1px 6px;margin:2px 4px 0 0;font-size:11px}.cmd-legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}.modal{display:none;position:fixed;inset:0;z-index:20;background:rgba(0,0,0,.72);padding:32px}.modal.open{display:flex}.modal-panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;width:min(1180px,100%);max-height:calc(100vh - 64px);margin:auto;display:flex;flex-direction:column;box-shadow:0 16px 48px rgba(0,0,0,.45)}.modal-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 16px;border-bottom:1px solid var(--border)}.modal-head h2{font-size:14px;margin:0}.modal-body{overflow:auto;padding:0 16px 16px}.flag-guide{display:grid;grid-template-columns:minmax(140px,170px) minmax(180px,260px) minmax(460px,1fr);gap:12px;align-items:start;padding:8px 0;border-bottom:1px solid var(--border);font-size:12px;min-width:820px}.flag-guide.head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700;z-index:1}.flag-help{color:var(--text);line-height:1.4}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
</style>
</head>
<body>
<div class="header"><h1 id="title">Observer</h1><div class="meta"><span id="model" class="model">--</span> · Uptime <span id="uptime">0s</span> · <span id="updated">--</span></div></div>
<div class="grid">
<section class="card"><h2>GPU</h2><div id="gpuGrid" class="gpu-grid"></div></section>
<section class="card"><h2>Summary</h2><div class="summary">
<div class="summary-item"><div id="active" class="summary-value">0</div><div class="summary-label">Active</div></div>
<div class="summary-item"><div id="requests" class="summary-value">0</div><div class="summary-label">Completed</div></div>
<div class="summary-item"><div id="gpuTemp" class="summary-value">--</div><div class="summary-label">GPU Temp</div></div>
<div class="summary-item"><div id="memTemp" class="summary-value">--</div><div class="summary-label">VRAM Temp</div></div>
<div class="summary-item"><div id="avgTps" class="summary-value">0</div><div class="summary-label">Avg Gen t/s</div></div>
<div class="summary-item"><div id="queued" class="summary-value">-</div><div class="summary-label">Queued</div></div>
</div></section>
<section class="card"><h2>Context / KV Cache</h2><div id="slotInfo" class="gpu-grid"></div></section>
<section class="card"><h2>Model Config</h2><div id="modelInfo"></div>
<div class="controls"><select id="presetSel" class="btn" onchange="renderModelInfoFromState()" title="baseline: club-3090 verbatim · insight: +metrics/props/trace logs · insight-cache: insight + cache-ram 8192">
<option value="baseline">baseline</option><option value="insight">insight</option><option value="insight-cache" selected>insight+cache</option><option value="insight-debug">insight+debug</option>
</select><button class="btn" onclick="doRestart()">Restart model</button><button class="btn" onclick="doUpdate()">Update club-3090</button><span id="ctlStatus" class="label"></span></div></section>
<section class="card"><h2>club-3090 Catalog</h2><div id="catalogInfo"></div></section>
<section class="card"><h2>Server Metrics</h2><div id="metricsInfo"></div></section>
<section class="card"><h2>Inference Health</h2><div class="summary">
<div class="summary-item"><div id="truncRate" class="summary-value">0%</div><div class="summary-label">Truncated</div></div>
<div class="summary-item"><div id="cancelled" class="summary-value">0</div><div class="summary-label">Cancelled</div></div>
<div class="summary-item"><div id="cacheDefeat" class="summary-value">0</div><div class="summary-label">Cache Defeated</div></div>
<div class="summary-item"><div id="ctxShift" class="summary-value">0</div><div class="summary-label">Ctx Shifts</div></div>
<div class="summary-item"><div id="draftAccept" class="summary-value">-</div><div class="summary-label">Avg Draft Accept</div></div>
<div class="summary-item"><div id="httpErrors" class="summary-value">0</div><div class="summary-label">HTTP Errors</div></div>
<div class="summary-item"><div id="budgetHits" class="summary-value">0</div><div class="summary-label">Budget Hits</div></div>
</div></section>
<section class="card full"><h2>Recent Requests</h2><div id="requestList" class="requests"></div></section>
</div>
<div id="flagModal" class="modal" onclick="if(event.target===this)closeFlagGuide()"><div class="modal-panel">
<div class="modal-head"><h2 id="flagModalTitle">Flag guide</h2><button class="btn" onclick="closeFlagGuide()">Close</button></div>
<div id="flagModalBody" class="modal-body"></div>
</div></div>
<script>
let es;let lastModelInfo={};let lastRenderData=null;let flagModalOpen=false;
const PRESET_LABELS={'baseline':'baseline','insight':'insight','insight-cache':'insight+cache','insight-debug':'insight+debug','custom':'custom'};
const PRESET_DESCRIPTIONS={
baseline:'club-3090 compose command with no observer insight flags added',
insight:'adds /metrics, /props, verbosity 4, and timestamps for observer visibility',
'insight-cache':'insight plus cache-ram 8192 for prompt-cache experiments',
'insight-debug':'cache plus verbosity 5, including very chatty request/response body logs',
custom:'live command has observer-managed flags but does not exactly match a preset'
};
const PRESET_OPTIONS={
baseline:{},
insight:{'--metrics':null,'--props':null,'--log-verbosity':'4','--log-timestamps':null},
'insight-cache':{'--metrics':null,'--props':null,'--log-verbosity':'4','--log-timestamps':null,'--cache-ram':'8192'},
'insight-debug':{'--metrics':null,'--props':null,'--log-verbosity':'5','--log-timestamps':null,'--cache-ram':'8192'}
};
const MANAGED_FLAGS=new Set(Object.values(PRESET_OPTIONS).flatMap(o=>Object.keys(o)));
const PRESET_FLAG_ALIASES={'-lv':'--log-verbosity'};
function presetFlag(flag){return PRESET_FLAG_ALIASES[flag]||flag}
function pct(v,max){return Math.max(0,Math.min(100,(v/max)*100))}
function cls(t){return t>85?'critical':t>75?'hot':''}
function connect(){if(es)es.close();es=new EventSource('/observer/sse');es.onmessage=e=>render(JSON.parse(e.data));es.onerror=()=>{es.close();setTimeout(connect,3000)}}
let lastActive=0;
function render(d){lastRenderData=d;lastActive=(d.active_requests||[]).length;renderHeader(d);renderGpu(d.gpu_stats||[]);renderSummary(d);renderSlots(d);renderModelInfo(d);renderCatalog(d);renderMetrics(d);renderHealth(d);renderRequests(d.requests||[],d.active_requests||[]);document.getElementById('uptime').textContent=d.uptime_human;document.getElementById('updated').textContent=new Date().toLocaleTimeString()}
function renderModelInfoFromState(){if(lastRenderData)renderModelInfo(lastRenderData)}
function renderMetrics(d){let m=d.metrics||{};let el=document.getElementById('metricsInfo');let q=document.getElementById('queued');
if(!m.available){q.textContent='-';q.className='summary-value';el.innerHTML='<div class="row"><span class="label">/metrics disabled — restart with an insight preset to enable</span></div>';return}
q.textContent=m.queued??'-';q.className='summary-value '+((m.queued||0)>0?'hot':'');
let rows='';
rows+=infoRow('Processing / queued',`${m.processing??'-'} / ${(m.queued||0)>0?`<span class="hot">${m.queued}</span>`:m.queued??'-'}`,'requests_processing / requests_deferred — queued means all slots are busy');
if(m.prompt_tps_avg!=null)rows+=infoRow('Avg prompt t/s',Number(m.prompt_tps_avg).toFixed(1),'server-lifetime average prompt processing throughput');
if(m.gen_tps_avg!=null)rows+=infoRow('Avg gen t/s',Number(m.gen_tps_avg).toFixed(1),'server-lifetime average generation throughput');
if(m.prompt_tokens_total!=null)rows+=infoRow('Prompt tokens',Number(m.prompt_tokens_total).toLocaleString()+(m.prompt_seconds_total!=null?` <span class="label">(${Number(m.prompt_seconds_total).toFixed(0)}s)</span>`:''));
if(m.gen_tokens_total!=null)rows+=infoRow('Generated tokens',Number(m.gen_tokens_total).toLocaleString()+(m.gen_seconds_total!=null?` <span class="label">(${Number(m.gen_seconds_total).toFixed(0)}s)</span>`:''));
if(m.decode_calls_total!=null)rows+=infoRow('Decode calls',Number(m.decode_calls_total).toLocaleString());
if(m.busy_slots_per_decode!=null)rows+=infoRow('Busy slots / decode',Number(m.busy_slots_per_decode).toFixed(2));
if(m.kv_cache_usage_ratio!=null)rows+=infoRow('KV cache usage',(100*m.kv_cache_usage_ratio).toFixed(1)+'%');
el.innerHTML=rows}
async function ctlPost(path,body){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});let j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||('HTTP '+r.status));return j}
async function ctlRun(btn,msg,fn){if(!confirm(msg))return;let st=document.getElementById('ctlStatus');document.querySelectorAll('.controls .btn').forEach(b=>b.disabled=true);st.textContent='working…';try{st.textContent=await fn()}catch(e){st.textContent='failed: '+e.message}finally{document.querySelectorAll('.controls .btn').forEach(b=>b.disabled=false)}}
function doUpdate(){ctlRun(this,'git pull club-3090 (fast-forward only)?',async()=>{let r=await ctlPost('/observer/api/update');return r.updated?`updated ${r.from} → ${r.to} (${(r.commits||[]).length} commits)`:(r.detail||'already up to date')})}
function doRestart(){let p=document.getElementById('presetSel').value;let warn=lastActive?` ⚠ ${lastActive} request(s) in flight will be killed!`:'';ctlRun(this,`Restart the model with preset '${p}'?${warn}`,async()=>{let r=await ctlPost('/observer/api/restart',{preset:p,force:lastActive>0});return `restarted with ${r.preset} (model reloading…)`})}
function statusSpan(s){let c=s==='production'?'good':(s==='caveats'?'hot':'critical');return `<span class="${c}">${esc(s||'?')}</span>`}
function renderCatalog(d){let c=d.catalog||{};let diff=d.catalog_diff||{};let mi=d.model_info||{};let ri=d.repo_info||{};let el=document.getElementById('catalogInfo');
if(c.error){el.innerHTML=infoRow('Catalog',`<span class="critical">${esc(c.error)}</span>`);return}
let vars=c.variants||{};let keys=Object.keys(vars);
if(!keys.length){el.innerHTML='<div class="row"><span class="label">No catalog yet</span></div>';return}
let rows='';
let runKey=keys.find(k=>vars[k].compose_path&&mi.compose_file&&mi.compose_file.indexOf(vars[k].compose_path)>=0);
if(runKey){let v=vars[runKey];
rows+=infoRow('Running variant',esc(runKey)+' '+statusSpan(v.status),v.status_note);
if(v.workload)rows+=infoRow('Workload',esc(v.workload));
let cur=Number((mi.flags||{}).ctx_size||0);
if(v.max_ctx)rows+=infoRow('Validated ctx',Number(v.max_ctx).toLocaleString()+(cur&&v.max_ctx>cur?` <span class="hot">(running ${cur.toLocaleString()} — can raise)</span>`:''));
let topo=(v.compose_path||'').indexOf('/dual/')>=0?'dual':'single';
let def=(c.defaults||{})[`${v.model}/${runKey.split('/')[0]}/${topo}`];
if(def)rows+=infoRow('Curated default',esc(def)+(def===runKey?' <span class="good">(running it)</span>':' <span class="hot">(differs)</span>'));}
let counts={};keys.forEach(k=>{let s=vars[k].status||'other';counts[s]=(counts[s]||0)+1});
rows+=infoRow('Variants',`${keys.length} (${counts.production||0} production · ${counts.caveats||0} caveats)`);
let ngpu=(d.gpu_stats||[]).length||1;
let reqGpus=p=>p.indexOf('/multi4/')>=0?4:(p.indexOf('/dual/')>=0?2:1);
let fits=keys.filter(k=>reqGpus(vars[k].compose_path||'')<=ngpu&&(vars[k].tp||1)<=ngpu);
let order={production:0,caveats:1};
fits.sort((a,b)=>(order[vars[a].status]??2)-(order[vars[b].status]??2)||(vars[a].model||'').localeCompare(vars[b].model||'')||a.localeCompare(b));
let defSet=new Set(Object.values(c.defaults||{}));
let items=fits.map(k=>{let v=vars[k];let mark=k===runKey?'▶ ':(defSet.has(k)?'⭐ ':'');let ctx=v.max_ctx?` · ${Math.round(v.max_ctx/1024)}K`:'';
return `<div class="row" style="font-size:12px"><span class="label" title="${esc(v.status_note||'')}">${mark}${esc(v.model)} · ${esc(k)}${ctx}${v.workload?' · '+esc(v.workload):''}</span>${statusSpan(v.status)}</div>`}).join('');
rows+=det('detVariants',false,`variants for this machine (${fits.length} of ${keys.length}, ${ngpu} GPU)`,items);
let dl=[];(diff.added||[]).forEach(k=>dl.push('new: '+k));
(diff.removed||[]).forEach(k=>dl.push('removed: '+k));
(diff.changed||[]).forEach(ch=>dl.push(ch.key+': '+Object.entries(ch.fields).map(([f,v])=>`${f} ${v[0]??'-'} → ${v[1]??'-'}`).join(', ')));
Object.entries(diff.default_changes||{}).forEach(([k,v])=>dl.push(`default ${k}: ${v[0]??'-'} → ${v[1]??'-'}`));
if(dl.length)rows+=det('detUpstream',true,`upstream recommendation changes (${dl.length})`,dl.map(t=>`<div class="row" style="font-size:12px"><span class="hot">${esc(t)}</span></div>`).join(''));
else if(ri.behind>0)rows+=infoRow('Upstream',`<span class="good">no recommendation changes in ${ri.behind} pending commits</span>`);
el.innerHTML=rows}
function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function infoRow(label,value,title){return `<div class="row"><span class="label">${label}</span><span class="value"${title?` title="${esc(title)}"`:''}>${value}</span></div>`}
// SSE re-renders replace innerHTML, which would reset <details> to its
// default state; remember each toggle by id and re-apply it.
let detailsState={};
document.addEventListener('toggle',e=>{if(e.target.id)detailsState[e.target.id]=e.target.open},true);
function det(id,defOpen,summary,body){let open=detailsState[id]!==undefined?detailsState[id]:defOpen;return `<details id="${id}"${open?' open':''}><summary class="label" style="cursor:pointer;font-size:12px;padding:4px 0">${summary}</summary>${body}</details>`}
function flagGuideTitle(mi){let guide=mi.command_guide||[];let help=mi.help||{};return help.error?`Flag guide (${guide.length}, help failed)`:help.source?`Flag guide (${guide.length}, from ${esc(help.source)})`:`Flag guide (${guide.length})`}
function flagGuideRows(mi){let guide=mi.command_guide||[];if(!guide.length)return '<div class="row"><span class="label">No command guide yet</span></div>';let rows='<div class="flag-guide head"><span>Flag</span><span>Value</span><span>Help text</span></div>';rows+=guide.map(r=>{let desc=r.known?esc(r.description||'listed in --help without description'):'<span class="hot">not found in --help</span>';let aliases=(r.aliases||[]).filter(a=>a!==r.flag).join(', ');let title=aliases?` title="aliases: ${esc(aliases)}"`:'';return `<div class="flag-guide"><span class="mono"${title}>${esc(r.flag)}</span><span class="mono">${r.value==null?'<span class="label">switch</span>':esc(r.value)}</span><span class="flag-help">${desc}</span></div>`}).join('');let help=mi.help||{};if(help.error)rows+=`<div class="row"><span class="critical">${esc(help.error)}</span></div>`;return rows}
function renderFlagGuideModal(mi){document.getElementById('flagModalTitle').innerHTML=flagGuideTitle(mi);document.getElementById('flagModalBody').innerHTML=flagGuideRows(mi)}
function openFlagGuide(){flagModalOpen=true;renderFlagGuideModal(lastModelInfo);document.getElementById('flagModal').classList.add('open')}
function closeFlagGuide(){flagModalOpen=false;document.getElementById('flagModal').classList.remove('open')}
function renderCommandGuideButton(mi){let guide=mi.command_guide||[];if(!guide.length)return '';return `<div class="row"><span class="label">Flags</span><span class="value"><button class="btn" onclick="openFlagGuide()">${flagGuideTitle(mi)}</button></span></div>`}
function presetLabel(name){return PRESET_LABELS[name]||name||'unknown'}
function liveOptionMap(cmd){let out={};let argv=cmd||[];for(let i=0;i<argv.length;i++){let tok=String(argv[i]);if(!tok.startsWith('-'))continue;if(tok.startsWith('--')&&tok.includes('=')){let parts=tok.split(/=(.*)/s);out[presetFlag(parts[0])]=parts[1]??''}else if(i+1<argv.length&&!String(argv[i+1]).startsWith('-')){out[presetFlag(tok)]=String(argv[i+1]);i++}else out[presetFlag(tok)]=null}if(out['--cache-ram']==='0')delete out['--cache-ram'];return out}
function optionText(flag,value){return value==null?flag:`${flag} ${value}`}
function selectedPreset(){let el=document.getElementById('presetSel');return el?el.value:'insight-cache'}
function presetDiff(mi,selected){let live=liveOptionMap(mi.command||[]);let want=PRESET_OPTIONS[selected]||{};let rows=[];MANAGED_FLAGS.forEach(flag=>{let hasLive=Object.prototype.hasOwnProperty.call(live,flag);let hasWant=Object.prototype.hasOwnProperty.call(want,flag);if(hasLive&&hasWant&&String(live[flag])!==String(want[flag]))rows.push(`<span class="cmd-add">${esc(flag)}: ${esc(live[flag]??'switch')} → ${esc(want[flag]??'switch')}</span>`);else if(!hasLive&&hasWant)rows.push(`<span class="cmd-add">add ${esc(optionText(flag,want[flag]))}</span>`);else if(hasLive&&!hasWant)rows.push(`<span class="cmd-add">remove ${esc(optionText(flag,live[flag]))}</span>`)});return rows.join('')}
function renderPresetStatus(mi){let running=mi.preset||'unknown';let selected=selectedPreset();let cls=running==='custom'?'preset-custom':(running===selected?'preset-match':'preset-diff');let rows='';
rows+=infoRow('Running mode',`<span class="preset-pill ${cls}">${esc(presetLabel(running))}</span>`,PRESET_DESCRIPTIONS[running]||'inferred from the live container command');
rows+=infoRow('Selected mode',`<span class="preset-pill ${running===selected?'preset-match':'preset-diff'}">${esc(presetLabel(selected))}</span>`,PRESET_DESCRIPTIONS[selected]||'');
rows+=infoRow('Mode difference',`<span class="label preset-desc">${esc(PRESET_DESCRIPTIONS[selected]||'')}</span>`);
let diff=presetDiff(mi,selected);if(diff)rows+=`<div class="row"><span class="label">Selected changes</span><span class="value" style="text-align:right">${diff}</span></div>`;
return rows}
function commandFlagAt(argv,i){let tok=String(argv[i]);if(tok.startsWith('--')&&tok.includes('='))return presetFlag(tok.split('=')[0]);return tok.startsWith('-')?presetFlag(tok):null}
function commandTokenClass(argv,i,want,live){let flag=commandFlagAt(argv,i);let prev=i>0?commandFlagAt(argv,i-1):null;if(prev&&Object.prototype.hasOwnProperty.call(live,prev)&&String(argv[i])===String(live[prev]))flag=prev;if(!flag||!MANAGED_FLAGS.has(flag))return '';let hasWant=Object.prototype.hasOwnProperty.call(want,flag);if(!hasWant)return 'cmd-remove';let liveVal=live[flag];let wantVal=want[flag];if(String(liveVal)===String(wantVal))return 'cmd-same';return 'cmd-change'}
function renderCommandLine(mi){let argv=mi.command||[];let selected=selectedPreset();let want=PRESET_OPTIONS[selected]||{};let live=liveOptionMap(argv);let html=argv.map((tok,i)=>{let c=commandTokenClass(argv,i,want,live);return c?`<span class="cmd-token ${c}">${esc(tok)}</span>`:esc(tok)}).join(' ');
let additions=[];MANAGED_FLAGS.forEach(flag=>{if(!Object.prototype.hasOwnProperty.call(live,flag)&&Object.prototype.hasOwnProperty.call(want,flag))additions.push(`<span class="cmd-add">+ ${esc(optionText(flag,want[flag]))}</span>`)});
let legend='<div class="cmd-legend"><span class="cmd-token cmd-same">same in selected mode</span><span class="cmd-token cmd-change">value changes</span><span class="cmd-token cmd-remove">removed by selected mode</span></div>';
if(additions.length)legend+=`<div class="cmd-legend">${additions.join('')}</div>`;
return `<div class="cmd-line">${html}</div>${legend}`}
function renderModelInfo(d){let mi=d.model_info||{};let ri=d.repo_info||{};let f=mi.flags||{};let rows='';
lastModelInfo=mi;if(flagModalOpen)renderFlagGuideModal(mi);
if(mi.variant)rows+=infoRow('Variant',esc(mi.variant),mi.compose_file);
if(mi.image)rows+=infoRow('Image',esc((mi.image||'').split('/').pop()),mi.image);
rows+=renderPresetStatus(mi);
if(f.ctx_size)rows+=infoRow('Context',Number(f.ctx_size).toLocaleString());
if(f.parallel)rows+=infoRow('Slots',esc(f.parallel));
if(f.kv_type_k||f.kv_type_v)rows+=infoRow('KV quant',esc((f.kv_type_k||'f16')+' / '+(f.kv_type_v||'f16')));
if(f.spec_type)rows+=infoRow('Spec decode',esc(f.spec_type));
if(f.cache_ram_mib!==undefined)rows+=infoRow('Prompt cache',f.cache_ram_mib==='0'?'<span class="hot">off (cache-ram 0)</span>':esc(f.cache_ram_mib)+' MiB');
if(ri.head){rows+=infoRow('club-3090 HEAD',esc(ri.head)+' '+esc(ri.head_subject||''),ri.head_date);
let st=ri.error?`<span class="critical">${esc(ri.error)}</span>`:(ri.behind>0?`<span class="hot">${ri.behind} commits behind</span>`:(ri.behind===0?'<span class="good">up to date</span>':'-'));
rows+=infoRow('Upstream',st,(ri.upstream_commits||[]).join('\\n')||ri.fetch_error||'');}
else if(ri.error)rows+=infoRow('club-3090',`<span class="critical">${esc(ri.error)}</span>`,ri.path);
rows+=renderCommandGuideButton(mi);
if(mi.command&&mi.command.length)rows+=det('detCmd',false,'full server command',renderCommandLine(mi));
document.getElementById('modelInfo').innerHTML=rows||'<div class="row"><span class="label">No model info yet</span></div>'}
function renderHealth(d){let reqs=d.requests||[];let comp=reqs.filter(r=>r.status==='completed');let trunc=comp.filter(r=>r.truncated).length;document.getElementById('truncRate').textContent=comp.length?(100*trunc/comp.length).toFixed(0)+'%':'0%';document.getElementById('cancelled').textContent=d.cancelled_count||0;document.getElementById('cacheDefeat').textContent=d.cache_defeated_count||0;document.getElementById('ctxShift').textContent=d.context_shift_count||0;let dr=reqs.filter(r=>r.draft_acceptance!=null);document.getElementById('draftAccept').textContent=dr.length?(100*dr.reduce((s,r)=>s+r.draft_acceptance,0)/dr.length).toFixed(0)+'%':'-';
let hs=d.http_statuses||{};let errs=0;let breakdown=[];Object.keys(hs).sort().forEach(k=>{breakdown.push(k+': '+hs[k]);if(Number(k)>=400)errs+=hs[k]});
let he=document.getElementById('httpErrors');he.textContent=errs;he.className='summary-value '+(errs?'critical':'');he.parentElement.title=breakdown.join(', ')||'no completion POSTs seen';
let bh=document.getElementById('budgetHits');bh.textContent=d.budget_hit_count||0;bh.parentElement.title='reasoning-budget deactivations other than a natural end'}
function renderHeader(d){let host=d.hostname||'';let name=host?host+' Observer':'Observer';document.getElementById('title').textContent=name;document.title=name;document.getElementById('model').textContent=d.model||'no model';}
function renderGpu(gpus){document.getElementById('gpuGrid').innerHTML=gpus.map(g=>{let mt=g.mem_temp_c>=0?`${g.mem_temp_c}°C`:'N/A';return `<div class="gpu-card"><div class="gpu-name">GPU ${g.index}: ${g.name}</div>
<div class="row"><span class="label">GPU Temp</span><span class="value ${cls(g.temp_c)}">${g.temp_c}°C</span></div><div class="bar"><div class="fill" style="width:${pct(g.temp_c,100)}%"></div></div>
<div class="row"><span class="label">VRAM Temp</span><span class="value ${cls(g.mem_temp_c)}">${mt}</span></div>
<div class="row"><span class="label">GPU Util</span><span class="value">${g.gpu_util_pct}%</span></div><div class="bar"><div class="fill" style="width:${g.gpu_util_pct}%"></div></div>
<div class="row"><span class="label">VRAM</span><span class="value">${(g.mem_used_mib/1024).toFixed(1)} / ${(g.mem_total_mib/1024).toFixed(1)} GB</span></div><div class="bar"><div class="fill mem" style="width:${g.mem_util_pct}%"></div></div>
<div class="row"><span class="label">Fan</span><span class="value">${g.fan_pct}%</span></div><div class="bar"><div class="fill fan" style="width:${g.fan_pct}%"></div></div>
<div class="row"><span class="label">Power</span><span class="value">${g.power_w} / ${g.power_limit_w} W</span></div><div class="bar"><div class="fill power" style="width:${pct(g.power_w,g.power_limit_w)}%"></div></div></div>`}).join('')}
function renderSummary(d){document.getElementById('active').textContent=d.active_count;document.getElementById('requests').textContent=d.requests.length;if(d.gpu_stats&&d.gpu_stats.length){let g=d.gpu_stats[0];gpuTemp.textContent=`${g.temp_c}°C`;gpuTemp.className='summary-value '+cls(g.temp_c);memTemp.textContent=g.mem_temp_c>=0?`${g.mem_temp_c}°C`:'N/A';memTemp.className='summary-value '+cls(g.mem_temp_c)}let done=d.requests.filter(r=>r.status==='completed'&&r.gen_tps>0);avgTps.textContent=done.length?(done.reduce((s,r)=>s+r.gen_tps,0)/done.length).toFixed(1):'0'}
function renderSlots(d){let slots=d.slots||[];let nctx=d.n_ctx||0;document.getElementById('slotInfo').innerHTML=slots.length?slots.map(s=>{let hit=(s.cache_hit_pct==null)?'-':s.cache_hit_pct+'%';let badge=s.is_processing?'<span class="status processing">busy</span>':'<span class="status completed">idle</span>';return `<div class="gpu-card"><div class="gpu-name">Slot ${s.id} ${badge}</div>
<div class="row"><span class="label">Context</span><span class="value ${cls(s.kv_pct)}">${(s.kv_used||0).toLocaleString()} / ${(s.n_ctx||nctx).toLocaleString()} (${s.kv_pct}%)</span></div><div class="bar"><div class="fill mem" style="width:${pct(s.kv_pct,100)}%"></div></div>
<div class="row"><span class="label">Prompt cache hit</span><span class="value">${hit}</span></div><div class="bar"><div class="fill fan" style="width:${s.cache_hit_pct||0}%"></div></div>
<div class="row"><span class="label">Cached / reproc.</span><span class="value">${(s.cache_tokens||0).toLocaleString()} / ${(s.processed_tokens||0).toLocaleString()}</span></div></div>`}).join(''):'<div class="row"><span class="label">No slot data</span></div>'}
function formatDuration(ms){if(!ms&&ms!==0)return '-';ms=Number(ms);if(!Number.isFinite(ms))return '-';if(ms>=60000)return (ms/60000).toFixed(ms>=600000?1:2)+' min';if(ms>=1000)return (ms/1000).toFixed(ms>=10000?1:2)+' sec';return ms.toFixed(0)+' ms'}
function formatPhaseDuration(ms){return Number(ms)>0?formatDuration(ms):'-'}
function liveElapsed(r){return formatDuration(Date.now()-(r.start_time||0)*1000)}
function cacheCell(r){let t=r.slot_route?` title="slot route: ${r.slot_route}${r.route_similarity!=null?' (sim '+Number(r.route_similarity).toFixed(2)+')':''}"`:'';if(r.cache_hit_pct==null)return `<span${t}>-</span>`;let p=Number(r.cache_hit_pct);let c=p>=70?'good':p>=30?'hot':'critical';return `<span class="${c}"${t}>${p.toFixed(0)}%</span>`}
function statusCell(r,label){let shifts=r.context_shifts||0;let t=shifts?` title="${shifts} context shift(s): oldest history was dropped to keep generating"`:'';return `<span class="status ${r.status||'processing'}"${t}>${label}${shifts?' ⇄':''}</span>`}
function groupCell(r){let label=r.request_group_label||'-';let bits=[];if(r.request_group_id)bits.push('group '+r.request_group_id);if(r.request_message_count)bits.push(r.request_message_count+' messages');if(r.request_has_tools)bits.push('tools');if(r.request_has_response_format)bits.push('response_format');if(r.request_stream)bits.push('stream');let title=bits.length?` title="${esc(bits.join(' · '))}"`:'';let cls=r.request_group_id?'group-label':'label';return `<span class="${cls}"${title}>${esc(label)}</span>`}
function renderRequests(reqs,active){let head='<div class="request-row request-head"><span>Status</span><span>Time</span><span>Group</span><span>PT</span><span>Cache</span><span>TTFT</span><span>P t/s</span><span>P time</span><span>G t/s</span><span>GT</span><span>G time</span><span>Total</span></div>';let act=(active||[]).slice().reverse();let actRows=act.map(r=>{let phase=r.phase==='generating'?'generating':(r.phase==='prefill'?'prefill':'processing');let ptime=r.phase==='prefill'?`<div class="bar" title="${r.prefill_pct||0}%"><div class="fill" style="width:${r.prefill_pct||0}%"></div></div>`:'-';return `<div class="request-row live">${statusCell(r,phase)}<span>${r.start_time_str||'--'}</span>${groupCell(r)}<span>${r.prompt_tokens||0}</span>${cacheCell(r)}<span>${formatPhaseDuration(r.ttft_ms)}</span><span>${r.prompt_tps?Number(r.prompt_tps).toFixed(1):'-'}</span><span>${ptime}</span><span>-</span><span>${r.completion_tokens||0}</span><span>-</span><span>${liveElapsed(r)}</span></div>`}).join('');let recent=reqs.slice(-40).reverse();let doneRows=recent.map(r=>`<div class="request-row">${statusCell(r,r.status)}<span>${r.end_time_str||r.start_time_str||'--'}</span>${groupCell(r)}<span>${r.prompt_tokens||0}</span>${cacheCell(r)}<span>${formatPhaseDuration(r.ttft_ms)}</span><span>${r.prompt_tps?Number(r.prompt_tps).toFixed(1):'-'}</span><span>${formatPhaseDuration(r.prompt_eval_ms)}</span><span>${r.gen_tps?Number(r.gen_tps).toFixed(1):'-'}</span><span>${r.completion_tokens||0}</span><span>${formatPhaseDuration(r.eval_ms)}</span><span>${formatDuration(r.total_ms||r.elapsed_ms)}</span></div>`).join('');let body=actRows+doneRows;document.getElementById('requestList').innerHTML=head+(body||'<div class="request-row"><span class="label">No requests yet</span></div>')}
connect();
</script>
</body></html>"""


def handle_observer_get(handler):
    path = handler.path.split("?", 1)[0]
    if path in ("/observer", "/observer/"):
        body = DASHBOARD_HTML.encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True
    if path == "/observer/api/snapshot":
        body = json.dumps(state.snapshot()).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True
    if path == "/observer/sse":
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        evt = state.subscribe_sse()
        try:
            while True:
                evt.wait(timeout=1)
                evt.clear()
                body = f"data: {json.dumps(state.snapshot())}\n\n".encode()
                handler.wfile.write(body)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, Exception):
            pass
        finally:
            state.unsubscribe_sse(evt)
        return True
    return False


def _send_json(handler, code, payload):
    body = json.dumps(payload).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def handle_observer_post(handler):
    """Control endpoints: update the club-3090 checkout / restart the model.

    Single-flight: only one control action at a time per host. The Tailscale
    binding is the auth boundary, matching the existing PUT /curve design.
    """
    path = handler.path.split("?", 1)[0]
    if path not in ("/observer/api/update", "/observer/api/restart"):
        return False
    body = {}
    length = int(handler.headers.get("Content-Length") or 0)
    if length > 4096:
        _send_json(handler, 400, {"error": "oversized body"})
        return True
    if length > 0:
        try:
            body = json.loads(handler.rfile.read(length))
        except json.JSONDecodeError as e:
            _send_json(handler, 400, {"error": f"invalid JSON: {e}"})
            return True
    if not _control_lock.acquire(blocking=False):
        _send_json(handler, 409, {"error": "another control action is running"})
        return True
    try:
        if path == "/observer/api/update":
            repo = _config.get("model_repo")
            if not repo:
                _send_json(handler, 503, {"error": "no model repo configured"})
                return True
            result = update_repo(repo)
            _repo_wake.set()
        else:
            check_restart_allowed(state, force=bool(body.get("force")))
            result = restart_model(str(body.get("preset", "insight")))
        _send_json(handler, 200, result)
    except ValueError as e:
        _send_json(handler, 400, {"error": str(e)})
    except Exception as e:
        _send_json(handler, 409, {"error": str(e)})
    finally:
        _control_lock.release()
    return True


def start_observer(monitor_port=DEFAULT_MONITOR_PORT, container=DEFAULT_CONTAINER,
                   model_repo=DEFAULT_MODEL_REPO):
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    _config["monitor_port"] = monitor_port
    _config["model_repo"] = model_repo or None
    threading.Thread(target=poll_gpu_stats, name="observer-gpu", daemon=True).start()
    threading.Thread(
        target=monitor_connections,
        args=(monitor_port,),
        name="observer-connections",
        daemon=True,
    ).start()
    threading.Thread(
        target=poll_model,
        args=(monitor_port,),
        name="observer-model",
        daemon=True,
    ).start()
    threading.Thread(
        target=poll_slots,
        args=(monitor_port,),
        name="observer-slots",
        daemon=True,
    ).start()
    threading.Thread(
        target=poll_metrics,
        args=(monitor_port,),
        name="observer-metrics",
        daemon=True,
    ).start()
    threading.Thread(
        target=tail_docker_logs,
        args=(container, monitor_port),
        name="observer-docker-logs",
        daemon=True,
    ).start()
    threading.Thread(
        target=poll_model_info,
        args=(monitor_port,),
        name="observer-model-info",
        daemon=True,
    ).start()
    if model_repo:
        threading.Thread(
            target=poll_repo,
            args=(model_repo,),
            name="observer-repo",
            daemon=True,
        ).start()
    print(
        f"Observer enabled at /observer (host {HOSTNAME}, monitor :{monitor_port}, "
        f"container {container or 'auto-detect'})",
        flush=True,
    )
