#!/usr/bin/env python3
"""Integrated aipc observer dashboard.

This module folds the old aipc-observer sidecar into the fan controller's HTTP
server. It monitors llama.cpp docker logs, active TCP connections, and nvidia-smi
GPU stats, then serves a small dashboard at /observer with SSE updates.
"""

import ast
import contextlib
import json
import hashlib
import os
import pwd
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime

DEFAULT_MONITOR_PORT = 8020
# None => auto-detect the container publishing the monitor port.
DEFAULT_CONTAINER = None
# The club-3090 checkout that launches the model server; the observer reports
# its version and how far it is behind upstream. Empty/None disables this.
DEFAULT_MODEL_REPO = "/home/graywzc/projects/club-3090"
DEFAULT_MODEL_CACHE_DIR = "models-cache"
GPU_POLL_INTERVAL = 2.0
CONN_CHECK_INTERVAL = 3.0
MODEL_POLL_INTERVAL = 30.0
SLOTS_POLL_INTERVAL = 2.0
METRICS_POLL_INTERVAL = 2.0
CONTAINER_DETECT_INTERVAL = 5.0
MODEL_INFO_POLL_INTERVAL = 30.0
REPO_POLL_INTERVAL = 900.0
REQUEST_LOG_MAX = 200
DETAIL_TEXT_MAX = 12000
DETAIL_JSON_MAX = 24000
MESSAGE_TEXT_MAX = 4000
# JSON is valid YAML, so the generated compose override is written as JSON.
OVERRIDE_FILE = "/tmp/aipc-observer-compose-override.yml"
AUDIT_LOG = "/var/log/aipc-observer-actions.log"
# Docker log rotation applied on every controlled restart. The container
# otherwise runs json-file with no limits, and debug mode logs
# full request bodies — a long context is a ~400 KB log line.
LOG_ROTATE_MAX_SIZE = "100m"
LOG_ROTATE_MAX_FILE = "5"
OBSERVER_PRESET_LABEL = "aipc.observer.preset"
OBSERVER_CACHE_RAM_LABEL = "aipc.observer.cache_ram"
# How many recent model-load profiles to retain for the dashboard/history.
LOAD_PROFILE_HISTORY = 20
# Readiness gate: how long to wait for the model to answer /v1/models after a
# (re)start before giving up, and how often to poll for it.
SERVING_READY_TIMEOUT = 900.0
SERVING_POLL_INTERVAL = 2.0
# /proc/diskstats reports I/O in 512-byte sectors regardless of device geometry.
DISK_SECTOR_BYTES = 512

# Oncall watchdog: detect a crashed/OOM'd model, alert ntfy, and auto-revive.
# All overridable via env so a deploy can retarget the alert without a code edit.
NTFY_SERVER = os.environ.get("AIPC_OBSERVER_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.environ.get("AIPC_OBSERVER_NTFY_TOPIC", "oncall-alert")
# Optional bearer token for protected ntfy topics; the public ntfy.sh topic
# needs none.
NTFY_TOKEN = os.environ.get("AIPC_OBSERVER_NTFY_TOKEN") or None
WATCHDOG_ENABLED = os.environ.get("AIPC_OBSERVER_WATCHDOG", "1") != "0"
WATCHDOG_INTERVAL = 5.0
# How long /v1/models must stay unreachable (after the model was healthy)
# before we call it a crash. Long enough to ride out a stray failed poll.
WATCHDOG_DOWN_GRACE = 30.0
# Auto-revive attempts per crash episode before we give up and stay quiet
# (avoids hammering a GPU that OOMs on every boot).
MAX_REVIVE_ATTEMPTS = 3
# Backoff between revive attempts: REVIVE_BACKOFF_BASE * 2**attempt seconds.
REVIVE_BACKOFF_BASE = 30.0
# Docker log substrings that mark a crash as an out-of-memory kill.
OOM_LOG_SIGNATURES = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
    "out of memory",
)

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
        # Case-fan readings (liquidctl/hwmon) pushed by the fan controller.
        self.case_fans = []
        # docker-inspect view of the model container (image, flags, variant).
        self.model_info = {}
        # club-3090 checkout state (HEAD, commits behind upstream).
        self.repo_info = {}
        # Variant catalog extracted from the repo's compose registry, and the
        # recommendation diff between local HEAD and the fetched upstream.
        self.catalog = {}
        self.catalog_diff = {}
        self.installed_assets = {}
        # Latest scrape of the llama.cpp Prometheus /metrics endpoint.
        self.metrics = {}
        # Rolling history of vLLM metric samples for the live activity timeline.
        self.vllm_history = deque(maxlen=180)
        # Progress of the current/last async control action (model switch).
        self.control_status = {}
        # Recent raw docker log lines for the Docker Logs card.
        self.docker_logs = deque(maxlen=500)
        # Recent end-to-end model-load profiles (button click -> serving).
        self.load_profiles = deque(maxlen=LOAD_PROFILE_HISTORY)
        # Most recent model start action (merged install+start/switch).
        self.last_start = {}
        self.last_start_log = deque(maxlen=400)

    def add_request(self, req):
        with self.lock:
            self.requests.append(req)

    def add_vllm_sample(self, sample):
        with self.lock:
            self.vllm_history.append(sample)

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

    def prune_inactive_requests(self):
        """Drop stale active rows when no matching slot is processing.

        This is the control-plane guard companion to enrich_active_from_slots().
        If a stop/restart request arrives between slot polls, the guard should
        not be held hostage by a missed release log line when /slots already
        shows the task is idle.
        """
        now = time.time()
        processing = {s.get("id_task") for s in self.slots if s.get("is_processing")}
        with self.lock:
            for task_id in list(self.active_requests):
                req = self.active_requests[task_id]
                if task_id not in processing and (
                    now - req.get("start_time", now)
                ) > SLOTS_POLL_INTERVAL * 2:
                    del self.active_requests[task_id]

    def prune_vllm_inactive_requests(self, metrics=None):
        """Drop vLLM rows that the authoritative engine gauges prove inactive.

        vLLM does not expose per-request slots, and the log tail can miss the
        final "Generated response" line that normally clears an active row.
        When a fresh metrics scrape reports zero running and zero waiting
        requests, any row that already existed at the scrape time is a ghost.
        """
        metrics = metrics or self.metrics
        if metrics.get("engine") != "vllm" or not metrics.get("available"):
            return 0
        if metrics.get("processing") != 0 or metrics.get("queued") != 0:
            return 0
        scraped_at = metrics.get("scraped_at") or time.time()
        removed = 0
        with self.lock:
            for task_id in list(self.active_requests):
                req = self.active_requests[task_id]
                if req.get("request_id") and req.get("start_time", 0) <= scraped_at:
                    del self.active_requests[task_id]
                    removed += 1
        return removed

    def set_vram_temps(self, mapping):
        with self.lock:
            self.vram_temps = dict(mapping)

    def set_case_fans(self, fans):
        with self.lock:
            self.case_fans = list(fans)

    def set_model_info(self, info):
        with self.lock:
            self.model_info = dict(info)
            self._seed_running_installed_locked()

    def set_repo_info(self, info):
        with self.lock:
            self.repo_info = dict(info)

    def set_catalog(self, catalog):
        with self.lock:
            self.catalog = dict(catalog)
            self._seed_running_installed_locked()

    def _seed_running_installed_locked(self):
        """Mark the running variant's assets as installed; caller holds the lock.

        The container that's actually serving necessarily has its weights on
        disk, but installed_assets otherwise only records observer-driven
        installs — so without this the running model wrongly shows an "install"
        button. Matches the catalog key the dashboard uses (compose_path is a
        substring of the running container's compose_file). Additive only:
        switching away leaves the prior variant marked, since it's still on disk.
        """
        compose_file = (self.model_info or {}).get("compose_file") or ""
        if not compose_file:
            return
        variants = (self.catalog or {}).get("variants") or {}
        for key, entry in variants.items():
            cp = (entry or {}).get("compose_path")
            if cp and cp in compose_file and key not in self.installed_assets:
                self.installed_assets[key] = {
                    "installed_at": time.time(),
                    "source": "running",
                }
                return

    def set_catalog_diff(self, diff):
        with self.lock:
            self.catalog_diff = dict(diff)

    def mark_assets_installed(self, variant, detail=None):
        with self.lock:
            self.installed_assets[str(variant)] = {
                "installed_at": time.time(),
                **(detail or {}),
            }

    def merge_installed_assets(self, assets):
        with self.lock:
            for variant, detail in (assets or {}).items():
                if variant not in self.installed_assets:
                    self.installed_assets[str(variant)] = {
                        "installed_at": time.time(),
                        **(detail or {}),
                    }

    def set_metrics(self, metrics):
        with self.lock:
            self.metrics = dict(metrics)

    def set_control_status(self, status):
        with self.lock:
            self.control_status = dict(status)

    def add_docker_log(self, line):
        with self.lock:
            self.docker_logs.append(line.rstrip())

    def add_load_profile(self, record):
        with self.lock:
            self.load_profiles.append(record)

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

    def start_run(self, action, variant, preset, cache_ram):
        """Begin tracking a model start/switch run."""
        with self.lock:
            self.last_start_log.clear()
            self.last_start = {
                "action": action,
                "variant": variant,
                "preset": preset,
                "cache_ram": cache_ram,
                "started_at": time.time(),
                "finished_at": None,
                "ok": None,
                "detail": "",
            }

    def append_start_log(self, line):
        """Append a raw log line to the start log buffer."""
        with self.lock:
            self.last_start_log.append(str(line).rstrip())

    def finish_run(self, ok, detail):
        """Mark the current start run as completed."""
        with self.lock:
            self.last_start["finished_at"] = time.time()
            self.last_start["ok"] = ok
            self.last_start["detail"] = str(detail)

    def snapshot(self):
        # Serialize any in-flight load before taking self.lock: as_dict() samples
        # VRAM, which itself grabs self.lock — doing it inside would deadlock.
        active = get_active_profile()
        active_profile = active.as_dict() if active is not None else None
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
                "installed_assets": dict(self.installed_assets),
                "metrics": dict(self.metrics),
                "vllm_history": list(self.vllm_history),
                "control_status": dict(self.control_status),
                "control_busy": _control_lock.locked(),
                "load_profiles": list(self.load_profiles),
                "active_load_profile": active_profile,
                "watchdog": _watchdog.summary(),
                "docker_logs": list(self.docker_logs)[-100:],
                "uptime_seconds": uptime,
                "uptime_human": format_duration(uptime),
                "gpu_stats": list(self.gpu_stats),
                "case_fans": list(self.case_fans),
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
                "last_start": {
                    **self.last_start,
                    "log": list(self.last_start_log),
                },
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


def _bounded_text(text, limit):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "\n… truncated …"


def _bounded_json(value, limit=DETAIL_JSON_MAX):
    try:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return _bounded_text(text, limit)


def request_detail_metadata(payload):
    if not isinstance(payload, dict):
        return {}
    messages = payload.get("messages")
    detail = {
        "request_model": payload.get("model"),
        "request_stream": bool(payload.get("stream")),
        "request_temperature": payload.get("temperature"),
        "request_tools_count": len(payload.get("tools") or []),
        "request_has_response_format": bool(payload.get("response_format")),
        "request_detail_json": _bounded_json(payload),
    }
    if isinstance(messages, list):
        detail["request_messages"] = [
            {
                "role": msg.get("role") or "",
                "name": msg.get("name") or "",
                "content": _bounded_text(
                    _text_from_content(msg.get("content")), MESSAGE_TEXT_MAX
                ),
            }
            for msg in messages if isinstance(msg, dict)
        ]
    return detail


def response_detail_metadata(payload):
    if not isinstance(payload, dict):
        return {}
    outputs = []
    finish_reason = None
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        finish_reason = finish_reason or choice.get("finish_reason")
        msg = choice.get("message") or {}
        if isinstance(msg, dict):
            content = _text_from_content(msg.get("content"))
            if content:
                outputs.append(content)
            if msg.get("tool_calls"):
                outputs.append(_bounded_json(msg.get("tool_calls"), MESSAGE_TEXT_MAX))
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = _text_from_content(delta.get("content"))
            if content:
                outputs.append(content)
    if not outputs and payload.get("content") is not None:
        outputs.append(_text_from_content(payload.get("content")))
    if not outputs and payload.get("text") is not None:
        outputs.append(_text_from_content(payload.get("text")))
    output = "\n\n".join(o for o in outputs if o)
    return {
        "response_output": _bounded_text(output, DETAIL_TEXT_MAX),
        "response_finish_reason": finish_reason,
        "response_detail_json": _bounded_json(payload),
    }


def request_group_metadata(payload):
    """Derive passive conversation grouping from a debug-mode request body.

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


def _command_argv(command):
    if command is None:
        return []
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return [command]
    return [str(tok) for tok in command if str(tok)]


def _command_option_values(command):
    values = {}
    argv = _command_argv(command)
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("-"):
            i += 1
            continue
        if tok.startswith("--") and "=" in tok:
            flag, value = tok.split("=", 1)
        else:
            flag, value = tok, None
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                value = argv[i + 1]
                i += 1
        values[flag] = value
        i += 1
    return values


def build_flag_compare_matrix(results, help_index):
    per_variant = {
        item["variant"]: _command_option_values(item.get("command"))
        for item in results if not item.get("error")
    }
    flags = sorted({f for values in per_variant.values() for f in values})
    rows = []
    for flag in flags:
        help_entry = (help_index or {}).get(flag) or {}
        rows.append({
            "flag": flag,
            "description": help_entry.get("description") or "",
            "aliases": help_entry.get("aliases") or [flag],
            "known": bool(help_entry),
            "values": {
                variant: values.get(flag)
                for variant, values in per_variant.items()
                if flag in values
            },
        })
    return rows


# vLLM's `serve --help` can't be probed by exec-ing the running container: its
# entrypoint is a `bash -c <launch script>`, so appending --help just re-runs
# the launcher. Even a direct `vllm serve --help` fails in a docker-exec'd
# process because NVML can't initialise there (CUDA platform -> unavailable ->
# device inference raises "Failed to infer device type"). A throwaway
# `docker run --gpus all` wires NVML up correctly. v0.22.0 also paginates help,
# so the full flag list needs `--help=all`. The probe spins a fresh container
# (slow), so results are cached per image — help only changes when the image
# does. Failures aren't cached, so a transient error retries on the next probe.
VLLM_HELP_PROBE_TIMEOUT = 120
_vllm_help_cache = {}


def _inspect_vllm_help(image, runner=subprocess.run):
    if not image:
        return None
    cached = _vllm_help_cache.get(image)
    if cached is not None:
        return cached
    argv = ["docker", "run", "--rm", "--gpus", "all",
            "--entrypoint", "vllm", image, "serve", "--help=all"]
    source = " ".join(argv)
    try:
        result = runner(argv, capture_output=True, text=True,
                        timeout=VLLM_HELP_PROBE_TIMEOUT)
    except Exception as e:
        return {"source": source, "error": str(e)}
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    index = parse_help_flags(output)
    if result.returncode != 0 and not index:
        return {
            "source": source,
            "error": output.strip()[-400:] or "vllm serve --help failed",
        }
    info = {
        "source": source,
        "flag_count": len(index),
        "flags": index,
        **({"warning": f"--help exited {result.returncode}"} if result.returncode != 0 else {}),
    }
    _vllm_help_cache[image] = info
    return info


def inspect_container_help(name, entrypoint, engine=None, image=None):
    """Run the server's --help and parse flag descriptions.

    vLLM is special-cased to a throwaway-container probe by image (see
    _inspect_vllm_help); every other engine execs the running container.
    """
    if engine == "vllm":
        return _inspect_vllm_help(image)
    argv = _entrypoint_argv(entrypoint)
    if not name or not argv:
        return None
    source = " ".join([*argv, "--help"])
    try:
        result = subprocess.run(
            ["docker", "exec", name, *argv, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return {"source": source, "error": str(e)}
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    index = parse_help_flags(output)
    if result.returncode != 0 and not index:
        return {
            "source": source,
            "error": output.strip()[-400:] or "server --help failed",
        }
    return {
        "source": source,
        "flag_count": len(index),
        "flags": index,
        **({"warning": f"--help exited {result.returncode}"} if result.returncode != 0 else {}),
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
                '"labels":{{json .Config.Labels}},"env":{{json .Config.Env}},'
                '"mounts":{{json .Mounts}},'
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
    env_map = _env_list_to_map(data.get("env"))
    compose_file = labels.get("com.docker.compose.project.config_files") or ""
    cmd = data.get("cmd") or []
    engine_probe = {
        "compose_file": compose_file,
        "variant": variant_from_compose_path(compose_file),
        "image": data.get("image"),
    }
    engine = infer_engine(engine_probe)
    labeled_preset = normalize_observer_mode(labels.get(OBSERVER_PRESET_LABEL))
    labeled_cache = labels.get(OBSERVER_CACHE_RAM_LABEL)
    if labeled_cache is None:
        labeled_cache = infer_cache_ram_enabled(cmd)
    else:
        labeled_cache = labeled_cache.lower() == "true"
    cache_supported = engine != "vllm"
    if not cache_supported:
        labeled_cache = False
    help_info = inspect_container_help(
        name, data.get("entrypoint"), engine=engine, image=data.get("image")
    )
    help_index = (help_info or {}).get("flags") or {}
    if help_info and "flags" in help_info:
        help_info = {k: v for k, v in help_info.items() if k != "flags"}
    info = {
        "container": name,
        "image": data.get("image"),
        "entrypoint": _entrypoint_argv(data.get("entrypoint")),
        "compose_file": compose_file,
        "variant": variant_from_compose_path(compose_file),
        "engine": engine,
        "command": cmd,
        "preset": labeled_preset or infer_insight_preset(
            cmd, engine=engine, env=env_map
        ),
        "cache_ram_enabled": labeled_cache,
        "cache_ram_supported": cache_supported,
        "vllm_logging_level": env_map.get("VLLM_LOGGING_LEVEL"),
        "flags": summarize_command(cmd),
        "help": help_info,
        "command_guide": command_guide(cmd, help_index),
        "project": labels.get("com.docker.compose.project"),
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


# Make git non-interactive so a control action can never block on a prompt,
# and make the network give up quickly instead of stalling: a hung fetch used
# to wedge the single _control_lock forever (the daemon captured git's pipes
# through a runuser wrapper, so subprocess.run's timeout could not reap the
# surviving ssh child and communicate() blocked past the timeout).
_GIT_NONINTERACTIVE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_SSH_COMMAND": (
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
    ),
}


def _kill_process_group(proc):
    """SIGKILL the child's whole process group, reaping orphaned ssh/git."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def repo_git(repo, *args, timeout=60):
    """Run git in the model repo, dropping to the repo owner when root.

    Hardened so a stalled remote can never wedge a control action: git runs
    non-interactively with stdin closed (no credential/host-key prompt can
    block), and in its own process group so a timeout SIGKILLs the whole tree
    -- including any orphaned ssh -- instead of blocking forever on a surviving
    grandchild that still holds the output pipe.
    """
    cmd = _repo_owner_cmd(repo, ["git", "-C", repo, *args])
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, **_GIT_NONINTERACTIVE_ENV}, start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(f"git {args[0]} timed out after {timeout}s (killed)")
    if proc.returncode != 0:
        raise RuntimeError((err or "").strip() or f"git {args[0]} failed")
    return (out or "").strip()


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
DUAL_CARD_DOC_PATH = "docs/DUAL_CARD.md"

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
        attach_dual_card_docs(repo, ref, catalog)
        catalog["ref"] = ref
        return catalog
    except Exception as e:
        return {"error": str(e), "ref": ref}


def _clean_md_cell(value):
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", str(value or ""))
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = value.replace("**", "").replace("~~", "")
    value = value.replace("`", "")
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _dual_card_rows(markdown):
    """Extract pipe-table data rows from docs/DUAL_CARD.md."""
    rows = []
    current = None
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|"):
            if current:
                rows.append(current)
            current = line
        elif current and "|" in line:
            # Some long cells wrap in markdown. Keep wrapped notes attached to
            # the row before parsing cells.
            current += " " + line
    if current:
        rows.append(current)

    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        first = _clean_md_cell(cells[0]).lower()
        if first in ("what you're doing", "---") or set(first) <= {"-", " "}:
            continue
        parsed.append({
            "workload_label": _clean_md_cell(cells[0]),
            "compose": _clean_md_cell(cells[1]),
            "max_ctx_doc": _clean_md_cell(cells[2]),
            "tps": _clean_md_cell(cells[3]),
            "vram": _clean_md_cell(cells[4]),
            "why": _clean_md_cell(" | ".join(cells[5:])),
            "_raw": row,
        })
    return parsed


def _doc_row_matches_variant(row, key, entry):
    raw = row.get("_raw") or ""
    compose_path = entry.get("compose_path") or ""
    if key and re.search(rf"(?<![\w/-]){re.escape(key)}(?![\w/-])", raw):
        return True
    if compose_path and compose_path in raw:
        return True
    base = os.path.basename(compose_path)
    if base and base in raw:
        return True
    return False


def attach_dual_card_docs(repo, ref, catalog):
    """Attach curated DUAL_CARD table metadata to matching variants."""
    variants = catalog.get("variants") or {}
    if not variants:
        return
    try:
        src = repo_git(repo, "show", f"{ref}:{DUAL_CARD_DOC_PATH}")
    except Exception:
        return
    rows = _dual_card_rows(src)
    for key, entry in variants.items():
        for row in rows:
            if _doc_row_matches_variant(row, key, entry):
                entry["doc"] = {
                    k: row[k] for k in (
                        "workload_label", "compose", "max_ctx_doc",
                        "tps", "vram", "why"
                    )
                }
                break


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

# Observer modes are command-line flag tweaks, not env vars: llama.cpp CLI
# args take precedence over LLAMA_ARG_* env, and flags like --cache-ram are
# already present in the compose command, so env could never override them.
# (flag, None) appends a switch if absent; (flag, value) replaces or appends.
# Modes are defined by *capability* (what we want on), not by hard flags,
# because llama.cpp forks spell the same capability differently (ik-llama uses
# --verbosity where mainline/beellama use --log-verbosity, etc.). Each
# capability lists candidate (flag, value) pairs in preference order; at
# restart time the first candidate the running build advertises in --help wins
# (see resolve_preset). The trace/verbose distinction is the log level.
CAPABILITY_FLAGS = {
    "metrics": [("--metrics", None)],
    "props": [("--props", None)],
    "ram_cache": [("--cache-ram", "8192")],
    "verbose_logging": [("--log-verbosity", "4"), ("--verbosity", "4")],
    "trace_logging": [("--log-verbosity", "5"), ("--verbosity", "5")],
    "timestamps": [("--log-timestamps", None), ("--log-format", "json")],
}

MODE_CAPABILITIES = {
    "baseline": [],
    # Debug verbosity logs full request/response bodies where the engine
    # supports it. Very chatty; use for experiments.
    "debug": ["metrics", "props", "trace_logging", "timestamps"],
}
LEGACY_PRESET_TO_MODE = {
    "baseline": "baseline",
    "insight": "debug",
    "insight-cache": "debug",
    "insight-debug": "debug",
    "debug": "debug",
}
LEGACY_PRESET_CACHE_DEFAULT = {
    "baseline": False,
    "insight": False,
    "insight-cache": True,
    "insight-debug": True,
    "debug": True,
}

# Legacy flag view of each observer mode (the first/default candidate per
# capability). Kept for validation call sites that historically referenced
# INSIGHT_PRESETS.
INSIGHT_PRESETS = {
    name: [CAPABILITY_FLAGS[cap][0] for cap in caps]
    for name, caps in MODE_CAPABILITIES.items()
}

# vLLM speaks its own flags, and the llama.cpp capability/--help machinery
# doesn't apply (vLLM's `serve --help` can't even run in a busy container — it
# re-infers the GPU device and errors). So debug mode turns on vLLM's
# per-request logging, which feeds the dashboard's request rows. Output logging
# is what carries the finish reason + output token ids per request.
VLLM_REQUEST_LOG_FLAGS = [("--enable-log-requests", None),
                          ("--enable-log-outputs", None)]
VLLM_PRESET_FLAGS = {
    "baseline": [],
    "debug": VLLM_REQUEST_LOG_FLAGS,
}
VLLM_PRESET_ENV = {
    "debug": {"VLLM_LOGGING_LEVEL": "DEBUG"},
}
CACHE_RAM_TWEAK = ("--cache-ram", "8192")

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


def remove_command_options(cmd, flags):
    """Remove flag/value options from argv, including --flag=value spelling."""
    flags = {normalize_preset_flag(f) for f in flags}
    argv = list(cmd or [])
    out = []
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        flag = tok.split("=", 1)[0] if tok.startswith("--") else tok
        if normalize_preset_flag(flag) in flags:
            if "=" not in tok and i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                i += 2
            else:
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


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


def command_capabilities(cmd):
    """Map a live server argv to the set of insight capabilities it enables.

    Capability-based (not flag-based) so a translated build — ik-llama running
    --verbosity 5 instead of --log-verbosity 5 — still reads as trace_logging.
    """
    options = _command_option_map(cmd)
    caps = set()
    if "--metrics" in options:
        caps.add("metrics")
    if "--props" in options:
        caps.add("props")
    if options.get("--cache-ram") not in (None, "0"):
        caps.add("ram_cache")
    if "--log-timestamps" in options or options.get("--log-format") == "json":
        caps.add("timestamps")
    level = options.get("--log-verbosity", options.get("--verbosity"))
    try:
        level = int(level)
    except (TypeError, ValueError):
        level = None
    if level is not None and level >= 5:
        caps.add("trace_logging")
    elif level is not None and level >= 1:
        caps.add("verbose_logging")
    return caps


def normalize_observer_mode(name):
    return LEGACY_PRESET_TO_MODE.get(str(name or ""), None)


def infer_cache_ram_enabled(cmd):
    options = _command_option_map(cmd)
    return options.get("--cache-ram") not in (None, "0")


def _env_list_to_map(items):
    out = {}
    for item in items or []:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def infer_vllm_preset(cmd, env=None):
    """Infer the high-level observer mode from vLLM-specific argv/env state."""
    options = _command_option_map(cmd)
    has_request_logs = (
        "--enable-log-requests" in options and "--enable-log-outputs" in options
    )
    if not has_request_logs:
        return "baseline"
    return "debug"


def infer_insight_preset(cmd, engine=None, env=None):
    """Infer which observer-managed observability mode the live argv matches."""
    if engine == "vllm":
        return infer_vllm_preset(cmd, env)
    caps = command_capabilities(cmd)
    mode_caps = caps - {"ram_cache"}
    for name in ("debug",):
        if mode_caps == set(MODE_CAPABILITIES[name]):
            return name
    if not mode_caps:
        return "baseline"
    return "custom"


def build_compose_override(service, argv=None, image=None, environment=None,
                           labels=None):
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
    if environment:
        svc["environment"] = environment
    if labels:
        svc["labels"] = labels
    return {"services": {service: svc}}


def check_restart_allowed(observer_state, force=False):
    observer_state.prune_inactive_requests()
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


def _run_with_progress(cmd, env=None, cwd=None, timeout=600, on_line=None):
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines = queue.Queue()

    def read_stdout():
        try:
            buf = []
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    if buf:
                        lines.put("".join(buf))
                    break
                if ch in ("\n", "\r"):
                    if buf:
                        lines.put("".join(buf))
                        buf = []
                else:
                    buf.append(ch)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    tail = deque(maxlen=20)
    deadline = time.monotonic() + timeout if timeout else None
    saw_eof = False

    while not saw_eof:
        if deadline and time.monotonic() > deadline:
            _kill_process_group(proc)
            if proc.stdout:
                proc.stdout.close()
            raise RuntimeError(
                f"command timed out after {timeout}s (killed)"
                + (": " + " | ".join(tail) if tail else "")
            )
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if proc.poll() is not None and not reader.is_alive():
                break
            continue
        if line is None:
            saw_eof = True
            continue
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        if on_line:
            on_line(line)

    if proc.stdout:
        proc.stdout.close()
    try:
        rc = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        raise RuntimeError("command did not exit after output closed (killed)")
    output = "\n".join(tail)
    if rc != 0:
        raise RuntimeError(output[-800:] or "command failed")
    return output


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


def _compose_config_for_variant(repo, key, entry, runner=_run):
    compose_path = (entry or {}).get("compose_path")
    if not compose_path:
        raise ValueError(f"variant {key!r} has no compose_path")
    env = dict(os.environ)
    if entry.get("default_port"):
        env["PORT"] = env["ESTATE_PORT"] = str(entry["default_port"])
    out = runner(
        ["docker", "compose", "-f", compose_path, "config", "--format", "json"],
        env=env, cwd=repo, timeout=60,
    )
    cfg = json.loads(out)
    services = cfg.get("services") or {}
    if not services:
        raise RuntimeError(f"{key}: compose has no services")
    command_services = [
        (name, svc) for name, svc in services.items() if svc.get("command")
    ]
    if command_services:
        service, svc = command_services[0]
    else:
        service, svc = next(iter(services.items()))
    return service, svc


def compare_variant_commands(repo, variants, catalog, runner=_run, help_info=None):
    variants = [str(v) for v in (variants or []) if str(v)]
    if len(variants) < 2:
        raise ValueError("select at least two variants to compare")
    if len(variants) > 4:
        raise ValueError("compare at most four variants at a time")
    results = []
    seen = set()
    entries = (catalog or {}).get("variants") or {}
    if not entries:
        raise RuntimeError("variant catalog not loaded yet; try again shortly")
    for raw in variants:
        key = normalize_switch_variant(raw, catalog)
        if key in seen:
            continue
        seen.add(key)
        entry = entries.get(key)
        if entry is None:
            raise ValueError(f"unknown variant {raw!r}")
        try:
            service, svc = _compose_config_for_variant(
                repo, key, entry, runner=runner
            )
            result = {
                "variant": key,
                "service": service,
                "compose_path": entry.get("compose_path"),
                "image": svc.get("image"),
                "entrypoint": svc.get("entrypoint"),
                "command": svc.get("command"),
                "environment": svc.get("environment") or {},
                "status": entry.get("status"),
                "model": entry.get("model"),
                "engine": entry.get("engine"),
                "tp": entry.get("tp"),
                "error": None,
            }
        except Exception as e:
            result = {
                "variant": key,
                "compose_path": entry.get("compose_path"),
                "status": entry.get("status"),
                "model": entry.get("model"),
                "engine": entry.get("engine"),
                "tp": entry.get("tp"),
                "error": str(e),
            }
        results.append(result)
    help_info = dict(help_info or {})
    help_index = help_info.pop("flags", {}) or {}
    return {
        "variants": results,
        "flag_matrix": build_flag_compare_matrix(results, help_index),
        "help": help_info,
    }


def _supported_flags(model_info, help_getter=inspect_container_help):
    """Flags the running build's --help advertises, or None when unknown.

    Returns None — so callers fail open and apply the preset unchanged — when
    the container can't be inspected. Dropping a flag only happens on a
    positively parsed help screen, never on missing information.
    """
    name = (model_info or {}).get("container")
    if not name:
        return None
    help_info = help_getter(name, (model_info or {}).get("entrypoint"))
    flags = (help_info or {}).get("flags") or {}
    if not flags or (help_info or {}).get("error"):
        return None
    return set(flags)


def resolve_preset(preset_name, supported):
    """Resolve an observer mode's capabilities to flags this build supports.

    For each capability, pick the first candidate flag the build advertises in
    --help (`supported`); other forks spell the same capability differently
    (ik-llama: --verbosity, mainline: --log-verbosity). When help is unknown
    (supported is None) fall back to the first/default candidate so behaviour
    matches the legacy hardcoded preset. Returns (tweaks, dropped) where dropped
    names capabilities no candidate flag could satisfy on this build.
    """
    mode = normalize_observer_mode(preset_name)
    caps = MODE_CAPABILITIES.get(mode)
    if caps is None:
        return None, []
    tweaks, dropped = [], []
    for cap in caps:
        chosen = next(
            ((flag, value) for flag, value in CAPABILITY_FLAGS[cap]
             if supported is None or flag in supported),
            None,
        )
        if chosen is not None:
            tweaks.append(chosen)
        else:
            dropped.append(cap)
    return tweaks, dropped


def restart_model(preset_name, model_info=None, runner=_run, cache_ram=None,
                  override_path=OVERRIDE_FILE, help_getter=inspect_container_help):
    """Recreate the model container with an observer mode/cache toggle applied.

    Presets always build on the compose file's own command (resolved via
    `docker compose config`), never the running container's — so switching
    insight -> baseline sheds flags instead of accumulating them. Each preset
    capability is resolved to a flag the running build advertises in --help, so
    a fork that spells (or lacks) a flag differently boots cleanly instead of
    crash-looping (see resolve_preset).
    """
    mode = normalize_observer_mode(preset_name)
    if mode not in MODE_CAPABILITIES:
        raise ValueError(f"unknown preset {preset_name!r}; "
                         f"known: {', '.join(MODE_CAPABILITIES)}")
    if cache_ram is None:
        cache_ram = LEGACY_PRESET_CACHE_DEFAULT.get(str(preset_name), True)
    mi = model_info if model_info is not None else state.model_info
    missing = [k for k in ("compose_file", "service", "working_dir", "command")
               if not mi.get(k)]
    if missing:
        raise RuntimeError(f"model info incomplete ({', '.join(missing)}); "
                           "is the model container running?")
    compose_file = mi["compose_file"].split(",")[0]
    env = _compose_env(mi)
    cwd = mi["working_dir"]
    engine = infer_engine(mi)
    effective_cache_ram = bool(cache_ram) and engine != "vllm"
    argv = None
    preset_env = None
    dropped = []
    if mode != "baseline" or effective_cache_ram:
        baseline = compose_baseline_command(compose_file, mi["service"], env,
                                            cwd, runner=runner)
        if engine == "vllm":
            # vLLM: inject its own request-logging flags directly; the llama.cpp
            # --help capability probe doesn't apply (and can't run here).
            tweaks = VLLM_PRESET_FLAGS.get(mode, [])
            preset_env = VLLM_PRESET_ENV.get(mode)
        else:
            tweaks, dropped = resolve_preset(
                mode, _supported_flags(mi, help_getter))
            if effective_cache_ram:
                tweaks = [*tweaks, CACHE_RAM_TWEAK]
        argv = apply_preset_to_command(baseline, tweaks)
        if engine == "vllm":
            argv = remove_command_options(argv, ["--cache-ram"])
    # Always pin the running image: launchers may have injected a different
    # image (e.g. BEELLAMA_IMAGE) than the compose file's fallback, and a
    # restart must never silently switch images.
    with open(override_path, "w") as f:
        json.dump(
            build_compose_override(
                mi["service"], argv, image=mi.get("image"),
                environment=preset_env,
                labels={
                    OBSERVER_PRESET_LABEL: mode,
                    OBSERVER_CACHE_RAM_LABEL: str(effective_cache_ram).lower(),
                },
            ),
            f, indent=1,
        )
    os.chmod(override_path, 0o644)
    detail = (f"preset={mode} cache_ram={effective_cache_ram} "
              f"variant={mi.get('variant')}")
    if dropped:
        detail += f" dropped={','.join(dropped)}"
    audit("restart", detail)
    runner(
        ["docker", "compose", "-f", compose_file, "-f", override_path,
         "up", "-d", "--remove-orphans"],
        env=env, cwd=cwd,
    )
    return {"restarted": True, "preset": mode, "cache_ram": effective_cache_ram,
            "variant": mi.get("variant"), "dropped_capabilities": dropped}


def _compose_file_args(compose_file):
    files = [p.strip() for p in (compose_file or "").split(",") if p.strip()]
    args = []
    for path in files:
        args.extend(["-f", path])
    return args


def stop_model(model_info=None, repo=None, runner=_run):
    """Stop the running model compose project without starting a replacement."""
    repo = repo if repo is not None else _config.get("model_repo")
    mi = model_info if model_info is not None else state.model_info
    container = (mi or {}).get("container")
    if model_info is None and not container:
        container = state.container_name
    if not container:
        return {"stopped": False, "detail": "model container is not running"}

    if repo:
        switch_script = os.path.join(repo, "scripts", "switch.sh")
        if os.path.exists(switch_script):
            audit("stop", f"repo={repo} via switch.sh --down")
            _watchdog.mark_deliberately_stopped()
            try:
                runner(
                    _repo_owner_cmd(repo, ["bash", "scripts/switch.sh", "--down"]),
                    env=dict(os.environ),
                    cwd=repo,
                    timeout=300,
                )
            except Exception:
                _watchdog.clear_deliberately_stopped()
                raise
            return {"stopped": True, "detail": "club-3090 switch.sh --down ran"}

    missing = [k for k in ("compose_file", "working_dir") if not mi.get(k)]
    if missing:
        raise RuntimeError(f"model info incomplete ({', '.join(missing)}); "
                           "cannot identify compose project to stop")
    cmd = ["docker", "compose"]
    if mi.get("project"):
        cmd.extend(["--project-name", mi["project"]])
    cmd.extend(_compose_file_args(mi["compose_file"]))
    cmd.append("down")
    audit("stop", f"variant={mi.get('variant')} container={mi.get('container')}")
    _watchdog.mark_deliberately_stopped()
    try:
        runner(cmd, env=_compose_env(mi), cwd=mi["working_dir"], timeout=300)
    except Exception:
        _watchdog.clear_deliberately_stopped()
        raise
    return {"stopped": True, "variant": mi.get("variant"),
            "container": mi.get("container")}


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


# --- model-load profiling ---------------------------------------------------
#
# Times one end-to-end model load: from the control request (the dashboard
# button click) until the model actually answers on /v1/models. Each control
# path drives a LoadProfile through named phases; host resource counters (disk
# read, page cache, GPU VRAM) are diffed across the load window so the slow part
# is attributable, and the engine's own startup log lines add an in-container
# breakdown. All parsing/math is in pure functions so it is testable without a
# GPU or docker.

# Whole physical disks in /proc/diskstats (skip partitions and loop/dm/ram/sr,
# which would double-count or add noise to the read total).
WHOLE_DISK_RE = re.compile(r"^(nvme\d+n\d+|mmcblk\d+|(?:sd|vd|hd|xvd)[a-z]+)$")


def read_disk_read_bytes(text):
    """Total bytes read across whole disks from /proc/diskstats text.

    Columns are: major minor name reads merged sectors_read ... — so field 5
    (0-indexed) is sectors read; multiply by the 512-byte sector unit.
    """
    total = 0
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 6 or not WHOLE_DISK_RE.match(parts[2]):
            continue
        try:
            total += int(parts[5]) * DISK_SECTOR_BYTES
        except ValueError:
            continue
    return total


def parse_meminfo(text):
    """Parse /proc/meminfo (kB fields) into a {name: bytes} dict."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"(\w+):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def _current_vram_used_mib():
    """Sum VRAM used across GPUs from the latest nvidia-smi sample, or None."""
    with state.lock:
        gpus = list(state.gpu_stats)
    if not gpus:
        return None
    return sum(g.get("mem_used_mib") or 0 for g in gpus)


def sample_resources():
    """Snapshot host counters used for load attribution; missing -> None.

    Linux-only files degrade to None elsewhere (e.g. the macOS test box), so the
    profiler still records phases and timings without resource attribution.
    """
    snap = {"t": time.time(), "disk_read_bytes": None,
            "mem_cached": None, "mem_available": None}
    try:
        with open("/proc/diskstats") as f:
            snap["disk_read_bytes"] = read_disk_read_bytes(f.read())
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem = parse_meminfo(f.read())
        snap["mem_cached"] = mem.get("Cached")
        snap["mem_available"] = mem.get("MemAvailable")
    except OSError:
        pass
    snap["vram_used_mib"] = _current_vram_used_mib()
    return snap


# Engine startup log markers -> (sub-phase label, seconds). Matched only while a
# load profile is active. Formats vary across engine versions, so these are
# best-effort and tolerant; "weights load" is the disk->RAM/GPU read.
ENGINE_LOAD_PATTERNS = [
    # vLLM. These are non-overlapping milestones except engine_init, which is a
    # rollup (profile + KV cache + torch.compile + CUDA graph capture + warmup).
    (re.compile(r"Model loading took [\d.]+ ?[GM]i?B(?: memory)? and ([\d.]+) ?(?:s\b|secs?|seconds)"),
     "weights_load"),
    (re.compile(r"[Ll]oading.*weights took ([\d.]+) ?(?:s\b|secs?|seconds)"),
     "weights_load"),
    (re.compile(r"Dynamo bytecode transform time: ([\d.]+) ?(?:s\b|secs?|seconds)"),
     "compile"),
    (re.compile(r"[Gg]raph capturing finished in ([\d.]+) ?(?:s\b|secs?|seconds)"),
     "cuda_graphs"),
    (re.compile(r"[Cc]apturing CUDA graph[s]?.*?(?:in|took) ([\d.]+) ?(?:s\b|secs?|seconds)"),
     "cuda_graphs"),
    (re.compile(r"init engine \([^)]*\) took ([\d.]+) ?(?:s\b|secs?|seconds)"),
     "engine_init"),
    # llama.cpp family (reports ms)
    (re.compile(r"load time\s*=\s*([\d.]+) ?ms"),
     "llama_load_ms"),
]


def parse_engine_load_line(line):
    """Return (sub_phase, seconds) if a startup line reports a load duration."""
    for rx, name in ENGINE_LOAD_PATTERNS:
        m = rx.search(line)
        if not m:
            continue
        try:
            secs = float(m.group(1))
        except (ValueError, IndexError):
            return None
        if name == "llama_load_ms":
            return "weights_load", secs / 1000.0
        return name, secs
    return None


class LoadProfile:
    """Times one end-to-end model load and attributes it to phases/resources."""

    def __init__(self, trigger, variant=None, preset=None, started_at=None):
        self.lock = threading.Lock()
        self.trigger = trigger
        self.variant = variant
        self.preset = preset
        self.started_at = started_at or time.time()
        self.phases = []            # [{name, start, end, duration}]
        self.engine_phases = {}     # sub_phase -> seconds (from container logs)
        self.start_resources = sample_resources()
        self.end_resources = None
        self.peak_vram_mib = self.start_resources.get("vram_used_mib")
        # Track the trough too: on a switch the start sample still includes the
        # outgoing model, so VRAM dips when it stops and rises as the new one
        # loads. "Filled" is peak - trough, which captures the new model even
        # when start was already high.
        self.min_vram_mib = self.start_resources.get("vram_used_mib")
        self.peak_disk_read_bps = 0.0
        self.total = None
        self.ok = None
        self.error = None

    @contextlib.contextmanager
    def phase(self, name):
        start = time.time()
        rec = {"name": name, "start": start, "end": None, "duration": None,
               "children": {}}
        with self.lock:
            self.phases.append(rec)
        try:
            yield self
        finally:
            end = time.time()
            with self.lock:
                rec["end"] = end
                rec["duration"] = end - start
            self.observe_vram()

    def add_engine_phase(self, name, seconds):
        with self.lock:
            # Keep the largest sample if a marker repeats across the double load.
            if seconds > self.engine_phases.get(name, 0):
                self.engine_phases[name] = seconds
            # Nest under the observer phase open when the log line arrived
            # (switch.sh for the 1st load, ready_wait for the preset re-up's
            # 2nd load), which is what makes the dashboard view hierarchical.
            parent = next((p for p in reversed(self.phases)
                           if p.get("end") is None), None)
            if parent is not None:
                children = parent.setdefault("children", {})
                if seconds > children.get(name, 0):
                    children[name] = seconds

    def observe_vram(self):
        used = _current_vram_used_mib()
        if used is None:
            return
        with self.lock:
            if self.peak_vram_mib is None or used > self.peak_vram_mib:
                self.peak_vram_mib = used
            if self.min_vram_mib is None or used < self.min_vram_mib:
                self.min_vram_mib = used

    def observe_disk_rate(self, bps):
        with self.lock:
            if bps > self.peak_disk_read_bps:
                self.peak_disk_read_bps = bps

    def fail(self, msg):
        with self.lock:
            self.ok = False
            self.error = str(msg)[-300:]

    def finalize(self):
        self.observe_vram()
        with self.lock:
            if self.ok is None:
                self.ok = True
            self.total = time.time() - self.started_at
        self.end_resources = sample_resources()
        return self.as_dict()

    def _resource_summary(self):
        s = self.start_resources
        e = self.end_resources or sample_resources()
        out = {}
        if s.get("disk_read_bytes") is not None and e.get("disk_read_bytes") is not None:
            out["disk_read_bytes"] = max(0, e["disk_read_bytes"] - s["disk_read_bytes"])
            out["peak_disk_read_bps"] = self.peak_disk_read_bps
        if s.get("mem_cached") is not None and e.get("mem_cached") is not None:
            out["page_cache_delta_bytes"] = e["mem_cached"] - s["mem_cached"]
        if self.peak_vram_mib is not None and self.min_vram_mib is not None:
            # Fill = peak minus trough, so a switch (which starts with the old
            # model still resident) still shows the new model's footprint.
            out["vram_delta_mib"] = self.peak_vram_mib - self.min_vram_mib
            out["vram_peak_mib"] = self.peak_vram_mib
        return out

    def as_dict(self):
        # While the load is still running the open phase has no end yet, so
        # report its live elapsed time (and a live total) so the dashboard can
        # tick through each step as it happens.
        now = time.time()
        with self.lock:
            phases = []
            for p in self.phases:
                d = dict(p)
                d["children"] = dict(p.get("children") or {})
                if d["end"] is None and d["duration"] is None:
                    d["duration"] = now - d["start"]
                    d["running"] = True
                phases.append(d)
            running = self.total is None
            total = self.total if self.total is not None else now - self.started_at
            return {
                "trigger": self.trigger,
                "variant": self.variant,
                "preset": self.preset,
                "started_at": self.started_at,
                "started_at_str": local_time_str(self.started_at),
                "phases": phases,
                "engine_phases": dict(self.engine_phases),
                "resources": self._resource_summary(),
                "total": total,
                "running": running,
                "ok": self.ok,
                "error": self.error,
            }


# The profile currently capturing engine startup log lines, if any.
_active_profile = None
_active_profile_lock = threading.Lock()


def get_active_profile():
    with _active_profile_lock:
        return _active_profile


def _set_active_profile(profile):
    global _active_profile
    with _active_profile_lock:
        _active_profile = profile


def _profile_sampler(profile, stop):
    """Track peak VRAM and disk-read rate for the duration of a load."""
    last = profile.start_resources
    while not stop.wait(SERVING_POLL_INTERVAL):
        profile.observe_vram()
        cur = sample_resources()
        lb, cb = last.get("disk_read_bytes"), cur.get("disk_read_bytes")
        dt = cur["t"] - last["t"]
        if lb is not None and cb is not None and dt > 0:
            profile.observe_disk_rate((cb - lb) / dt)
        last = cur


def _profile_audit_line(record):
    phases = " ".join(
        f"{p['name']}={p['duration']:.1f}s"
        for p in record.get("phases", []) if p.get("duration") is not None
    )
    res = record.get("resources") or {}
    bits = []
    if res.get("disk_read_bytes") is not None:
        bits.append(f"disk_read={res['disk_read_bytes'] / 1e6:.0f}MB")
    if res.get("vram_delta_mib") is not None:
        bits.append(f"vram+={res['vram_delta_mib']:.0f}MiB")
    total = record.get("total")
    return (f"trigger={record.get('trigger')} variant={record.get('variant')} "
            f"ok={record.get('ok')} total={total:.1f}s "
            f"[{phases}]" + (f" {' '.join(bits)}" if bits else "")
            ) if total is not None else f"trigger={record.get('trigger')} (incomplete)"


@contextlib.contextmanager
def profiled_load(trigger, variant=None, preset=None, started_at=None):
    """Wrap a control action so it records an end-to-end LoadProfile.

    The caller marks failure via profile.fail(); an exception that escapes the
    block is also recorded as a failure. On exit the profile is finalized, pushed
    into state history, audited, and subscribers are notified.
    """
    profile = LoadProfile(trigger, variant=variant, preset=preset,
                          started_at=started_at)
    _set_active_profile(profile)
    stop = threading.Event()
    sampler = threading.Thread(target=_profile_sampler, args=(profile, stop),
                               name="observer-load-sampler", daemon=True)
    sampler.start()
    try:
        yield profile
    except BaseException as e:
        profile.fail(e)
        raise
    finally:
        stop.set()
        record = profile.finalize()
        _set_active_profile(None)
        state.add_load_profile(record)
        audit("load-profile", _profile_audit_line(record))
        state.notify_subscribers()


def wait_until_serving(monitor_port, timeout=SERVING_READY_TIMEOUT,
                       interval=SERVING_POLL_INTERVAL, fetch=None):
    """Block until the model answers on /v1/models. Returns seconds waited.

    This is the gate that defines "available for use": restart_model only runs
    `docker compose up -d` and returns before the weights finish loading, so the
    end-to-end profile needs this to find the true ready moment.
    """
    fetch = fetch or fetch_json
    start = time.time()
    deadline = start + timeout
    url = f"http://127.0.0.1:{monitor_port}/v1/models"
    while time.time() < deadline:
        data = fetch(url)
        if data and (data.get("data") or []):
            return time.time() - start
        time.sleep(interval)
    raise RuntimeError(f"model did not start serving within {timeout:.0f}s")


# --- model switching (variant change via club-3090's switch.sh) -------------

# switch.sh waits for readiness itself (READY_TIMEOUT defaults to 600 s),
# plus image pull and model load on top.
SWITCH_TIMEOUT = 1200.0
INSTALL_TIMEOUT = 7200.0


def infer_variant_setup(entry):
    """Best-effort setup.sh model/weight key from a registry entry."""
    model = (entry or {}).get("model")
    compose_path = (entry or {}).get("compose_path") or ""
    if not model:
        return {}
    parts = compose_path.split("/")
    profile = None
    if "compose" in parts:
        idx = parts.index("compose")
        # models/<model>/<engine>/compose/<topology>/<profile>/<file>.yml
        if idx + 2 < len(parts):
            profile = parts[idx + 2]
    hint = {"model": model}
    if profile:
        hint["weight_key"] = f"{model}:{profile}"
    return hint


def _model_cache_roots(repo, model_info=None):
    roots = []
    model_dir = (model_info or {}).get("model_dir")
    if model_dir:
        roots.append(model_dir)
    if repo:
        roots.append(os.path.join(repo, DEFAULT_MODEL_CACHE_DIR))
    seen = set()
    for root in roots:
        root = os.path.abspath(os.path.expanduser(str(root)))
        if root not in seen:
            seen.add(root)
            yield root


def _asset_path_has_files(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_file(follow_symlinks=False):
                    return True
                if entry.is_dir(follow_symlinks=False):
                    return _asset_path_has_files(entry.path)
    except OSError:
        return False
    return False


def _variant_asset_candidates(root, hint):
    model = hint.get("model")
    weight_key = hint.get("weight_key")
    if not model:
        return []
    candidates = []
    if weight_key:
        profile = weight_key.split(":", 1)[1] if ":" in weight_key else weight_key
        candidates.extend([
            os.path.join(root, weight_key),
            os.path.join(root, weight_key.replace(":", os.sep)),
            os.path.join(root, weight_key.replace(":", "_")),
            os.path.join(root, weight_key.replace(":", "-")),
            os.path.join(root, model, profile),
            os.path.join(root, model, "weights", profile),
            os.path.join(root, model, "profiles", profile),
        ])
    candidates.append(os.path.join(root, model))
    seen = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]


def detect_installed_assets(repo, catalog, model_info=None):
    """Detect catalog variants whose model assets already exist on disk."""
    variants = (catalog or {}).get("variants") or {}
    detected = {}
    roots = list(_model_cache_roots(repo, model_info))
    for key, entry in variants.items():
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


def parse_setup_hint(text):
    """Extract setup.sh instructions from club-3090 preflight output."""
    msg = re.sub(r"\s+", " ", str(text or ""))
    m = re.search(
        r"(?:MODEL_DIR=(?P<model_dir>\S+)\s+)?"
        r"WEIGHT_KEY=(?P<weight_key>\S+)\s+"
        r"bash\s+scripts/setup\.sh\s+(?P<model>\S+)",
        msg,
    )
    if not m:
        return {}
    return {k: v for k, v in m.groupdict().items() if v}


def validate_switch(variant, catalog, force=False):
    """Check the requested variant against the extracted compose registry."""
    variants = (catalog or {}).get("variants") or {}
    if not variants:
        print(f"[observer] validate_switch: rejected — catalog not loaded", flush=True)
        raise RuntimeError("variant catalog not loaded yet; try again shortly")
    entry = variants.get(variant)
    if entry is None:
        print(f"[observer] validate_switch: rejected — unknown variant {variant!r}", flush=True)
        raise ValueError(f"unknown variant {variant!r}")
    status = entry.get("status")
    if status not in ("production", "caveats") and not force:
        print(f"[observer] validate_switch: rejected — status={status!r} force={force}", flush=True)
        raise ValueError(
            f"variant {variant!r} has status {status!r}; pass force to switch anyway"
        )
    return entry


def normalize_switch_variant(variant, catalog):
    """Accept either a registry slug or a compose-derived variant path."""
    variants = (catalog or {}).get("variants") or {}
    if variant in variants:
        return variant
    for key, entry in variants.items():
        if variant == variant_from_compose_path(entry.get("compose_path") or ""):
            print(f"[observer] normalize_switch_variant: mapped {variant!r} -> {key!r}", flush=True)
            return key
    if variant not in variants:
        print(f"[observer] normalize_switch_variant: could not resolve {variant!r}, returning as-is", flush=True)
    return variant


def switch_model(repo, variant, monitor_port, force=False, runner=_run):
    """Boot a different club-3090 variant via scripts/switch.sh.

    switch.sh is the repo's own gated path: it stops the running compose,
    runs hardware/weights preflight, boots the variant, and waits for the
    server to answer on READY_URL.
    """
    env = dict(os.environ)
    env["PORT"] = str(monitor_port)
    env["READY_URL"] = f"http://localhost:{monitor_port}/v1/models"
    cmd = ["bash", "scripts/switch.sh"]
    if force:
        cmd.append("--force")
    cmd.append(variant)
    print(f"[observer] switch_model: running {' '.join(cmd)} in {repo}", flush=True)
    runner(_repo_owner_cmd(repo, cmd), env=env, cwd=repo, timeout=SWITCH_TIMEOUT)


def _install_progress_detail(variant, line):
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    if len(line) > 220:
        line = line[-220:]
    return f"installing assets for {variant}: {line}"


def install_variant_assets(repo, variant, catalog, setup=None, runner=_run,
                           progress=None):
    """Run club-3090 setup.sh for a variant's missing model assets."""
    variants = (catalog or {}).get("variants") or {}
    entry = variants.get(variant) or {}
    hint = dict(infer_variant_setup(entry))
    hint.update({k: v for k, v in (setup or {}).items() if v})
    model = hint.get("model") or entry.get("model")
    if not model:
        raise ValueError("install needs a model name")
    env = dict(os.environ)
    if hint.get("model_dir"):
        env["MODEL_DIR"] = hint["model_dir"]
    if hint.get("weight_key"):
        env["WEIGHT_KEY"] = hint["weight_key"]
    if variant_is_gguf(entry):
        env["VERIFY_GLOB_OVERRIDE"] = "*.gguf"
        env["WEIGHT_VERIFY_GLOB"] = "*.gguf"
    detail = f"variant={variant} model={model}"
    if hint.get("weight_key"):
        detail += f" weight_key={hint['weight_key']}"
    if variant_is_gguf(entry):
        detail += " verify_glob=*.gguf"
    audit("install", detail)
    cmd = _repo_owner_cmd(repo, ["bash", "scripts/setup.sh", model])
    if runner is _run:
        _run_with_progress(
            cmd, env=env, cwd=repo, timeout=INSTALL_TIMEOUT,
            on_line=progress,
        )
    else:
        runner(cmd, env=env, cwd=repo, timeout=INSTALL_TIMEOUT)
    return {"installed": True, "variant": variant, **hint}


def _set_control_status(action, detail, done=False, ok=None, **extra):
    payload = {
        "action": action,
        "detail": detail,
        "done": done,
        "ok": ok,
        "updated_at": time.time(),
    }
    payload.update(extra)
    state.set_control_status(payload)
    state.notify_subscribers()


def _start_step(action, detail, **extra):
    """Log a step to both the start log and the control status."""
    state.append_start_log(detail)
    _set_control_status(action, detail, **extra)


def _wait_for_model_info(monitor_port, timeout=120):
    """Wait for the freshly booted container to become inspectable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        name = detect_container(monitor_port)
        if name:
            info = inspect_container(name)
            if info and all(info.get(k) for k in
                            ("compose_file", "service", "working_dir", "command")):
                return info
        time.sleep(2)
    raise RuntimeError("new container did not become inspectable in time")


def _switch_worker(repo, variant, preset, monitor_port, force, runner=_run,
                   info_getter=None, override_path=OVERRIDE_FILE,
                   cache_ram=False, started_at=None, ready_waiter=None):
    """Background body of a model switch; releases the control lock when done.

    switch.sh always boots the variant verbatim; the preset re-up afterwards
    runs for every preset — baseline included — because the override is also
    what applies log rotation and the image pin. The whole sequence is timed as
    a LoadProfile: switch.sh, wait-for-info, the preset re-up (a second model
    load), and the final ready-wait that marks the model actually serving.
    """
    ready_waiter = ready_waiter or wait_until_serving
    state.start_run("start", variant, preset, cache_ram)
    try:
        with profiled_load("switch", variant=variant, preset=preset,
                           started_at=started_at) as profile:
            try:
                print(f"[observer] _switch_worker: START variant={variant!r} preset={preset!r} cache_ram={cache_ram} force={force}", flush=True)
                _start_step(
                    "switch", f"switching to {variant} — old model stopping, "
                              "new model loading (takes a few minutes)…"
                )
                print(f"[observer] _switch_worker: calling switch_model()", flush=True)
                with profile.phase("switch.sh"):
                    switch_model(repo, variant, monitor_port, force=force,
                                 runner=runner)
                print(f"[observer] _switch_worker: switch_model() returned OK", flush=True)
                _start_step(
                    "switch", f"{variant} is up; applying mode {preset} + cache "
                              f"{'on' if cache_ram else 'off'} + log rotation "
                              "(one more model reload)…"
                )
                print(f"[observer] _switch_worker: calling _wait_for_model_info()", flush=True)
                with profile.phase("wait_model_info"):
                    info = (info_getter or _wait_for_model_info)(monitor_port)
                print(f"[observer] _switch_worker: model_info returned, calling restart_model()", flush=True)
                with profile.phase("restart_preset"):
                    result = restart_model(preset, model_info=info, runner=runner,
                                           cache_ram=cache_ram,
                                           override_path=override_path)
                with profile.phase("ready_wait"):
                    ready_waiter(monitor_port)
                dropped = result.get("dropped_capabilities") or []
                note = (f" — this build can't do {', '.join(dropped)}"
                        if dropped else "")
                detail = f"switched to {variant} (mode {result['preset']}, " \
                         f"cache {'on' if result.get('cache_ram') else 'off'}){note}"
                _start_step(
                    "switch", detail,
                    done=True, ok=True
                )
                state.finish_run(True, detail)
                print(f"[observer] _switch_worker: DONE variant={variant!r}", flush=True)
                audit("switch-done", f"variant={variant} preset={preset} "
                                     f"cache_ram={cache_ram}")
            except Exception as e:
                profile.fail(e)
                print(f"[observer] _switch_worker: FAILED variant={variant!r} error={e!r}", flush=True)
                audit("switch-failed", f"variant={variant}: {e}")
                setup = parse_setup_hint(str(e))
                if setup:
                    setup.update({"variant": variant, "preset": preset,
                                  "force": force, "cache_ram": cache_ram})
                state.finish_run(False, str(e))
                _set_control_status(
                    "switch", f"switch to {variant} failed: {e}", done=True,
                    ok=False, install_hint=setup or None,
                )
    finally:
        _control_lock.release()


def _install_worker(repo, variant, preset, monitor_port, force, retry,
                    setup, runner=_run, info_getter=None,
                    override_path=OVERRIDE_FILE, cache_ram=False,
                    started_at=None, ready_waiter=None):
    ready_waiter = ready_waiter or wait_until_serving
    state.start_run("start", variant, preset, cache_ram)
    try:
        with profiled_load("install", variant=variant, preset=preset,
                           started_at=started_at) as profile:
            try:
                print(f"[observer] _install_worker: START variant={variant!r} preset={preset!r} retry={retry}", flush=True)
                _start_step(
                    "install", f"installing assets for {variant} (can take a while)…"
                )
                last_progress_at = [0.0]

                def progress(line):
                    # Append every raw line to the start log immediately.
                    state.append_start_log(line)
                    now = time.monotonic()
                    if now - last_progress_at[0] < 0.75:
                        return
                    last_progress_at[0] = now
                    clean = re.sub(r"\s+", " ", str(line or "")).strip()
                    _start_step(
                        "install", _install_progress_detail(variant, clean),
                        progress_line=clean[-500:],
                    )

                with profile.phase("install_assets"):
                    result = install_variant_assets(
                        repo, variant, state.catalog, setup=setup, runner=runner,
                        progress=progress)
                print(f"[observer] _install_worker: install_variant_assets() returned OK", flush=True)
                state.mark_assets_installed(variant, {
                    k: v for k, v in result.items()
                    if k in ("model", "model_dir", "weight_key")
                })
                state.notify_subscribers()
                if not retry:
                    detail = f"installed assets for {variant}"
                    _start_step(
                        "install", detail,
                        done=True, ok=True, installed_variant=variant,
                    )
                    state.finish_run(True, detail)
                    audit("install-done", f"variant={variant}")
                    print(f"[observer] _install_worker: DONE (no retry) variant={variant!r}", flush=True)
                    return
                print(f"[observer] _install_worker: assets installed, calling switch_model() for retry", flush=True)
                _start_step(
                    "install", f"installed assets for {variant}; retrying switch…",
                    installed_variant=variant,
                )
                with profile.phase("switch.sh"):
                    switch_model(repo, variant, monitor_port, force=force,
                                 runner=runner)
                print(f"[observer] _install_worker: switch_model() returned OK after install", flush=True)
                _start_step(
                    "switch", f"{variant} is up; applying mode {preset} + cache "
                              f"{'on' if cache_ram else 'off'} + log rotation "
                              "(one more model reload)…"
                )
                print(f"[observer] _install_worker: calling _wait_for_model_info() after switch", flush=True)
                with profile.phase("wait_model_info"):
                    info = (info_getter or _wait_for_model_info)(monitor_port)
                print(f"[observer] _install_worker: calling restart_model() after switch", flush=True)
                with profile.phase("restart_preset"):
                    restart_model(preset, model_info=info, runner=runner,
                                  cache_ram=cache_ram,
                                  override_path=override_path)
                with profile.phase("ready_wait"):
                    ready_waiter(monitor_port)
                detail = f"switched to {variant} (mode {preset}, " \
                         f"cache {'on' if cache_ram else 'off'})"
                _start_step(
                    "switch", detail,
                    done=True, ok=True
                )
                state.finish_run(True, detail)
                print(f"[observer] _install_worker: DONE (with retry) variant={variant!r}", flush=True)
                audit("install-switch-done", f"variant={variant} preset={preset} "
                                             f"cache_ram={cache_ram}")
            except Exception as e:
                profile.fail(e)
                print(f"[observer] _install_worker: FAILED variant={variant!r} error={e!r}", flush=True)
                audit("install-failed", f"variant={variant}: {e}")
                state.finish_run(False, str(e))
                _set_control_status(
                    "install", f"install for {variant} failed: {e}",
                    done=True, ok=False,
                )
    finally:
        _control_lock.release()


def _restart_worker(preset, monitor_port, cache_ram=None, runner=_run,
                    override_path=OVERRIDE_FILE, started_at=None,
                    ready_waiter=None):
    """Background body of an in-place restart; releases the control lock.

    restart_model only re-ups the compose project (returns once the container is
    recreated), so the readiness wait is what actually marks the model serving
    again — and what makes the profile's total a true end-to-end measurement.
    """
    ready_waiter = ready_waiter or wait_until_serving
    variant = (state.model_info or {}).get("variant")
    try:
        with profiled_load("restart", variant=variant, preset=preset,
                           started_at=started_at) as profile:
            try:
                _set_control_status(
                    "restart", f"restarting with mode {preset} "
                               "(model reloading)…"
                )
                with profile.phase("restart"):
                    result = restart_model(preset, runner=runner,
                                           cache_ram=cache_ram,
                                           override_path=override_path)
                with profile.phase("ready_wait"):
                    ready_waiter(monitor_port)
                dropped = result.get("dropped_capabilities") or []
                note = (f" — this build can't do {', '.join(dropped)}"
                        if dropped else "")
                _set_control_status(
                    "restart", f"restarted with mode {result['preset']}, cache "
                               f"{'on' if result.get('cache_ram') else 'off'}{note}",
                    done=True, ok=True
                )
                audit("restart-done", f"preset={result['preset']} "
                                      f"cache_ram={result.get('cache_ram')}")
            except Exception as e:
                profile.fail(e)
                audit("restart-failed", f"preset={preset}: {e}")
                _set_control_status(
                    "restart", f"restart failed: {e}", done=True, ok=False,
                )
    finally:
        _control_lock.release()


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


def http_status(url, timeout=5):
    """GET a URL and return its HTTP status code, or None if unreachable.

    Unlike fetch_json/fetch_text, this distinguishes a server that *answers*
    (even with a 5xx) from a dead/refused port — needed to tell "loading" (503)
    apart from "process gone" (connection refused).
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def send_ntfy(message, *, title=None, priority="high", tags=None, timeout=5):
    """Best-effort push to the oncall ntfy topic. Never raises.

    Notifications are advisory: a failed POST must not wedge the watchdog, so
    every error is swallowed (logged to stderr) like the read-only fetch_*
    helpers above. Returns True only on a confirmed 2xx.
    """
    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
    headers = {"Priority": str(priority)}
    if title:
        # ntfy headers must be latin-1 safe; emoji in titles would 400.
        headers["Title"] = title.encode("ascii", "ignore").decode() or "alert"
    if tags:
        headers["Tags"] = ",".join(tags)
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
    try:
        req = urllib.request.Request(
            url, data=message.encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"WARNING: ntfy notify failed: {e}", file=sys.stderr)
        return False


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


def infer_engine(info):
    """Identify the serving engine (vllm vs the llama.cpp family) from a
    model_info dict's compose path / variant / image.

    vLLM speaks a different metrics+logging dialect than llama.cpp/ik-llama, so
    the observer scrapes and renders it differently. ik-llama and llama.cpp both
    expose llama.cpp-style /metrics and trace logs, so they collapse to one
    "llamacpp" engine here. Returns None when the engine can't be determined yet
    (model_info not populated), which the callers treat as the llama.cpp default.
    """
    blob = " ".join(
        str((info or {}).get(k) or "")
        for k in ("compose_file", "variant", "image")
    ).lower()
    if "vllm" in blob:
        return "vllm"
    if any(s in blob for s in ("llamacpp", "llama.cpp", "ik-llama", "ik_llama")):
        return "llamacpp"
    return None


def variant_is_gguf(entry):
    """Return True when a catalog variant entry is a llama.cpp-family (GGUF) engine.

    Uses the same marker vocabulary as infer_engine but operates on a catalog
    entry dict (engine/compose_path/model fields) rather than model_info.
    Returns False for vLLM, SGLang, or empty/unknown entries.
    """
    blob = " ".join(
        str((entry or {}).get(k) or "")
        for k in ("engine", "compose_path", "model")
    ).lower()
    return any(s in blob for s in ("llama-cpp", "llamacpp", "llama.cpp", "ik-llama", "ik_llama", "beellama"))


# request_success_total carries a finished_reason label, so parse_prometheus
# (which strips labels) can't break it down — scan the raw text for the split.
RE_VLLM_SUCCESS = re.compile(
    r'vllm:request_success_total\{[^}]*finished_reason="([^"]+)"[^}]*\}\s+([\d.eE+]+)'
)


def summarize_vllm_metrics(text, prev=None):
    """Map vLLM's Prometheus /metrics into the dashboard's metric snapshot.

    vLLM exposes no live tokens/s gauge, so throughput is derived from the delta
    of the cumulative token counters against the previous scrape (`prev`). Keys
    reuse the llama.cpp snapshot names (processing/queued/*_tokens_total/...) so
    the existing Summary + Server Metrics cards light up unchanged; vLLM-only
    extras (prefix-cache hit, spec-decode acceptance, latency averages, success
    breakdown) carry their own keys and are gated on engine=='vllm' in the UI.
    """
    values = parse_prometheus(text or "")
    out = {"available": True, "engine": "vllm", "scraped_at": time.time()}

    def num(name):
        return values.get(name)

    if num("vllm:num_requests_running") is not None:
        out["processing"] = int(num("vllm:num_requests_running"))
    if num("vllm:num_requests_waiting") is not None:
        out["queued"] = int(num("vllm:num_requests_waiting"))
    if num("vllm:kv_cache_usage_perc") is not None:
        out["kv_cache_usage_ratio"] = num("vllm:kv_cache_usage_perc")
    pt = num("vllm:prompt_tokens_total")
    gt = num("vllm:generation_tokens_total")
    if pt is not None:
        out["prompt_tokens_total"] = int(pt)
    if gt is not None:
        out["gen_tokens_total"] = int(gt)
    pq = num("vllm:prefix_cache_queries_total")
    ph = num("vllm:prefix_cache_hits_total")
    if pq:
        out["prefix_cache_hit_pct"] = round(100.0 * (ph or 0) / pq, 1)
    drafted = num("vllm:spec_decode_num_draft_tokens_total")
    accepted = num("vllm:spec_decode_num_accepted_tokens_total")
    if drafted:
        out["spec_accept_pct"] = round(100.0 * (accepted or 0) / drafted, 1)
    if num("vllm:num_preemptions_total") is not None:
        out["preemptions_total"] = int(num("vllm:num_preemptions_total"))
    # Histogram averages: sum / count (seconds -> ms).
    for key, base in (
        ("avg_ttft_ms", "vllm:time_to_first_token_seconds"),
        ("avg_tpot_ms", "vllm:request_time_per_output_token_seconds"),
        ("avg_e2e_ms", "vllm:e2e_request_latency_seconds"),
    ):
        s = values.get(base + "_sum")
        c = values.get(base + "_count")
        if s is not None and c:
            out[key] = round(1000.0 * s / c, 1)
    # Successful-request count, broken down by finish reason.
    total = 0.0
    by_reason = {}
    for reason, raw in RE_VLLM_SUCCESS.findall(text or ""):
        v = safe_float(raw) or 0.0
        by_reason[reason] = int(v)
        total += v
    if by_reason:
        out["requests_total"] = int(total)
        out["success_by_reason"] = by_reason
    # Throughput from cumulative-counter deltas vs the previous scrape.
    if prev and prev.get("scraped_at"):
        dt = out["scraped_at"] - prev["scraped_at"]
        if dt > 0:
            if pt is not None and prev.get("prompt_tokens_total") is not None:
                out["prompt_tps_avg"] = max(
                    0.0, (out["prompt_tokens_total"] - prev["prompt_tokens_total"]) / dt)
            if gt is not None and prev.get("gen_tokens_total") is not None:
                out["gen_tps_avg"] = max(
                    0.0, (out["gen_tokens_total"] - prev["gen_tokens_total"]) / dt)
    return out


def vllm_timeline_sample(metrics):
    """Compact a vLLM metric snapshot into one point on the activity timeline."""
    return {
        "t": metrics.get("scraped_at") or time.time(),
        "running": metrics.get("processing") or 0,
        "waiting": metrics.get("queued") or 0,
        "gen_tps": round(metrics.get("gen_tps_avg") or 0, 1),
        "prompt_tps": round(metrics.get("prompt_tps_avg") or 0, 1),
        "kv": round(100 * (metrics.get("kv_cache_usage_ratio") or 0), 1),
        "spec": metrics.get("spec_accept_pct"),
    }


def poll_metrics(monitor_port):
    """Scrape the model server's /metrics endpoint.

    llama.cpp only serves /metrics under debug mode (else it 404s →
    available=False, so the dashboard hints at the preset). vLLM serves a richer
    /metrics unconditionally but in its own dialect, so the engine (derived from
    the live model_info) selects the summarizer.
    """
    url = f"http://127.0.0.1:{monitor_port}/metrics"
    last_sig = None
    prev_vllm = None
    while True:
        engine = infer_engine(state.model_info)
        text = fetch_text(url)
        if engine == "vllm":
            if text:
                metrics = summarize_vllm_metrics(text, prev_vllm)
                prev_vllm = metrics
                state.set_metrics(metrics)
                pruned = state.prune_vllm_inactive_requests(metrics)
                state.add_vllm_sample(vllm_timeline_sample(metrics))
                # Push while there's activity so throughput updates live, not
                # only when the running/queued counts flip.
                sig = (metrics.get("queued"), metrics.get("processing"),
                       round(metrics.get("gen_tps_avg") or 0))
                if sig != last_sig or pruned:
                    last_sig = sig
                    state.notify_subscribers()
            else:
                prev_vllm = None
                if state.metrics.get("available"):
                    state.set_metrics({"available": False, "engine": "vllm"})
                    state.notify_subscribers()
                elif not state.metrics:
                    state.set_metrics({"available": False, "engine": "vllm"})
            time.sleep(METRICS_POLL_INTERVAL)
            continue
        values = parse_prometheus(text) if text else {}
        if values:
            metrics = summarize_metrics(values)
            state.set_metrics(metrics)
            queue_state = (metrics.get("queued"), metrics.get("processing"))
            if queue_state != last_sig:
                last_sig = queue_state
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


def probe_model_state(monitor_port):
    """Liveness probe for the watchdog, combining both health signals.

    /health is the engine-blessed endpoint (llama.cpp and vLLM both expose it)
    and uniquely flags the *loading* state — llama.cpp returns 503 while weights
    load — which must NOT be mistaken for a crash. /v1/models is the fallback
    for builds/engines without /health and a corroborating second opinion, so
    one flaky endpoint can't fake a crash. Returns "ready" | "loading" | "down".
    """
    code = http_status(f"http://127.0.0.1:{monitor_port}/health")
    if code == 200:
        return "ready"
    if code == 503:
        return "loading"
    # /health refused, timed out, or unexpected (e.g. a build without it):
    # defer to the OpenAI endpoint before concluding the process is gone.
    if detect_model(monitor_port):
        return "ready"
    return "down"


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
    if "error" not in local:
        st.merge_installed_assets(
            detect_installed_assets(repo, local, st.model_info)
        )
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
                    mem_used = safe_float(parts[6])
                    mem_total = safe_float(parts[7])
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "temp_c": safe_float(parts[2]),
                        "mem_temp_c": mem_temp,
                        "gpu_util_pct": safe_float(parts[4]),
                        "mem_util_pct": (mem_total > 0) and (mem_used / mem_total * 100) or 0,
                        "mem_used_mib": mem_used,
                        "mem_total_mib": mem_total,
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
# Trace/debug-level lines from debug mode:
RE_ACCESS = re.compile(r"done request:\s+(\w+)\s+(\S+)\s+\S+\s+(\d{3})")
RE_REQUEST_BODY = re.compile(r"\brequest:\s*(\{.*\})\s*$")
RE_RESPONSE_BODY = re.compile(r"\bresponse:\s*(\{.*\})\s*$")
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
        self.last_finalized = None

    def process_line(self, line):
        m = RE_REQUEST_BODY.search(line)
        if m:
            try:
                payload = json.loads(m.group(1))
                meta = request_group_metadata(payload)
                meta.update(request_detail_metadata(payload))
            except json.JSONDecodeError:
                meta = {}
            if meta:
                self.pending_request_meta.append(meta)
            return

        m = RE_RESPONSE_BODY.search(line)
        if m:
            try:
                detail = response_detail_metadata(json.loads(m.group(1)))
            except json.JSONDecodeError:
                detail = {}
            if detail:
                self._attach_response_detail(line, detail)
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
        self.last_finalized = req
        self.state.notify_subscribers()

    def _attach_response_detail(self, line, detail):
        req = self._current_request_for_line(line)
        if req is None:
            req = self.last_finalized
        if req is None:
            return
        req.update(detail)
        if req.get("task_id") in self.active:
            self.state.update_active_request(req["task_id"], detail)
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


# vLLM's --enable-log-requests / --enable-log-outputs lines (see RequestLogger
# in vllm/entrypoints/logger.py). Arrival logs params (not the prompt, at INFO).
# Streaming responses log one line per token delta (each with its own
# output_token_ids chunk + finish_reason: None) then a final "streaming
# complete" line whose own output_token_ids is None and finish_reason is the
# literal "streaming_complete" — so the real output token count is summed from
# the deltas and the timing comes from the delta timestamps. Non-streaming
# responses log a single line carrying the full token id list + finish_reason.
RE_VLLM_RECV = re.compile(r"Received request (\S+?): params: (.*), lora_request:")
RE_VLLM_DEBUG = re.compile(
    r"Request (\S+?) details: prompt: (.*), prompt_token_ids: (.*), "
    r"prompt_embeds shape:"
)
RE_VLLM_RESP_ID = re.compile(
    r"Generated response (\S+?)(?: \(streaming[^)]*\))?: output:")
RE_VLLM_IDS = re.compile(r"output_token_ids: (\[[\d,\s]*\]|None)")
RE_VLLM_FINISH = re.compile(r"finish_reason: (\S+)\s*$")
RE_VLLM_MAXTOK = re.compile(r"max_tokens=(\d+)")
RE_VLLM_TEMP = re.compile(r"temperature=([\d.]+)")
VLLM_OUTPUT_PREVIEW_MAX = 2000


def _vllm_count_ids(line):
    """Count the token ids in a 'Generated response' line (0 for None/empty)."""
    m = RE_VLLM_IDS.search(line)
    if not m or m.group(1) == "None" or not m.group(1).strip("[] "):
        return 0
    return m.group(1).count(",") + 1


def _vllm_extract_output(line, limit=VLLM_OUTPUT_PREVIEW_MAX):
    """Pull the response text out of a 'Generated response' line's output: %r."""
    head, sep, _ = line.rpartition(", output_token_ids:")
    if not sep:
        return None
    i = head.find("output: ")
    if i < 0:
        return None
    s = head[i + len("output: "):].strip()
    if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        s = s[1:-1]
    s = (s.replace("\\n", "\n").replace("\\t", "\t")
          .replace("\\'", "'").replace('\\"', '"'))
    return s[:limit]


def _vllm_debug_prompt_metadata(prompt_repr, token_ids_repr):
    try:
        prompt = ast.literal_eval(prompt_repr)
    except (SyntaxError, ValueError):
        prompt = prompt_repr
    if prompt is None:
        prompt = ""
    prompt = _bounded_text(str(prompt), MESSAGE_TEXT_MAX)
    meta = {
        "request_messages": [{"role": "prompt", "name": "", "content": prompt}],
        "request_message_count": 1 if prompt else 0,
        "request_detail_json": _bounded_json({"prompt": prompt}),
    }
    if token_ids_repr not in ("None", ""):
        meta["prompt_tokens"] = _vllm_count_debug_ids(token_ids_repr)
        meta["total_tokens"] = meta["prompt_tokens"]
    if prompt:
        meta["request_group_label"] = _shorten(prompt)
        meta["request_group_id"] = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return meta


def _vllm_count_debug_ids(token_ids_repr):
    try:
        token_ids = ast.literal_eval(token_ids_repr)
    except (SyntaxError, ValueError):
        token_ids = None
    if isinstance(token_ids, list):
        return len(token_ids)
    return 0


class VllmLogTracker:
    """Turn vLLM request/response log lines into the dashboard's request rows.

    Active only under debug mode (which injects --enable-log-requests /
    --enable-log-outputs). Output tokens are accumulated across streaming delta
    lines (the final line reports None); TTFT is arrival -> first delta, and the
    generation rate is tokens over the delta span. Prompt tokens aren't logged
    at INFO so PT stays blank; the streaming finish reason is always reported as
    "stop" since vLLM only logs "streaming_complete" there.
    """

    def __init__(self, observer_state=None):
        self.state = observer_state or state
        self.active = {}   # request_id -> display row (shared into state)
        self.meta = {}     # request_id -> delta accounting {tokens,first,last,push}
        self.pending_debug = {}
        self.counter = 0

    def _new_row(self, rid, now, params=None):
        self.counter += 1
        mt = RE_VLLM_MAXTOK.search(params) if params else None
        tp = RE_VLLM_TEMP.search(params) if params else None
        return {
            "id": self.counter, "task_id": rid, "request_id": rid,
            "start_time": now, "start_time_str": local_time_str(now),
            "status": "processing", "path": "/v1/chat/completions",
            "model": self.state.model_name or "vllm",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "ttft_ms": 0, "prompt_tps": 0, "gen_tps": 0,
            "prompt_eval_ms": 0, "eval_ms": 0, "total_ms": 0,
            "cache_hit_pct": None, "finish_reason": None,
            "max_tokens": int(mt.group(1)) if mt else None,
            "temperature": float(tp.group(1)) if tp else None,
        }

    def process_line(self, line):
        m = RE_VLLM_DEBUG.search(line)
        if m:
            rid = m.group(1)
            detail = _vllm_debug_prompt_metadata(m.group(2), m.group(3))
            req = self.active.get(rid)
            if req is None:
                self.pending_debug[rid] = detail
            else:
                req.update(detail)
                self.state.update_active_request(rid, detail)
                self.state.notify_subscribers()
            return
        m = RE_VLLM_RECV.search(line)
        if m:
            rid, now = m.group(1), time.time()
            req = self._new_row(rid, now, m.group(2))
            req.update(self.pending_debug.pop(rid, {}))
            self.active[rid] = req
            self.meta[rid] = {"tokens": 0, "first": None, "last": now, "push": 0.0}
            self.state.add_active_request(req)
            self.state.notify_subscribers()
            return
        if "Generated response" not in line:
            return
        m = RE_VLLM_RESP_ID.search(line)
        if not m:
            return
        rid, now = m.group(1), time.time()
        if "(streaming delta)" in line:
            meta = self.meta.get(rid)
            if meta is None:
                return  # arrival not seen (observer started mid-stream)
            meta["tokens"] += _vllm_count_ids(line)
            if meta["first"] is None:
                meta["first"] = now
            meta["last"] = now
            req = self.active.get(rid)
            if req is not None and now - meta["push"] > 0.5:
                meta["push"] = now  # throttle live pushes, not one per token
                self.state.update_active_request(rid, {
                    "completion_tokens": meta["tokens"],
                    "ttft_ms": round((meta["first"] - req["start_time"]) * 1000, 1),
                })
                self.state.notify_subscribers()
            return
        # Completion: a streaming-complete line, or a whole non-streaming response.
        req = self.active.pop(rid, None) or self._new_row(rid, now)
        req.update(self.pending_debug.pop(rid, {}))
        meta = self.meta.pop(rid, {})
        toks = _vllm_count_ids(line) or meta.get("tokens", 0)
        fin = RE_VLLM_FINISH.search(line)
        reason = fin.group(1) if fin else None
        if reason in (None, "None", "streaming_complete"):
            reason = "stop"
        first = meta.get("first")
        if first:
            req["ttft_ms"] = round((first - req["start_time"]) * 1000, 1)
        gen_secs = (now - first) if first else (now - req.get("start_time", now))
        output = _vllm_extract_output(line)
        req.update({
            "status": "cancelled" if reason in ("abort", "error") else "completed",
            "completion_tokens": toks,
            "total_tokens": (req.get("prompt_tokens") or 0) + toks,
            "finish_reason": reason,
            "total_ms": round((now - req.get("start_time", now)) * 1000, 1),
            "end_time": now, "end_time_str": local_time_str(now),
            "gen_tps": round(toks / gen_secs, 1) if gen_secs > 0 and toks else 0,
        })
        if output:
            req["response_output"] = output
        self.state.remove_active_request(rid)
        self.state.add_request(req)
        self.state.notify_subscribers()


vllm_log_tracker = VllmLogTracker()


def log_tracker_for(model_info):
    """Pick the per-line log parser for the running engine."""
    return (vllm_log_tracker if infer_engine(model_info) == "vllm"
            else request_tracker)


def tail_docker_logs(container, monitor_port):
    """Tail the model container logs, auto-detecting it when not pinned.

    Dispatches each line to the engine-appropriate parser (vLLM vs the llama.cpp
    family). Re-detects after the stream ends (container restart) so the observer
    keeps working across redeploys without a hardcoded container name.
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
                state.add_docker_log(line)
                # During a load, mine the engine's own startup lines for an
                # in-container breakdown (weights load, KV cache, warmup).
                profile = get_active_profile()
                if profile is not None:
                    hit = parse_engine_load_line(line)
                    if hit:
                        profile.add_engine_phase(*hit)
                # Pick the parser per line: model_info often isn't populated yet
                # when the stream starts, so choosing once would wrongly pin the
                # llama.cpp tracker until the next container restart. The check is
                # cheap, so re-evaluating self-corrects as soon as the engine is
                # known.
                tracker = log_tracker_for(state.model_info)
                try:
                    tracker.process_line(line)
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
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.header,.card{background:var(--surface);border:1px solid var(--border);border-radius:8px}.header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;margin-bottom:16px}.header h1{font-size:18px;margin:0}.meta,.label{color:var(--dim)}.model{color:var(--accent);font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{padding:16px}.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 12px}.gpu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.gpu-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px}.gpu-name{font-weight:650;color:var(--accent);margin-bottom:8px}.row{display:flex;justify-content:space-between;gap:16px;padding:4px 0;font-size:13px}.value{font-variant-numeric:tabular-nums;font-weight:600}.bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden}.fill{height:100%;background:var(--accent);border-radius:3px}.fill.mem{background:var(--purple)}.fill.power{background:var(--yellow)}.fill.fan{background:var(--green)}.hot{color:var(--yellow)}.critical{color:var(--red)}.summary{display:flex;gap:24px;flex-wrap:wrap}.summary-item{text-align:center}.summary-value{font-size:28px;font-weight:750;font-variant-numeric:tabular-nums}.summary-label{font-size:11px;text-transform:uppercase;color:var(--dim);letter-spacing:.05em}.full{grid-column:1/-1}.requests{flex:1;min-height:0;overflow:auto}.request-row{display:grid;grid-template-columns:88px 150px minmax(150px,1.4fr) 60px 56px 70px 74px 78px 74px 60px 78px minmax(80px,1fr);gap:8px;align-items:center;padding:7px 8px;border-bottom:1px solid var(--border);font-size:12px}.group-label{color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.good{color:var(--green)}.request-head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700}.status{border-radius:999px;padding:2px 8px;text-align:center;font-size:10px;text-transform:uppercase;font-weight:700}.completed{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}.processing{background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.3)}.cancelled{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}.request-row.live{box-shadow:inset 3px 0 0 var(--accent)}
.btn{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px;font-family:inherit}.btn:hover{border-color:var(--accent)}.btn:disabled{opacity:.5;cursor:wait}.docker-logs{flex:1;min-height:0;overflow:auto;font-family:monospace;font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-word}.docker-logs .log-line{padding:1px 0;border-bottom:1px solid rgba(48,54,61,.3)}.docker-logs .log-line:last-child{border-bottom:none}.controls{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}.request-row:not(.request-head){cursor:pointer}.request-row:not(.request-head):hover{background:rgba(88,166,255,.06)}.preset-pill{border:1px solid var(--border);border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700}.preset-match{color:var(--green);border-color:rgba(63,185,80,.45);background:rgba(63,185,80,.12)}.preset-diff{color:var(--yellow);border-color:rgba(210,153,34,.45);background:rgba(210,153,34,.12)}.preset-custom{color:var(--purple);border-color:rgba(188,140,255,.45);background:rgba(188,140,255,.12)}.preset-desc{line-height:1.35;max-width:360px;text-align:right}.cmd-line{font-size:11px;color:var(--dim);word-break:break-word;padding:4px 0;line-height:1.75}.cmd-token{display:inline-block;border-radius:4px;padding:0 3px;margin:1px 0}.cmd-same{color:var(--green);background:rgba(63,185,80,.12);outline:1px solid rgba(63,185,80,.25)}.cmd-change{color:var(--yellow);background:rgba(210,153,34,.13);outline:1px solid rgba(210,153,34,.32)}.cmd-remove{color:var(--red);background:rgba(248,81,73,.12);outline:1px solid rgba(248,81,73,.28);text-decoration:line-through}.cmd-add{display:inline-block;border-radius:999px;border:1px solid rgba(88,166,255,.38);background:rgba(88,166,255,.1);color:var(--accent);padding:1px 6px;margin:2px 4px 0 0;font-size:11px}.cmd-legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}.modal{display:none;position:fixed;inset:0;z-index:20;background:rgba(0,0,0,.72);padding:32px}.modal.open{display:flex}.modal-panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;width:min(1180px,100%);max-height:calc(100vh - 64px);margin:auto;display:flex;flex-direction:column;box-shadow:0 16px 48px rgba(0,0,0,.45)}.modal-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 16px;border-bottom:1px solid var(--border)}.modal-head h2{font-size:14px;margin:0}.modal-body{overflow:auto;padding:0 16px 16px}.detail-grid{display:grid;grid-template-columns:minmax(320px,1fr) minmax(320px,1fr);gap:12px;padding-top:12px}.detail-section{border:1px solid var(--border);border-radius:6px;background:var(--bg);padding:10px;min-width:0}.detail-section h3{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 8px}.message-card{border-top:1px solid var(--border);padding:8px 0}.message-card:first-child{border-top:0}.message-role{font-size:11px;font-weight:700;color:var(--accent);margin-bottom:4px}.prewrap{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;line-height:1.45}.flag-guide{display:grid;grid-template-columns:minmax(140px,170px) minmax(180px,260px) minmax(460px,1fr);gap:12px;align-items:start;padding:8px 0;border-bottom:1px solid var(--border);font-size:12px;min-width:820px}.flag-guide.head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700;z-index:1}.flag-help{color:var(--text);line-height:1.4}.variant-table{min-width:1050px}.variant-row{display:grid;grid-template-columns:minmax(230px,1.1fr) 86px 86px minmax(140px,.8fr) minmax(260px,1.5fr) 120px;gap:10px;align-items:start;padding:9px 0;border-bottom:1px solid var(--border);font-size:12px}.variant-row.head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700;z-index:1}.variant-name{font-weight:650;color:var(--accent);overflow-wrap:anywhere}.variant-note{color:var(--dim);line-height:1.35;margin-top:2px}.variant-pick{display:flex;gap:8px;align-items:flex-start}.variant-pick input{margin-top:2px}.compare-toolbar{display:flex;align-items:center;gap:8px;padding:10px 0;position:sticky;top:0;background:var(--surface);z-index:2;border-bottom:1px solid var(--border)}.compare-grid{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(260px,1fr);gap:12px;min-width:720px;padding-top:12px}.compare-card{border:1px solid var(--border);border-radius:6px;background:var(--bg);padding:10px;min-width:0}.compare-card h3{font-size:12px;color:var(--accent);margin:0 0 8px;overflow-wrap:anywhere}.compare-field{border-top:1px solid var(--border);padding:8px 0}.compare-field:first-of-type{border-top:0}.compare-field .label{display:block;font-size:10px;text-transform:uppercase;font-weight:700;margin-bottom:4px}.flag-compare{display:grid;min-width:860px;border:1px solid var(--border);border-radius:6px;overflow:hidden;background:var(--bg);margin-top:12px}.flag-cell{border-right:1px solid var(--border);border-bottom:1px solid var(--border);padding:7px 8px;font-size:12px;min-width:0}.flag-cell.head{position:sticky;top:0;background:var(--surface);z-index:1;color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700}.flag-cell.changed{background:rgba(210,153,34,.12)}.flag-cell.same{color:var(--dim)}.flag-desc{line-height:1.35}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
@media(max-width:900px){.detail-grid{grid-template-columns:1fr}.request-row{grid-template-columns:88px 120px minmax(140px,1fr) 52px 52px 64px 68px 72px 68px 52px 72px minmax(70px,1fr)}}
.gpu-history-chart{position:relative;width:100%;height:160px;margin-bottom:12px;border-radius:6px;overflow:hidden;background:var(--bg);border:1px solid var(--border)}.gpu-history-chart:last-child{margin-bottom:0}.gpu-history-chart canvas{display:block;width:100%;height:100%}.gpu-history-label{position:absolute;top:8px;left:10px;font-size:11px;font-weight:650;color:var(--accent);pointer-events:none;z-index:1;text-shadow:0 1px 3px rgba(0,0,0,.7)}.gpu-history-legend{position:absolute;top:8px;right:10px;display:flex;gap:12px;font-size:10px;font-weight:600;pointer-events:none;z-index:1;text-shadow:0 1px 3px rgba(0,0,0,.7)}.gpu-history-legend span{display:flex;align-items:center;gap:4px}.legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
/* Drag & resize */
/* Cards get a uniform height (280px) with internal scroll.
   .height-expanded removes the cap; .card-height-exempt skips it entirely (GPU History). */
.card{transition:box-shadow .15s,opacity .15s;height:280px;overflow:auto;display:flex;flex-direction:column}.card.height-expanded{height:auto;max-height:none}.card.card-height-exempt{height:auto;max-height:none}.card.dragging{opacity:.4;box-shadow:0 0 0 2px var(--accent)}.card.drag-over{box-shadow:0 0 0 2px var(--green)}.card h2{display:flex;align-items:center;justify-content:space-between;flex-shrink:0}.drag-handle{cursor:grab;color:var(--dim);font-size:14px;user-select:none;display:inline-block;width:20px;text-align:center}.drag-handle:active{cursor:grabbing}.resize-btn{background:transparent;border:1px solid var(--border);color:var(--dim);border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px;font-family:inherit;line-height:1.6}.resize-btn:hover{border-color:var(--accent);color:var(--accent)}.resize-btn-h.active{color:var(--accent);border-color:var(--accent)}.half{grid-column:span 2}
</style>
</head>
<body>
<div class="header"><h1 id="title">Observer</h1><div class="meta"><span id="model" class="model">--</span> · Uptime <span id="uptime">0s</span> · <span id="updated">--</span></div></div>
<div class="grid">
<section class="card"><h2><span>GPU</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="gpuGrid" class="gpu-grid"></div></section>
<section class="card" id="caseFanCard" style="display:none"><h2><span>Case Fans</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="caseFanGrid" class="gpu-grid"></div></section>
<section class="card full card-height-exempt" id="gpuHistoryCard" style="display:none"><h2><span>GPU History</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="gpuHistoryCharts"></div></section>
<section class="card"><h2><span>Summary</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div class="summary">
<div class="summary-item"><div id="active" class="summary-value">0</div><div class="summary-label">Active</div></div>
<div class="summary-item"><div id="requests" class="summary-value">0</div><div class="summary-label">Completed</div></div>
<div class="summary-item"><div id="gpuTemp" class="summary-value">--</div><div class="summary-label">GPU Temp</div></div>
<div class="summary-item"><div id="memTemp" class="summary-value">--</div><div class="summary-label">VRAM Temp</div></div>
<div class="summary-item"><div id="avgTps" class="summary-value">0</div><div class="summary-label">Avg Gen t/s</div></div>
<div class="summary-item"><div id="queued" class="summary-value">-</div><div class="summary-label">Queued</div></div>
</div></section>
<section class="card"><h2><span>Context / KV Cache</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="slotInfo" class="gpu-grid"></div></section>
<section class="card"><h2><span>Model Config</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="modelInfo"></div>
<div class="controls"><select id="presetSel" class="btn" onchange="renderModelInfoFromState()" title="baseline: club-3090 verbatim · debug: add engine-specific observability/debug flags">
<option value="baseline">baseline</option><option value="debug" selected>debug</option>
</select><label id="cacheRamLabel" class="btn" title="llama.cpp only: set --cache-ram 8192 for host-RAM prompt cache"><input id="cacheRamChk" type="checkbox" checked onchange="cacheRamUserEdited=true;renderModelInfoFromState()"> cache-ram 8192</label><button id="btnRestart" class="btn" onclick="doRestart()">Restart model</button><button id="btnStop" class="btn" onclick="doStop()">Stop model</button><button class="btn" onclick="doUpdate()">Update club-3090</button><span id="ctlStatus" class="label"></span></div>
<div id="lastStartSummary" style="padding:8px 0;font-size:12px"></div>
<div id="lastStartLog" class="docker-logs" style="max-height:180px;min-height:24px;border-top:1px solid var(--border);padding:4px 0"></div>
</section>
<section class="card"><h2><span>club-3090 Catalog</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="catalogInfo"></div></section>
<section class="card"><h2><span>Server Metrics</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="metricsInfo"></div></section>
<section class="card full" id="vllmTimelineCard"><h2><span>vLLM Activity (live)</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="vllmTimeline"></div></section>
<section class="card full" id="loadProfileCard" style="display:none"><h2><span>Model Load Profile</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="loadProfile"></div></section>
<section class="card"><h2><span>Inference Health</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div class="summary">
<div class="summary-item"><div id="truncRate" class="summary-value">0%</div><div class="summary-label">Truncated</div></div>
<div class="summary-item"><div id="cancelled" class="summary-value">0</div><div class="summary-label">Cancelled</div></div>
<div class="summary-item"><div id="cacheDefeat" class="summary-value">0</div><div class="summary-label">Cache Defeated</div></div>
<div class="summary-item"><div id="ctxShift" class="summary-value">0</div><div class="summary-label">Ctx Shifts</div></div>
<div class="summary-item"><div id="draftAccept" class="summary-value">-</div><div class="summary-label">Avg Draft Accept</div></div>
<div class="summary-item"><div id="httpErrors" class="summary-value">0</div><div class="summary-label">HTTP Errors</div></div>
<div class="summary-item"><div id="budgetHits" class="summary-value">0</div><div class="summary-label">Budget Hits</div></div>
</div></section>
<section class="card full"><h2><span>Recent Requests</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="requestList" class="requests"></div></section>
<section class="card full"><h2><span>Docker Logs</span><div><span class="drag-handle" title="Drag to reorder">⠿</span><button class="resize-btn resize-btn-w" title="Toggle width">⇔</button><button class="resize-btn resize-btn-h" title="Toggle height">⇕</button></div></h2><div id="dockerLogs" class="docker-logs"></div></section>
</div>
<div id="flagModal" class="modal" onclick="if(event.target===this)closeFlagGuide()"><div class="modal-panel">
<div class="modal-head"><h2 id="flagModalTitle">Flag guide</h2><button class="btn" onclick="closeFlagGuide()">Close</button></div>
<div id="flagModalBody" class="modal-body"></div>
</div></div>
<div id="variantModal" class="modal" onclick="if(event.target===this)closeVariantList()"><div class="modal-panel">
<div class="modal-head"><h2 id="variantModalTitle">Variants</h2><button class="btn" onclick="closeVariantList()">Close</button></div>
<div id="variantModalStatus" style="display:none;padding:8px 16px;font-size:12px;border-bottom:1px solid var(--border);color:var(--accent);background:rgba(88,166,255,.06);overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></div>
<div id="variantModalBody" class="modal-body"></div>
</div></div>
<div id="compareModal" class="modal" onclick="if(event.target===this)closeCompareModal()"><div class="modal-panel">
<div class="modal-head"><h2 id="compareModalTitle">Command compare</h2><button class="btn" onclick="closeCompareModal()">Close</button></div>
<div id="compareModalBody" class="modal-body"></div>
</div></div>
<div id="requestModal" class="modal" onclick="if(event.target===this)closeRequestDetail()"><div class="modal-panel">
<div class="modal-head"><h2 id="requestModalTitle">Request detail</h2><button class="btn" onclick="closeRequestDetail()">Close</button></div>
<div id="requestModalBody" class="modal-body"></div>
</div></div>
<script>
let es;let lastModelInfo={};let lastRenderData=null;let flagModalOpen=false;let requestRowsByKey={};let cacheRamUserEdited=false;let lastCacheSourceKey='';let compareSelections=new Set();
const PRESET_LABELS={'baseline':'baseline','debug':'debug','insight':'debug','insight-cache':'debug','insight-debug':'debug','custom':'custom'};
const PRESET_DESCRIPTIONS={
baseline:'club-3090 compose command with no observer insight flags added',
debug:'engine-specific metrics, request logging, timestamps, and debug details where supported',
custom:'live command has observer-managed flags but does not exactly match a preset'
};
const PRESET_OPTIONS={
baseline:{},
debug:{'--metrics':null,'--props':null,'--log-verbosity':'5','--log-timestamps':null}
};
const VLLM_PRESET_OPTIONS={
baseline:{},
debug:{'--enable-log-requests':null,'--enable-log-outputs':null}
};
const VLLM_PRESET_ENV={baseline:{},debug:{'VLLM_LOGGING_LEVEL':'DEBUG'}};
const MANAGED_FLAGS=new Set([...Object.values(PRESET_OPTIONS),...Object.values(VLLM_PRESET_OPTIONS)].flatMap(o=>Object.keys(o)).concat(['--cache-ram']));
const PRESET_FLAG_ALIASES={'-lv':'--log-verbosity'};
function presetFlag(flag){return PRESET_FLAG_ALIASES[flag]||flag}
function pct(v,max){return Math.max(0,Math.min(100,(v/max)*100))}
function cls(t){return t>85?'critical':t>75?'hot':''}
function connect(){if(es)es.close();es=new EventSource('/observer/sse');es.onmessage=e=>render(JSON.parse(e.data));es.onerror=()=>{es.close();setTimeout(connect,3000)}}
let lastActive=0;
let lastBusy=false;
function render(d){lastRenderData=d;lastActive=(d.active_requests||[]).length;lastBusy=!!d.control_busy;renderHeader(d);renderGpu(d.gpu_stats||[]);renderCaseFans(d.case_fans||[]);renderGpuHistory(d);renderSummary(d);renderSlots(d);renderModelInfo(d);renderCatalog(d);renderMetrics(d);renderVllmTimeline(d);renderLoadProfile(d);renderHealth(d);renderControl(d);renderLastStart(d);renderRequests(d.requests||[],d.active_requests||[]);renderDockerLogs(d);refreshVariantListIfOpen();document.getElementById('uptime').textContent=d.uptime_human;document.getElementById('updated').textContent=new Date().toLocaleTimeString()}
function renderControl(d){let cs=d.control_status||{};let st=document.getElementById('ctlStatus');
if(cs.action&&(!cs.done||(Date.now()/1000-(cs.updated_at||0))<120)){let msg=(cs.done?(cs.ok?'✓ ':'✗ '):'⏳ ')+esc(cs.detail||'');let h=cs.install_hint;if(h&&!cs.ok){msg+=` <button class="btn" style="padding:2px 8px" onclick="doStartOrSwitch('${esc(h.variant)}','${esc(h.status||'')}')">Install + retry</button>`}st.innerHTML=msg}
document.querySelectorAll('.controls .btn').forEach(b=>b.disabled=lastBusy);
// Restart/Stop act on a running container; when nothing is up they 409 or
// no-op, so disable them and point at the catalog's Start buttons instead.
let running=!!d.container;let rb=document.getElementById('btnRestart'),sb=document.getElementById('btnStop');
if(rb){rb.disabled=lastBusy||!running;rb.title=running?'':'no model running — use Start in the Catalog below'}
if(sb){sb.disabled=lastBusy||!running;sb.title=running?'':'no model running'}}
function doStartOrSwitch(v,status){let p=selectedPreset(),cache=selectedCacheRam();let warn=lastActive?`\n⚠ ${lastActive} request(s) in flight will be killed!`:'';let exp=(status!=='production'&&status!=='caveats')?`\n⚠ status is '${status}' — will pass --force.`:'';
let confirmMsg=`Start/switch model '${v}' with mode '${p}' and cache ${cache?'on':'off'}?\nThis downloads any missing files, then boots — takes a few minutes.${exp}${warn}`;
if(!confirm(confirmMsg))return;
let body={variant:v,preset:p,cache_ram:cache,force:lastActive>0||(status!=='production'&&status!=='caveats'),retry:true};
ctlPost('/observer/api/install',body).then(()=>{document.getElementById('ctlStatus').textContent='⏳ preparing & starting…'}).catch(e=>{document.getElementById('ctlStatus').textContent='✗ '+e.message})}
function renderLastStart(d){let ls=d.last_start||{};let sumEl=document.getElementById('lastStartSummary');let logEl=document.getElementById('lastStartLog');if(!sumEl||!logEl)return;if(!ls.started_at){sumEl.innerHTML='<span class="label">No model start recorded yet.</span>';logEl.innerHTML='';return}let icon=ls.ok===true?'✓':(ls.ok===false?'✗':'⏳');let variant=ls.variant||'?';let preset=ls.preset||'?';let cacheLabel=ls.cache_ram?'on':'off';let endAt=ls.finished_at||Date.now()/1000;let dur=endAt-ls.started_at;let dm=Math.floor(dur/60);let ds=Math.floor(dur%60);let durStr=dm>0?`${dm}m ${ds}s`:`${ds}s`;let cls=ls.ok===true?'good':(ls.ok===false?'critical':'hot');sumEl.innerHTML=`<span class="${cls}">${icon} ${esc(variant)} · mode ${esc(preset)} · cache ${cacheLabel} · ${durStr}</span>`;let lines=ls.log||[];if(!lines.length){logEl.innerHTML='<div class="label" style="padding:4px 0">No log captured yet.</div>';return}let html=lines.map(l=>`<div class="log-line">${esc(l)}</div>`).join('');logEl.innerHTML=html;logEl.scrollTop=logEl.scrollHeight}
function renderModelInfoFromState(){if(lastRenderData)renderModelInfo(lastRenderData)}
function renderMetrics(d){let m=d.metrics||{};let el=document.getElementById('metricsInfo');let q=document.getElementById('queued');
let vllm=m.engine==='vllm';
if(!m.available){q.textContent='-';q.className='summary-value';el.innerHTML='<div class="row"><span class="label">'+(vllm?'vLLM /metrics not reachable yet — model may still be loading':'/metrics disabled — restart with debug mode to enable')+'</span></div>';return}
q.textContent=m.queued??'-';q.className='summary-value '+((m.queued||0)>0?'hot':'');
let rows='';
rows+=infoRow('Running / waiting',`${m.processing??'-'} / ${(m.queued||0)>0?`<span class="hot">${m.queued}</span>`:m.queued??'-'}`,vllm?'num_requests_running / num_requests_waiting':'requests_processing / requests_deferred — queued means all slots are busy');
let tpsNote=vllm?'throughput over the last scrape interval (derived from token counters)':'server-lifetime average';
if(m.prompt_tps_avg!=null)rows+=infoRow('Prompt t/s',Number(m.prompt_tps_avg).toFixed(1),tpsNote+' prompt processing throughput');
if(m.gen_tps_avg!=null)rows+=infoRow('Gen t/s',Number(m.gen_tps_avg).toFixed(1),tpsNote+' generation throughput');
if(m.prompt_tokens_total!=null)rows+=infoRow('Prompt tokens',Number(m.prompt_tokens_total).toLocaleString()+(m.prompt_seconds_total!=null?` <span class="label">(${Number(m.prompt_seconds_total).toFixed(0)}s)</span>`:''));
if(m.gen_tokens_total!=null)rows+=infoRow('Generated tokens',Number(m.gen_tokens_total).toLocaleString()+(m.gen_seconds_total!=null?` <span class="label">(${Number(m.gen_seconds_total).toFixed(0)}s)</span>`:''));
if(m.decode_calls_total!=null)rows+=infoRow('Decode calls',Number(m.decode_calls_total).toLocaleString());
if(m.busy_slots_per_decode!=null)rows+=infoRow('Busy slots / decode',Number(m.busy_slots_per_decode).toFixed(2));
if(m.kv_cache_usage_ratio!=null)rows+=infoRow('KV cache usage',(100*m.kv_cache_usage_ratio).toFixed(1)+'%');
if(vllm){
if(m.prefix_cache_hit_pct!=null)rows+=infoRow('Prefix cache hit',Number(m.prefix_cache_hit_pct).toFixed(1)+'%','prefix_cache_hits_total / prefix_cache_queries_total');
if(m.spec_accept_pct!=null)rows+=infoRow('Spec-decode acceptance',Number(m.spec_accept_pct).toFixed(1)+'%','accepted / drafted speculative tokens');
if(m.avg_ttft_ms!=null)rows+=infoRow('Avg TTFT',Number(m.avg_ttft_ms).toFixed(0)+' ms','time_to_first_token histogram mean');
if(m.avg_tpot_ms!=null)rows+=infoRow('Avg time / output tok',Number(m.avg_tpot_ms).toFixed(1)+' ms','request_time_per_output_token histogram mean');
if(m.avg_e2e_ms!=null)rows+=infoRow('Avg e2e latency',(m.avg_e2e_ms/1000).toFixed(2)+' s','e2e_request_latency histogram mean');
if(m.requests_total!=null){let br=m.success_by_reason||{};let parts=Object.keys(br).filter(k=>br[k]>0).map(k=>`${esc(k)}: ${br[k]}`).join(', ');rows+=infoRow('Completed',Number(m.requests_total).toLocaleString()+(parts?` <span class="label">(${parts})</span>`:''),'request_success_total by finish reason')}
if(m.preemptions_total)rows+=infoRow('Preemptions',Number(m.preemptions_total).toLocaleString(),'num_preemptions_total — requests evicted under KV pressure');
}
el.innerHTML=rows}
function renderVllmTimeline(d){let card=document.getElementById('vllmTimelineCard');let el=document.getElementById('vllmTimeline');let m=d.metrics||{};if(m.engine!=='vllm'){card.style.display='none';return}card.style.display='';let h=d.vllm_history||[];if(!h.length){el.innerHTML='<div class="row"><span class="label">no samples yet</span></div>';return}
let spark=(key,color,unit,dec)=>{let vals=h.map(s=>Number(s[key])||0);let max=Math.max(1,...vals);let bars=vals.map(v=>`<span style="display:inline-block;width:3px;margin-right:1px;background:${color};height:${Math.max(1,Math.round(30*v/max))}px;vertical-align:bottom" title="${v}${unit}"></span>`).join('');let last=vals[vals.length-1];let peak=Math.max(...vals);return `<div class="row" style="align-items:flex-end"><span class="label" style="min-width:130px">${key}</span><span style="height:32px;line-height:0;flex:1;overflow:hidden;white-space:nowrap">${bars}</span><span class="value" style="min-width:120px;text-align:right">${last.toFixed(dec||0)}${unit} <span class="label">(peak ${peak.toFixed(dec||0)})</span></span></div>`};
let rows='';rows+=spark('gen_tps','var(--green)',' t/s',1);rows+=spark('prompt_tps','var(--accent)',' t/s',0);rows+=spark('running','var(--purple)','',0);rows+=spark('kv','var(--yellow)','%',0);if(h.some(s=>s.spec!=null))rows+=spark('spec','var(--green)','%',0);rows+='<div class="row"><span class="label">~'+Math.round(h.length*2)+'s of history · newest at right</span></div>';el.innerHTML=rows}
let _loadAnchor=null,_loadTimer=null,_loadCollapsed=new Set();
function renderLoadProfile(d){let card=document.getElementById('loadProfileCard');
let active=d.active_load_profile;let ps=d.load_profiles||[];
let p=active||(ps.length?ps[ps.length-1]:null);
if(!p){card.style.display='none';_loadAnchor=null;if(_loadTimer){clearInterval(_loadTimer);_loadTimer=null}return}
card.style.display='';
// During a live load, re-anchor to now so the ticker can extrapolate the open
// phase between snapshots (which only arrive every ~2s).
_loadAnchor={p:p,active:!!active,at:Date.now(),history:active?ps.slice(-6).reverse():ps.slice(0,-1).slice(-6).reverse()};
paintLoadProfile(0);
if(active){if(!_loadTimer)_loadTimer=setInterval(()=>paintLoadProfile(_loadAnchor?(Date.now()-_loadAnchor.at)/1000:0),250)}
else if(_loadTimer){clearInterval(_loadTimer);_loadTimer=null}}
function paintLoadProfile(extra){let a=_loadAnchor;if(!a)return;let el=document.getElementById('loadProfile');if(!el)return;let p=a.p;
let fs=s=>s==null?'-':(s>=60?Math.floor(s/60)+'m '+Math.round(s%60)+'s':s.toFixed(1)+'s');
let mb=b=>b==null?'-':(Math.abs(b)>=1e9?(b/1e9).toFixed(2)+' GB':(b/1e6).toFixed(0)+' MB');
let hdr=t=>`<div class="row"><span class="label" style="font-weight:600">${t}</span></div>`;
let plabel={'switch.sh':'switch.sh (stop + boot + 1st load)','wait_model_info':'detect new container','restart_preset':'preset re-up (compose up)','restart':'compose up','ready_wait':'wait until serving (2nd load)','install_assets':'download weights'};
let clabel={weights_load:'weights load (SSD → GPU)',model_load:'model load (total)',compile:'torch.compile (Dynamo)',cuda_graphs:'CUDA graph capture',engine_init:'engine init: KV cache + compile + warmup'};
let corder=['weights_load','model_load','compile','cuda_graphs','engine_init'];
let phs=p.phases||[];let live=x=>(x.duration||0)+(x.running?extra:0);
let liveTotal=(p.total||phs.reduce((s,x)=>s+(x.duration||0),0))+(a.active?extra:0);
let denom=Math.max(liveTotal,phs.reduce((s,x)=>s+live(x),0),1);
let bars=phs.map(x=>{let dur=live(x);let w=Math.max(1,Math.round(100*dur/denom));let col=x.running?'var(--yellow)':'var(--accent)';
let kids=x.children||{};let kk=corder.filter(k=>kids[k]!=null).concat(Object.keys(kids).filter(k=>corder.indexOf(k)<0));
let key=esc((p.trigger||'')+'|'+x.name);let collapsed=_loadCollapsed.has(key);
let tog=kk.length?`<span style="cursor:pointer;display:inline-block;width:14px;color:var(--label,#9aa)" onclick="toggleLoadPhase('${key}')">${collapsed?'▶':'▼'}</span>`:'<span style="display:inline-block;width:14px"></span>';
let row=`<div class="row" style="align-items:center">${tog}<span class="label" style="min-width:196px">${x.running?'⏳ ':''}${esc(plabel[x.name]||x.name)}</span><span style="flex:1"><span style="display:inline-block;height:14px;width:${w}%;background:${col};border-radius:3px;vertical-align:middle"></span></span><span class="value" style="min-width:80px;text-align:right">${fs(dur)}</span></div>`;
if(kk.length&&!collapsed)row+=kk.map(k=>{let cd=kids[k];let cw=Math.max(1,Math.round(100*cd/Math.max(dur,cd,0.001)));return `<div class="row" style="align-items:center;opacity:.85"><span style="display:inline-block;width:30px"></span><span class="label" style="min-width:180px;font-size:11px">${esc(clabel[k]||k)}</span><span style="flex:1"><span style="display:inline-block;height:9px;width:${cw}%;background:var(--purple);border-radius:3px;vertical-align:middle"></span></span><span class="value" style="min-width:80px;text-align:right;font-size:11px">${fs(cd)}</span></div>`}).join('');
return row}).join('');
let r=p.resources||{};let resRows='';
if(r.disk_read_bytes!=null)resRows+=infoRow('Disk read',mb(r.disk_read_bytes)+(r.peak_disk_read_bps?` <span class="label">(peak ${(r.peak_disk_read_bps/1e6).toFixed(0)} MB/s)</span>`:''),'/proc/diskstats read delta over the load window');
if(r.vram_delta_mib!=null){let gib=v=>Math.abs(v)>=1024?(v/1024).toFixed(2)+' GiB':Number(v).toFixed(0)+' MiB';resRows+=infoRow('VRAM filled',(r.vram_delta_mib>=0?'+':'')+gib(r.vram_delta_mib)+(r.vram_peak_mib!=null?` <span class="label">(peak ${gib(r.vram_peak_mib)})</span>`:''),'peak minus trough VRAM over the load — survives a switch where the old model was still resident at start (nvidia-smi)');}
if(r.page_cache_delta_bytes!=null)resRows+=infoRow('Page cache',(r.page_cache_delta_bytes>=0?'+':'-')+mb(Math.abs(r.page_cache_delta_bytes)),'/proc/meminfo Cached delta (host RAM)');
let status=a.active?`<span class="value" style="color:var(--yellow)">⏳ loading… ${fs(liveTotal)}</span>`:`<span class="value ${p.ok?'good':'critical'}">${p.ok?'✓':'✗'} ${fs(liveTotal)} total</span>`;
let head=`<div class="row"><span class="label">${esc(p.trigger)}${p.variant?' · '+esc(p.variant):''}${p.preset?' · '+esc(p.preset):''} <span class="label">${esc(p.started_at_str||'')}</span></span>${status}</div>`;
if(!a.active&&!p.ok&&p.error)head+=`<div class="row"><span class="value critical">${esc(p.error)}</span></div>`;
let hist=(a.history||[]).map(x=>infoRow((esc(x.started_at_str||'')+' · '+esc(x.trigger)),`<span class="${x.ok?'good':'critical'}">${x.ok?'✓':'✗'}</span> ${fs(x.total)}`)).join('');
el.innerHTML=head+'<div style="margin-top:8px">'+bars+'</div>'+(resRows?hdr('Resource attribution')+resRows:'')+(hist?hdr('Recent loads')+hist:'')}
function toggleLoadPhase(key){if(_loadCollapsed.has(key))_loadCollapsed.delete(key);else _loadCollapsed.add(key);paintLoadProfile(0)}
async function ctlPost(path,body){console.log('[observer] ctlPost:',path,JSON.stringify(body));let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});console.log('[observer] ctlPost response:',r.status,r.ok);let j=await r.json().catch(()=>({}));if(!r.ok){console.error('[observer] ctlPost error:',j.error||'HTTP '+r.status);throw new Error(j.error||('HTTP '+r.status))}console.log('[observer] ctlPost result:',JSON.stringify(j));return j}
async function ctlRun(btn,msg,fn){if(!confirm(msg))return;let st=document.getElementById('ctlStatus');document.querySelectorAll('.controls .btn').forEach(b=>b.disabled=true);st.textContent='working…';try{st.textContent=await fn()}catch(e){st.textContent='failed: '+e.message}finally{document.querySelectorAll('.controls .btn').forEach(b=>b.disabled=false)}}
function doUpdate(){ctlRun(this,'git pull club-3090 (fast-forward only)?',async()=>{let r=await ctlPost('/observer/api/update');return r.updated?`updated ${r.from} → ${r.to} (${(r.commits||[]).length} commits)`:(r.detail||'already up to date')})}
function doRestart(){let p=selectedPreset(),cache=selectedCacheRam();let warn=lastActive?` ⚠ ${lastActive} request(s) in flight will be killed!`:'';if(!confirm(`Restart the model with mode '${p}' and cache ${cache?'on':'off'}?${warn}`))return;ctlPost('/observer/api/restart',{preset:p,cache_ram:cache,force:lastActive>0}).then(()=>{document.getElementById('ctlStatus').textContent='⏳ restart started…'}).catch(e=>{document.getElementById('ctlStatus').textContent='✗ '+e.message})}
function doStop(){let warn=lastActive?` ⚠ ${lastActive} request(s) in flight will be killed!`:'';ctlRun(this,`Stop model serving on this host?${warn}`,async()=>{let r=await ctlPost('/observer/api/stop',{force:lastActive>0});return r.stopped?`stopped ${r.container||'model'}`:(r.detail||'already stopped')})}
function statusSpan(s){let c=s==='production'?'good':(s==='caveats'?'hot':'critical');return `<span class="${c}">${esc(s||'?')}</span>`}
function variantDoc(v){return v.doc||{}}
function variantCtx(v){let doc=variantDoc(v);return doc.max_ctx_doc||`${v.max_ctx?Number(v.max_ctx).toLocaleString():'-'}`}
function variantTps(v){return variantDoc(v).tps||'-'}
function variantWhy(v){return variantDoc(v).why||v.status_note||''}
function variantTopology(v){let p=(v&&v.compose_path)||'';return p.indexOf('/multi4/')>=0?'multi4':(p.indexOf('/dual/')>=0?'dual':'single')}
function machineTopology(d){let ngpu=(d.gpu_stats||[]).length||1;return ngpu>=4?'multi4':(ngpu>=2?'dual':'single')}
function topologyLabel(t){return t==='multi4'?'4-GPU':(t==='dual'?'dual-GPU':'single-GPU')}
function defaultForVariant(c,k,v,topo){return (c.defaults||{})[`${v.model}/${k.split('/')[0]}/${topo}`]}
function isTopologyDefault(c,k,v,topo){return defaultForVariant(c,k,v,topo)===k}
function updateCompareToolbar(){let count=compareSelections.size;let n=document.getElementById('compareCount'),b=document.getElementById('btnCompareVariants');if(n)n.textContent=`${count} selected`;if(b)b.disabled=count<2||count>4}
function toggleCompareVariant(k,on){if(on)compareSelections.add(k);else compareSelections.delete(k);updateCompareToolbar()}
function renderVariantListModal(d,fits,runKey,running,topo){let c=d.catalog||{};let vars=c.variants||{};let installed=d.installed_assets||{};let order={production:0,caveats:1};
fits=fits.slice().sort((a,b)=>(order[vars[a].status]??2)-(order[vars[b].status]??2)||(vars[a].model||'').localeCompare(vars[b].model||'')||a.localeCompare(b));
let visible=new Set(fits);compareSelections.forEach(k=>{if(!visible.has(k))compareSelections.delete(k)});
let rows='<div class="compare-toolbar"><button id="btnCompareVariants" class="btn" onclick="compareSelectedVariants()">Compare commands</button><span id="compareCount" class="label">0 selected</span><span class="label">select 2-4 variants</span></div><div class="variant-table"><div class="variant-row head"><span>Variant</span><span>Max ctx</span><span>Narr / Code</span><span>Workload</span><span>Why / comments</span><span>Action</span></div>';
rows+=fits.map(k=>{let v=vars[k]||{};let doc=variantDoc(v);let mark=k===runKey?'▶ ':(isTopologyDefault(c,k,v,topo)?'⭐ ':'');let note=v.status_note&&!doc.why?`<div class="variant-note">${esc(v.status_note)}</div>`:'';let installedLabel=installed[k]?'<span class="good">installed</span>':'<span class="label">needs download</span>';let action=k===runKey?`<button class="btn switch-btn" style="padding:2px 8px;font-size:11px" onclick="doStop()"${lastBusy?' disabled':''}>Stop</button>`:`<button class="btn switch-btn" style="padding:2px 8px;font-size:11px" onclick="doStartOrSwitch('${esc(k)}','${esc(v.status)}')"${lastBusy?' disabled':''}>${running?'switch':'start'}</button>`;let cs=d.control_status||{};let installingVariant=cs.action==='install'&&!cs.done&&cs.detail&&cs.detail.indexOf(k)>=0?k:null;let installing=installingVariant?`<span class="hot">installing…</span>`:'';let checked=compareSelections.has(k)?' checked':'';
return `<div class="variant-row"><span class="variant-pick"><input type="checkbox"${checked} onchange="toggleCompareVariant('${esc(k)}',this.checked)"><span><div class="variant-name">${mark}${esc(k)}</div><div class="variant-note">${esc(v.model||'')}${v.kv_format?' · '+esc(v.kv_format):''}${v.tp?` · TP=${esc(v.tp)}`:''}</div></span></span><span class="value">${esc(variantCtx(v))}</span><span class="value">${esc(variantTps(v))}</span><span>${esc(doc.workload_label||v.workload||'-')}</span><span>${esc(variantWhy(v))}${note}</span><span style="display:flex;gap:8px;align-items:center;justify-content:flex-end">${statusSpan(v.status)}${installing}${k!==runKey?installedLabel:''}${action}</span></div>`}).join('');
rows+='</div>';
document.getElementById('variantModalTitle').textContent=`${topologyLabel(topo)} variants for this machine (${fits.length})`;
document.getElementById('variantModalBody').innerHTML=rows;
updateCompareToolbar();
let cs=d.control_status||{};let vms=document.getElementById('variantModalStatus');
if(cs.action&&!cs.done){let icon=cs.action==='install'?'📦 ':'⏳ ';vms.textContent=icon+esc(cs.detail||cs.action+'…');vms.style.display=''}else{vms.style.display='none'}
}
function openVariantList(){if(!lastRenderData)return;let d=lastRenderData;let c=d.catalog||{};let vars=c.variants||{};let keys=Object.keys(vars);let mi=d.model_info||{};let running=!!d.container;let runKey=running?keys.find(k=>vars[k].compose_path&&mi.compose_file&&mi.compose_file.indexOf(vars[k].compose_path)>=0):null;let ngpu=(d.gpu_stats||[]).length||1;let topo=machineTopology(d);let fits=keys.filter(k=>variantTopology(vars[k])===topo&&(vars[k].tp||1)<=ngpu);renderVariantListModal(d,fits,runKey,running,topo);document.getElementById('variantModal').classList.add('open')}
function refreshVariantListIfOpen(){let m=document.getElementById('variantModal');if(m&&m.classList.contains('open'))openVariantList()}
function closeVariantList(){document.getElementById('variantModal').classList.remove('open')}
function closeCompareModal(){document.getElementById('compareModal').classList.remove('open')}
function commandText(v){let c=v.command;if(Array.isArray(c))return c.join(' ');if(c==null)return '-';return String(c)}
function entrypointText(v){let e=v.entrypoint;if(Array.isArray(e))return e.join(' ');if(e==null)return '-';return String(e)}
function envText(env){if(!env)return '-';if(Array.isArray(env))return env.join('\\n');let keys=Object.keys(env).sort();return keys.length?keys.map(k=>`${k}=${env[k]}`).join('\\n'):'-'}
function compareCard(v){if(v.error)return `<div class="compare-card"><h3>${esc(v.variant)}</h3><div class="compare-field"><span class="label">Error</span><div class="critical mono">${esc(v.error)}</div></div></div>`;return `<div class="compare-card"><h3>${esc(v.variant)}</h3><div class="compare-field"><span class="label">Status</span><div>${statusSpan(v.status)} <span class="label">TP=${esc(v.tp||'-')}</span></div></div><div class="compare-field"><span class="label">Compose</span><div class="mono">${esc(v.compose_path||'-')}</div></div><div class="compare-field"><span class="label">Service</span><div class="mono">${esc(v.service||'-')}</div></div><div class="compare-field"><span class="label">Image</span><div class="mono">${esc(v.image||'-')}</div></div><div class="compare-field"><span class="label">Entrypoint</span><div class="mono">${esc(entrypointText(v))}</div></div><div class="compare-field"><span class="label">Command</span><div class="mono">${esc(commandText(v))}</div></div><div class="compare-field"><span class="label">Environment</span><div class="mono prewrap">${esc(envText(v.environment))}</div></div></div>`}
function flagValue(v){return v===undefined?'<span class="label">-</span>':(v===null?'<span class="label">switch</span>':esc(v))}
function flagCompareRows(vars,rows,help){if(!rows||!rows.length)return '<div class="row"><span class="label">No command flags found</span></div>';let helpFailed=!!(help&&help.error);let missingDesc=helpFailed?'<span class="hot">help unavailable</span>':'<span class="hot">not found in --help</span>';let cols=`minmax(150px,.8fr) minmax(260px,1.3fr) repeat(${vars.length}, minmax(180px,1fr))`;let html=`<div class="flag-compare" style="grid-template-columns:${cols}"><div class="flag-cell head">Flag</div><div class="flag-cell head">Definition</div>`+vars.map(v=>`<div class="flag-cell head">${esc(v.variant)}</div>`).join('');html+=rows.map(r=>{let vals=vars.map(v=>Object.prototype.hasOwnProperty.call(r.values||{},v.variant)?r.values[v.variant]:undefined);let present=vals.filter(v=>v!==undefined).map(v=>v===null?'':String(v));let changed=new Set(present).size>1||present.length!==vals.length;let desc=r.known?esc(r.description||'listed in --help without description'):missingDesc;let aliases=(r.aliases||[]).filter(a=>a!==r.flag).join(', ');let title=aliases?` title="aliases: ${esc(aliases)}"`:'';return `<div class="flag-cell mono"${title}>${esc(r.flag)}</div><div class="flag-cell flag-desc">${desc}</div>`+vals.map(v=>`<div class="flag-cell mono ${changed?'changed':'same'}">${flagValue(v)}</div>`).join('')}).join('');return html+'</div>'}
function renderCompareModal(data){let vars=data.variants||[];let help=data.help||{};let helpNote=help.error?`<div class="row"><span class="label">Help source</span><span class="value mono">${esc(help.source||'not available')}</span></div><div class="row"><span class="critical">help: ${esc(help.error)}</span></div>`:(help.source?`<div class="row"><span class="label">Help source</span><span class="value mono">${esc(help.source)}${help.warning?' · '+esc(help.warning):''}</span></div>`:'<div class="row"><span class="label">Help source</span><span class="value">not available</span></div>');document.getElementById('compareModalTitle').textContent=`Command compare (${vars.length})`;document.getElementById('compareModalBody').innerHTML=helpNote+flagCompareRows(vars,data.flag_matrix||[],help)+det('detCompareRaw',false,'raw compose details',`<div class="compare-grid">${vars.map(compareCard).join('')}</div>`);document.getElementById('compareModal').classList.add('open')}
function compareSelectedVariants(){let variants=[...compareSelections];if(variants.length<2||variants.length>4){updateCompareToolbar();return}let btn=document.getElementById('btnCompareVariants');if(btn)btn.disabled=true;ctlPost('/observer/api/compare',{variants}).then(renderCompareModal).catch(e=>{document.getElementById('compareModalTitle').textContent='Command compare';document.getElementById('compareModalBody').innerHTML=`<div class="row"><span class="critical">${esc(e.message)}</span></div>`;document.getElementById('compareModal').classList.add('open')}).finally(()=>updateCompareToolbar())}
function renderCatalog(d){let c=d.catalog||{};let diff=d.catalog_diff||{};let mi=d.model_info||{};let ri=d.repo_info||{};let el=document.getElementById('catalogInfo');
if(c.error){el.innerHTML=infoRow('Catalog',`<span class="critical">${esc(c.error)}</span>`);return}
let vars=c.variants||{};let keys=Object.keys(vars);
if(!keys.length){el.innerHTML='<div class="row"><span class="label">No catalog yet</span></div>';return}
let rows='';
// model_info persists stale after a stop; d.container is the live signal,
// so only treat a variant as "running" when a container is actually up.
let running=!!d.container;
let runKey=running?keys.find(k=>vars[k].compose_path&&mi.compose_file&&mi.compose_file.indexOf(vars[k].compose_path)>=0):null;
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
let topo=machineTopology(d);
let fits=keys.filter(k=>variantTopology(vars[k])===topo&&(vars[k].tp||1)<=ngpu);
let order={production:0,caveats:1};
fits.sort((a,b)=>(order[vars[a].status]??2)-(order[vars[b].status]??2)||(vars[a].model||'').localeCompare(vars[b].model||'')||a.localeCompare(b));
rows+=infoRow('Variants for this machine',`${fits.length} of ${keys.length} <button class="btn" onclick="openVariantList()">View variants</button>`,`${ngpu} GPU detected · ${topologyLabel(topo)}`);
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
function selectedPreset(){let el=document.getElementById('presetSel');return el?el.value:'debug'}
function selectedCacheRam(){if(lastModelInfo&&lastModelInfo.cache_ram_supported===false)return false;let el=document.getElementById('cacheRamChk');return el?el.checked:true}
function syncCacheControl(mi){let lab=document.getElementById('cacheRamLabel'),chk=document.getElementById('cacheRamChk');if(!lab||!chk)return;let supported=mi.cache_ram_supported!==false&&engineOf(mi)!=='vllm';lab.style.display=supported?'':'none';let key=[mi.container||'',engineOf(mi),supported,!!mi.cache_ram_enabled].join('|');if(key!==lastCacheSourceKey){lastCacheSourceKey=key;cacheRamUserEdited=false}if(!cacheRamUserEdited)chk.checked=supported?!!mi.cache_ram_enabled:true}
function engineOf(mi){if(mi.engine)return mi.engine;let blob=[mi.image,mi.compose_file,mi.variant].join(' ').toLowerCase();return blob.includes('vllm')?'vllm':'llamacpp'}
function presetOptionsFor(mi,selected,cacheRam){let opts=engineOf(mi)==='vllm'?(VLLM_PRESET_OPTIONS[selected]||{}):(PRESET_OPTIONS[selected]||{});opts={...opts};if(engineOf(mi)!=='vllm'&&cacheRam)opts['--cache-ram']='8192';return opts}
function presetEnvFor(mi,selected){return engineOf(mi)==='vllm'?(VLLM_PRESET_ENV[selected]||{}):{}}
function liveEnvMap(mi){let out={};if(mi.vllm_logging_level)out.VLLM_LOGGING_LEVEL=mi.vllm_logging_level;return out}
function presetDiff(mi,selected){let live=liveOptionMap(mi.command||[]);let want=presetOptionsFor(mi,selected,selectedCacheRam());let envLive=liveEnvMap(mi);let envWant=presetEnvFor(mi,selected);let flags=new Set([...Object.keys(live),...Object.keys(want)].filter(f=>MANAGED_FLAGS.has(f)));let rows=[];flags.forEach(flag=>{let hasLive=Object.prototype.hasOwnProperty.call(live,flag);let hasWant=Object.prototype.hasOwnProperty.call(want,flag);if(hasLive&&hasWant&&String(live[flag])!==String(want[flag]))rows.push(`<span class="cmd-add">${esc(flag)}: ${esc(live[flag]??'switch')} → ${esc(want[flag]??'switch')}</span>`);else if(!hasLive&&hasWant)rows.push(`<span class="cmd-add">add ${esc(optionText(flag,want[flag]))}</span>`);else if(hasLive&&!hasWant)rows.push(`<span class="cmd-add">remove ${esc(optionText(flag,live[flag]))}</span>`)});Object.keys(envWant).forEach(k=>{if(String(envLive[k]||'')!==String(envWant[k]))rows.push(`<span class="cmd-add">set ${esc(k)}=${esc(envWant[k])}</span>`)});Object.keys(envLive).forEach(k=>{if(!Object.prototype.hasOwnProperty.call(envWant,k))rows.push(`<span class="cmd-add">unset ${esc(k)}</span>`)});return rows.join('')}
function renderPresetStatus(mi){let running=mi.preset||'unknown';let selected=selectedPreset();let cls=running==='custom'?'preset-custom':(running===selected?'preset-match':'preset-diff');let rows='';
rows+=infoRow('Running mode',`<span class="preset-pill ${cls}">${esc(presetLabel(running))}</span>`,PRESET_DESCRIPTIONS[running]||'inferred from the live container command');
rows+=infoRow('Selected mode',`<span class="preset-pill ${running===selected?'preset-match':'preset-diff'}">${esc(presetLabel(selected))}</span>`,PRESET_DESCRIPTIONS[selected]||'');
rows+=infoRow('Cache option',`<span class="preset-pill ${mi.cache_ram_enabled===selectedCacheRam()?'preset-match':'preset-diff'}">${selectedCacheRam()?'cache on':'cache off'}</span>`,engineOf(mi)==='vllm'?'No vLLM equivalent; ignored for vLLM':'llama.cpp --cache-ram 8192 toggle');
rows+=infoRow('Mode difference',`<span class="label preset-desc">${esc(PRESET_DESCRIPTIONS[selected]||'')}</span>`);
let diff=presetDiff(mi,selected);if(diff)rows+=`<div class="row"><span class="label">Selected changes</span><span class="value" style="text-align:right">${diff}</span></div>`;
return rows}
function commandFlagAt(argv,i){let tok=String(argv[i]);if(tok.startsWith('--')&&tok.includes('='))return presetFlag(tok.split('=')[0]);return tok.startsWith('-')?presetFlag(tok):null}
function commandTokenClass(argv,i,want,live){let flag=commandFlagAt(argv,i);let prev=i>0?commandFlagAt(argv,i-1):null;if(prev&&Object.prototype.hasOwnProperty.call(live,prev)&&String(argv[i])===String(live[prev]))flag=prev;if(!flag||!MANAGED_FLAGS.has(flag))return '';let hasWant=Object.prototype.hasOwnProperty.call(want,flag);if(!hasWant)return 'cmd-remove';let liveVal=live[flag];let wantVal=want[flag];if(String(liveVal)===String(wantVal))return 'cmd-same';return 'cmd-change'}
function renderCommandLine(mi){let argv=mi.command||[];let selected=selectedPreset();let want=presetOptionsFor(mi,selected,selectedCacheRam());let live=liveOptionMap(argv);let html=argv.map((tok,i)=>{let c=commandTokenClass(argv,i,want,live);return c?`<span class="cmd-token ${c}">${esc(tok)}</span>`:esc(tok)}).join(' ');
let additions=[];MANAGED_FLAGS.forEach(flag=>{if(!Object.prototype.hasOwnProperty.call(live,flag)&&Object.prototype.hasOwnProperty.call(want,flag))additions.push(`<span class="cmd-add">+ ${esc(optionText(flag,want[flag]))}</span>`)});
let envWant=presetEnvFor(mi,selected);let envLive=liveEnvMap(mi);Object.keys(envWant).forEach(k=>{if(String(envLive[k]||'')!==String(envWant[k]))additions.push(`<span class="cmd-add">+ ${esc(k)}=${esc(envWant[k])}</span>`)});
let legend='<div class="cmd-legend"><span class="cmd-token cmd-same">same in selected mode</span><span class="cmd-token cmd-change">value changes</span><span class="cmd-token cmd-remove">removed by selected mode</span></div>';
if(additions.length)legend+=`<div class="cmd-legend">${additions.join('')}</div>`;
return `<div class="cmd-line">${html}</div>${legend}`}
function renderModelInfo(d){let mi=d.model_info||{};let ri=d.repo_info||{};let f=mi.flags||{};let rows='';
lastModelInfo=mi;syncCacheControl(mi);if(flagModalOpen)renderFlagGuideModal(mi);
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
var caseFanFreezeUntil=0;
function setCaseFanDuty(fanId,duty,idx){caseFanFreezeUntil=Date.now()+3000;var lab=document.getElementById('cfval-'+idx);if(lab)lab.textContent=duty+'%…';fetch('/case-fans',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({fan:fanId,duty_pct:Number(duty)})}).then(function(r){return r.json().then(function(j){return{ok:r.ok,j:j}})}).then(function(res){if(!res.ok){if(lab)lab.textContent='error';console.error('[case-fans] set failed:',(res.j&&res.j.error)||('HTTP '));}else if(lab){lab.textContent=res.j.duty_pct+'%';}}).catch(function(e){if(lab)lab.textContent='error';console.error('[case-fans] set error:',e);});}
function renderCaseFans(fans){var card=document.getElementById('caseFanCard');if(Date.now()<caseFanFreezeUntil)return;fans=(fans||[]).filter(function(f){return f.rpm!==0});if(!fans.length){card.style.display='none';return}card.style.display='';var grid=document.getElementById('caseFanGrid');grid.innerHTML=fans.map(function(f,i){var pump=f.kind==='pump';var settable=f.settable===true;var rpm=(f.rpm==null)?'N/A':f.rpm+' RPM';var duty=(f.duty_pct==null)?null:f.duty_pct;var src=esc(f.backend||'')+(pump?' · pump':(settable?'':' · read-only'));var showDuty=settable||duty!=null;var dutyTxt=(duty==null)?'not set':(duty+'%');var sliderVal=(duty==null)?50:duty;var ctrl=settable?('<input type="range" min="0" max="100" value="'+sliderVal+'" data-fan="'+esc(f.id)+'" data-idx="'+i+'" class="cf-slider" style="width:100%;margin-top:6px">'):(duty==null?'':'<div class="bar"><div class="fill fan" style="width:'+duty+'%"></div></div>');return '<div class="gpu-card"><div class="gpu-name">'+esc(f.label||f.id)+'</div>'+
'<div class="row"><span class="label">'+(pump?'Pump':'Speed')+'</span><span class="value">'+rpm+'</span></div>'+
(showDuty?('<div class="row"><span class="label">Duty</span><span class="value" id="cfval-'+i+'">'+dutyTxt+'</span></div>'+ctrl):'')+
'<div class="row"><span class="label">Source</span><span class="value" style="font-weight:500;color:var(--dim)">'+src+'</span></div></div>'}).join('');
grid.querySelectorAll('input.cf-slider').forEach(function(sl){sl.addEventListener('input',function(){caseFanFreezeUntil=Date.now()+3000;var lab=document.getElementById('cfval-'+sl.dataset.idx);if(lab)lab.textContent=sl.value+'%';});sl.addEventListener('change',function(){setCaseFanDuty(sl.dataset.fan,sl.value,sl.dataset.idx);});});}
function renderSummary(d){let m=d.metrics||{};let vllm=m.engine==='vllm';let activeEl=document.getElementById('active');let requestsEl=document.getElementById('requests');let gpuTempEl=document.getElementById('gpuTemp');let memTempEl=document.getElementById('memTemp');let avgTpsEl=document.getElementById('avgTps');activeEl.textContent=vllm?(m.processing??0):d.active_count;requestsEl.textContent=vllm?Number(m.requests_total||0).toLocaleString():(d.requests||[]).length;if(d.gpu_stats&&d.gpu_stats.length){let g=d.gpu_stats[0];gpuTempEl.textContent=`${g.temp_c}°C`;gpuTempEl.className='summary-value '+cls(g.temp_c);memTempEl.textContent=g.mem_temp_c>=0?`${g.mem_temp_c}°C`:'N/A';memTempEl.className='summary-value '+cls(g.mem_temp_c)}let done=(d.requests||[]).filter(r=>r.status==='completed'&&r.gen_tps>0);avgTpsEl.textContent=done.length?(done.reduce((s,r)=>s+r.gen_tps,0)/done.length).toFixed(1):(vllm&&m.gen_tps_avg!=null?Number(m.gen_tps_avg).toFixed(1):'0')}
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
function requestKey(r){return String((r.status||'processing')+'-'+(r.task_id??'x')+'-'+(r.start_time||r.end_time||r.id||0))}
function rowAttrs(r){let k=requestKey(r);requestRowsByKey[k]=r;return ` role="button" tabindex="0" onclick="openRequestDetail('${esc(k)}')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openRequestDetail('${esc(k)}')}"`}
function detailField(label,value){if(value===undefined||value===null||value==='')return '';return `<div class="row"><span class="label">${label}</span><span class="value">${esc(value)}</span></div>`}
function renderMessages(messages){if(!messages||!messages.length)return '<div class="label">No request body captured. Restart with debug mode to log messages.</div>';return messages.map(m=>`<div class="message-card"><div class="message-role">${esc(m.role||'message')}${m.name?' · '+esc(m.name):''}</div><div class="prewrap">${esc(m.content||'')}</div></div>`).join('')}
function renderRequestDetail(r){let meta='';meta+=detailField('Status',r.status);meta+=detailField('Task',r.task_id);meta+=detailField('Group',r.request_group_label);meta+=detailField('Group id',r.request_group_id);meta+=detailField('Model',r.request_model||r.model);meta+=detailField('Messages',r.request_message_count);meta+=detailField('Tools',r.request_tools_count);meta+=detailField('Response format',r.request_has_response_format?'yes':'');meta+=detailField('Stream',r.request_stream?'yes':'');meta+=detailField('Finish reason',r.response_finish_reason||r.finish_reason);
let vllm=!!r.request_id;
if(vllm){meta+=detailField('Max tokens',r.max_tokens);meta+=detailField('Temperature',r.temperature);meta+=detailField('Output tokens',r.completion_tokens);meta+=detailField('TTFT',r.ttft_ms?Math.round(r.ttft_ms)+' ms':'');meta+=detailField('Gen t/s',r.gen_tps?Number(r.gen_tps).toFixed(1):'');meta+=detailField('Latency',r.total_ms?formatDuration(r.total_ms):'')}
let out=r.response_output?`<div class="prewrap">${esc(r.response_output)}</div>`:`<div class="label">${vllm?'No output captured — enable debug mode (--enable-log-outputs) to log responses.':'No response body captured yet. Non-streaming responses require debug logs.'}</div>`;
let rawReq=r.request_detail_json?`<details><summary class="label" style="cursor:pointer;padding:8px 0">raw request JSON</summary><div class="prewrap">${esc(r.request_detail_json)}</div></details>`:'';
let rawResp=r.response_detail_json?`<details><summary class="label" style="cursor:pointer;padding:8px 0">raw response JSON</summary><div class="prewrap">${esc(r.response_detail_json)}</div></details>`:'';
let reqBody=(vllm&&!(r.request_messages&&r.request_messages.length))?'<div class="label">Prompt not logged — restart with debug mode to set VLLM_LOGGING_LEVEL=DEBUG.</div>':renderMessages(r.request_messages);
return `<div class="detail-grid"><div class="detail-section"><h3>Request</h3>${meta}${reqBody}${rawReq}</div><div class="detail-section"><h3>Output</h3>${out}${rawResp}</div></div>`}
function openRequestDetail(key){let r=requestRowsByKey[key];if(!r)return;document.getElementById('requestModalTitle').textContent=`Request ${r.task_id??''} · ${r.status||'processing'}`;document.getElementById('requestModalBody').innerHTML=renderRequestDetail(r);document.getElementById('requestModal').classList.add('open')}
function closeRequestDetail(){document.getElementById('requestModal').classList.remove('open')}
function renderRequests(reqs,active){requestRowsByKey={};let head='<div class="request-row request-head"><span>Status</span><span>Time</span><span>Group</span><span>PT</span><span>Cache</span><span>TTFT</span><span>P t/s</span><span>P time</span><span>G t/s</span><span>GT</span><span>G time</span><span>Total</span></div>';let act=(active||[]).slice().reverse();let actRows=act.map(r=>{let phase=r.phase==='generating'?'generating':(r.phase==='prefill'?'prefill':'processing');let ptime=r.phase==='prefill'?`<div class="bar" title="${r.prefill_pct||0}%"><div class="fill" style="width:${r.prefill_pct||0}%"></div></div>`:'-';return `<div class="request-row live"${rowAttrs(r)}>${statusCell(r,phase)}<span>${r.start_time_str||'--'}</span>${groupCell(r)}<span>${r.prompt_tokens||0}</span>${cacheCell(r)}<span>${formatPhaseDuration(r.ttft_ms)}</span><span>${r.prompt_tps?Number(r.prompt_tps).toFixed(1):'-'}</span><span>${ptime}</span><span>-</span><span>${r.completion_tokens||0}</span><span>-</span><span>${liveElapsed(r)}</span></div>`}).join('');let recent=reqs.slice(-40).reverse();let doneRows=recent.map(r=>`<div class="request-row"${rowAttrs(r)}>${statusCell(r,r.status)}<span>${r.end_time_str||r.start_time_str||'--'}</span>${groupCell(r)}<span>${r.prompt_tokens||0}</span>${cacheCell(r)}<span>${formatPhaseDuration(r.ttft_ms)}</span><span>${r.prompt_tps?Number(r.prompt_tps).toFixed(1):'-'}</span><span>${formatPhaseDuration(r.prompt_eval_ms)}</span><span>${r.gen_tps?Number(r.gen_tps).toFixed(1):'-'}</span><span>${r.completion_tokens||0}</span><span>${formatPhaseDuration(r.eval_ms)}</span><span>${formatDuration(r.total_ms||r.elapsed_ms)}</span></div>`).join('');let body=actRows+doneRows;let vllm=((lastRenderData||{}).metrics||{}).engine==='vllm';let empty=vllm?'<div class="request-row"><span class="label">No request rows yet — pick debug mode and Restart model to enable vLLM request logging.</span></div>':'<div class="request-row"><span class="label">No requests yet</span></div>';document.getElementById('requestList').innerHTML=head+(body||empty)}
function renderDockerLogs(d){let logs=d.docker_logs||[];let el=document.getElementById('dockerLogs');if(!el)return;if(!logs.length){el.innerHTML='<div class="label" style="padding:8px 0">No docker logs yet</div>';return}let html=logs.map(l=>`<div class="log-line">${esc(l)}</div>`).join('');el.innerHTML=html;let container=el;if(container._autoScroll){container.scrollTop=container.scrollHeight}}
function renderGpuHistory(d){
let card=document.getElementById('gpuHistoryCard');
let container=document.getElementById('gpuHistoryCharts');
let hist=d.gpu_history||[];
if(!hist.length||!(d.gpu_stats||[]).length){card.style.display='none';return}
card.style.display='';
// group history by gpu index
let gpuIds=[...new Set((d.gpu_stats||[]).map(g=>g.index))].sort((a,b)=>a-b);
// hist entries are {timestamp, gpus:[{index, ...}, ...]} — extract per-gpu samples
let byGpu={};
gpuIds.forEach(id=>{byGpu[id]=[]});
hist.forEach(entry=>{
let gpus=Array.isArray(entry)?entry:(entry.gpus||[]);
if(!Array.isArray(gpus))return;
gpus.forEach(g=>{if(g.index!=null&&byGpu[g.index]!=null)byGpu[g.index].push(g)});
});
// ensure container has correct number of chart divs
while(container.children.length<gpuIds.length){
let wrap=document.createElement('div');
wrap.className='gpu-history-chart';
let lbl=document.createElement('div');
lbl.className='gpu-history-label';
let leg=document.createElement('div');
leg.className='gpu-history-legend';
let cvs=document.createElement('canvas');
wrap.appendChild(lbl);
wrap.appendChild(leg);
wrap.appendChild(cvs);
container.appendChild(wrap);
}
while(container.children.length>gpuIds.length)container.removeChild(container.lastChild);
gpuIds.forEach((gpuId,ci)=>{
let wrap=container.children[ci];
let cvs=wrap.querySelector('canvas');
let lbl=wrap.querySelector('.gpu-history-label');
let leg=wrap.querySelector('.gpu-history-legend');
let samples=byGpu[gpuId]||[];
if(!samples.length)return;
let gpuName=samples[samples.length-1].name||('GPU '+gpuId);
let lastUtil=samples[samples.length-1].gpu_util_pct||0;
let lastMemUsed=samples[samples.length-1].mem_used_mib||0;
let lastMemTotal=samples[samples.length-1].mem_total_mib||1;
let lastMemPct=Math.round(100*lastMemUsed/lastMemTotal);
lbl.textContent='GPU '+gpuId+': '+gpuName;
leg.innerHTML='<span><span class="legend-dot" style="background:#58a6ff"></span>GPU '+lastUtil+'%</span>'+
'<span><span class="legend-dot" style="background:#bc8cff"></span>VRAM '+lastMemPct+'% ('+(lastMemUsed/1024).toFixed(1)+'G)</span>';
// draw chart
let rect=wrap.getBoundingClientRect();
let dpr=window.devicePixelRatio||1;
let w=Math.round(rect.width*dpr);
let h=Math.round(rect.height*dpr);
if(cvs.width!==w||cvs.height!==h){cvs.width=w;cvs.height=h}
let ctx=cvs.getContext('2d');
ctx.clearRect(0,0,w,h);
// margins
let ml=0,mr=0,mt=30*dpr,mb=18*dpr;
let cw=w-ml-mr,ch=h-mt-mb;
// gridlines
ctx.strokeStyle='rgba(48,54,61,0.8)';
ctx.lineWidth=dpr;
ctx.setLineDash([3*dpr,3*dpr]);
for(let p=0;p<=100;p+=25){
let y=mt+ch*(1-p/100);
ctx.beginPath();ctx.moveTo(ml,y);ctx.lineTo(ml+cw,y);ctx.stroke();
// grid label
ctx.fillStyle='rgba(139,148,158,0.6)';
ctx.font=(9*dpr)+'px -apple-system,BlinkMacSystemFont,sans-serif';
ctx.textAlign='left';
ctx.fillText(p+'%',ml+3*dpr,y-2*dpr);
}
ctx.setLineDash([]);
// time labels
let nSamples=samples.length;
let totalSec=nSamples*2;
ctx.fillStyle='rgba(139,148,158,0.5)';
ctx.font=(9*dpr)+'px -apple-system,BlinkMacSystemFont,sans-serif';
ctx.textAlign='center';
for(let i=0;i<nSamples;i+=Math.max(1,Math.floor(nSamples/5))){
let x=ml+(i/(nSamples-1||1))*cw;
let secAgo=Math.round((nSamples-1-i)*2);
ctx.fillText(secAgo+'s',x,h-2*dpr);
}
// draw a filled area series
function drawSeries(values,strokeColor,fillColor){
if(values.length<2)return;
ctx.beginPath();
for(let i=0;i<values.length;i++){
let x=ml+(i/(values.length-1||1))*cw;
let y=mt+ch*(1-Math.min(100,Math.max(0,values[i]))/100);
if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
}
ctx.strokeStyle=strokeColor;
ctx.lineWidth=1.5*dpr;
ctx.stroke();
// fill
let last=values.length-1;
ctx.lineTo(ml+(last/(values.length-1||1))*cw,mt+ch);
ctx.lineTo(ml,mt+ch);
ctx.closePath();
ctx.fillStyle=fillColor;
ctx.fill();
}
let gpuUtils=samples.map(s=>s.gpu_util_pct||0);
let vramPcts=samples.map(s=>{
let used=s.mem_used_mib||0,total=s.mem_total_mib||1;
return 100*used/total;
});
drawSeries(vramPcts,'#bc8cff','rgba(188,140,255,0.15)');
drawSeries(gpuUtils,'#58a6ff','rgba(88,166,255,0.12)');
});
}
// Drag-and-drop + resize
(function(){
var STORAGE_KEY='observer-card-order';
var SIZE_KEY='observer-card-sizes';
var HEIGHT_KEY='observer-card-heights';
function cardId(c){var h=c.querySelector('h2 span:first-child');return h?h.textContent.trim():''}
function saveOrder(){var g=document.querySelector('.grid');if(!g)return;var ids=[];for(var i=0;i<g.children.length;i++)ids.push(cardId(g.children[i]));localStorage.setItem(STORAGE_KEY,JSON.stringify(ids))}
function saveSizes(){var g=document.querySelector('.grid');if(!g)return;var s={};for(var i=0;i<g.children.length;i++){var c=g.children[i];var id=cardId(c);if(c.classList.contains('full'))s[id]='full';else if(c.classList.contains('half'))s[id]='half';else s[id]='default'}localStorage.setItem(SIZE_KEY,JSON.stringify(s))}
function saveHeights(){var g=document.querySelector('.grid');if(!g)return;var h={};for(var i=0;i<g.children.length;i++){var c=g.children[i];var id=cardId(c);if(c.classList.contains('height-expanded'))h[id]=true}localStorage.setItem(HEIGHT_KEY,JSON.stringify(h))}
function applyOrder(){var raw=localStorage.getItem(STORAGE_KEY);if(!raw)return false;var g=document.querySelector('.grid');if(!g)return false;var order;try{order=JSON.parse(raw)}catch(e){return false}if(!order||!order.length)return false;var map={};for(var i=0;i<g.children.length;i++)map[cardId(g.children[i])]=g.children[i];var newCards=[];for(var j=0;j<g.children.length;j++){var id=cardId(g.children[j]);if(order.indexOf(id)===-1)newCards.push(id)}for(var k=0;k<order.length;k++){var el=map[order[k]];if(el)g.appendChild(el)}for(var m=0;m<newCards.length;m++){var el2=map[newCards[m]];if(el2)g.appendChild(el2)}return true}
function applySizes(){var raw=localStorage.getItem(SIZE_KEY);if(!raw)return;var s;try{s=JSON.parse(raw)}catch(e){return}if(!s)return;var g=document.querySelector('.grid');if(!g)return;for(var i=0;i<g.children.length;i++){var c=g.children[i];var id=cardId(c);var sz=s[id];c.classList.remove('full','half');if(sz==='full')c.classList.add('full');else if(sz==='half')c.classList.add('half')}}
function applyHeights(){var raw=localStorage.getItem(HEIGHT_KEY);if(!raw)return;var h;try{h=JSON.parse(raw)}catch(e){return}if(!h)return;var g=document.querySelector('.grid');if(!g)return;for(var i=0;i<g.children.length;i++){var c=g.children[i];var id=cardId(c);if(h[id]){c.classList.add('height-expanded');var hb=c.querySelector('.resize-btn-h');if(hb)hb.classList.add('active')}}}
applyOrder();applySizes();applyHeights();
var grid=document.querySelector('.grid');if(!grid)return;
var dragSrc=null;
for(var i=0;i<grid.children.length;i++)grid.children[i].draggable=true;
grid.addEventListener('dragstart',function(e){var card=e.target.closest('.card');if(!card)return;dragSrc=card;card.classList.add('dragging');e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('text/plain','')});
grid.addEventListener('dragend',function(e){var card=e.target.closest('.card');if(!card)return;card.classList.remove('dragging');for(var i=0;i<grid.children.length;i++)grid.children[i].classList.remove('drag-over');dragSrc=null;saveOrder()});
grid.addEventListener('dragover',function(e){e.preventDefault();e.dataTransfer.dropEffect='move';var card=e.target.closest('.card');if(!card||card===dragSrc)return;for(var i=0;i<grid.children.length;i++)grid.children[i].classList.remove('drag-over');card.classList.add('drag-over')});
grid.addEventListener('dragleave',function(e){if(e.target.classList.contains('card'))e.target.classList.remove('drag-over')});
grid.addEventListener('drop',function(e){e.preventDefault();var card=e.target.closest('.card');if(!card||!dragSrc||card===dragSrc)return;card.classList.remove('drag-over');var children=[].slice.call(grid.children);var fromIdx=children.indexOf(dragSrc);var toIdx=children.indexOf(card);if(fromIdx<toIdx)card.after(dragSrc);else card.before(dragSrc)});
grid.addEventListener('click',function(e){var btn=e.target.closest('.resize-btn');if(!btn)return;var card=btn.closest('.card');if(!card)return;if(btn.classList.contains('resize-btn-w')){if(!card.classList.contains('half')&&!card.classList.contains('full'))card.classList.add('half');else if(card.classList.contains('half')){card.classList.remove('half');card.classList.add('full')}else card.classList.remove('full');saveSizes()}else if(btn.classList.contains('resize-btn-h')){card.classList.toggle('height-expanded');btn.classList.toggle('active');saveHeights()}});
var obs=new MutationObserver(function(mutations){mutations.forEach(function(m){for(var i=0;i<m.addedNodes.length;i++){var n=m.addedNodes[i];if(n.nodeType===1&&n.classList&&n.classList.contains('card'))n.draggable=true}})});
obs.observe(grid,{childList:true});
})();
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


def observer_control_options(body, default_mode="debug"):
    raw = str(body.get("preset", default_mode))
    mode = normalize_observer_mode(raw)
    if mode not in MODE_CAPABILITIES:
        raise ValueError(f"unknown preset {raw!r}")
    cache_ram = body.get("cache_ram")
    if cache_ram is None:
        cache_ram = LEGACY_PRESET_CACHE_DEFAULT.get(raw, True)
    return mode, bool(cache_ram)


def handle_observer_post(handler):
    """Control endpoints: update the club-3090 checkout / restart the model.

    Single-flight: only one control action at a time per host. The Tailscale
    binding is the auth boundary, matching the existing PUT /curve design.
    """
    path = handler.path.split("?", 1)[0]
    if path not in ("/observer/api/update", "/observer/api/restart",
                    "/observer/api/stop", "/observer/api/switch",
                    "/observer/api/install", "/observer/api/compare"):
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
    if path == "/observer/api/compare":
        repo = _config.get("model_repo")
        if not repo:
            _send_json(handler, 503, {"error": "no model repo configured"})
            return True
        try:
            mi = state.model_info or {}
            help_info = inspect_container_help(
                mi.get("container"), mi.get("entrypoint"),
                engine=infer_engine(mi), image=mi.get("image"),
            ) or {}
            _send_json(
                handler, 200,
                compare_variant_commands(
                    repo, body.get("variants"), state.catalog,
                    help_info=help_info,
                ),
            )
        except ValueError as e:
            _send_json(handler, 400, {"error": str(e)})
        except Exception as e:
            _send_json(handler, 409, {"error": str(e)})
        return True
    if not _control_lock.acquire(blocking=False):
        print(f"[observer] {path}: rejected — another control action is running", flush=True)
        _send_json(handler, 409, {"error": "another control action is running"})
        return True
    release = True
    # Stamp the moment the action was triggered (button click) so the load
    # profile measures true end-to-end wall-clock, not just the worker runtime.
    started_at = time.time()
    try:
        if path == "/observer/api/update":
            repo = _config.get("model_repo")
            if not repo:
                _send_json(handler, 503, {"error": "no model repo configured"})
                return True
            result = update_repo(repo)
            _repo_wake.set()
        elif path == "/observer/api/stop":
            check_restart_allowed(state, force=bool(body.get("force")))
            result = stop_model(repo=_config.get("model_repo"))
        elif path == "/observer/api/switch":
            print(f"[observer] SWITCH request: body={body}", flush=True)
            repo = _config.get("model_repo")
            if not repo:
                print(f"[observer] SWITCH rejected: no model repo configured", flush=True)
                _send_json(handler, 503, {"error": "no model repo configured"})
                return True
            variant = str(body.get("variant", ""))
            preset, cache_ram = observer_control_options(body, "baseline")
            force = bool(body.get("force"))
            print(f"[observer] SWITCH: variant={variant!r} preset={preset!r} cache_ram={cache_ram} force={force}", flush=True)
            variant = normalize_switch_variant(variant, state.catalog)
            print(f"[observer] SWITCH: normalized variant={variant!r}", flush=True)
            validate_switch(variant, state.catalog, force=force)
            check_restart_allowed(state, force=force)
            audit("switch", f"variant={variant} preset={preset} "
                            f"cache_ram={cache_ram} force={force}")
            print(f"[observer] SWITCH: spawning worker thread", flush=True)
            # Long-running (minutes): hand off to a worker that owns the
            # control lock; progress streams via control_status over SSE.
            threading.Thread(
                target=_switch_worker,
                args=(repo, variant, preset, _config["monitor_port"], force),
                kwargs={"cache_ram": cache_ram, "started_at": started_at},
                name="observer-switch",
                daemon=True,
            ).start()
            release = False
            result = {"started": True, "variant": variant, "preset": preset,
                      "cache_ram": cache_ram}
        elif path == "/observer/api/install":
            print(f"[observer] INSTALL request: body={body}", flush=True)
            repo = _config.get("model_repo")
            if not repo:
                print(f"[observer] INSTALL rejected: no model repo configured", flush=True)
                _send_json(handler, 503, {"error": "no model repo configured"})
                return True
            variant = str(body.get("variant", ""))
            preset, cache_ram = observer_control_options(body, "baseline")
            force = bool(body.get("force"))
            retry = bool(body.get("retry"))
            variant = normalize_switch_variant(variant, state.catalog)
            print(f"[observer] INSTALL: variant={variant!r} preset={preset!r} cache_ram={cache_ram} force={force} retry={retry}", flush=True)
            validate_switch(variant, state.catalog, force=True)
            if retry:
                check_restart_allowed(state, force=force)
            setup = {
                k: body.get(k)
                for k in ("model", "weight_key", "model_dir")
                if body.get(k)
            }
            audit("install", f"variant={variant} retry={retry}")
            print(f"[observer] INSTALL: spawning worker thread", flush=True)
            threading.Thread(
                target=_install_worker,
                args=(
                    repo, variant, preset, _config["monitor_port"], force,
                    retry, setup,
                ),
                kwargs={"cache_ram": cache_ram, "started_at": started_at},
                name="observer-install",
                daemon=True,
            ).start()
            release = False
            result = {"started": True, "variant": variant, "retry": retry,
                      "preset": preset, "cache_ram": cache_ram}
        else:
            check_restart_allowed(state, force=bool(body.get("force")))
            preset, cache_ram = observer_control_options(body, "debug")
            # Hand off like switch/install so the profile captures the reload +
            # readiness wait end-to-end instead of returning before it loads.
            threading.Thread(
                target=_restart_worker,
                args=(preset, _config["monitor_port"]),
                kwargs={"cache_ram": cache_ram, "started_at": started_at},
                name="observer-restart",
                daemon=True,
            ).start()
            release = False
            result = {"started": True, "preset": preset, "cache_ram": cache_ram}
        _send_json(
            handler,
            202 if path.endswith(("/switch", "/install", "/restart")) else 200,
            result,
        )
    except ValueError as e:
        print(f"[observer] {path}: ValueError — {e}", flush=True)
        _send_json(handler, 400, {"error": str(e)})
    except Exception as e:
        print(f"[observer] {path}: Exception — {e}", flush=True)
        _send_json(handler, 409, {"error": str(e)})
    finally:
        if release:
            _control_lock.release()
    return True


def classify_crash(container_name, logs, runner=_run):
    """Best-effort label for why the model died: OOM, a crash, or unresponsive.

    The alert/revive fire regardless of cause (OOM is just the common one); this
    only decorates the notification. Scans the tail of the docker logs for an
    OOM signature first, then falls back to the exited container's state
    (OOMKilled / exit 137 both mean the kernel reaped it).
    """
    text = "\n".join(logs[-80:]).lower() if logs else ""
    for sig in OOM_LOG_SIGNATURES:
        if sig in text:
            return f"OOM ({sig})"
    if container_name:
        try:
            out = runner(
                ["docker", "inspect", "--format",
                 "{{.State.OOMKilled}} {{.State.ExitCode}}", container_name]
            ).strip().split()
            oom = bool(out) and out[0].lower() == "true"
            code = out[1] if len(out) > 1 else ""
            if oom:
                return "OOM (killed)"
            if code == "137":
                return "OOM (exit 137)"
            if code and code != "0":
                return f"crash (exit {code})"
        except Exception:
            pass
    return "unresponsive"


def _classify_crash_now():
    return classify_crash(state.container_name, list(state.docker_logs))


def _revive_model():
    """Recreate the crashed model container via the same path as the dashboard
    "Restart model" button. Returns (ok, detail). Never raises."""
    if not _control_lock.acquire(blocking=False):
        return False, "another control action is running"
    try:
        mi = state.model_info
        preset = mi.get("preset") or "baseline"
        cache_ram = mi.get("cache_ram_enabled")
        restart_model(preset, cache_ram=cache_ram)
        return True, f"preset={preset}"
    except Exception as e:
        return False, str(e)
    finally:
        _control_lock.release()


def _watchdog_notify(event, **info):
    """Format and push an oncall ntfy alert for a watchdog event."""
    host = HOSTNAME
    if event == "down":
        reason = info.get("reason", "")
        tags = ["rotating_light"] + (["warning"] if reason.startswith("OOM") else [])
        send_ntfy(
            f"Model on {host} is DOWN: {reason}. "
            f"model={info.get('model') or '?'}. Attempting auto-revive.",
            title=f"{host} model down", priority="urgent", tags=tags,
        )
    elif event == "revived":
        send_ntfy(
            f"Auto-revive attempt {info.get('attempt')} on {host} triggered "
            f"({info.get('detail')}). Watching for recovery.",
            title=f"{host} model revive", priority="default", tags=["recycle"],
        )
    elif event == "revive_failed":
        send_ntfy(
            f"Auto-revive attempt {info.get('attempt')} on {host} FAILED: "
            f"{info.get('detail')}.",
            title=f"{host} revive failed", priority="high", tags=["warning"],
        )
    elif event == "gave_up":
        send_ntfy(
            f"Gave up auto-reviving model on {host} after "
            f"{MAX_REVIVE_ATTEMPTS} attempts. Manual intervention needed.",
            title=f"{host} revive gave up", priority="urgent",
            tags=["rotating_light", "sos"],
        )


class WatchdogState:
    """State machine that turns a stream of (healthy?) probes into alerts and
    bounded auto-revive attempts. Kept side-effect-injectable so the transition
    logic is unit-testable without threads, sleeps, docker, or the network."""

    def __init__(self):
        # The model has come up at least once this episode. Until it does we
        # never alarm — a never-healthy probe is first boot, not a crash.
        self.seen_healthy = False
        # Ready to fire the "down" alert for the current crash episode.
        self.armed = True
        self.attempts = 0
        self.down_since = None       # monotonic time the probe first failed
        self.next_attempt_at = 0.0   # backoff gate for the next revive
        self.gave_up = False
        self.last_reason = None
        self.deliberately_stopped = False

    def summary(self, now=None):
        now = time.monotonic() if now is None else now
        return {
            "armed": self.armed,
            "attempts": self.attempts,
            "max_attempts": MAX_REVIVE_ATTEMPTS,
            "down_seconds": (now - self.down_since) if self.down_since else 0,
            "gave_up": self.gave_up,
            "last_reason": self.last_reason,
            "deliberately_stopped": self.deliberately_stopped,
        }

    def _reset(self):
        self.armed = True
        self.attempts = 0
        self.down_since = None
        self.next_attempt_at = 0.0
        self.gave_up = False

    def mark_deliberately_stopped(self):
        self.deliberately_stopped = True
        self._reset()

    def clear_deliberately_stopped(self):
        self.deliberately_stopped = False

    def tick(self, now, status, control_busy, *,
             classify=None, revive=None, notify=None, model=None):
        """Advance the machine one probe. `status` is "ready"|"loading"|"down"."""
        classify = classify or _classify_crash_now
        revive = revive or _revive_model
        notify = notify or _watchdog_notify

        if status == "ready":
            self.seen_healthy = True
            if self.deliberately_stopped:
                self.deliberately_stopped = False
                self._reset()
            elif not self.armed or self.attempts or self.down_since is not None:
                self._reset()
            return
        if self.deliberately_stopped and status == "down":
            self.down_since = None
            return
        if control_busy or status == "loading":
            # A switch/install/restart — or a model still loading its weights —
            # is legitimately not serving; don't count that as a crash. Pause
            # the down clock and wait.
            self.down_since = None
            return
        # status == "down": the model is unreachable.
        if not self.seen_healthy:
            return
        if self.down_since is None:
            self.down_since = now
            return
        if now - self.down_since < WATCHDOG_DOWN_GRACE:
            return  # ride out a transient blip before declaring a crash
        # Crash confirmed.
        if self.armed:
            self.armed = False
            reason = classify()
            self.last_reason = reason
            audit("watchdog", f"model down: {reason}")
            notify("down", reason=reason, model=model)
        if self.attempts >= MAX_REVIVE_ATTEMPTS:
            if not self.gave_up:
                self.gave_up = True
                audit("watchdog", f"giving up after {self.attempts} revive attempts")
                notify("gave_up")
            return
        if now < self.next_attempt_at:
            return  # backing off between attempts
        self.attempts += 1
        ok, detail = revive()
        audit("watchdog",
              f"revive attempt {self.attempts}: {'ok' if ok else 'failed'} {detail}")
        notify("revived" if ok else "revive_failed",
               attempt=self.attempts, detail=detail)
        # Restart the grace + backoff clocks so the model gets time to boot
        # before we count it down again or retry.
        self.down_since = now
        self.next_attempt_at = now + REVIVE_BACKOFF_BASE * (2 ** self.attempts)


_watchdog = WatchdogState()


def watch_model_health(monitor_port):
    """Daemon loop: probe the model and drive the crash/alert/revive machine."""
    while True:
        try:
            status = probe_model_state(monitor_port)
            _watchdog.tick(
                time.monotonic(), status, _control_lock.locked(),
                model=state.model_name,
            )
        except Exception as e:
            print(f"WARNING: watchdog tick error: {e}", file=sys.stderr)
        time.sleep(WATCHDOG_INTERVAL)


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
    if WATCHDOG_ENABLED:
        threading.Thread(
            target=watch_model_health,
            args=(monitor_port,),
            name="observer-watchdog",
            daemon=True,
        ).start()
    print(
        f"Observer enabled at /observer (host {HOSTNAME}, monitor :{monitor_port}, "
        f"container {container or 'auto-detect'})",
        flush=True,
    )
