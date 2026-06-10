#!/usr/bin/env python3
"""Integrated aipc observer dashboard.

This module folds the old aipc-observer sidecar into the fan controller's HTTP
server. It monitors llama.cpp docker logs, active TCP connections, and nvidia-smi
GPU stats, then serves a small dashboard at /observer with SSE updates.
"""

import json
import re
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
GPU_POLL_INTERVAL = 2.0
CONN_CHECK_INTERVAL = 3.0
MODEL_POLL_INTERVAL = 30.0
SLOTS_POLL_INTERVAL = 2.0
CONTAINER_DETECT_INTERVAL = 5.0
REQUEST_LOG_MAX = 200

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
        # task_id -> in-flight request dict, shown as live "processing" rows.
        self.active_requests = {}
        # GPU index -> real VRAM temp (°C) from the gddr6 reader, since
        # nvidia-smi reports temperature.memory as N/A on consumer cards.
        self.vram_temps = {}

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

    def incr_cancelled(self):
        with self.lock:
            self.cancelled_count += 1

    def incr_cache_defeated(self):
        with self.lock:
            self.cache_defeated_count += 1

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


def fetch_json(url, timeout=5):
    """GET a JSON endpoint, returning the parsed body or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


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
RE_TASK_ID = re.compile(r"\btask\s+(\d+)\b")


class RequestTracker:
    def __init__(self, observer_state=None):
        self.state = observer_state or state
        self.active = {}
        self.task_counter = 0
        self.current_timing_task_id = None

    def process_line(self, line):
        m = RE_LAUNCH_SLOT.search(line)
        if m:
            slot_id, task_id = int(m.group(1)), int(m.group(2))
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
                "phase": "starting",
                "prefill_pct": 0,
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

        m = RE_DRAFT.search(line)
        if m and req is not None:
            req["draft_acceptance"] = safe_float(m.group(1))
            req["draft_accepted"] = int(m.group(2))
            req["draft_generated"] = int(m.group(3))
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
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.header,.card{background:var(--surface);border:1px solid var(--border);border-radius:8px}.header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;margin-bottom:16px}.header h1{font-size:18px;margin:0}.meta,.label{color:var(--dim)}.model{color:var(--accent);font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{padding:16px}.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 12px}.gpu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.gpu-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px}.gpu-name{font-weight:650;color:var(--accent);margin-bottom:8px}.row{display:flex;justify-content:space-between;gap:16px;padding:4px 0;font-size:13px}.value{font-variant-numeric:tabular-nums;font-weight:600}.bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden}.fill{height:100%;background:var(--accent);border-radius:3px}.fill.mem{background:var(--purple)}.fill.power{background:var(--yellow)}.fill.fan{background:var(--green)}.hot{color:var(--yellow)}.critical{color:var(--red)}.summary{display:flex;gap:24px;flex-wrap:wrap}.summary-item{text-align:center}.summary-value{font-size:28px;font-weight:750;font-variant-numeric:tabular-nums}.summary-label{font-size:11px;text-transform:uppercase;color:var(--dim);letter-spacing:.05em}.full{grid-column:1/-1}.requests{max-height:520px;overflow:auto}.request-row{display:grid;grid-template-columns:88px 150px 60px 56px 70px 74px 78px 74px 60px 78px 1fr;gap:8px;align-items:center;padding:7px 8px;border-bottom:1px solid var(--border);font-size:12px}.good{color:var(--green)}.request-head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700}.status{border-radius:999px;padding:2px 8px;text-align:center;font-size:10px;text-transform:uppercase;font-weight:700}.completed{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}.processing{background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.3)}.cancelled{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}.request-row.live{box-shadow:inset 3px 0 0 var(--accent)}
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
</div></section>
<section class="card"><h2>Context / KV Cache</h2><div id="slotInfo" class="gpu-grid"></div></section>
<section class="card"><h2>Inference Health</h2><div class="summary">
<div class="summary-item"><div id="truncRate" class="summary-value">0%</div><div class="summary-label">Truncated</div></div>
<div class="summary-item"><div id="cancelled" class="summary-value">0</div><div class="summary-label">Cancelled</div></div>
<div class="summary-item"><div id="cacheDefeat" class="summary-value">0</div><div class="summary-label">Cache Defeated</div></div>
<div class="summary-item"><div id="draftAccept" class="summary-value">-</div><div class="summary-label">Avg Draft Accept</div></div>
</div></section>
<section class="card full"><h2>Recent Requests</h2><div id="requestList" class="requests"></div></section>
</div>
<script>
let es;function pct(v,max){return Math.max(0,Math.min(100,(v/max)*100))}
function cls(t){return t>85?'critical':t>75?'hot':''}
function connect(){if(es)es.close();es=new EventSource('/observer/sse');es.onmessage=e=>render(JSON.parse(e.data));es.onerror=()=>{es.close();setTimeout(connect,3000)}}
function render(d){renderHeader(d);renderGpu(d.gpu_stats||[]);renderSummary(d);renderSlots(d);renderHealth(d);renderRequests(d.requests||[],d.active_requests||[]);document.getElementById('uptime').textContent=d.uptime_human;document.getElementById('updated').textContent=new Date().toLocaleTimeString()}
function renderHealth(d){let reqs=d.requests||[];let comp=reqs.filter(r=>r.status==='completed');let trunc=comp.filter(r=>r.truncated).length;document.getElementById('truncRate').textContent=comp.length?(100*trunc/comp.length).toFixed(0)+'%':'0%';document.getElementById('cancelled').textContent=d.cancelled_count||0;document.getElementById('cacheDefeat').textContent=d.cache_defeated_count||0;let dr=reqs.filter(r=>r.draft_acceptance!=null);document.getElementById('draftAccept').textContent=dr.length?(100*dr.reduce((s,r)=>s+r.draft_acceptance,0)/dr.length).toFixed(0)+'%':'-'}
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
function cacheCell(r){if(r.cache_hit_pct==null)return '<span>-</span>';let p=Number(r.cache_hit_pct);let c=p>=70?'good':p>=30?'hot':'critical';return `<span class="${c}">${p.toFixed(0)}%</span>`}
function renderRequests(reqs,active){let head='<div class="request-row request-head"><span>Status</span><span>Time</span><span>PT</span><span>Cache</span><span>TTFT</span><span>P t/s</span><span>P time</span><span>G t/s</span><span>GT</span><span>G time</span><span>Total</span></div>';let act=(active||[]).slice().reverse();let actRows=act.map(r=>{let phase=r.phase==='generating'?'generating':(r.phase==='prefill'?'prefill':'processing');let ptime=r.phase==='prefill'?`<div class="bar" title="${r.prefill_pct||0}%"><div class="fill" style="width:${r.prefill_pct||0}%"></div></div>`:'-';return `<div class="request-row live"><span class="status processing">${phase}</span><span>${r.start_time_str||'--'}</span><span>${r.prompt_tokens||0}</span>${cacheCell(r)}<span>${formatPhaseDuration(r.ttft_ms)}</span><span>-</span><span>${ptime}</span><span>-</span><span>${r.completion_tokens||0}</span><span>-</span><span>${liveElapsed(r)}</span></div>`}).join('');let recent=reqs.slice(-40).reverse();let doneRows=recent.map(r=>`<div class="request-row"><span class="status ${r.status}">${r.status}</span><span>${r.end_time_str||r.start_time_str||'--'}</span><span>${r.prompt_tokens||0}</span>${cacheCell(r)}<span>${formatPhaseDuration(r.ttft_ms)}</span><span>${r.prompt_tps?Number(r.prompt_tps).toFixed(1):'-'}</span><span>${formatPhaseDuration(r.prompt_eval_ms)}</span><span>${r.gen_tps?Number(r.gen_tps).toFixed(1):'-'}</span><span>${r.completion_tokens||0}</span><span>${formatPhaseDuration(r.eval_ms)}</span><span>${formatDuration(r.total_ms||r.elapsed_ms)}</span></div>`).join('');let body=actRows+doneRows;document.getElementById('requestList').innerHTML=head+(body||'<div class="request-row"><span class="label">No requests yet</span></div>')}
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


def start_observer(monitor_port=DEFAULT_MONITOR_PORT, container=DEFAULT_CONTAINER):
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
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
        target=tail_docker_logs,
        args=(container, monitor_port),
        name="observer-docker-logs",
        daemon=True,
    ).start()
    print(
        f"Observer enabled at /observer (host {HOSTNAME}, monitor :{monitor_port}, "
        f"container {container or 'auto-detect'})",
        flush=True,
    )
