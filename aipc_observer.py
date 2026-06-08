#!/usr/bin/env python3
"""Integrated aipc observer dashboard.

This module folds the old aipc-observer sidecar into the fan controller's HTTP
server. It monitors llama.cpp docker logs, active TCP connections, and nvidia-smi
GPU stats, then serves a small dashboard at /observer with SSE updates.
"""

import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

DEFAULT_MONITOR_PORT = 8020
DEFAULT_CONTAINER = "beellama-qwen36-27b"
GPU_POLL_INTERVAL = 2.0
CONN_CHECK_INTERVAL = 3.0
REQUEST_LOG_MAX = 200


class ObserverState:
    def __init__(self):
        self.lock = threading.Lock()
        self.gpu_stats = []
        self.gpu_history = deque(maxlen=600)
        self.requests = deque(maxlen=REQUEST_LOG_MAX)
        self.active_connections = {}
        self.start_time = time.time()
        self.sse_subscribers = []

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
                "uptime_seconds": uptime,
                "uptime_human": format_duration(uptime),
                "gpu_stats": list(self.gpu_stats),
                "gpu_history": list(self.gpu_history)[-100:],
                "active_connections": dict(self.active_connections),
                "active_count": len(self.active_connections),
                "requests": list(self.requests),
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
)
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
                "model": "llama.cpp",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_eval_ms": 0,
                "prompt_tps": 0,
                "eval_ms": 0,
                "gen_tps": 0,
                "total_ms": 0,
            }
            self.task_counter += 1
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

        m = RE_RELEASE.search(line)
        if m:
            task_id = int(m.group(2))
            n_tokens = int(m.group(3))
            if task_id not in self.active:
                return
            req = self.active.pop(task_id)
            if self.current_timing_task_id == task_id:
                self.current_timing_task_id = None
            req["status"] = "completed"
            req["end_time"] = time.time()
            req["end_time_str"] = local_time_str(req["end_time"])
            req["elapsed_ms"] = (req["end_time"] - req["start_time"]) * 1000
            req["total_tokens"] = max(req["total_tokens"], n_tokens)
            with self.state.lock:
                conns = list(self.state.active_connections.values())
            req["client_ip"] = conns[0]["peer_ip"] if conns else "unknown"
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


def tail_docker_logs(container):
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "--tail", "100", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"Observer tailing docker logs for {container}", file=sys.stderr)
        for line in proc.stdout:
            try:
                request_tracker.process_line(line)
            except Exception as e:
                print(f"Observer log parse error: {e} ({line[:100]})", file=sys.stderr)
    except FileNotFoundError:
        print("WARNING: docker not found; observer request tracking disabled.",
              file=sys.stderr)
    except Exception as e:
        print(f"WARNING: observer docker log tail error: {e}", file=sys.stderr)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aipc Observer</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;--accent:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;--purple:#bc8cff}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.header,.card{background:var(--surface);border:1px solid var(--border);border-radius:8px}.header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;margin-bottom:16px}.header h1{font-size:18px;margin:0}.meta,.label{color:var(--dim)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{padding:16px}.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 12px}.gpu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.gpu-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px}.gpu-name{font-weight:650;color:var(--accent);margin-bottom:8px}.row{display:flex;justify-content:space-between;gap:16px;padding:4px 0;font-size:13px}.value{font-variant-numeric:tabular-nums;font-weight:600}.bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden}.fill{height:100%;background:var(--accent);border-radius:3px}.fill.mem{background:var(--purple)}.fill.power{background:var(--yellow)}.fill.fan{background:var(--green)}.hot{color:var(--yellow)}.critical{color:var(--red)}.summary{display:flex;gap:24px;flex-wrap:wrap}.summary-item{text-align:center}.summary-value{font-size:28px;font-weight:750;font-variant-numeric:tabular-nums}.summary-label{font-size:11px;text-transform:uppercase;color:var(--dim);letter-spacing:.05em}.full{grid-column:1/-1}.requests{max-height:520px;overflow:auto}.request-row{display:grid;grid-template-columns:88px 150px 60px 74px 74px 60px 1fr;gap:8px;align-items:center;padding:7px 8px;border-bottom:1px solid var(--border);font-size:12px}.request-head{position:sticky;top:0;background:var(--surface);color:var(--dim);font-size:11px;text-transform:uppercase;font-weight:700}.status{border-radius:999px;padding:2px 8px;text-align:center;font-size:10px;text-transform:uppercase;font-weight:700}.completed{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}.processing{background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.3)}
</style>
</head>
<body>
<div class="header"><h1>aipc Observer</h1><div class="meta">Uptime <span id="uptime">0s</span> · <span id="updated">--</span></div></div>
<div class="grid">
<section class="card"><h2>GPU</h2><div id="gpuGrid" class="gpu-grid"></div></section>
<section class="card"><h2>Summary</h2><div class="summary">
<div class="summary-item"><div id="active" class="summary-value">0</div><div class="summary-label">Active</div></div>
<div class="summary-item"><div id="requests" class="summary-value">0</div><div class="summary-label">Completed</div></div>
<div class="summary-item"><div id="gpuTemp" class="summary-value">--</div><div class="summary-label">GPU Temp</div></div>
<div class="summary-item"><div id="memTemp" class="summary-value">--</div><div class="summary-label">VRAM Temp</div></div>
<div class="summary-item"><div id="avgTps" class="summary-value">0</div><div class="summary-label">Avg Gen t/s</div></div>
</div></section>
<section class="card full"><h2>Recent Requests</h2><div id="requestList" class="requests"></div></section>
</div>
<script>
let es;function pct(v,max){return Math.max(0,Math.min(100,(v/max)*100))}
function cls(t){return t>85?'critical':t>75?'hot':''}
function connect(){if(es)es.close();es=new EventSource('/observer/sse');es.onmessage=e=>render(JSON.parse(e.data));es.onerror=()=>{es.close();setTimeout(connect,3000)}}
function render(d){renderGpu(d.gpu_stats||[]);renderSummary(d);renderRequests(d.requests||[]);document.getElementById('uptime').textContent=d.uptime_human;document.getElementById('updated').textContent=new Date().toLocaleTimeString()}
function renderGpu(gpus){document.getElementById('gpuGrid').innerHTML=gpus.map(g=>{let mt=g.mem_temp_c>=0?`${g.mem_temp_c}°C`:'N/A';return `<div class="gpu-card"><div class="gpu-name">GPU ${g.index}: ${g.name}</div>
<div class="row"><span class="label">GPU Temp</span><span class="value ${cls(g.temp_c)}">${g.temp_c}°C</span></div><div class="bar"><div class="fill" style="width:${pct(g.temp_c,100)}%"></div></div>
<div class="row"><span class="label">VRAM Temp</span><span class="value ${cls(g.mem_temp_c)}">${mt}</span></div>
<div class="row"><span class="label">GPU Util</span><span class="value">${g.gpu_util_pct}%</span></div><div class="bar"><div class="fill" style="width:${g.gpu_util_pct}%"></div></div>
<div class="row"><span class="label">VRAM</span><span class="value">${(g.mem_used_mib/1024).toFixed(1)} / ${(g.mem_total_mib/1024).toFixed(1)} GB</span></div><div class="bar"><div class="fill mem" style="width:${g.mem_util_pct}%"></div></div>
<div class="row"><span class="label">Fan</span><span class="value">${g.fan_pct}%</span></div><div class="bar"><div class="fill fan" style="width:${g.fan_pct}%"></div></div>
<div class="row"><span class="label">Power</span><span class="value">${g.power_w} / ${g.power_limit_w} W</span></div><div class="bar"><div class="fill power" style="width:${pct(g.power_w,g.power_limit_w)}%"></div></div></div>`}).join('')}
function renderSummary(d){document.getElementById('active').textContent=d.active_count;document.getElementById('requests').textContent=d.requests.length;if(d.gpu_stats&&d.gpu_stats.length){let g=d.gpu_stats[0];gpuTemp.textContent=`${g.temp_c}°C`;gpuTemp.className='summary-value '+cls(g.temp_c);memTemp.textContent=g.mem_temp_c>=0?`${g.mem_temp_c}°C`:'N/A';memTemp.className='summary-value '+cls(g.mem_temp_c)}let done=d.requests.filter(r=>r.status==='completed'&&r.gen_tps>0);avgTps.textContent=done.length?(done.reduce((s,r)=>s+r.gen_tps,0)/done.length).toFixed(1):'0'}
function renderRequests(reqs){let recent=reqs.slice(-40).reverse();let head='<div class="request-row request-head"><span>Status</span><span>Time</span><span>PT</span><span>P t/s</span><span>G t/s</span><span>GT</span><span>Duration</span></div>';document.getElementById('requestList').innerHTML=head+(recent.length?recent.map(r=>`<div class="request-row"><span class="status ${r.status}">${r.status}</span><span>${r.end_time_str||r.start_time_str||'--'}</span><span>${r.prompt_tokens||0}</span><span>${r.prompt_tps?Number(r.prompt_tps).toFixed(1):'-'}</span><span>${r.gen_tps?Number(r.gen_tps).toFixed(1):'-'}</span><span>${r.completion_tokens||0}</span><span>${r.total_ms?Number(r.total_ms).toFixed(0)+'ms':r.elapsed_ms?Number(r.elapsed_ms).toFixed(0)+'ms':'-'}</span></div>`).join(''):'<div class="request-row"><span class="label">No requests yet</span></div>')}
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
        target=tail_docker_logs,
        args=(container,),
        name="observer-docker-logs",
        daemon=True,
    ).start()
    print(
        f"Observer enabled at /observer (monitor :{monitor_port}, container {container})",
        flush=True,
    )
