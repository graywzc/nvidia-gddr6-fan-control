#!/usr/bin/env python3
"""
nvidia-gddr6-fan-control

Drives NVIDIA GPU fans from GDDR6/GDDR6X VRAM junction temperature instead of
core temp. Reads VRAM temp by parsing the output of the `gddr6` binary
(https://github.com/olealgoritme/gddr6) and applies the fan curve via NVML
(libnvidia-ml.so) directly — no X server required.

Requires:
  - root (gddr6 needs /dev/mem; NVML SetFanSpeed needs root)
  - NVIDIA driver new enough to expose nvmlDeviceSetFanSpeed_v2 (~525+)
  - The `gddr6` binary installed in PATH or at --gddr6-bin
"""

import argparse
import atexit
import ctypes
import hmac
import http.server
import json
import os
import re
import signal
import socketserver
import subprocess
import sys
import threading
import time

import aipc_observer

# (vram_temp_C, fan_percent). Linear interpolation between points.
# Clamped to first/last entry outside the range.
DEFAULT_CURVE = [
    (60, 40),
    (80, 55),
    (90, 75),
    (95, 90),
    (100, 100),
]

# Only push a new fan target if it differs from the last applied target
# by at least this many %. Prevents jitter from sub-degree temp wobble.
HYSTERESIS_PCT = 3

# Re-apply the current fan target at least this often even if unchanged,
# in case something else (driver, another tool) reset it.
REAPPLY_INTERVAL_S = 30

VRAM_RE = re.compile(r"(\d+)\s*°C")


# --- NVML bindings via ctypes ---------------------------------------------

class _Utilization(ctypes.Structure):
    """Mirrors nvmlUtilization_t: percent busy over the last sample period."""
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NVML:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self._call("nvmlInit_v2")

    def _call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc != 0:
            msg = self.lib.nvmlErrorString(rc).decode()
            raise RuntimeError(f"{name} failed (rc={rc}): {msg}")

    def get_handle(self, index):
        h = ctypes.c_void_p()
        self._call("nvmlDeviceGetHandleByIndex_v2", index, ctypes.byref(h))
        return h

    def get_name(self, handle):
        buf = ctypes.create_string_buffer(96)
        self._call("nvmlDeviceGetName", handle, buf, 96)
        return buf.value.decode()

    def get_num_fans(self, handle):
        n = ctypes.c_uint()
        self._call("nvmlDeviceGetNumFans", handle, ctypes.byref(n))
        return n.value

    def get_fan_speed(self, handle, fan_index):
        s = ctypes.c_uint()
        self._call("nvmlDeviceGetFanSpeed_v2", handle, fan_index, ctypes.byref(s))
        return s.value

    def get_power_usage_w(self, handle):
        """Current board power draw in watts (NVML reports milliwatts)."""
        mw = ctypes.c_uint()
        self._call("nvmlDeviceGetPowerUsage", handle, ctypes.byref(mw))
        return mw.value / 1000.0

    def get_power_limit_w(self, handle):
        """Current board power-management limit in watts."""
        mw = ctypes.c_uint()
        self._call("nvmlDeviceGetPowerManagementLimit", handle, ctypes.byref(mw))
        return mw.value / 1000.0

    def get_default_power_limit_w(self, handle):
        """Default board power-management limit in watts."""
        mw = ctypes.c_uint()
        self._call("nvmlDeviceGetPowerManagementDefaultLimit", handle, ctypes.byref(mw))
        return mw.value / 1000.0

    def get_power_limit_constraints_w(self, handle):
        """Allowed power-management limit range in watts: (min, max)."""
        min_mw = ctypes.c_uint()
        max_mw = ctypes.c_uint()
        self._call(
            "nvmlDeviceGetPowerManagementLimitConstraints",
            handle,
            ctypes.byref(min_mw),
            ctypes.byref(max_mw),
        )
        return (min_mw.value / 1000.0, max_mw.value / 1000.0)

    def set_power_limit_w(self, handle, watts):
        self._call("nvmlDeviceSetPowerManagementLimit", handle, int(round(watts * 1000)))

    def get_utilization_pct(self, handle):
        """Current GPU core utilization, percent (0..100)."""
        u = _Utilization()
        self._call("nvmlDeviceGetUtilizationRates", handle, ctypes.byref(u))
        return u.gpu

    def set_fan_speed(self, handle, fan_index, percent):
        self._call("nvmlDeviceSetFanSpeed_v2", handle, fan_index, percent)

    def set_default_fan_speed(self, handle, fan_index):
        self._call("nvmlDeviceSetDefaultFanSpeed_v2", handle, fan_index)

    def shutdown(self):
        try:
            self._call("nvmlShutdown")
        except RuntimeError:
            pass


# --- Shared state (updated by control loop, read by HTTP server) ---------

class State:
    """Thread-safe snapshot of the controller's current view."""

    def __init__(self):
        self._lock = threading.Lock()
        self._d = {
            "vram_temp_c": None,
            "power_w": None,        # current board power draw, watts
            "power_limit_w": None,  # current board power cap, watts
            "power_limit_min_w": None,
            "power_limit_max_w": None,
            "power_limit_default_w": None,
            "tdp_w": None,          # default VBIOS/driver board power target
            "gpu_util_pct": None,   # current GPU core utilization, percent
            "fan_pct": None,        # last applied fan target
            "gpu_name": None,
            "num_fans": None,
            "curve": None,
            "power_limit_supported": None,
            "updated_at": 0.0,      # monotonic ts of last successful update
            "wall_time": 0.0,       # unix ts of last successful update
            "dry_run": False,
        }

    def update(self, **kwargs):
        with self._lock:
            self._d.update(kwargs)
            self._d["updated_at"] = time.monotonic()
            self._d["wall_time"] = time.time()

    def snapshot(self):
        with self._lock:
            return dict(self._d)


# --- HTTP server ----------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    state: "State" = None
    token: "str | None" = None
    state_file: "str | None" = None
    apply_power_limit = None

    def log_message(self, fmt, *args):
        # Quieter default logging — control loop has its own prints.
        return

    def _authorized(self):
        if self.token is None:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:], self.token)

    def _write_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        if self.path == "/observer" or self.path.startswith("/observer/"):
            if not aipc_observer.handle_observer_get(self):
                self.send_response(404)
                self.end_headers()
            return
        if self.path == "/status":
            self._write_json(200, self.state.snapshot())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        if self.path.startswith("/observer/"):
            if not aipc_observer.handle_observer_post(self):
                self.send_response(404)
                self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_PUT(self):
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        if self.path not in ("/curve", "/power-limit"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 4096:
            self._write_json(400, {"error": "missing or oversized body"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._write_json(400, {"error": f"invalid JSON: {e}"})
            return
        if self.path == "/curve":
            try:
                new_curve = validate_curve(body)
            except ValueError as e:
                self._write_json(400, {"error": str(e)})
                return
            self.state.update(curve=new_curve)
            if self.state_file:
                save_persisted_state(self.state_file, self.state.snapshot())
            self._write_json(200, {"curve": new_curve})
            return

        try:
            limit_w = validate_power_limit_request(body, self.state.snapshot())
        except ValueError as e:
            self._write_json(400, {"error": str(e)})
            return
        if self.apply_power_limit is None:
            self._write_json(503, {"error": "power limit control unavailable"})
            return
        try:
            applied_w = self.apply_power_limit(limit_w)
        except RuntimeError as e:
            self._write_json(400, {"error": str(e)})
            return
        self.state.update(power_limit_w=applied_w)
        if self.state_file:
            save_persisted_state(self.state_file, self.state.snapshot())
        self._write_json(200, {"power_limit_w": applied_w})


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_http_server(host, port, state, token, state_file, apply_power_limit=None):
    handler = type(
        "BoundHandler",
        (_Handler,),
        {
            "state": state,
            "token": token,
            "state_file": state_file,
            "apply_power_limit": staticmethod(apply_power_limit),
        },
    )
    server = _ThreadedHTTPServer((host, port), handler)
    thread = threading.Thread(
        target=server.serve_forever, name="http-server", daemon=True
    )
    thread.start()
    return server


# --- Curve & parsing ------------------------------------------------------

def validate_curve(value):
    """Validate a curve from JSON: list of [temp, fan_pct] pairs.

    Rules: at least 2 points; temps strictly ascending; temps in 0..150;
    fan_pct in 0..100. Returns a normalized list of [int, int] lists.
    """
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("curve must be a list of at least 2 [temp, fan_pct] pairs")
    out = []
    last_temp = None
    for i, pt in enumerate(value):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"point {i} must be a [temp, fan_pct] pair")
        try:
            t = int(pt[0])
            p = int(pt[1])
        except (TypeError, ValueError):
            raise ValueError(f"point {i} values must be integers")
        if not (0 <= t <= 150):
            raise ValueError(f"point {i} temp {t} out of range 0..150")
        if not (0 <= p <= 100):
            raise ValueError(f"point {i} fan_pct {p} out of range 0..100")
        if last_temp is not None and t <= last_temp:
            raise ValueError(
                f"temps must be strictly ascending; got {t} after {last_temp}"
            )
        last_temp = t
        out.append([t, p])
    return out


def validate_power_limit_w(value, min_w=None, max_w=None):
    """Validate a requested power-management limit in watts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("power limit must be a number of watts or null")
    watts = float(value)
    if watts <= 0:
        raise ValueError("power limit must be greater than 0 watts")
    if min_w is not None and watts < min_w:
        raise ValueError(f"power limit {watts:g}W is below minimum {min_w:g}W")
    if max_w is not None and watts > max_w:
        raise ValueError(f"power limit {watts:g}W is above maximum {max_w:g}W")
    return round(watts, 1)


def validate_power_limit_request(value, state_snapshot):
    """Validate PUT /power-limit JSON.

    Accepts either a raw number/null or {"power_limit_w": number|null}. null
    restores the GPU's default power-management limit.
    """
    if isinstance(value, dict):
        if "power_limit_w" in value:
            raw = value["power_limit_w"]
        elif "limit_w" in value:
            raw = value["limit_w"]
        else:
            raise ValueError("body must include power_limit_w")
    else:
        raw = value
    if raw is None:
        return None
    return validate_power_limit_w(
        raw,
        state_snapshot.get("power_limit_min_w"),
        state_snapshot.get("power_limit_max_w"),
    )


def interp_curve(curve, temp):
    """Linearly interpolate fan % from the curve at the given temp."""
    if temp <= curve[0][0]:
        return curve[0][1]
    if temp >= curve[-1][0]:
        return curve[-1][1]
    for (t0, f0), (t1, f1) in zip(curve, curve[1:]):
        if t0 <= temp <= t1:
            ratio = (temp - t0) / (t1 - t0)
            return round(f0 + ratio * (f1 - f0))
    return curve[-1][1]


def parse_vram_temps(chunk):
    """Pull integers out of a 'VRAM Temps: | XX°C | XX°C |' chunk."""
    return [int(m) for m in VRAM_RE.findall(chunk)]


def _get_tailscale_ipv4():
    """Return the host's first Tailscale IPv4, or None if unavailable."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    ips = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ips[0] if ips else None


# --- State persistence ----------------------------------------------------

DEFAULT_STATE_FILE = "/var/lib/nvidia-gddr6-fan-control/curve.json"


def load_persisted_state(path):
    """Load persisted settings.

    Older installs wrote the curve JSON directly as a list. Newer installs
    write {"curve": [...], "power_limit_w": number|null}.
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARN: failed to read {path}: {e}", file=sys.stderr, flush=True)
        return {}
    if isinstance(raw, list):
        raw = {"curve": raw}
    if not isinstance(raw, dict):
        print(f"WARN: persisted state in {path} is invalid; using defaults",
              file=sys.stderr, flush=True)
        return {}

    out = {}
    if "curve" in raw:
        try:
            out["curve"] = validate_curve(raw["curve"])
        except ValueError as e:
            print(
                f"WARN: persisted curve in {path} is invalid ({e}); using default",
                file=sys.stderr,
                flush=True,
            )
    if "power_limit_w" in raw:
        if raw["power_limit_w"] is None:
            out["power_limit_w"] = None
        else:
            try:
                out["power_limit_w"] = validate_power_limit_w(raw["power_limit_w"])
            except ValueError as e:
                print(
                    f"WARN: persisted power limit in {path} is invalid ({e}); "
                    "leaving unchanged",
                    file=sys.stderr,
                    flush=True,
                )
    return out


def save_persisted_state(path, snapshot):
    """Atomically write persistent settings. Logs and swallows write errors."""
    data = {
        "curve": snapshot.get("curve"),
        "power_limit_w": snapshot.get("power_limit_w"),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError as e:
        print(f"WARN: failed to persist state to {path}: {e}",
              file=sys.stderr, flush=True)


def load_persisted_curve(path):
    """Backward-compatible helper retained for tests or external imports."""
    return load_persisted_state(path).get("curve")


def save_persisted_curve(path, curve):
    """Backward-compatible helper retained for tests or external imports."""
    save_persisted_state(path, {"curve": curve, "power_limit_w": None})


# --- Main loop ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--gddr6-bin",
        default="/usr/local/bin/gddr6",
        help="Path to the gddr6 binary",
    )
    parser.add_argument(
        "--vram-source-index",
        type=int,
        default=0,
        help="Which VRAM temp from gddr6 output to use, if multiple GPUs (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fan decisions but do not actually change fan speed",
    )
    parser.add_argument(
        "--power-limit-w",
        type=float,
        default=None,
        help="Set GPU board power limit in watts at startup. Overrides the "
        "persisted power limit for this run.",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="HTTP API bind address (default: 0.0.0.0; use 'off' to disable). "
        "Overridden if --listen-tailscale is given.",
    )
    parser.add_argument(
        "--listen-tailscale",
        action="store_true",
        help="Bind only to the host's Tailscale IPv4 (resolved via "
        "`tailscale ip -4`). LAN/public interfaces stay unreachable.",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=8765,
        help="HTTP API port (default: 8765)",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Path to a file containing the bearer token for the HTTP API. "
        "If unset, the API has no auth (only safe if bound to a trusted "
        "interface like Tailscale or localhost).",
    )
    parser.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        help=f"Path to the persisted curve file (default: {DEFAULT_STATE_FILE}). "
        "Use 'off' to disable persistence (curve lives in memory only).",
    )
    parser.add_argument(
        "--observer",
        dest="observer",
        action="store_true",
        default=True,
        help="Enable integrated aipc observer dashboard at /observer (default: on)",
    )
    parser.add_argument(
        "--no-observer",
        dest="observer",
        action="store_false",
        help="Disable integrated aipc observer dashboard",
    )
    parser.add_argument(
        "--observer-monitor-port",
        type=int,
        default=aipc_observer.DEFAULT_MONITOR_PORT,
        help=f"llama.cpp frontend port to monitor (default: {aipc_observer.DEFAULT_MONITOR_PORT})",
    )
    parser.add_argument(
        "--observer-container",
        default=aipc_observer.DEFAULT_CONTAINER,
        help="Docker container whose llama.cpp logs are tailed "
        "(default: auto-detect the container publishing the monitor port)",
    )
    parser.add_argument(
        "--observer-repo",
        default=aipc_observer.DEFAULT_MODEL_REPO,
        help="club-3090 checkout whose version/upstream status the observer "
        f"reports (default: {aipc_observer.DEFAULT_MODEL_REPO}; empty to disable)",
    )
    args = parser.parse_args()
    state_file = None if args.state_file.lower() == "off" else args.state_file

    if args.listen_tailscale:
        ts_ip = _get_tailscale_ipv4()
        if ts_ip is None:
            print(
                "ERROR: --listen-tailscale set but `tailscale ip -4` failed. "
                "Is tailscaled running and logged in?",
                file=sys.stderr,
            )
            sys.exit(1)
        args.listen_host = ts_ip

    if not args.dry_run and os.geteuid() != 0:
        print(
            "ERROR: must run as root (gddr6 needs /dev/mem; "
            "NVML SetFanSpeed needs root).",
            file=sys.stderr,
        )
        sys.exit(1)

    # The active curve lives in shared State so HTTP PUT /curve can update it
    # without restarting the controller. The control loop reads it on each tick.
    state = State()
    persisted_state = {}
    if state_file:
        persisted_state = load_persisted_state(state_file)
    initial_curve = persisted_state.get("curve")
    persisted_power_limit_w = persisted_state.get("power_limit_w")
    if args.power_limit_w is not None:
        persisted_power_limit_w = validate_power_limit_w(args.power_limit_w)
    if initial_curve is None:
        initial_curve = validate_curve(DEFAULT_CURVE)
        print(f"Fan curve (default): {initial_curve}", flush=True)
    else:
        print(f"Fan curve (loaded from {state_file}): {initial_curve}", flush=True)
    state.update(
        curve=initial_curve,
        power_limit_w=persisted_power_limit_w,
        dry_run=args.dry_run,
    )

    token = None
    if args.token_file:
        with open(args.token_file, "r") as f:
            token = f.read().strip()
        if not token:
            print(
                f"ERROR: token file {args.token_file} is empty", file=sys.stderr
            )
            sys.exit(1)

    nvml = NVML()
    try:
        handle = nvml.get_handle(args.gpu)
        gpu_name = nvml.get_name(handle)
        num_fans = nvml.get_num_fans(handle)
        power_limit_supported = True
        try:
            min_w, max_w = nvml.get_power_limit_constraints_w(handle)
            default_w = nvml.get_default_power_limit_w(handle)
            current_limit_w = nvml.get_power_limit_w(handle)
            state.update(
                power_limit_w=round(current_limit_w, 1),
                power_limit_min_w=round(min_w, 1),
                power_limit_max_w=round(max_w, 1),
                power_limit_default_w=round(default_w, 1),
                tdp_w=round(default_w, 1),
                power_limit_supported=True,
            )
        except RuntimeError as e:
            power_limit_supported = False
            print(f"WARN: power-limit control unavailable: {e}", file=sys.stderr)
            state.update(power_limit_supported=False)
        print(
            f"Controlling GPU {args.gpu}: {gpu_name} ({num_fans} fan(s)), "
            f"dry_run={args.dry_run}",
            flush=True,
        )
        state.update(gpu_name=gpu_name, num_fans=num_fans)
    except RuntimeError as e:
        print(f"NVML init failed: {e}", file=sys.stderr)
        nvml.shutdown()
        sys.exit(1)

    def apply_power_limit(limit_w):
        if not power_limit_supported:
            raise RuntimeError("power limit control is not supported by this GPU/driver")
        if limit_w is None:
            limit_w = state.snapshot()["power_limit_default_w"]
        snap = state.snapshot()
        limit_w = validate_power_limit_w(
            limit_w,
            snap.get("power_limit_min_w"),
            snap.get("power_limit_max_w"),
        )
        if args.dry_run:
            print(f"[dry-run] would set power limit={limit_w:g}W", flush=True)
            return limit_w
        nvml.set_power_limit_w(handle, limit_w)
        return round(nvml.get_power_limit_w(handle), 1)

    if persisted_power_limit_w is not None:
        try:
            applied_w = apply_power_limit(persisted_power_limit_w)
            state.update(power_limit_w=applied_w)
            print(f"Power limit set to {applied_w:g}W", flush=True)
        except RuntimeError as e:
            print(f"WARN: failed to apply power limit: {e}", file=sys.stderr)

    if args.observer:
        aipc_observer.start_observer(
            monitor_port=args.observer_monitor_port,
            container=args.observer_container,
            model_repo=args.observer_repo,
        )

    if args.listen_host.lower() != "off":
        try:
            server = start_http_server(
                args.listen_host,
                args.listen_port,
                state,
                token,
                state_file,
                apply_power_limit,
            )
            auth_note = "with bearer-token auth" if token else "WITHOUT auth"
            print(
                f"HTTP API listening on {args.listen_host}:{args.listen_port} "
                f"({auth_note})",
                flush=True,
            )
        except OSError as e:
            print(
                f"ERROR: failed to bind HTTP API: {e}", file=sys.stderr
            )
            sys.exit(1)
    else:
        print("HTTP API disabled (--listen-host=off)", flush=True)

    last_applied_pct = None
    last_apply_ts = 0.0
    gddr6_proc = None
    shutting_down = False

    def restore_auto():
        for i in range(num_fans):
            try:
                nvml.set_default_fan_speed(handle, i)
            except RuntimeError as e:
                print(f"WARN: restore fan {i} failed: {e}", file=sys.stderr)

    def cleanup(*_):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print("\nShutting down — restoring auto fan control.", flush=True)
        if gddr6_proc and gddr6_proc.poll() is None:
            gddr6_proc.terminate()
            try:
                gddr6_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                gddr6_proc.kill()
        if not args.dry_run:
            restore_auto()
        nvml.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    # SIGHUP fires when the controlling terminal goes away (e.g. SSH drops).
    # Without this, an interrupted session leaves the fans stuck in manual mode.
    signal.signal(signal.SIGHUP, cleanup)
    # atexit catches normal Python exits and unhandled exceptions, but not
    # SIGKILL or power loss.
    atexit.register(restore_auto if not args.dry_run else lambda: None)

    gddr6_proc = subprocess.Popen(
        [args.gddr6_bin],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    buf = b""
    while True:
        chunk = gddr6_proc.stdout.read(256)
        if not chunk:
            print("gddr6 exited unexpectedly.", file=sys.stderr, flush=True)
            cleanup()
        buf += chunk
        if b"\r" not in buf and b"\n" not in buf:
            continue
        parts = re.split(rb"[\r\n]", buf)
        buf = parts[-1]
        latest = next(
            (p for p in reversed(parts[:-1]) if b"VRAM Temps" in p), None
        )
        if latest is None:
            continue
        temps = parse_vram_temps(latest.decode("utf-8", errors="ignore"))
        if not temps:
            continue
        if args.vram_source_index >= len(temps):
            print(
                f"vram-source-index {args.vram_source_index} out of range "
                f"({len(temps)} temps available)",
                file=sys.stderr,
                flush=True,
            )
            continue
        temp = temps[args.vram_source_index]
        # Read the live curve so remote PUT /curve updates take effect immediately.
        current_curve = state.snapshot()["curve"]
        target_pct = interp_curve(current_curve, temp)
        # Board power draw, if the GPU/driver exposes it. Not fatal if it
        # doesn't (some cards return NotSupported) — just publish None.
        try:
            power_w = round(nvml.get_power_usage_w(handle), 1)
        except RuntimeError:
            power_w = None
        try:
            current_power_limit_w = round(nvml.get_power_limit_w(handle), 1)
        except RuntimeError:
            current_power_limit_w = None
        # GPU core utilization, same NotSupported tolerance as power above.
        try:
            gpu_util = nvml.get_utilization_pct(handle)
        except RuntimeError:
            gpu_util = None
        # Publish to HTTP API readers — temp/power/util every tick, fan_pct after apply.
        state.update(
            vram_temp_c=temp,
            power_w=power_w,
            power_limit_w=current_power_limit_w,
            gpu_util_pct=gpu_util,
        )
        # Hand the real VRAM temps to the observer; nvidia-smi reports
        # temperature.memory as N/A on consumer cards. Indices line up with
        # nvidia-smi's GPU order (both enumerate in PCI order).
        if args.observer:
            aipc_observer.state.set_vram_temps(dict(enumerate(temps)))
        now = time.monotonic()
        needs_update = (
            last_applied_pct is None
            or abs(target_pct - last_applied_pct) >= HYSTERESIS_PCT
            or (now - last_apply_ts) >= REAPPLY_INTERVAL_S
        )
        if needs_update:
            if args.dry_run:
                print(
                    f"[dry-run] VRAM={temp}°C -> would set fan={target_pct}% "
                    f"on {num_fans} fan(s)",
                    flush=True,
                )
                last_applied_pct = target_pct
                last_apply_ts = now
                state.update(fan_pct=target_pct)
            else:
                ok = True
                err = ""
                for i in range(num_fans):
                    try:
                        nvml.set_fan_speed(handle, i, target_pct)
                    except RuntimeError as e:
                        ok = False
                        err = str(e)
                        break
                status = "OK" if ok else f"FAIL ({err})"
                print(
                    f"VRAM={temp}°C  ->  fan={target_pct}%  [{status}]",
                    flush=True,
                )
                if ok:
                    last_applied_pct = target_pct
                    last_apply_ts = now
                    state.update(fan_pct=target_pct)
        else:
            print(
                f"VRAM={temp}°C  (target {target_pct}%, holding "
                f"{last_applied_pct}%)",
                flush=True,
            )


if __name__ == "__main__":
    main()
