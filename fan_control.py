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
import case_fans

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

    def get_device_count(self):
        n = ctypes.c_uint()
        self._call("nvmlDeviceGetCount_v2", ctypes.byref(n))
        return n.value

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
            "gpus": [],             # per-GPU view when controlling several cards
            "curve": None,          # primary GPU's curve (back-compat mirror)
            "curves": {},           # per-GPU fan curves, keyed by str(gpu index)
            "power_limits": {},     # per-GPU applied power caps, keyed by str(index)
            "power_limit_supported": None,
            "case_fans": [],        # current case-fan readings (liquidctl/hwmon)
            "case_fan_duties": {},  # manually set case-fan duties, keyed by fan id
            "updated_at": 0.0,      # monotonic ts of last successful update
            "wall_time": 0.0,       # unix ts of last successful update
            "dry_run": False,
        }

    def update(self, **kwargs):
        with self._lock:
            self._d.update(kwargs)
            self._d["updated_at"] = time.monotonic()
            self._d["wall_time"] = time.time()

    def set_gpu_curve(self, index, curve, mirror_primary=False):
        """Set one GPU's curve (read-modify-write under the lock)."""
        with self._lock:
            curves = dict(self._d.get("curves") or {})
            curves[str(index)] = curve
            self._d["curves"] = curves
            if mirror_primary:
                self._d["curve"] = curve
            self._d["updated_at"] = time.monotonic()
            self._d["wall_time"] = time.time()

    def set_all_curves(self, indices, curve):
        """Apply one curve to every controlled GPU (legacy PUT /curve)."""
        with self._lock:
            self._d["curves"] = {str(i): curve for i in indices}
            self._d["curve"] = curve
            self._d["updated_at"] = time.monotonic()
            self._d["wall_time"] = time.time()

    def gpu_curve(self, index):
        with self._lock:
            curves = self._d.get("curves") or {}
            return curves.get(str(index)) or self._d.get("curve")

    def set_case_fans(self, fans):
        """Replace the current case-fan readings (called by the poll thread)."""
        with self._lock:
            self._d["case_fans"] = fans
            self._d["updated_at"] = time.monotonic()
            self._d["wall_time"] = time.time()

    def record_case_fan_duty(self, fan_id, duty_pct):
        """Remember a manually applied case-fan duty (read-modify-write)."""
        with self._lock:
            duties = dict(self._d.get("case_fan_duties") or {})
            duties[str(fan_id)] = duty_pct
            self._d["case_fan_duties"] = duties
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
    set_case_fan_duty = None

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
        if self.path not in ("/curve", "/power-limit", "/case-fans"):
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
                gpu_index, new_curve = validate_curve_request(body)
            except ValueError as e:
                self._write_json(400, {"error": str(e)})
                return
            snap = self.state.snapshot()
            gpu_indices = [g["index"] for g in snap.get("gpus") or []]
            if gpu_index is None:
                if gpu_indices:
                    self.state.set_all_curves(gpu_indices, new_curve)
                else:
                    self.state.update(curve=new_curve)
            else:
                if gpu_indices and gpu_index not in gpu_indices:
                    self._write_json(
                        400, {"error": f"GPU {gpu_index} is not under control"}
                    )
                    return
                mirror_primary = not gpu_indices or gpu_index == gpu_indices[0]
                self.state.set_gpu_curve(gpu_index, new_curve, mirror_primary)
            if self.state_file:
                save_persisted_state(self.state_file, self.state.snapshot())
            payload = {"curve": new_curve}
            if gpu_index is not None:
                payload["gpu_index"] = gpu_index
            self._write_json(200, payload)
            return

        if self.path == "/case-fans":
            try:
                fan_id, duty_pct = validate_case_fan_request(body)
            except ValueError as e:
                self._write_json(400, {"error": str(e)})
                return
            if self.set_case_fan_duty is None:
                self._write_json(503, {"error": "case fan control unavailable"})
                return
            try:
                applied = self.set_case_fan_duty(fan_id, duty_pct)
            except ValueError as e:
                self._write_json(400, {"error": str(e)})
                return
            except RuntimeError as e:
                self._write_json(502, {"error": str(e)})
                return
            self.state.record_case_fan_duty(fan_id, applied)
            if self.state_file:
                save_persisted_state(self.state_file, self.state.snapshot())
            self._write_json(200, {"fan": fan_id, "duty_pct": applied})
            return

        try:
            snap = self.state.snapshot()
            gpu_index = request_gpu_index(body)
            limit_w = validate_power_limit_request(body, snap, gpu_index)
        except ValueError as e:
            self._write_json(400, {"error": str(e)})
            return
        if self.apply_power_limit is None:
            self._write_json(503, {"error": "power limit control unavailable"})
            return
        try:
            if gpu_index is None:
                applied_w = self.apply_power_limit(limit_w)
            else:
                applied_w = self.apply_power_limit(limit_w, gpu_index)
        except RuntimeError as e:
            self._write_json(400, {"error": str(e)})
            return
        self._record_power_limit(gpu_index, applied_w)
        if self.state_file:
            save_persisted_state(self.state_file, self.state.snapshot())
        payload = {"power_limit_w": applied_w}
        if gpu_index is not None:
            payload["gpu_index"] = gpu_index
        self._write_json(200, payload)

    def _record_power_limit(self, gpu_index, applied_w):
        snap = self.state.snapshot()
        gpus = [dict(g) for g in snap.get("gpus") or []]
        primary_index = gpus[0]["index"] if gpus else None
        update = {}
        if gpu_index is not None:
            power_limits = dict(snap.get("power_limits") or {})
            power_limits[str(gpu_index)] = applied_w
            update["power_limits"] = power_limits
            for g in gpus:
                if g["index"] == gpu_index:
                    g["power_limit_w"] = applied_w
            if not gpus or gpu_index == primary_index:
                update["power_limit_w"] = applied_w
        else:
            update["power_limit_w"] = applied_w
        if gpus:
            update["gpus"] = gpus
        self.state.update(**update)


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_http_server(host, port, state, token, state_file, apply_power_limit=None,
                      set_case_fan_duty=None):
    handler = type(
        "BoundHandler",
        (_Handler,),
        {
            "state": state,
            "token": token,
            "state_file": state_file,
            "apply_power_limit": staticmethod(apply_power_limit),
            "set_case_fan_duty": staticmethod(set_case_fan_duty),
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


def request_gpu_index(value):
    """Return an optional GPU index from an object request body."""
    if not isinstance(value, dict):
        return None
    for key in ("gpu_index", "gpu", "index"):
        if key not in value:
            continue
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{key} must be an integer GPU index")
        if raw < 0:
            raise ValueError(f"{key} must be >= 0")
        return raw
    return None


def validate_curve_request(value):
    """Validate PUT /curve JSON.

    Accepts either a raw curve list or {"curve": [...], "gpu_index": N}. A raw
    curve preserves the legacy behavior and applies to every controlled GPU.
    """
    gpu_index = request_gpu_index(value)
    if isinstance(value, dict):
        if "curve" not in value:
            raise ValueError("body must include curve")
        value = value["curve"]
    return gpu_index, validate_curve(value)


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


def validate_case_fan_request(value):
    """Validate PUT /case-fans JSON: {"fan": "<id>", "duty_pct": 0..100}."""
    if not isinstance(value, dict):
        raise ValueError("body must be an object with fan and duty_pct")
    fan_id = value.get("fan", value.get("id"))
    if not isinstance(fan_id, str) or not fan_id.strip():
        raise ValueError("fan must be a non-empty fan id string")
    if "duty_pct" not in value and "duty" not in value:
        raise ValueError("body must include duty_pct")
    raw = value.get("duty_pct", value.get("duty"))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("duty_pct must be a number between 0 and 100")
    if not (0 <= raw <= 100):
        raise ValueError("duty_pct must be between 0 and 100")
    return fan_id.strip(), int(round(raw))


def validate_case_fan_duties(raw):
    """Validate a persisted {fan_id: duty_pct} map; drop invalid entries."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not (0 <= value <= 100):
            continue
        out[str(key)] = int(round(value))
    return out


def _power_limit_bounds(state_snapshot, gpu_index=None):
    if gpu_index is not None:
        for g in state_snapshot.get("gpus") or []:
            if g.get("index") == gpu_index:
                return (g.get("power_limit_min_w"), g.get("power_limit_max_w"))
        raise ValueError(f"GPU {gpu_index} is not under control")
    return (
        state_snapshot.get("power_limit_min_w"),
        state_snapshot.get("power_limit_max_w"),
    )


def validate_power_limit_request(value, state_snapshot, gpu_index=None):
    """Validate PUT /power-limit JSON.

    Accepts either a raw number/null or {"power_limit_w": number|null,
    "gpu_index": N}. null restores the selected GPU's default power-management
    limit.
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
    min_w, max_w = _power_limit_bounds(state_snapshot, gpu_index)
    return validate_power_limit_w(raw, min_w, max_w)


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


def resolve_gpu_targets(gpus_spec, device_count, single_gpu=None,
                        single_vram_index=None):
    """Resolve which GPUs to control into (gpu_index, vram_index) pairs.

    Each controlled GPU is matched to a gddr6 VRAM-temp slot. gddr6 lists VRAM
    temps in PCI/GPU order, so vram_index == gpu_index by default.

    - --gpu N (single_gpu) is the back-compat single-GPU path; it honours an
      explicit --vram-source-index.
    - --gpus accepts "all" (every device) or a comma list like "0,1".
    """
    if single_gpu is not None:
        vidx = single_gpu if single_vram_index is None else single_vram_index
        return [(single_gpu, vidx)]
    spec = (gpus_spec or "all").strip().lower()
    if spec == "all":
        indices = list(range(device_count))
    else:
        indices = [int(tok) for tok in spec.split(",") if tok.strip() != ""]
    return [(i, i) for i in indices]


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
    # Per-GPU overrides (newer multi-GPU installs). Keyed by str(gpu index).
    if isinstance(raw.get("curves"), dict):
        curves = {}
        for key, value in raw["curves"].items():
            try:
                curves[str(key)] = validate_curve(value)
            except ValueError as e:
                print(f"WARN: persisted curve for GPU {key} in {path} is "
                      f"invalid ({e}); skipping", file=sys.stderr, flush=True)
        if curves:
            out["curves"] = curves
    if isinstance(raw.get("power_limits"), dict):
        limits = {}
        for key, value in raw["power_limits"].items():
            if value is None:
                continue
            try:
                limits[str(key)] = validate_power_limit_w(value)
            except ValueError as e:
                print(f"WARN: persisted power limit for GPU {key} in {path} is "
                      f"invalid ({e}); skipping", file=sys.stderr, flush=True)
        if limits:
            out["power_limits"] = limits
    if isinstance(raw.get("case_fan_duties"), dict):
        duties = validate_case_fan_duties(raw["case_fan_duties"])
        if duties:
            out["case_fan_duties"] = duties
    return out


def save_persisted_state(path, snapshot):
    """Atomically write persistent settings. Logs and swallows write errors."""
    data = {
        "curve": snapshot.get("curve"),
        "power_limit_w": snapshot.get("power_limit_w"),
        "curves": snapshot.get("curves") or {},
        "power_limits": snapshot.get("power_limits") or {},
        "case_fan_duties": snapshot.get("case_fan_duties") or {},
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
    parser.add_argument(
        "--gpus",
        default="all",
        help="GPUs to control: 'all' (default) or a comma list like '0,1'. "
        "Each GPU's fans follow its own VRAM temp.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Deprecated single-GPU mode: control only this index "
        "(equivalent to --gpus <N>; honours --vram-source-index).",
    )
    parser.add_argument(
        "--gddr6-bin",
        default="/usr/local/bin/gddr6",
        help="Path to the gddr6 binary",
    )
    parser.add_argument(
        "--vram-source-index",
        type=int,
        default=None,
        help="With --gpu, which gddr6 VRAM temp index to drive it from "
        "(default: the GPU index).",
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
        "--case-fans",
        dest="case_fans",
        action="store_true",
        default=True,
        help="Enable case-fan query/control via liquidctl + hwmon (default: on)",
    )
    parser.add_argument(
        "--no-case-fans",
        dest="case_fans",
        action="store_false",
        help="Disable case-fan query/control entirely",
    )
    parser.add_argument(
        "--case-fan-config",
        default=case_fans.DEFAULT_CONFIG_FILE,
        help="Per-host case-fan config (labels, settable allowlist, poll "
        f"interval). Default: {case_fans.DEFAULT_CONFIG_FILE}; missing file is "
        "fine (pure auto-discovery).",
    )
    parser.add_argument(
        "--liquidctl-bin",
        default=case_fans.DEFAULT_LIQUIDCTL_BIN,
        help="Path to the liquidctl binary for Corsair/AIO case fans "
        f"(default: {case_fans.DEFAULT_LIQUIDCTL_BIN})",
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
    default_curve = persisted_state.get("curve")
    if default_curve is None:
        default_curve = validate_curve(DEFAULT_CURVE)
        print(f"Fan curve (default): {default_curve}", flush=True)
    else:
        print(f"Fan curve (loaded from {state_file}): {default_curve}", flush=True)
    persisted_curves = persisted_state.get("curves") or {}
    persisted_power_limits = persisted_state.get("power_limits") or {}
    persisted_power_limit_w = persisted_state.get("power_limit_w")
    if args.power_limit_w is not None:
        persisted_power_limit_w = validate_power_limit_w(args.power_limit_w)
    persisted_case_fan_duties = persisted_state.get("case_fan_duties") or {}
    state.update(
        curve=default_curve, dry_run=args.dry_run,
        case_fan_duties=dict(persisted_case_fan_duties),
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
        targets = resolve_gpu_targets(
            args.gpus, nvml.get_device_count(),
            single_gpu=args.gpu, single_vram_index=args.vram_source_index,
        )
        if not targets:
            raise RuntimeError("no GPUs selected to control")
        gpus = []
        for gidx, vidx in targets:
            h = nvml.get_handle(gidx)
            g = {
                "index": gidx,
                "vram_index": vidx,
                "handle": h,
                "name": nvml.get_name(h),
                "num_fans": nvml.get_num_fans(h),
                "last_applied_pct": None,
                "last_apply_ts": 0.0,
                "power_supported": True,
                "power_min_w": None,
                "power_max_w": None,
                "power_default_w": None,
            }
            try:
                mn, mx = nvml.get_power_limit_constraints_w(h)
                g["power_min_w"] = round(mn, 1)
                g["power_max_w"] = round(mx, 1)
                g["power_default_w"] = round(nvml.get_default_power_limit_w(h), 1)
            except RuntimeError as e:
                g["power_supported"] = False
                print(f"WARN: GPU {gidx} power-limit control unavailable: {e}",
                      file=sys.stderr)
            gpus.append(g)
        # Seed each GPU's curve: a persisted per-GPU override, else the default.
        state.update(curves={
            str(g["index"]): persisted_curves.get(str(g["index"]), default_curve)
            for g in gpus
        })
        # Top-level fields mirror the primary GPU for back-compat readers.
        primary = gpus[0]
        handle = primary["handle"]
        power_limit_supported = primary["power_supported"]
        if primary["power_supported"]:
            state.update(
                power_limit_min_w=primary["power_min_w"],
                power_limit_max_w=primary["power_max_w"],
                power_limit_default_w=primary["power_default_w"],
                tdp_w=primary["power_default_w"],
                power_limit_supported=True,
            )
        else:
            state.update(power_limit_supported=False)
        summary = ", ".join(
            f"GPU {g['index']} ({g['name']}, {g['num_fans']} fan(s), "
            f"VRAM temp #{g['vram_index']})" for g in gpus
        )
        print(f"Controlling {summary}; dry_run={args.dry_run}", flush=True)
        state.update(gpu_name=primary["name"], num_fans=primary["num_fans"])
    except RuntimeError as e:
        print(f"NVML init failed: {e}", file=sys.stderr)
        nvml.shutdown()
        sys.exit(1)

    def gpu_by_index(idx):
        for g in gpus:
            if g["index"] == idx:
                return g
        raise RuntimeError(f"GPU {idx} is not under control")

    def apply_power_limit(limit_w, gpu_index=None):
        g = gpus[0] if gpu_index is None else gpu_by_index(int(gpu_index))
        if not g["power_supported"]:
            raise RuntimeError(
                f"power limit control is not supported on GPU {g['index']}")
        if limit_w is None:
            limit_w = g["power_default_w"]
        limit_w = validate_power_limit_w(
            limit_w, g["power_min_w"], g["power_max_w"])
        if args.dry_run:
            print(f"[dry-run] would set GPU {g['index']} power limit="
                  f"{limit_w:g}W", flush=True)
            return limit_w
        nvml.set_power_limit_w(g["handle"], limit_w)
        return round(nvml.get_power_limit_w(g["handle"]), 1)

    # Re-apply persisted/CLI power limits: a per-GPU override wins, else the
    # legacy single value applies to every card.
    for g in gpus:
        target_w = persisted_power_limits.get(str(g["index"]),
                                               persisted_power_limit_w)
        if target_w is None or not g["power_supported"]:
            continue
        try:
            applied_w = apply_power_limit(target_w, g["index"])
            snap = state.snapshot()
            power_limits = dict(snap.get("power_limits") or {})
            power_limits[str(g["index"])] = applied_w
            update = {"power_limits": power_limits}
            if g["index"] == primary["index"]:
                update["power_limit_w"] = applied_w
            state.update(**update)
            print(f"GPU {g['index']} power limit set to {applied_w:g}W", flush=True)
        except RuntimeError as e:
            print(f"WARN: GPU {g['index']} failed to apply power limit: {e}",
                  file=sys.stderr)

    # Case-fan controller (liquidctl + hwmon). Optional and isolated: any
    # backend/config failure here must not affect GPU fan control.
    case_fan_controller = None
    set_case_fan_duty = None
    if args.case_fans:
        cf_config = case_fans.load_config(args.case_fan_config)
        case_fan_controller = case_fans.CaseFanController(
            config=cf_config,
            liquidctl_bin=args.liquidctl_bin,
            dry_run=args.dry_run,
        )
        if case_fan_controller.has_backends():
            def set_case_fan_duty(fan_id, duty_pct):
                return case_fan_controller.set_duty(fan_id, duty_pct)

            def publish_case_fans(fans):
                state.set_case_fans(fans)
                if args.observer:
                    # Mirror into the observer so the /observer dashboard can
                    # render the Case Fans card.
                    aipc_observer.state.set_case_fans(fans)

            case_fan_controller.apply_persisted(persisted_case_fan_duties)
            case_fan_controller.start(on_update=publish_case_fans)
            print(
                f"Case-fan control active "
                f"({len(case_fan_controller.backends)} backend(s); "
                f"poll {case_fan_controller.poll_interval_s}s)",
                flush=True,
            )
        else:
            print("No case-fan backends found; case-fan control disabled",
                  flush=True)
            case_fan_controller = None

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
                set_case_fan_duty,
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

    gddr6_proc = None
    shutting_down = False

    def restore_auto():
        for g in gpus:
            for i in range(g["num_fans"]):
                try:
                    nvml.set_default_fan_speed(g["handle"], i)
                except RuntimeError as e:
                    print(f"WARN: restore GPU {g['index']} fan {i} failed: {e}",
                          file=sys.stderr)

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
        if case_fan_controller is not None:
            case_fan_controller.stop()
            case_fan_controller.restore_all()
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
    if case_fan_controller is not None:
        atexit.register(case_fan_controller.restore_all)

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
        # Hand the real VRAM temps to the observer; nvidia-smi reports
        # temperature.memory as N/A on consumer cards. Indices line up with
        # nvidia-smi's GPU order (both enumerate in PCI order).
        if args.observer:
            aipc_observer.state.set_vram_temps(dict(enumerate(temps)))
        # Snapshot once so remote PUT /curve updates take effect immediately;
        # each GPU follows its own curve.
        snap = state.snapshot()
        now = time.monotonic()
        per_gpu = []
        for g in gpus:
            vidx = g["vram_index"]
            if vidx >= len(temps):
                print(
                    f"GPU {g['index']}: VRAM temp #{vidx} out of range "
                    f"({len(temps)} available)",
                    file=sys.stderr, flush=True,
                )
                continue
            temp = temps[vidx]
            curve = (snap.get("curves") or {}).get(str(g["index"])) \
                or snap.get("curve")
            target_pct = interp_curve(curve, temp)
            # Power/util are best-effort; some cards return NotSupported.
            try:
                power_w = round(nvml.get_power_usage_w(g["handle"]), 1)
            except RuntimeError:
                power_w = None
            try:
                limit_w = round(nvml.get_power_limit_w(g["handle"]), 1)
            except RuntimeError:
                limit_w = None
            try:
                gpu_util = nvml.get_utilization_pct(g["handle"])
            except RuntimeError:
                gpu_util = None
            needs_update = (
                g["last_applied_pct"] is None
                or abs(target_pct - g["last_applied_pct"]) >= HYSTERESIS_PCT
                or (now - g["last_apply_ts"]) >= REAPPLY_INTERVAL_S
            )
            if needs_update and args.dry_run:
                print(
                    f"[dry-run] GPU {g['index']} VRAM={temp}°C -> would set "
                    f"fan={target_pct}% on {g['num_fans']} fan(s)",
                    flush=True,
                )
                g["last_applied_pct"] = target_pct
                g["last_apply_ts"] = now
            elif needs_update:
                ok, err = True, ""
                for i in range(g["num_fans"]):
                    try:
                        nvml.set_fan_speed(g["handle"], i, target_pct)
                    except RuntimeError as e:
                        ok, err = False, str(e)
                        break
                status = "OK" if ok else f"FAIL ({err})"
                print(
                    f"GPU {g['index']} VRAM={temp}°C  ->  fan={target_pct}%  "
                    f"[{status}]",
                    flush=True,
                )
                if ok:
                    g["last_applied_pct"] = target_pct
                    g["last_apply_ts"] = now
            else:
                print(
                    f"GPU {g['index']} VRAM={temp}°C  (target {target_pct}%, "
                    f"holding {g['last_applied_pct']}%)",
                    flush=True,
                )
            per_gpu.append({
                "index": g["index"],
                "name": g["name"],
                "vram_temp_c": temp,
                "fan_pct": g["last_applied_pct"],
                "num_fans": g["num_fans"],
                "power_w": power_w,
                "gpu_util_pct": gpu_util,
                "curve": curve,
                "power_limit_w": limit_w,
                "power_limit_min_w": g["power_min_w"],
                "power_limit_max_w": g["power_max_w"],
                "power_limit_default_w": g["power_default_w"],
                "power_limit_supported": g["power_supported"],
            })
        if not per_gpu:
            continue
        # `gpus` carries every controlled card; the legacy top-level fields
        # mirror the primary GPU so existing /status readers keep working.
        head = per_gpu[0]
        state.update(
            gpus=per_gpu,
            vram_temp_c=head["vram_temp_c"],
            fan_pct=head["fan_pct"],
            power_w=head["power_w"],
            gpu_util_pct=head["gpu_util_pct"],
            power_limit_w=head["power_limit_w"],
        )


if __name__ == "__main__":
    main()
