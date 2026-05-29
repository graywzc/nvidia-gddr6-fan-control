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
            "fan_pct": None,        # last applied fan target
            "gpu_name": None,
            "num_fans": None,
            "curve": None,
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

    def do_GET(self):
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        if self.path == "/status":
            snap = self.state.snapshot()
            body = json.dumps(snap).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_http_server(host, port, state, token):
    handler = type(
        "BoundHandler",
        (_Handler,),
        {"state": state, "token": token},
    )
    server = _ThreadedHTTPServer((host, port), handler)
    thread = threading.Thread(
        target=server.serve_forever, name="http-server", daemon=True
    )
    thread.start()
    return server


# --- Curve & parsing ------------------------------------------------------

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
        "--listen-host",
        default="0.0.0.0",
        help="HTTP API bind address (default: 0.0.0.0; use 'off' to disable)",
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
    args = parser.parse_args()

    if not args.dry_run and os.geteuid() != 0:
        print(
            "ERROR: must run as root (gddr6 needs /dev/mem; "
            "NVML SetFanSpeed needs root).",
            file=sys.stderr,
        )
        sys.exit(1)

    curve = DEFAULT_CURVE
    print(f"Fan curve (VRAM°C -> fan%): {curve}", flush=True)

    state = State()
    state.update(curve=curve, dry_run=args.dry_run)

    token = None
    if args.token_file:
        with open(args.token_file, "r") as f:
            token = f.read().strip()
        if not token:
            print(
                f"ERROR: token file {args.token_file} is empty", file=sys.stderr
            )
            sys.exit(1)

    if args.listen_host.lower() != "off":
        try:
            server = start_http_server(
                args.listen_host, args.listen_port, state, token
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

    nvml = NVML()
    try:
        handle = nvml.get_handle(args.gpu)
        gpu_name = nvml.get_name(handle)
        num_fans = nvml.get_num_fans(handle)
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
        target_pct = interp_curve(curve, temp)
        # Publish to HTTP API readers — temp every tick, fan_pct after apply.
        state.update(vram_temp_c=temp)
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
