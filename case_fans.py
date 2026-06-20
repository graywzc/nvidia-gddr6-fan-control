#!/usr/bin/env python3
"""
case_fans.py — query and manual control of *case* fans.

The GPU fan loop in fan_control.py drives only the NVIDIA card's fans. Case
fans (AIO radiator fans, chassis fans) hang off separate controllers that vary
per host, so they need their own backends:

  - LiquidctlBackend: Corsair Commander Core / Commander Pro and the AIO pumps
    they host (iCUE H100i Capellix, MSI Coreliquid, ...). Driven by shelling out
    to the `liquidctl` CLI — same subprocess pattern as the `gddr6` binary, so
    fan_control.py stays stdlib-only and liquidctl is just a runtime dependency.
  - HwmonBackend: motherboard fan headers exposed via the Linux hwmon sysfs
    (e.g. an Asus B550-E's Nuvoton Super-I/O). Pure /sys reads and writes.

This first cut only *reads* fan speeds and applies a *manual* duty percent — no
temperature-driven curves yet. Backends and the controller are written so a
faked sysfs tree / stubbed liquidctl CLI can drive them under test.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading


DEFAULT_CONFIG_FILE = "/etc/nvidia-gddr6-fan-control/case-fans.json"
DEFAULT_POLL_INTERVAL_S = 5
DEFAULT_LIQUIDCTL_BIN = "liquidctl"
LIQUIDCTL_TIMEOUT_S = 30


def _warn(msg):
    print(f"WARN: {msg}", file=sys.stderr, flush=True)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "device"


def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path):
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _write_int(path, value):
    with open(path, "w") as f:
        f.write(str(int(value)))


def _duty_from_pwm(raw):
    """hwmon pwm is 0..255; report it as a 0..100 percent."""
    if raw is None:
        return None
    return round(max(0, min(255, raw)) / 255 * 100)


def _pwm_from_duty(pct):
    return round(max(0, min(100, pct)) / 100 * 255)


def _clamp_pct(pct):
    if isinstance(pct, bool) or not isinstance(pct, (int, float)):
        raise ValueError("duty_pct must be a number between 0 and 100")
    if pct < 0 or pct > 100:
        raise ValueError("duty_pct must be between 0 and 100")
    return int(round(pct))


# --- Backends -------------------------------------------------------------

class FanBackend:
    """A source of case fans that can be read and (optionally) driven."""

    name = "base"

    def list_fans(self):
        """Return current readings: list of fan dicts.

        Each dict has: id, label, backend, kind ("fan"|"pump"), rpm, duty_pct,
        settable. rpm/duty_pct may be None when the hardware doesn't report them.
        """
        raise NotImplementedError

    def set_duty(self, fan_id, pct):
        """Apply a 0..100 duty to fan_id; return the applied percent."""
        raise NotImplementedError

    def restore(self, fan_id):
        """Return a channel to its pre-existing/automatic control mode."""

    def restore_all(self):
        pass


class HwmonBackend(FanBackend):
    """Motherboard fan headers via /sys/class/hwmon/*/{pwmN,pwmN_enable,fanN_input}."""

    name = "hwmon"

    def __init__(self, root="/sys/class/hwmon"):
        self.root = root
        # fan_id -> channel paths + captured original pwm_enable mode
        self._channels = {}
        self._discover()

    def _discover(self):
        if not os.path.isdir(self.root):
            return
        for entry in sorted(os.listdir(self.root)):
            base = os.path.join(self.root, entry)
            chip = _read_text(os.path.join(base, "name")) or entry
            try:
                files = os.listdir(base)
            except OSError:
                continue
            for fn in sorted(files):
                m = re.fullmatch(r"pwm(\d+)", fn)
                if not m:
                    continue
                n = m.group(1)
                pwm_path = os.path.join(base, fn)
                enable_path = os.path.join(base, f"pwm{n}_enable")
                fan_input = os.path.join(base, f"fan{n}_input")
                fid = f"hwmon:{chip}:pwm{n}"
                self._channels[fid] = {
                    "chip": chip,
                    "channel": f"pwm{n}",
                    "pwm_path": pwm_path,
                    "enable_path": enable_path if os.path.exists(enable_path) else None,
                    "fan_input_path": fan_input if os.path.exists(fan_input) else None,
                    "orig_enable": (
                        _read_int(enable_path) if os.path.exists(enable_path) else None
                    ),
                }

    def channels(self):
        return dict(self._channels)

    def list_fans(self):
        out = []
        for fid, ch in self._channels.items():
            rpm = _read_int(ch["fan_input_path"]) if ch["fan_input_path"] else None
            out.append({
                "id": fid,
                "label": f"{ch['chip']} {ch['channel']}",
                "backend": self.name,
                "kind": "fan",  # hwmon can't tell a pump from a fan
                "rpm": rpm,
                "duty_pct": _duty_from_pwm(_read_int(ch["pwm_path"])),
                "settable": True,
            })
        return out

    def set_duty(self, fan_id, pct):
        ch = self._channels.get(fan_id)
        if ch is None:
            raise KeyError(fan_id)
        pct = _clamp_pct(pct)
        try:
            if ch["enable_path"]:
                _write_int(ch["enable_path"], 1)  # 1 = manual PWM
            raw = _pwm_from_duty(pct)
            _write_int(ch["pwm_path"], raw)
        except OSError as e:
            raise RuntimeError(f"hwmon write failed for {fan_id}: {e}")
        return _duty_from_pwm(raw)

    def restore(self, fan_id):
        ch = self._channels.get(fan_id)
        if not ch or not ch["enable_path"] or ch["orig_enable"] is None:
            return
        try:
            _write_int(ch["enable_path"], ch["orig_enable"])
        except OSError as e:
            _warn(f"hwmon restore failed for {fan_id}: {e}")

    def restore_all(self):
        for fid in self._channels:
            self.restore(fid)


class LiquidctlBackend(FanBackend):
    """Corsair/AIO controllers via the `liquidctl` CLI.

    `run(cmd)` runs an argv list and returns stdout (raising on non-zero exit);
    it's injectable so tests can stub the CLI without the binary or hardware.
    """

    name = "liquidctl"

    def __init__(self, binary=DEFAULT_LIQUIDCTL_BIN, run=None):
        self.binary = binary
        self._run = run or self._default_run
        # fan_id -> {address, bus, match, channel} for targeting `set`
        self._fan_channels = {}
        # last duty we applied, since status rarely reports per-fan duty
        self._last_duty = {}

    def _default_run(self, cmd):
        res = subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            timeout=LIQUIDCTL_TIMEOUT_S,
        )
        return res.stdout

    def available(self):
        return shutil.which(self.binary) is not None

    def _status_json(self):
        try:
            out = self._run([self.binary, "--json", "status"])
        except subprocess.CalledProcessError as e:
            _warn(f"liquidctl status failed: {e.stderr or e}")
            return []
        except (OSError, subprocess.SubprocessError) as e:
            _warn(f"liquidctl status failed: {e}")
            return []
        try:
            data = json.loads(out or "[]")
        except json.JSONDecodeError as e:
            _warn(f"liquidctl status returned invalid JSON: {e}")
            return []
        return data if isinstance(data, list) else []

    def list_fans(self):
        out = []
        channels = {}
        for dev in self._status_json():
            if not isinstance(dev, dict):
                continue
            desc = dev.get("description") or "device"
            slug = _slug(desc)
            target = {"address": dev.get("address"), "bus": dev.get("bus"),
                      "match": desc}
            for sensor in dev.get("status") or []:
                if not isinstance(sensor, dict):
                    continue
                key = str(sensor.get("key") or "")
                unit = str(sensor.get("unit") or "").lower()
                if unit != "rpm":
                    continue
                kl = key.lower()
                m = re.search(r"fan\s*(?:speed\s*)?(\d+)", kl)
                if m:
                    n = m.group(1)
                    fid = f"liquidctl:{slug}:fan{n}"
                    channels[fid] = dict(target, channel=f"fan{n}")
                    out.append({
                        "id": fid,
                        "label": f"{desc} fan {n}",
                        "backend": self.name,
                        "kind": "fan",
                        "rpm": _num(sensor.get("value")),
                        "duty_pct": self._last_duty.get(fid),
                        "settable": True,
                    })
                elif "pump" in kl:
                    fid = f"liquidctl:{slug}:pump"
                    out.append({
                        "id": fid,
                        "label": f"{desc} pump",
                        "backend": self.name,
                        "kind": "pump",
                        "rpm": _num(sensor.get("value")),
                        "duty_pct": self._last_duty.get(fid),
                        "settable": False,  # never auto-throttle a pump
                    })
        self._fan_channels = channels
        return out

    def set_duty(self, fan_id, pct):
        ch = self._fan_channels.get(fan_id)
        if ch is None:
            # mapping may be stale (set before first poll); refresh once.
            self.list_fans()
            ch = self._fan_channels.get(fan_id)
        if ch is None:
            raise KeyError(fan_id)
        pct = _clamp_pct(pct)
        cmd = [self.binary]
        if ch.get("address"):
            cmd += ["--address", ch["address"]]
        elif ch.get("match"):
            cmd += ["--match", ch["match"]]
        cmd += ["set", ch["channel"], "speed", str(pct)]
        try:
            self._run(cmd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"liquidctl set failed for {fan_id}: {e.stderr or e}")
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"liquidctl set failed for {fan_id}: {e}")
        self._last_duty[fan_id] = pct
        return pct

    # liquidctl devices have no firmware auto-curve to fall back to, so we
    # deliberately leave fans running at their last manual duty on shutdown
    # rather than dropping them. restore() is intentionally a no-op.


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


# --- Controller -----------------------------------------------------------

class CaseFanController:
    """Aggregates case-fan backends: polls readings and applies manual duties."""

    def __init__(self, config=None, hwmon_root="/sys/class/hwmon",
                 liquidctl_bin=DEFAULT_LIQUIDCTL_BIN, enable_hwmon=True,
                 enable_liquidctl=True, liquidctl_run=None, dry_run=False):
        self.config = config or {}
        self.dry_run = dry_run
        self.poll_interval_s = self.config.get(
            "poll_interval_s", DEFAULT_POLL_INTERVAL_S)
        self._overlay = self.config.get("fans") or {}
        self.backends = []
        if enable_hwmon:
            hb = HwmonBackend(hwmon_root)
            if hb.channels():
                self.backends.append(hb)
        if enable_liquidctl:
            lb = LiquidctlBackend(liquidctl_bin, run=liquidctl_run)
            if liquidctl_run is not None or lb.available():
                self.backends.append(lb)
            else:
                _warn(
                    f"liquidctl binary '{liquidctl_bin}' not found; "
                    "Corsair/AIO case fans will not be available"
                )

        self._lock = threading.Lock()
        self._by_id = {}            # fan_id -> backend
        self._snapshot = []         # last polled list of fan dicts
        self._duties = {}           # fan_id -> last applied duty
        self._on_update = None
        self._stop = threading.Event()
        self._thread = None

    def has_backends(self):
        return bool(self.backends)

    def _apply_overlay(self, fan):
        override = self._overlay.get(fan["id"]) or {}
        fan = dict(fan)
        if "label" in override:
            fan["label"] = override["label"]
        if "settable" in override:
            fan["settable"] = bool(override["settable"])
        if self._duties.get(fan["id"]) is not None and fan.get("duty_pct") is None:
            fan["duty_pct"] = self._duties[fan["id"]]
        return fan

    def poll_once(self):
        fans = []
        by_id = {}
        for backend in self.backends:
            try:
                readings = backend.list_fans()
            except Exception as e:  # a flaky backend must not kill the poll loop
                _warn(f"case-fan backend {backend.name} read failed: {e}")
                continue
            for fan in readings:
                fan = self._apply_overlay(fan)
                fans.append(fan)
                by_id[fan["id"]] = backend
        with self._lock:
            self._snapshot = fans
            self._by_id = by_id
        if self._on_update:
            self._on_update(fans)
        return fans

    def snapshot(self):
        with self._lock:
            return list(self._snapshot)

    def duties(self):
        with self._lock:
            return dict(self._duties)

    def set_duty(self, fan_id, pct):
        """Validate and apply a manual duty. Raises ValueError/RuntimeError."""
        pct = _clamp_pct(pct)
        with self._lock:
            backend = self._by_id.get(fan_id)
            fan = next((f for f in self._snapshot if f["id"] == fan_id), None)
        if backend is None or fan is None:
            # mapping may be empty before the first poll — refresh and retry.
            self.poll_once()
            with self._lock:
                backend = self._by_id.get(fan_id)
                fan = next(
                    (f for f in self._snapshot if f["id"] == fan_id), None)
        if backend is None or fan is None:
            raise ValueError(f"unknown case fan '{fan_id}'")
        if not fan.get("settable", False):
            raise ValueError(f"case fan '{fan_id}' is not settable")
        if self.dry_run:
            print(f"[dry-run] would set case fan {fan_id} -> {pct}%", flush=True)
            applied = pct
        else:
            applied = backend.set_duty(fan_id, pct)
        with self._lock:
            self._duties[fan_id] = applied
            for f in self._snapshot:
                if f["id"] == fan_id:
                    f["duty_pct"] = applied
        return applied

    def apply_persisted(self, duties):
        """Re-apply duties saved from a previous run (best-effort)."""
        if not duties:
            return
        self.poll_once()
        for fan_id, pct in duties.items():
            try:
                applied = self.set_duty(fan_id, pct)
                print(f"Case fan {fan_id} restored to {applied}%", flush=True)
            except (ValueError, RuntimeError) as e:
                _warn(f"could not restore case fan {fan_id}: {e}")

    def start(self, on_update=None):
        self._on_update = on_update
        self.poll_once()
        self._thread = threading.Thread(
            target=self._loop, name="case-fan-poll", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.poll_interval_s):
            self.poll_once()

    def stop(self):
        self._stop.set()

    def restore_all(self):
        if self.dry_run:
            return
        for backend in self.backends:
            try:
                backend.restore_all()
            except Exception as e:
                _warn(f"case-fan backend {backend.name} restore failed: {e}")


def load_config(path):
    """Load the optional per-host case-fan config. Missing file -> {}."""
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        _warn(f"failed to read case-fan config {path}: {e}")
        return {}
    if not isinstance(raw, dict):
        _warn(f"case-fan config {path} is not an object; ignoring")
        return {}
    return raw
