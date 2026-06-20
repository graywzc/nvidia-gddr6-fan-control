#!/usr/bin/env python3
"""Tests for case-fan backends and the controller.

hwmon is exercised against a faked /sys tree in a temp dir; liquidctl is
exercised against a stub `run(cmd)` so neither real hardware nor the binary is
needed.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import case_fans
import fan_control


def _make_hwmon_tree(base, chip, channels):
    """channels: {n: {"pwm":int, "enable":int|None, "fan_input":int|None}}."""
    hwmon = os.path.join(base, "hwmon0")
    os.makedirs(hwmon)
    with open(os.path.join(hwmon, "name"), "w") as f:
        f.write(chip + "\n")
    for n, ch in channels.items():
        with open(os.path.join(hwmon, f"pwm{n}"), "w") as f:
            f.write(str(ch["pwm"]))
        if ch.get("enable") is not None:
            with open(os.path.join(hwmon, f"pwm{n}_enable"), "w") as f:
                f.write(str(ch["enable"]))
        if ch.get("fan_input") is not None:
            with open(os.path.join(hwmon, f"fan{n}_input"), "w") as f:
                f.write(str(ch["fan_input"]))
    return hwmon


class HwmonBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        _make_hwmon_tree(self.root, "nct6798", {
            1: {"pwm": 255, "enable": 5, "fan_input": 1200},
            2: {"pwm": 128, "enable": 2, "fan_input": 800},
        })
        self.addCleanup(self.tmp.cleanup)

    def test_discovers_and_reads_channels(self):
        be = case_fans.HwmonBackend(self.root)
        fans = {f["id"]: f for f in be.list_fans()}
        self.assertEqual(set(fans), {"hwmon:nct6798:pwm1", "hwmon:nct6798:pwm2"})
        self.assertEqual(fans["hwmon:nct6798:pwm1"]["rpm"], 1200)
        self.assertEqual(fans["hwmon:nct6798:pwm1"]["duty_pct"], 100)
        self.assertEqual(fans["hwmon:nct6798:pwm2"]["duty_pct"], 50)
        self.assertTrue(fans["hwmon:nct6798:pwm2"]["settable"])

    def test_set_duty_writes_manual_enable_and_pwm(self):
        be = case_fans.HwmonBackend(self.root)
        applied = be.set_duty("hwmon:nct6798:pwm2", 80)
        self.assertEqual(applied, 80)
        hwmon = os.path.join(self.root, "hwmon0")
        with open(os.path.join(hwmon, "pwm2_enable")) as f:
            self.assertEqual(f.read().strip(), "1")  # manual
        with open(os.path.join(hwmon, "pwm2")) as f:
            self.assertEqual(int(f.read().strip()), round(80 / 100 * 255))

    def test_restore_returns_original_enable(self):
        be = case_fans.HwmonBackend(self.root)
        be.set_duty("hwmon:nct6798:pwm2", 80)
        be.restore_all()
        hwmon = os.path.join(self.root, "hwmon0")
        with open(os.path.join(hwmon, "pwm2_enable")) as f:
            self.assertEqual(f.read().strip(), "2")  # captured original mode

    def test_set_unknown_fan_raises(self):
        be = case_fans.HwmonBackend(self.root)
        with self.assertRaises(KeyError):
            be.set_duty("hwmon:nct6798:pwm9", 50)

    def test_missing_root_yields_no_channels(self):
        be = case_fans.HwmonBackend(os.path.join(self.root, "does-not-exist"))
        self.assertEqual(be.list_fans(), [])


# A representative Commander Core status payload (rpm-only, no per-fan duty).
COMMANDER_CORE_STATUS = json.dumps([{
    "bus": "hid",
    "address": "/dev/hidraw3",
    "description": "Corsair Commander Core",
    "status": [
        {"key": "Pump speed", "value": 2400, "unit": "rpm"},
        {"key": "Fan speed 1", "value": 1100, "unit": "rpm"},
        {"key": "Fan speed 2", "value": 0, "unit": "rpm"},
        {"key": "Water temperature", "value": 31.2, "unit": "°C"},
    ],
}])


class FakeLiquidctl:
    """Stub for LiquidctlBackend's run(cmd): records sets, replays status."""

    def __init__(self, status_json):
        self.status_json = status_json
        self.set_calls = []

    def __call__(self, cmd):
        if "status" in cmd:
            return self.status_json
        if "set" in cmd:
            self.set_calls.append(cmd)
            return ""
        return ""


class LiquidctlBackendTests(unittest.TestCase):
    def test_parses_fans_and_pump(self):
        be = case_fans.LiquidctlBackend(run=FakeLiquidctl(COMMANDER_CORE_STATUS))
        fans = {f["id"]: f for f in be.list_fans()}
        self.assertIn("liquidctl:corsair-commander-core:fan1", fans)
        self.assertIn("liquidctl:corsair-commander-core:fan2", fans)
        pump = fans["liquidctl:corsair-commander-core:pump"]
        self.assertEqual(pump["kind"], "pump")
        self.assertFalse(pump["settable"])  # pumps never auto-throttled
        self.assertEqual(fans["liquidctl:corsair-commander-core:fan1"]["rpm"], 1100)

    def test_set_duty_targets_device_by_address(self):
        fake = FakeLiquidctl(COMMANDER_CORE_STATUS)
        be = case_fans.LiquidctlBackend(run=fake)
        be.list_fans()
        applied = be.set_duty("liquidctl:corsair-commander-core:fan1", 65)
        self.assertEqual(applied, 65)
        self.assertEqual(len(fake.set_calls), 1)
        cmd = fake.set_calls[0]
        self.assertIn("--address", cmd)
        self.assertIn("/dev/hidraw3", cmd)
        self.assertIn("fan1", cmd)
        self.assertIn("65", cmd)

    def test_set_duty_refreshes_mapping_if_needed(self):
        # set before any explicit list_fans() still works (lazy refresh).
        be = case_fans.LiquidctlBackend(run=FakeLiquidctl(COMMANDER_CORE_STATUS))
        applied = be.set_duty("liquidctl:corsair-commander-core:fan2", 40)
        self.assertEqual(applied, 40)

    def test_invalid_status_json_degrades(self):
        be = case_fans.LiquidctlBackend(run=lambda cmd: "not json")
        self.assertEqual(be.list_fans(), [])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _make_hwmon_tree(self.tmp.name, "nct6798", {
            2: {"pwm": 128, "enable": 2, "fan_input": 800},
        })
        self.addCleanup(self.tmp.cleanup)

    def _controller(self, config=None, dry_run=False):
        return case_fans.CaseFanController(
            config=config,
            hwmon_root=self.tmp.name,
            enable_liquidctl=True,
            liquidctl_run=FakeLiquidctl(COMMANDER_CORE_STATUS),
            dry_run=dry_run,
        )

    def test_poll_aggregates_both_backends(self):
        c = self._controller()
        fans = {f["id"]: f for f in c.poll_once()}
        self.assertIn("hwmon:nct6798:pwm2", fans)
        self.assertIn("liquidctl:corsair-commander-core:fan1", fans)

    def test_set_duty_records_and_reflects(self):
        c = self._controller()
        c.poll_once()
        applied = c.set_duty("hwmon:nct6798:pwm2", 70)
        self.assertEqual(applied, 70)
        self.assertEqual(c.duties()["hwmon:nct6798:pwm2"], 70)
        snap = {f["id"]: f for f in c.snapshot()}
        self.assertEqual(snap["hwmon:nct6798:pwm2"]["duty_pct"], 70)

    def test_set_non_settable_pump_rejected(self):
        c = self._controller()
        c.poll_once()
        with self.assertRaises(ValueError):
            c.set_duty("liquidctl:corsair-commander-core:pump", 50)

    def test_set_unknown_fan_rejected(self):
        c = self._controller()
        c.poll_once()
        with self.assertRaises(ValueError):
            c.set_duty("hwmon:nope:pwm9", 50)

    def test_overlay_label_and_settable(self):
        c = self._controller(config={"fans": {
            "hwmon:nct6798:pwm2": {"label": "Front intake", "settable": False},
        }})
        fans = {f["id"]: f for f in c.poll_once()}
        self.assertEqual(fans["hwmon:nct6798:pwm2"]["label"], "Front intake")
        with self.assertRaises(ValueError):
            c.set_duty("hwmon:nct6798:pwm2", 50)

    def test_dry_run_does_not_write_hwmon(self):
        c = self._controller(dry_run=True)
        c.poll_once()
        applied = c.set_duty("hwmon:nct6798:pwm2", 90)
        self.assertEqual(applied, 90)
        # pwm2 file untouched (still its original 128).
        with open(os.path.join(self.tmp.name, "hwmon0", "pwm2")) as f:
            self.assertEqual(int(f.read().strip()), 128)

    def test_out_of_range_duty_rejected(self):
        c = self._controller()
        c.poll_once()
        with self.assertRaises(ValueError):
            c.set_duty("hwmon:nct6798:pwm2", 150)


class RequestValidationTests(unittest.TestCase):
    def test_valid_request(self):
        fan_id, duty = fan_control.validate_case_fan_request(
            {"fan": "hwmon:x:pwm2", "duty_pct": 60})
        self.assertEqual((fan_id, duty), ("hwmon:x:pwm2", 60))

    def test_rounds_float_duty(self):
        _, duty = fan_control.validate_case_fan_request(
            {"fan": "x", "duty_pct": 49.6})
        self.assertEqual(duty, 50)

    def test_missing_fan_rejected(self):
        with self.assertRaises(ValueError):
            fan_control.validate_case_fan_request({"duty_pct": 50})

    def test_missing_duty_rejected(self):
        with self.assertRaises(ValueError):
            fan_control.validate_case_fan_request({"fan": "x"})

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            fan_control.validate_case_fan_request({"fan": "x", "duty_pct": 101})

    def test_bool_duty_rejected(self):
        with self.assertRaises(ValueError):
            fan_control.validate_case_fan_request({"fan": "x", "duty_pct": True})


class PersistenceTests(unittest.TestCase):
    def test_round_trip_case_fan_duties(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "curve.json")
            fan_control.save_persisted_state(path, {
                "curve": [[60, 40], [100, 100]],
                "case_fan_duties": {"hwmon:x:pwm2": 70},
            })
            loaded = fan_control.load_persisted_state(path)
            self.assertEqual(loaded["case_fan_duties"], {"hwmon:x:pwm2": 70})

    def test_invalid_duties_dropped(self):
        self.assertEqual(
            fan_control.validate_case_fan_duties(
                {"a": 50, "b": 200, "c": "x", "d": True}),
            {"a": 50},
        )


if __name__ == "__main__":
    unittest.main()
