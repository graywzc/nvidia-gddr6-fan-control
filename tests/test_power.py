#!/usr/bin/env python3
"""Tests for the GPU power-draw plumbing added to fan_control.py.

NVML needs real hardware, so we exercise NVML.get_power_usage_w against a
fake ctypes lib that writes a known milliwatt value into the out-param.
Run with: python3 -m unittest discover -s tests
"""

import ctypes
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fan_control


class FakeLib:
    """Stands in for libnvidia-ml. Each NVML method name maps to a callable
    returning an NVML return code (0 == success)."""

    def __init__(self, power_mw=None, rc=0):
        self._power_mw = power_mw
        self._rc = rc

    def nvmlErrorString(self, rc):
        return b"fake error"

    def nvmlDeviceGetPowerUsage(self, handle, mw_ref):
        if self._rc == 0:
            mw_ref._obj.value = self._power_mw
        return self._rc


def make_nvml(lib):
    """Build an NVML instance without touching real hardware/CDLL."""
    nvml = object.__new__(fan_control.NVML)
    nvml.lib = lib
    return nvml


class GetPowerUsageTests(unittest.TestCase):
    def test_converts_milliwatts_to_watts(self):
        nvml = make_nvml(FakeLib(power_mw=215400))
        self.assertAlmostEqual(nvml.get_power_usage_w(ctypes.c_void_p()), 215.4)

    def test_zero_power(self):
        nvml = make_nvml(FakeLib(power_mw=0))
        self.assertEqual(nvml.get_power_usage_w(ctypes.c_void_p()), 0.0)

    def test_unsupported_raises_runtimeerror(self):
        # Some cards/drivers return NotSupported; main() catches this and
        # publishes power_w=None rather than crashing the control loop.
        nvml = make_nvml(FakeLib(rc=3))
        with self.assertRaises(RuntimeError):
            nvml.get_power_usage_w(ctypes.c_void_p())


class StateTests(unittest.TestCase):
    def test_power_w_defaults_to_none(self):
        self.assertIsNone(fan_control.State().snapshot()["power_w"])

    def test_power_w_round_trips_through_update(self):
        state = fan_control.State()
        state.update(vram_temp_c=88, power_w=215.4)
        snap = state.snapshot()
        self.assertEqual(snap["vram_temp_c"], 88)
        self.assertEqual(snap["power_w"], 215.4)


if __name__ == "__main__":
    unittest.main()
