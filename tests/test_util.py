#!/usr/bin/env python3
"""Tests for the GPU utilization plumbing added to fan_control.py.

NVML needs real hardware, so we exercise NVML.get_utilization_pct against a
fake ctypes lib that writes a known struct into the out-param.
Run with: python3 -m unittest discover -s tests
"""

import ctypes
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fan_control


class FakeLib:
    """Stands in for libnvidia-ml. Writes a known gpu/memory utilization
    into the _Utilization struct passed by reference."""

    def __init__(self, gpu=None, memory=0, rc=0):
        self._gpu = gpu
        self._memory = memory
        self._rc = rc

    def nvmlErrorString(self, rc):
        return b"fake error"

    def nvmlDeviceGetUtilizationRates(self, handle, util_ref):
        if self._rc == 0:
            util_ref._obj.gpu = self._gpu
            util_ref._obj.memory = self._memory
        return self._rc


def make_nvml(lib):
    """Build an NVML instance without touching real hardware/CDLL."""
    nvml = object.__new__(fan_control.NVML)
    nvml.lib = lib
    return nvml


class GetUtilizationTests(unittest.TestCase):
    def test_returns_gpu_percent(self):
        nvml = make_nvml(FakeLib(gpu=73, memory=40))
        self.assertEqual(nvml.get_utilization_pct(ctypes.c_void_p()), 73)

    def test_zero_utilization(self):
        nvml = make_nvml(FakeLib(gpu=0))
        self.assertEqual(nvml.get_utilization_pct(ctypes.c_void_p()), 0)

    def test_unsupported_raises_runtimeerror(self):
        # Some cards/drivers return NotSupported; main() catches this and
        # publishes gpu_util_pct=None rather than crashing the control loop.
        nvml = make_nvml(FakeLib(rc=3))
        with self.assertRaises(RuntimeError):
            nvml.get_utilization_pct(ctypes.c_void_p())


class ResolveGpuTargetsTests(unittest.TestCase):
    def test_all_expands_to_every_device(self):
        self.assertEqual(
            fan_control.resolve_gpu_targets("all", 2), [(0, 0), (1, 1)]
        )

    def test_all_on_single_gpu_host_is_unchanged(self):
        self.assertEqual(fan_control.resolve_gpu_targets("all", 1), [(0, 0)])

    def test_explicit_comma_list(self):
        self.assertEqual(
            fan_control.resolve_gpu_targets("0,1", 4), [(0, 0), (1, 1)]
        )

    def test_single_gpu_back_compat(self):
        # --gpu N controls only N; vram index defaults to N.
        self.assertEqual(
            fan_control.resolve_gpu_targets("all", 2, single_gpu=1), [(1, 1)]
        )

    def test_single_gpu_honours_explicit_vram_index(self):
        self.assertEqual(
            fan_control.resolve_gpu_targets(
                "all", 2, single_gpu=1, single_vram_index=0),
            [(1, 0)],
        )


class StateTests(unittest.TestCase):
    def test_gpu_util_pct_defaults_to_none(self):
        self.assertIsNone(fan_control.State().snapshot()["gpu_util_pct"])

    def test_gpus_list_defaults_empty(self):
        self.assertEqual(fan_control.State().snapshot()["gpus"], [])

    def test_gpu_util_pct_round_trips_through_update(self):
        state = fan_control.State()
        state.update(vram_temp_c=88, gpu_util_pct=73)
        snap = state.snapshot()
        self.assertEqual(snap["vram_temp_c"], 88)
        self.assertEqual(snap["gpu_util_pct"], 73)


if __name__ == "__main__":
    unittest.main()
