#!/usr/bin/env python3
"""Tests for GPU power-limit control plumbing."""

import ctypes
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fan_control


class FakeLib:
    def __init__(self, limit_mw=250000, default_mw=350000, min_mw=100000, max_mw=450000, rc=0):
        self.limit_mw = limit_mw
        self.default_mw = default_mw
        self.min_mw = min_mw
        self.max_mw = max_mw
        self.rc = rc
        self.set_limit_mw = None

    def nvmlErrorString(self, rc):
        return b"fake error"

    def nvmlDeviceGetPowerManagementLimit(self, handle, limit_ref):
        if self.rc == 0:
            limit_ref._obj.value = self.limit_mw
        return self.rc

    def nvmlDeviceGetPowerManagementDefaultLimit(self, handle, limit_ref):
        if self.rc == 0:
            limit_ref._obj.value = self.default_mw
        return self.rc

    def nvmlDeviceGetPowerManagementLimitConstraints(self, handle, min_ref, max_ref):
        if self.rc == 0:
            min_ref._obj.value = self.min_mw
            max_ref._obj.value = self.max_mw
        return self.rc

    def nvmlDeviceSetPowerManagementLimit(self, handle, limit_mw):
        if self.rc == 0:
            self.set_limit_mw = limit_mw
            self.limit_mw = limit_mw
        return self.rc


def make_nvml(lib):
    nvml = object.__new__(fan_control.NVML)
    nvml.lib = lib
    return nvml


class PowerLimitNVMLTests(unittest.TestCase):
    def test_reads_power_limit_in_watts(self):
        nvml = make_nvml(FakeLib(limit_mw=275500))
        self.assertEqual(nvml.get_power_limit_w(ctypes.c_void_p()), 275.5)

    def test_reads_default_power_limit_in_watts(self):
        nvml = make_nvml(FakeLib(default_mw=350000))
        self.assertEqual(nvml.get_default_power_limit_w(ctypes.c_void_p()), 350.0)

    def test_reads_limit_constraints_in_watts(self):
        nvml = make_nvml(FakeLib(min_mw=125000, max_mw=420000))
        self.assertEqual(
            nvml.get_power_limit_constraints_w(ctypes.c_void_p()),
            (125.0, 420.0),
        )

    def test_sets_power_limit_as_milliwatts(self):
        lib = FakeLib()
        nvml = make_nvml(lib)
        nvml.set_power_limit_w(ctypes.c_void_p(), 240.4)
        self.assertEqual(lib.set_limit_mw, 240400)

    def test_unsupported_raises_runtimeerror(self):
        nvml = make_nvml(FakeLib(rc=3))
        with self.assertRaises(RuntimeError):
            nvml.get_power_limit_w(ctypes.c_void_p())


class PowerLimitValidationTests(unittest.TestCase):
    def test_accepts_raw_number(self):
        state = {"power_limit_min_w": 100, "power_limit_max_w": 450}
        self.assertEqual(fan_control.validate_power_limit_request(250, state), 250.0)

    def test_accepts_object(self):
        state = {"power_limit_min_w": 100, "power_limit_max_w": 450}
        self.assertEqual(
            fan_control.validate_power_limit_request({"power_limit_w": 240.5}, state),
            240.5,
        )

    def test_accepts_null_for_default(self):
        self.assertIsNone(fan_control.validate_power_limit_request(None, {}))

    def test_rejects_out_of_range(self):
        state = {"power_limit_min_w": 100, "power_limit_max_w": 450}
        with self.assertRaises(ValueError):
            fan_control.validate_power_limit_request(90, state)
        with self.assertRaises(ValueError):
            fan_control.validate_power_limit_request(500, state)

    def test_state_defaults_include_power_limit_fields(self):
        snap = fan_control.State().snapshot()
        self.assertIsNone(snap["power_limit_w"])
        self.assertIsNone(snap["power_limit_min_w"])
        self.assertIsNone(snap["power_limit_max_w"])
        self.assertIsNone(snap["power_limit_default_w"])
        self.assertIsNone(snap["tdp_w"])


class PowerLimitHTTPTests(unittest.TestCase):
    def test_power_limit_callback_is_not_bound_to_handler_instance(self):
        state = fan_control.State()
        calls = []
        captured = {}

        def apply_power_limit(limit_w):
            calls.append(limit_w)
            return limit_w

        class FakeServer:
            def __init__(self, address, handler):
                captured["address"] = address
                captured["handler"] = handler

            def serve_forever(self):
                return None

        class FakeThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with mock.patch.object(fan_control, "_ThreadedHTTPServer", FakeServer), \
             mock.patch.object(fan_control.threading, "Thread", FakeThread):
            fan_control.start_http_server(
                "127.0.0.1",
                0,
                state,
                None,
                None,
                apply_power_limit,
            )

        handler_instance = object.__new__(captured["handler"])
        self.assertEqual(handler_instance.apply_power_limit(250.0), 250.0)
        self.assertEqual(calls, [250.0])


if __name__ == "__main__":
    unittest.main()
