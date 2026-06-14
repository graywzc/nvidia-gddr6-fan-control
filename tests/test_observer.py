#!/usr/bin/env python3
"""Tests for integrated aipc observer request parsing."""

import json
import sys
import time
import unittest

import aipc_observer


class RequestTrackerTests(unittest.TestCase):
    def setUp(self):
        self.state = aipc_observer.ObserverState()
        self.tracker = aipc_observer.RequestTracker(self.state)

    def test_timing_lines_are_applied_to_matching_task_only(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("I slot launch_slot_: id 1 | task 200 | processing task")

        self.tracker.process_line("I slot print_timing: id 0 | task 100 |")
        self.tracker.process_line(
            "prompt eval time = 100.0 ms / 10 tokens ( 10.0 ms per token, 100.0 tokens per second)"
        )
        self.tracker.process_line(
            "eval time = 200.0 ms / 20 tokens ( 10.0 ms per token, 100.0 tokens per second)"
        )
        self.tracker.process_line("total time = 300.0 ms / 30 tokens")

        self.tracker.process_line("I slot print_timing: id 1 | task 200 |")
        self.tracker.process_line(
            "prompt eval time = 400.0 ms / 40 tokens ( 10.0 ms per token, 100.0 tokens per second)"
        )
        self.tracker.process_line(
            "eval time = 500.0 ms / 50 tokens ( 10.0 ms per token, 100.0 tokens per second)"
        )
        self.tracker.process_line("total time = 900.0 ms / 90 tokens")

        task_100 = self.tracker.active[100]
        task_200 = self.tracker.active[200]

        self.assertEqual(task_100["prompt_tokens"], 10)
        self.assertEqual(task_100["completion_tokens"], 20)
        self.assertEqual(task_100["total_tokens"], 30)
        self.assertEqual(task_200["prompt_tokens"], 40)
        self.assertEqual(task_200["completion_tokens"], 50)
        self.assertEqual(task_200["total_tokens"], 90)

    def test_decoded_line_with_task_id_updates_that_task(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("I slot launch_slot_: id 1 | task 200 | processing task")

        self.tracker.process_line("I slot update_slots: id 1 | task 200 | n_decoded = 12, tg = 8.5 t/s")

        self.assertEqual(self.tracker.active[100]["completion_tokens"], 0)
        self.assertEqual(self.tracker.active[100]["gen_tps"], 0)
        self.assertEqual(self.tracker.active[200]["completion_tokens"], 12)
        self.assertEqual(self.tracker.active[200]["gen_tps"], 8.5)

    def test_prefixed_live_timing_lines_update_throughput(self):
        self.tracker.process_line(
            "8487.39.425.108 I slot launch_slot_: id  0 | task 380639 | processing task, is_child = 0"
        )
        self.tracker.process_line(
            "8487.39.817.791 I slot print_timing: id  0 | task 380639 | prompt eval time =     328.55 ms /   267 tokens (    1.23 ms per token,   812.65 tokens per second)"
        )
        self.tracker.process_line(
            "8487.39.817.793 I slot print_timing: id  0 | task 380639 |        eval time =      62.08 ms /     4 tokens (   15.52 ms per token,    64.43 tokens per second)"
        )
        self.tracker.process_line(
            "8487.39.817.794 I slot print_timing: id  0 | task 380639 |       total time =     390.63 ms /   271 tokens"
        )
        self.tracker.process_line(
            "8487.39.817.829 I slot      release: id  0 | task 380639 | stop processing: n_tokens = 272, truncated = 0"
        )

        request = list(self.state.requests)[0]
        self.assertEqual(request["prompt_tokens"], 267)
        self.assertEqual(request["completion_tokens"], 4)
        self.assertEqual(request["total_tokens"], 272)
        self.assertEqual(request["prompt_tps"], 812.65)
        self.assertEqual(request["gen_tps"], 64.43)

    def test_ambiguous_decoded_line_is_ignored_when_multiple_tasks_are_active(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("I slot launch_slot_: id 1 | task 200 | processing task")

        self.tracker.process_line("n_decoded = 12, tg = 8.5 t/s")

        self.assertEqual(self.tracker.active[100]["completion_tokens"], 0)
        self.assertEqual(self.tracker.active[200]["completion_tokens"], 0)

    def test_completion_records_only_the_released_task(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("I slot launch_slot_: id 1 | task 200 | processing task")
        self.tracker.process_line("I slot print_timing: id 0 | task 100 |")
        self.tracker.process_line("total time = 300.0 ms / 30 tokens")

        self.tracker.process_line("I slot release: id 0 | task 100 | stop processing: n_tokens = 30")

        requests = list(self.state.requests)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["task_id"], 100)
        self.assertEqual(requests[0]["total_tokens"], 30)
        self.assertIn(200, self.tracker.active)


    def test_truncated_release_sets_finish_reason(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 4096, truncated = 1"
        )
        request = list(self.state.requests)[0]
        self.assertTrue(request["truncated"])
        self.assertEqual(request["finish_reason"], "length")

    def test_untruncated_release_finish_reason_is_stop(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 50, truncated = 0"
        )
        request = list(self.state.requests)[0]
        self.assertFalse(request["truncated"])
        self.assertEqual(request["finish_reason"], "stop")

    def test_cancel_marks_active_request_and_counts(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("W srv stop: cancel task, id_task = 100")

        request = list(self.state.requests)[0]
        self.assertEqual(request["status"], "cancelled")
        self.assertEqual(request["finish_reason"], "cancelled")
        self.assertEqual(self.state.cancelled_count, 1)
        self.assertNotIn(100, self.tracker.active)

    def test_cancel_of_unknown_task_only_increments_counter(self):
        self.tracker.process_line("W srv stop: cancel task, id_task = 999")
        self.assertEqual(self.state.cancelled_count, 1)
        self.assertEqual(len(self.state.requests), 0)

    def test_draft_acceptance_is_parsed(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "I slot print_timing: id 0 | task 100 | draft acceptance = 0.20312 (  195 accepted /   960 generated)"
        )
        request = self.tracker.active[100]
        self.assertAlmostEqual(request["draft_acceptance"], 0.20312)
        self.assertEqual(request["draft_accepted"], 195)
        self.assertEqual(request["draft_generated"], 960)

    def test_ttft_is_set_from_prompt_eval_time(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("I slot print_timing: id 0 | task 100 |")
        self.tracker.process_line(
            "prompt eval time = 250.0 ms / 10 tokens ( 25.0 ms per token, 40.0 tokens per second)"
        )
        self.assertEqual(self.tracker.active[100]["ttft_ms"], 250.0)

    def test_finalize_preserves_slot_enrichment(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.state.enrich_active_from_slots([{
            "id_task": 100, "is_processing": True, "prompt_tokens": 8000,
            "processed_tokens": 2000, "cache_tokens": 6000, "decoded": 5,
            "kv_pct": 48.9, "cache_hit_pct": 75.0,
        }])
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 8005, truncated = 0"
        )
        request = list(self.state.requests)[0]
        self.assertEqual(request["cache_hit_pct"], 75.0)
        self.assertEqual(request["kv_pct"], 48.9)
        self.assertEqual(request["cached_tokens"], 6000)
        self.assertEqual(request["recomputed_tokens"], 2000)

    def test_ttft_fallback_survives_finalize_without_timing_lines(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.state.active_requests[100]["start_time"] -= 1.5
        self.state.enrich_active_from_slots([{
            "id_task": 100, "is_processing": True, "prompt_tokens": 100, "decoded": 3,
        }])
        self.assertGreater(self.state.active_requests[100]["ttft_ms"], 0)
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 103, truncated = 0"
        )
        request = list(self.state.requests)[0]
        self.assertGreater(request["ttft_ms"], 1000)

    def test_accurate_ttft_wins_over_fallback_estimate(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.state.enrich_active_from_slots([{
            "id_task": 100, "is_processing": True, "prompt_tokens": 100, "decoded": 3,
        }])
        self.tracker.process_line("I slot print_timing: id 0 | task 100 |")
        self.tracker.process_line(
            "prompt eval time = 250.0 ms / 100 tokens ( 2.5 ms per token, 400.0 tokens per second)"
        )
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 103, truncated = 0"
        )
        request = list(self.state.requests)[0]
        self.assertEqual(request["ttft_ms"], 250.0)

    def test_full_reprocess_marks_cache_defeated(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "W slot update_slots: id 0 | task 100 | forcing full prompt re-processing due to lack of cache data"
        )
        self.assertTrue(self.tracker.active[100]["cache_defeated"])
        self.assertEqual(self.state.cache_defeated_count, 1)

    def test_debug_request_body_groups_next_launched_task(self):
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "Hermes Agent Persona"},
                {"role": "user", "content": "Find MacBook Air M5 deals"},
                {"role": "assistant", "content": "I will check."},
            ],
            "tools": [{"type": "function"}],
            "response_format": {"type": "json_object"},
        }
        self.tracker.process_line(
            "D srv log_server_r: request: " + json.dumps(payload)
        )
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")

        req = self.tracker.active[100]
        self.assertEqual(req["request_group_label"], "Find MacBook Air M5 deals")
        self.assertEqual(req["request_message_count"], 3)
        self.assertEqual(req["request_messages"][1]["role"], "user")
        self.assertEqual(
            req["request_messages"][1]["content"], "Find MacBook Air M5 deals"
        )
        self.assertIn("Hermes Agent Persona", req["request_detail_json"])
        self.assertTrue(req["request_has_tools"])
        self.assertTrue(req["request_has_response_format"])
        self.assertIn("request_group_id", req)
        self.assertEqual(
            self.state.active_requests[100]["request_group_id"],
            req["request_group_id"],
        )

    def test_debug_request_group_survives_completion(self):
        payload = {
            "messages": [
                {"role": "system", "content": "Hermes Agent Persona"},
                {"role": "user", "content": "Summarize this page"},
            ],
        }
        self.tracker.process_line(
            "D srv log_server_r: request: " + json.dumps(payload)
        )
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 50, truncated = 0"
        )

        request = list(self.state.requests)[0]
        self.assertEqual(request["request_group_label"], "Summarize this page")
        self.assertEqual(request["request_message_count"], 2)
        self.assertIn("request_group_id", request)

    def test_debug_response_body_updates_active_request_output(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        payload = {
            "choices": [{
                "message": {"role": "assistant", "content": "The answer is 42."},
                "finish_reason": "stop",
            }]
        }
        self.tracker.process_line(
            "D srv log_server_r: response: " + json.dumps(payload)
        )

        req = self.tracker.active[100]
        self.assertEqual(req["response_output"], "The answer is 42.")
        self.assertEqual(req["response_finish_reason"], "stop")
        self.assertEqual(
            self.state.active_requests[100]["response_output"],
            "The answer is 42.",
        )

    def test_debug_response_body_updates_recent_completed_request(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 50, truncated = 0"
        )
        self.tracker.process_line(
            "D srv log_server_r: response: "
            + json.dumps({"choices": [{"message": {"content": "Done."}}]})
        )

        request = list(self.state.requests)[0]
        self.assertEqual(request["response_output"], "Done.")


class RequestGroupingTests(unittest.TestCase):
    def test_same_initial_conversation_prefix_gets_same_group_id(self):
        base = [
            {"role": "system", "content": "Hermes Agent Persona"},
            {"role": "user", "content": "Find MacBook Air M5 deals"},
        ]
        first = aipc_observer.request_group_metadata({
            "model": "qwen",
            "messages": base + [{"role": "assistant", "content": "Checking."}],
        })
        second = aipc_observer.request_group_metadata({
            "model": "qwen",
            "messages": base + [
                {"role": "assistant", "content": "Checking."},
                {"role": "user", "content": "Anything under 900?"},
            ],
        })

        self.assertEqual(first["request_group_id"], second["request_group_id"])
        self.assertEqual(
            first["request_group_label"], "Find MacBook Air M5 deals"
        )

    def test_different_first_user_message_gets_different_group_id(self):
        first = aipc_observer.request_group_metadata({
            "messages": [
                {"role": "system", "content": "Hermes Agent Persona"},
                {"role": "user", "content": "Find MacBook Air M5 deals"},
            ],
        })
        second = aipc_observer.request_group_metadata({
            "messages": [
                {"role": "system", "content": "Hermes Agent Persona"},
                {"role": "user", "content": "Watch RTX 5090 prices"},
            ],
        })

        self.assertNotEqual(first["request_group_id"], second["request_group_id"])

    def test_message_content_parts_are_used_for_label(self):
        meta = aipc_observer.request_group_metadata({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Evaluate deal"},
                        {"type": "text", "text": "MacBook Air"},
                    ],
                }
            ]
        })

        self.assertEqual(meta["request_group_label"], "Evaluate deal MacBook Air")

    def test_response_detail_extracts_tool_calls(self):
        detail = aipc_observer.response_detail_metadata({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }]
                }
            }]
        })

        self.assertIn("search", detail["response_output"])


class LogSignalTests(unittest.TestCase):
    def setUp(self):
        self.state = aipc_observer.ObserverState()
        self.tracker = aipc_observer.RequestTracker(self.state)

    def test_warm_route_is_attached_to_next_launch_on_that_slot(self):
        self.tracker.process_line(
            "I slot get_available_slot: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.873 (> 0.100 thold), f_keep = 0.950"
        )
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        req = self.tracker.active[100]
        self.assertEqual(req["slot_route"], "warm")
        self.assertAlmostEqual(req["route_similarity"], 0.873)

    def test_lru_route_is_cold(self):
        self.tracker.process_line(
            "I slot get_available_slot: id  1 | task 42 | selected slot by LRU, t_last = 1781130901"
        )
        self.tracker.process_line("I slot launch_slot_: id 1 | task 200 | processing task")
        req = self.tracker.active[200]
        self.assertEqual(req["slot_route"], "cold")
        self.assertIsNone(req["route_similarity"])

    def test_route_is_consumed_only_once(self):
        self.tracker.process_line(
            "I slot get_available_slot: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.873 (> 0.100 thold), f_keep = 0.950"
        )
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("I slot launch_slot_: id 0 | task 101 | processing task")
        self.assertIsNone(self.tracker.active[101]["slot_route"])

    def test_route_for_one_slot_does_not_leak_to_another(self):
        self.tracker.process_line(
            "I slot get_available_slot: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.873 (> 0.100 thold), f_keep = 0.950"
        )
        self.tracker.process_line("I slot launch_slot_: id 1 | task 200 | processing task")
        self.assertIsNone(self.tracker.active[200]["slot_route"])

    def test_context_shift_counts_and_persists_to_completed_row(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "W slot update_slots: id 0 | task 100 | slot context shift, n_keep = 1, n_left = 4094, n_discard = 2047"
        )
        self.tracker.process_line(
            "W slot update_slots: id 0 | task 100 | slot context shift, n_keep = 1, n_left = 4094, n_discard = 2047"
        )
        self.assertEqual(self.state.context_shift_count, 2)
        self.assertEqual(self.state.active_requests[100]["context_shifts"], 2)
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 8192, truncated = 0"
        )
        self.assertEqual(list(self.state.requests)[0]["context_shifts"], 2)

    def test_context_shift_without_known_task_still_counts_globally(self):
        self.tracker.process_line(
            "W slot update_slots: id 0 | task 999 | slot context shift, n_keep = 1, n_left = 4094, n_discard = 2047"
        )
        self.assertEqual(self.state.context_shift_count, 1)

    def test_prefill_progress_updates_live_prompt_tps(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.state.enrich_active_from_slots([{
            "id_task": 100, "is_processing": True, "prompt_tokens": 10000,
            "processed_tokens": 4500, "cache_tokens": 0, "decoded": 0,
            "cache_hit_pct": 0.0,
        }])
        self.tracker.process_line(
            "I slot update_slots: id 0 | task 100 | prompt processing, n_tokens =   4680, progress = 0.45, t =   3.50 s / 1337.14 tokens per second"
        )
        live = self.state.active_requests[100]
        self.assertAlmostEqual(live["prompt_tps"], 1337.14)
        self.assertEqual(live["prefill_pct"], 45)
        self.assertEqual(live["phase"], "prefill")
        # Slot enrichment must survive the merge.
        self.assertEqual(live["cache_hit_pct"], 0.0)

    def test_update_active_request_ignores_unknown_task(self):
        self.state.update_active_request(123, {"prompt_tps": 1.0})
        self.assertNotIn(123, self.state.active_requests)


class InFlightRequestTests(unittest.TestCase):
    def setUp(self):
        self.state = aipc_observer.ObserverState()
        self.tracker = aipc_observer.RequestTracker(self.state)

    def test_launch_registers_active_request(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.assertIn(100, self.state.active_requests)
        self.assertEqual(self.state.active_requests[100]["status"], "processing")

    def test_finalize_removes_active_request(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "I slot release: id 0 | task 100 | stop processing: n_tokens = 30, truncated = 0"
        )
        self.assertNotIn(100, self.state.active_requests)
        self.assertEqual(len(self.state.requests), 1)

    def test_cancel_removes_active_request(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line("W srv stop: cancel task, id_task = 100")
        self.assertNotIn(100, self.state.active_requests)

    def test_enrich_updates_live_decode_progress(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        slots = [{
            "id_task": 100, "is_processing": True,
            "prompt_tokens": 6000, "decoded": 42, "kv_pct": 5.9, "cache_hit_pct": 0.0,
        }]
        self.state.enrich_active_from_slots(slots)
        req = self.state.active_requests[100]
        self.assertEqual(req["completion_tokens"], 42)
        self.assertEqual(req["prompt_tokens"], 6000)
        self.assertEqual(req["kv_pct"], 5.9)

    def test_enrich_drops_ghost_request_when_no_slot_processing(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        # Simulate a missed release: age the request past the prune window.
        self.state.active_requests[100]["start_time"] -= 100
        self.state.enrich_active_from_slots([{"id_task": 100, "is_processing": False}])
        self.assertNotIn(100, self.state.active_requests)

    def test_enrich_reports_prefill_phase_and_progress(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        slots = [{
            "id_task": 100, "is_processing": True, "prompt_tokens": 10000,
            "processed_tokens": 2500, "cache_tokens": 0, "decoded": 0,
        }]
        self.state.enrich_active_from_slots(slots)
        req = self.state.active_requests[100]
        self.assertEqual(req["phase"], "prefill")
        self.assertEqual(req["prefill_pct"], 25)
        self.assertEqual(req["completion_tokens"], 0)

    def test_prefill_progress_counts_cached_tokens(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        slots = [{
            "id_task": 100, "is_processing": True, "prompt_tokens": 10000,
            "processed_tokens": 1000, "cache_tokens": 4000, "decoded": 0,
        }]
        self.state.enrich_active_from_slots(slots)
        self.assertEqual(self.state.active_requests[100]["prefill_pct"], 50)

    def test_enrich_switches_to_generating_once_decoding(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        slots = [{
            "id_task": 100, "is_processing": True, "prompt_tokens": 10000,
            "processed_tokens": 10000, "cache_tokens": 0, "decoded": 7,
        }]
        self.state.enrich_active_from_slots(slots)
        req = self.state.active_requests[100]
        self.assertEqual(req["phase"], "generating")
        self.assertEqual(req["prefill_pct"], 100)
        self.assertEqual(req["completion_tokens"], 7)

    def test_enrich_keeps_brand_new_request_not_yet_in_slots(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        # Just launched; /slots hasn't picked it up yet -> must not be pruned.
        self.state.enrich_active_from_slots([])
        self.assertIn(100, self.state.active_requests)

    def test_prune_inactive_requests_drops_idle_slot_ghost(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.state.active_requests[100]["start_time"] -= 100
        self.state.set_slots([{"id_task": 100, "is_processing": False}])
        self.state.prune_inactive_requests()
        self.assertNotIn(100, self.state.active_requests)

    def test_prune_inactive_requests_keeps_processing_slot(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.state.active_requests[100]["start_time"] -= 100
        self.state.set_slots([{"id_task": 100, "is_processing": True}])
        self.state.prune_inactive_requests()
        self.assertIn(100, self.state.active_requests)


class VramOverlayTests(unittest.TestCase):
    def test_overlay_replaces_na_with_gddr6_temp(self):
        gpus = [
            {"index": 0, "mem_temp_c": -1},
            {"index": 1, "mem_temp_c": -1},
        ]
        aipc_observer.overlay_vram_temps(gpus, {0: 30, 1: 42})
        self.assertEqual(gpus[0]["mem_temp_c"], 30.0)
        self.assertEqual(gpus[1]["mem_temp_c"], 42.0)

    def test_overlay_leaves_gpus_without_a_reading_untouched(self):
        gpus = [{"index": 0, "mem_temp_c": -1}, {"index": 1, "mem_temp_c": -1}]
        aipc_observer.overlay_vram_temps(gpus, {0: 30})
        self.assertEqual(gpus[0]["mem_temp_c"], 30.0)
        self.assertEqual(gpus[1]["mem_temp_c"], -1)

    def test_set_vram_temps_copies_mapping(self):
        state = aipc_observer.ObserverState()
        src = {0: 31}
        state.set_vram_temps(src)
        src[0] = 99
        self.assertEqual(state.vram_temps, {0: 31})


class ModelInfoTests(unittest.TestCase):
    def test_variant_from_compose_path(self):
        path = (
            "/home/u/projects/club-3090/models/qwen3.6-27b/beellama/compose/"
            "single/beellama-q5ks-dflash/dflash.yml"
        )
        self.assertEqual(
            aipc_observer.variant_from_compose_path(path),
            "qwen3.6-27b/beellama/single/beellama-q5ks-dflash/dflash",
        )

    def test_variant_uses_first_of_multiple_config_files(self):
        path = (
            "/r/models/m/eng/compose/single/v/base.yml,"
            "/r/models/m/eng/compose/single/v/override.yml"
        )
        self.assertEqual(
            aipc_observer.variant_from_compose_path(path), "m/eng/single/v/base"
        )

    def test_variant_of_empty_path_is_none(self):
        self.assertIsNone(aipc_observer.variant_from_compose_path(""))

    def test_summarize_command_extracts_notable_flags(self):
        cmd = [
            "--host", "0.0.0.0", "--port", "8080", "-m", "/models/x.gguf",
            "--spec-type", "dflash", "--ctx-size", "102400", "-np", "1",
            "--cache-type-k", "q5_0", "--cache-type-v", "q4_1",
            "--flash-attn", "on", "--cache-ram", "0", "--reasoning", "off",
        ]
        flags = aipc_observer.summarize_command(cmd)
        self.assertEqual(flags["ctx_size"], "102400")
        self.assertEqual(flags["parallel"], "1")
        self.assertEqual(flags["cache_ram_mib"], "0")
        self.assertEqual(flags["kv_type_k"], "q5_0")
        self.assertEqual(flags["kv_type_v"], "q4_1")
        self.assertEqual(flags["spec_type"], "dflash")
        self.assertEqual(flags["flash_attn"], "on")
        self.assertEqual(flags["reasoning"], "off")

    def test_summarize_command_handles_empty(self):
        self.assertEqual(aipc_observer.summarize_command([]), {})
        self.assertEqual(aipc_observer.summarize_command(None), {})

    def test_parse_help_flags_extracts_aliases_and_continuations(self):
        help_text = """
usage: llama-server [options]

  -m, --model FNAME              model path
                                 loaded from disk
  -h,    --help, --usage                  print usage and exit
      --ctx-size N               size of the prompt context
      --metrics                  enable prometheus endpoint
"""
        flags = aipc_observer.parse_help_flags(help_text)
        self.assertEqual(flags["--model"]["description"],
                         "model path loaded from disk")
        self.assertEqual(flags["-m"]["aliases"], ["-m", "--model"])
        self.assertEqual(flags["--usage"]["description"],
                         "print usage and exit")
        self.assertEqual(flags["--ctx-size"]["description"],
                         "size of the prompt context")
        self.assertEqual(flags["--metrics"]["description"],
                         "enable prometheus endpoint")

    def test_command_guide_uses_help_and_marks_unknown_flags(self):
        help_index = aipc_observer.parse_help_flags(
            "  --ctx-size N    size of the prompt context\n"
            "  --metrics       enable prometheus endpoint\n"
        )
        guide = aipc_observer.command_guide(
            ["--ctx-size", "102400", "--metrics", "--fork-only", "x"],
            help_index,
        )
        self.assertEqual(guide[0]["flag"], "--ctx-size")
        self.assertEqual(guide[0]["value"], "102400")
        self.assertTrue(guide[0]["known"])
        self.assertEqual(guide[0]["description"], "size of the prompt context")
        self.assertIsNone(guide[1]["value"])
        self.assertTrue(guide[1]["known"])
        self.assertEqual(guide[2]["flag"], "--fork-only")
        self.assertEqual(guide[2]["value"], "x")
        self.assertFalse(guide[2]["known"])

    def test_command_guide_handles_equals_values(self):
        help_index = aipc_observer.parse_help_flags(
            "  --host HOST    ip address to listen on\n"
        )
        guide = aipc_observer.command_guide(["--host=0.0.0.0"], help_index)
        self.assertEqual(guide[0]["flag"], "--host")
        self.assertEqual(guide[0]["value"], "0.0.0.0")
        self.assertTrue(guide[0]["known"])

    def test_entrypoint_argv_handles_strings_and_lists(self):
        self.assertEqual(
            aipc_observer._entrypoint_argv('/app/server --mode serve'),
            ["/app/server", "--mode", "serve"],
        )
        self.assertEqual(
            aipc_observer._entrypoint_argv(["/app/server"]),
            ["/app/server"],
        )

    def test_snapshot_includes_model_and_repo_info(self):
        state = aipc_observer.ObserverState()
        state.set_model_info({"image": "img", "flags": {"ctx_size": "1"}})
        state.set_repo_info({"head": "abc", "behind": 2})
        snap = state.snapshot()
        self.assertEqual(snap["model_info"]["image"], "img")
        self.assertEqual(snap["repo_info"]["behind"], 2)

    def test_snapshot_includes_catalog_and_diff(self):
        state = aipc_observer.ObserverState()
        state.set_catalog({"variants": {"e/v": {"status": "production"}}})
        state.set_catalog_diff({"added": ["e/w"]})
        snap = state.snapshot()
        self.assertEqual(snap["catalog"]["variants"]["e/v"]["status"], "production")
        self.assertEqual(snap["catalog_diff"]["added"], ["e/w"])

    def test_snapshot_includes_installed_assets(self):
        state = aipc_observer.ObserverState()
        state.mark_assets_installed("eng/prod", {"weight_key": "m1:aq4"})
        snap = state.snapshot()
        self.assertEqual(
            snap["installed_assets"]["eng/prod"]["weight_key"],
            "m1:aq4",
        )


class CatalogDiffTests(unittest.TestCase):
    def _catalog(self, variants, defaults=None):
        return {"variants": variants, "defaults": defaults or {}}

    def test_identical_catalogs_have_no_changes(self):
        cat = self._catalog({"e/v": {"status": "production", "max_ctx": 1000}})
        diff = aipc_observer.diff_catalogs(cat, cat)
        self.assertFalse(aipc_observer.catalog_has_changes(diff))

    def test_added_and_removed_variants(self):
        local = self._catalog({"e/old": {"status": "production"}})
        upstream = self._catalog({"e/new": {"status": "caveats"}})
        diff = aipc_observer.diff_catalogs(local, upstream)
        self.assertEqual(diff["added"], ["e/new"])
        self.assertEqual(diff["removed"], ["e/old"])
        self.assertTrue(aipc_observer.catalog_has_changes(diff))

    def test_status_and_ctx_changes_are_reported_per_field(self):
        local = self._catalog({"e/v": {"status": "caveats", "max_ctx": 1000}})
        upstream = self._catalog({"e/v": {"status": "production", "max_ctx": 2000}})
        diff = aipc_observer.diff_catalogs(local, upstream)
        self.assertEqual(len(diff["changed"]), 1)
        fields = diff["changed"][0]["fields"]
        self.assertEqual(fields["status"], ["caveats", "production"])
        self.assertEqual(fields["max_ctx"], [1000, 2000])

    def test_status_note_change_alone_is_not_a_recommendation_change(self):
        local = self._catalog({"e/v": {"status": "production", "status_note": "a"}})
        upstream = self._catalog({"e/v": {"status": "production", "status_note": "b"}})
        diff = aipc_observer.diff_catalogs(local, upstream)
        self.assertFalse(aipc_observer.catalog_has_changes(diff))

    def test_default_changes_are_reported(self):
        local = self._catalog({}, {"m/e/single": "e/old"})
        upstream = self._catalog({}, {"m/e/single": "e/new"})
        diff = aipc_observer.diff_catalogs(local, upstream)
        self.assertEqual(diff["default_changes"]["m/e/single"], ["e/old", "e/new"])
        self.assertTrue(aipc_observer.catalog_has_changes(diff))


class RepoInfoTests(unittest.TestCase):
    """collect_repo_info against real temporary git repos."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.origin = f"{self.tmp.name}/origin"
        self.clone = f"{self.tmp.name}/clone"
        self._git_in(self.tmp.name, "init", "-q", "-b", "main", self.origin)
        self._commit(self.origin, "first commit")
        self._git_in(self.tmp.name, "clone", "-q", self.origin, self.clone)

    def tearDown(self):
        self.tmp.cleanup()

    def _git_in(self, cwd, *args):
        import subprocess

        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )

    def _commit(self, repo, message):
        self._git_in(repo, "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-q", "--allow-empty", "-m", message)

    def test_up_to_date_clone_reports_zero_behind(self):
        info = aipc_observer.collect_repo_info(self.clone, fetch=True)
        self.assertNotIn("error", info)
        self.assertEqual(info["branch"], "main")
        self.assertEqual(info["behind"], 0)
        self.assertEqual(info["head_subject"], "first commit")

    def test_behind_clone_reports_count_and_subjects(self):
        self._commit(self.origin, "upstream change A")
        self._commit(self.origin, "upstream change B")
        info = aipc_observer.collect_repo_info(self.clone, fetch=True)
        self.assertEqual(info["behind"], 2)
        self.assertEqual(len(info["upstream_commits"]), 2)
        self.assertIn("upstream change B", info["upstream_commits"][0])

    def test_missing_repo_reports_error(self):
        info = aipc_observer.collect_repo_info(f"{self.tmp.name}/nope", fetch=False)
        self.assertIn("error", info)
        self.assertNotIn("head", info)


REGISTRY_V1 = '''
COMPOSE_REGISTRY = {
    "eng/var-a": {
        "model": "m1", "engine": "eng-local", "workload": "fast-chat",
        "status": "caveats", "status_note": "works with limits",
        "max_ctx": 1000, "compose_path": "models/m1/eng/compose/single/q/a.yml",
        "default_port": 8060, "kv_format": "q5_0", "tp": 1,
    },
}
DEFAULTS = {("m1", "eng", "single"): "eng/var-a"}
'''

REGISTRY_V2 = '''
COMPOSE_REGISTRY = {
    "eng/var-a": {
        "model": "m1", "engine": "eng-local", "workload": "fast-chat",
        "status": "production", "status_note": "now validated",
        "max_ctx": 2000, "compose_path": "models/m1/eng/compose/single/q/a.yml",
        "default_port": 8060, "kv_format": "q5_0", "tp": 1,
    },
    "eng/var-b": {
        "model": "m1", "engine": "eng-local", "workload": "tool-heavy",
        "status": "caveats", "status_note": "new variant",
        "max_ctx": 4000, "compose_path": "models/m1/eng/compose/single/q/b.yml",
        "default_port": 8060, "kv_format": "q8_0", "tp": 1,
    },
}
DEFAULTS = {("m1", "eng", "single"): "eng/var-b"}
'''

DUAL_CARD_DOC = '''
| What you're doing | Compose | Max ctx | Narr / Code TPS | VRAM per card | Why |
|---|---|---|---|---|---|
| General default | [`a.yml`](../models/m1/eng/compose/single/q/a.yml) (`eng/var-a`) | **1K** | **10 / 20** | ~1 / 2 GB | Good default for tests. |
'''


class CatalogExtractTests(unittest.TestCase):
    """extract_catalog/refresh_catalog against real temporary git repos."""

    def setUp(self):
        import os
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.origin = f"{self.tmp.name}/origin"
        self.clone = f"{self.tmp.name}/clone"
        self._git_in(self.tmp.name, "init", "-q", "-b", "main", self.origin)
        self._write_registry(self.origin, REGISTRY_V1)
        self._write_dual_card_doc(self.origin)
        self._commit(self.origin, "registry v1")
        self._git_in(self.tmp.name, "clone", "-q", self.origin, self.clone)
        self._write_registry(self.origin, REGISTRY_V2)
        self._write_dual_card_doc(self.origin)
        self._commit(self.origin, "registry v2")

    def tearDown(self):
        self.tmp.cleanup()

    def _git_in(self, cwd, *args):
        import subprocess

        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )

    def _commit(self, repo, message):
        self._git_in(repo, "add", "-A")
        self._git_in(repo, "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-q", "-m", message)

    def _write_registry(self, repo, content):
        import os

        path = os.path.join(repo, aipc_observer.REGISTRY_MODULE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _write_dual_card_doc(self, repo):
        import os

        path = os.path.join(repo, aipc_observer.DUAL_CARD_DOC_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(DUAL_CARD_DOC)

    def test_extracts_variants_and_tuple_defaults_at_head(self):
        cat = aipc_observer.extract_catalog(self.clone, "HEAD")
        self.assertNotIn("error", cat)
        self.assertEqual(cat["variants"]["eng/var-a"]["status"], "caveats")
        self.assertEqual(cat["variants"]["eng/var-a"]["max_ctx"], 1000)
        self.assertEqual(cat["defaults"]["m1/eng/single"], "eng/var-a")
        doc = cat["variants"]["eng/var-a"]["doc"]
        self.assertEqual(doc["max_ctx_doc"], "1K")
        self.assertEqual(doc["tps"], "10 / 20")
        self.assertIn("Good default", doc["why"])

    def test_extracts_upstream_ref_after_fetch(self):
        self._git_in(self.clone, "fetch", "-q")
        cat = aipc_observer.extract_catalog(self.clone, "@{upstream}")
        self.assertNotIn("error", cat)
        self.assertEqual(cat["variants"]["eng/var-a"]["status"], "production")
        self.assertIn("eng/var-b", cat["variants"])

    def test_missing_registry_reports_error(self):
        cat = aipc_observer.extract_catalog(self.tmp.name, "HEAD")
        self.assertIn("error", cat)

    def test_refresh_catalog_sets_state_and_upstream_diff(self):
        obs = aipc_observer.ObserverState()
        info = aipc_observer.collect_repo_info(self.clone, fetch=True)
        self.assertEqual(info["behind"], 1)
        cache = {}
        aipc_observer.refresh_catalog(self.clone, info, cache, observer_state=obs)
        self.assertEqual(obs.catalog["variants"]["eng/var-a"]["status"], "caveats")
        diff = obs.catalog_diff
        self.assertEqual(diff["added"], ["eng/var-b"])
        changed = {c["key"]: c["fields"] for c in diff["changed"]}
        self.assertEqual(changed["eng/var-a"]["status"], ["caveats", "production"])
        self.assertEqual(
            diff["default_changes"]["m1/eng/single"], ["eng/var-a", "eng/var-b"]
        )
        # Both refs are now cached; a second refresh must not re-extract.
        self.assertEqual(len(cache), 2)

    def test_refresh_catalog_clears_diff_when_up_to_date(self):
        self._git_in(self.clone, "pull", "-q")
        obs = aipc_observer.ObserverState()
        obs.set_catalog_diff({"added": ["stale"]})
        info = aipc_observer.collect_repo_info(self.clone, fetch=True)
        aipc_observer.refresh_catalog(self.clone, info, {}, observer_state=obs)
        self.assertEqual(obs.catalog["variants"]["eng/var-a"]["status"], "production")
        self.assertEqual(obs.catalog_diff, {})


class PresetTests(unittest.TestCase):
    def test_appends_missing_flags(self):
        argv = aipc_observer.apply_preset_to_command(
            ["--host", "0.0.0.0"], [("--metrics", None), ("--log-verbosity", "4")]
        )
        self.assertEqual(
            argv, ["--host", "0.0.0.0", "--metrics", "--log-verbosity", "4"]
        )

    def test_replaces_existing_flag_value(self):
        argv = aipc_observer.apply_preset_to_command(
            ["--cache-ram", "0", "--port", "8080"], [("--cache-ram", "8192")]
        )
        self.assertEqual(argv, ["--cache-ram", "8192", "--port", "8080"])

    def test_boolean_flag_is_not_duplicated(self):
        argv = aipc_observer.apply_preset_to_command(
            ["--metrics"], [("--metrics", None)]
        )
        self.assertEqual(argv, ["--metrics"])

    def test_replaces_existing_alias_value(self):
        argv = aipc_observer.apply_preset_to_command(
            ["-lv", "4"], [("--log-verbosity", "5")]
        )
        self.assertEqual(argv, ["-lv", "5"])

    def test_original_command_is_not_mutated(self):
        cmd = ["--cache-ram", "0"]
        aipc_observer.apply_preset_to_command(cmd, [("--cache-ram", "8192")])
        self.assertEqual(cmd, ["--cache-ram", "0"])

    def test_known_presets(self):
        self.assertEqual(
            set(aipc_observer.INSIGHT_PRESETS),
            {"baseline", "insight", "insight-cache", "insight-debug"},
        )
        self.assertEqual(aipc_observer.INSIGHT_PRESETS["baseline"], [])
        self.assertIn(
            ("--log-verbosity", "5"), aipc_observer.INSIGHT_PRESETS["insight-debug"]
        )

    def test_infers_baseline_when_no_managed_flags_are_present(self):
        self.assertEqual(
            aipc_observer.infer_insight_preset(
                ["--host", "0.0.0.0", "--cache-type-k", "q5_0"]
            ),
            "baseline",
        )

    def test_infers_insight_cache_from_live_command(self):
        cmd = [
            "--host", "0.0.0.0", "--metrics", "--props", "--log-verbosity",
            "4", "--log-timestamps", "--cache-ram", "8192",
        ]
        self.assertEqual(aipc_observer.infer_insight_preset(cmd), "insight-cache")

    def test_infers_debug_before_cache(self):
        cmd = [
            "--metrics", "--props", "--log-verbosity", "5",
            "--log-timestamps", "--cache-ram", "8192",
        ]
        self.assertEqual(aipc_observer.infer_insight_preset(cmd), "insight-debug")

    def test_infers_debug_from_live_alias_command(self):
        cmd = [
            "--host", "0.0.0.0", "--cache-ram", "8192", "--metrics",
            "--props", "-lv", "5", "--log-timestamps",
        ]
        self.assertEqual(aipc_observer.infer_insight_preset(cmd), "insight-debug")

    def test_infers_baseline_with_disabled_cache_ram(self):
        cmd = ["--host", "0.0.0.0", "--cache-ram", "0", "--reasoning", "off"]
        self.assertEqual(aipc_observer.infer_insight_preset(cmd), "baseline")

    def test_infers_custom_for_partial_managed_flags(self):
        cmd = ["--metrics", "--log-verbosity", "2"]
        self.assertEqual(aipc_observer.infer_insight_preset(cmd), "custom")

    def test_infers_equals_style_values(self):
        cmd = [
            "--metrics", "--props", "--log-verbosity=4",
            "--log-timestamps", "--cache-ram=8192",
        ]
        self.assertEqual(aipc_observer.infer_insight_preset(cmd), "insight-cache")

    def test_build_compose_override_shape(self):
        ov = aipc_observer.build_compose_override("svc", ["--a", "1"])
        svc = ov["services"]["svc"]
        self.assertEqual(svc["command"], ["--a", "1"])
        self.assertNotIn("image", svc)

    def test_override_always_caps_log_growth(self):
        for ov in (
            aipc_observer.build_compose_override("svc", ["--a"]),
            aipc_observer.build_compose_override("svc", None, image="img:1"),
        ):
            logging = ov["services"]["svc"]["logging"]
            self.assertEqual(logging["driver"], "json-file")
            self.assertEqual(
                logging["options"]["max-size"], aipc_observer.LOG_ROTATE_MAX_SIZE
            )
            self.assertEqual(
                logging["options"]["max-file"], aipc_observer.LOG_ROTATE_MAX_FILE
            )


class RestartGuardTests(unittest.TestCase):
    def test_blocks_when_requests_in_flight(self):
        st = aipc_observer.ObserverState()
        st.active_requests[1] = {"task_id": 1}
        with self.assertRaises(RuntimeError):
            aipc_observer.check_restart_allowed(st)

    def test_force_overrides_in_flight_guard(self):
        st = aipc_observer.ObserverState()
        st.active_requests[1] = {"task_id": 1}
        aipc_observer.check_restart_allowed(st, force=True)

    def test_allows_when_idle(self):
        aipc_observer.check_restart_allowed(aipc_observer.ObserverState())

    def test_prunes_idle_ghost_before_blocking(self):
        st = aipc_observer.ObserverState()
        st.active_requests[1] = {"task_id": 1, "start_time": time.time() - 100}
        st.set_slots([{"id_task": 1, "is_processing": False}])
        aipc_observer.check_restart_allowed(st)
        self.assertEqual(st.active_requests, {})


class FakeRunner:
    def __init__(self, config_json=""):
        self.calls = []
        self.config_json = config_json

    def __call__(self, cmd, env=None, cwd=None, timeout=600, input_text=None):
        self.calls.append({"cmd": list(cmd), "env": dict(env or {}), "cwd": cwd})
        if "config" in cmd:
            return self.config_json
        return ""


MODEL_INFO = {
    "container": "beellama-qwen36-27b",
    "image": "ghcr.io/anbeeld/beellama.cpp:server-cuda-v0.3.0",
    "compose_file": "/repo/models/m/eng/compose/single/q/dflash.yml",
    "service": "svc",
    "working_dir": "/repo/models/m/eng/compose/single/q",
    "command": ["--ctx-size", "102400"],
    "variant": "m/eng/single/q/dflash",
    "project": "q",
    "host_port": "8020",
    "model_dir": "/home/u/models",
    "gpu_ids": "0",
}

CONFIG_JSON = (
    '{"services": {"svc": {"command": '
    '["--host", "0.0.0.0", "--ctx-size", "102400", "--cache-ram", "0"]}}}'
)


class RestartModelTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".yml", delete=False)
        tmp.close()
        self.override_path = tmp.name
        self.runner = FakeRunner(CONFIG_JSON)

    def tearDown(self):
        import os

        os.unlink(self.override_path)

    def _restart(self, preset, model_info=None):
        return aipc_observer.restart_model(
            preset,
            model_info=dict(MODEL_INFO) if model_info is None else model_info,
            runner=self.runner,
            override_path=self.override_path,
        )

    def test_insight_cache_resolves_baseline_then_ups_with_override(self):
        import json

        result = self._restart("insight-cache")
        self.assertTrue(result["restarted"])
        self.assertEqual(len(self.runner.calls), 2)
        config_call, up_call = self.runner.calls
        self.assertIn("config", config_call["cmd"])
        self.assertEqual(
            up_call["cmd"],
            ["docker", "compose", "-f", MODEL_INFO["compose_file"],
             "-f", self.override_path, "up", "-d", "--remove-orphans"],
        )
        with open(self.override_path) as f:
            override = json.load(f)
        svc = override["services"]["svc"]
        argv = svc["command"]
        self.assertIn("--metrics", argv)
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "8192")
        # Built on the compose baseline, not the running command.
        self.assertIn("--host", argv)
        # The running image is pinned so a restart can't switch images.
        self.assertEqual(svc["image"], MODEL_INFO["image"])
        self.assertEqual(svc["logging"]["driver"], "json-file")
        self.assertEqual(
            svc["logging"]["options"]["max-size"],
            aipc_observer.LOG_ROTATE_MAX_SIZE,
        )
        self.assertEqual(
            svc["logging"]["options"]["max-file"],
            aipc_observer.LOG_ROTATE_MAX_FILE,
        )

    def test_compose_env_reproduces_boot_substitutions(self):
        self._restart("insight")
        env = self.runner.calls[-1]["env"]
        self.assertEqual(env["PORT"], "8020")
        self.assertEqual(env["ESTATE_PORT"], "8020")
        self.assertEqual(env["MODEL_DIR"], "/home/u/models")
        self.assertEqual(env["ESTATE_CONTAINER"], "beellama-qwen36-27b")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_baseline_skips_config_but_still_pins_image(self):
        import json

        self._restart("baseline")
        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(
            self.runner.calls[0]["cmd"],
            ["docker", "compose", "-f", MODEL_INFO["compose_file"],
             "-f", self.override_path, "up", "-d", "--remove-orphans"],
        )
        with open(self.override_path) as f:
            svc = json.load(f)["services"]["svc"]
        self.assertEqual(svc["image"], MODEL_INFO["image"])
        self.assertEqual(svc["logging"]["driver"], "json-file")
        self.assertEqual(
            svc["logging"]["options"]["max-size"],
            aipc_observer.LOG_ROTATE_MAX_SIZE,
        )
        self.assertEqual(
            svc["logging"]["options"]["max-file"],
            aipc_observer.LOG_ROTATE_MAX_FILE,
        )
        self.assertNotIn("command", svc)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            self._restart("turbo")
        self.assertEqual(self.runner.calls, [])

    def test_incomplete_model_info_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self._restart("insight", model_info={"command": ["--x"]})
        self.assertEqual(self.runner.calls, [])

    def test_multi_config_file_uses_first(self):
        mi = dict(MODEL_INFO)
        mi["compose_file"] = "/repo/a.yml,/repo/b.yml"
        self._restart("baseline", model_info=mi)
        self.assertEqual(self.runner.calls[0]["cmd"][3], "/repo/a.yml")

    def test_build_compose_override_image_only(self):
        ov = aipc_observer.build_compose_override("svc", None, image="img:1")
        svc = ov["services"]["svc"]
        self.assertEqual(svc["image"], "img:1")
        self.assertNotIn("command", svc)

    def test_vllm_insight_injects_request_logging_flags(self):
        import json

        mi = dict(MODEL_INFO)
        mi["image"] = "vllm/vllm-openai:v0.22.0"
        mi["compose_file"] = "/repo/models/m/vllm/compose/dual/fp8.yml"
        mi["working_dir"] = "/repo/models/m/vllm/compose/dual"
        result = self._restart("insight", model_info=mi)
        # vLLM never drops capabilities (it bypasses the llama.cpp --help probe).
        self.assertEqual(result["dropped_capabilities"], [])
        with open(self.override_path) as f:
            argv = json.load(f)["services"]["svc"]["command"]
        self.assertIn("--enable-log-requests", argv)
        self.assertIn("--enable-log-outputs", argv)
        # llama.cpp insight flags must NOT leak onto a vLLM command.
        self.assertNotIn("--metrics", argv)


class PresetResolveTests(unittest.TestCase):
    """Preset capabilities resolve to the flags each build advertises."""

    def setUp(self):
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".yml", delete=False)
        tmp.close()
        self.override_path = tmp.name
        self.runner = FakeRunner(CONFIG_JSON)

    def tearDown(self):
        import os

        os.unlink(self.override_path)

    def _restart(self, preset, help_flags):
        # Stand in for inspect_container_help: a build advertising exactly the
        # given flags, or an error (unknown -> fail open) when help_flags is None.
        def fake_help(name, entrypoint):
            if help_flags is None:
                return {"error": "help unavailable"}
            return {"flags": {f: {} for f in help_flags}}

        mi = dict(MODEL_INFO)
        mi["entrypoint"] = ["/app/llama-server"]
        return aipc_observer.restart_model(
            preset, model_info=mi, runner=self.runner,
            override_path=self.override_path, help_getter=fake_help,
        )

    def _override_argv(self):
        import json

        with open(self.override_path) as f:
            return json.load(f)["services"]["svc"]["command"]

    def test_translates_to_the_flag_the_build_advertises(self):
        # ik-llama: --verbosity (not --log-verbosity), no --props/--log-* flags.
        result = self._restart(
            "insight-debug",
            help_flags={"--metrics", "--cache-ram", "--verbosity", "--host"},
        )
        argv = self._override_argv()
        self.assertIn("--metrics", argv)
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "8192")
        # trace_logging resolves to --verbosity 5, not --log-verbosity.
        self.assertEqual(argv[argv.index("--verbosity") + 1], "5")
        self.assertNotIn("--log-verbosity", argv)
        # props and timestamps have no supported flag on this build.
        self.assertEqual(sorted(result["dropped_capabilities"]),
                         ["props", "timestamps"])

    def test_drops_capabilities_with_no_supported_flag(self):
        result = self._restart(
            "insight-cache",
            help_flags={"--metrics", "--props", "--cache-ram", "--host"},
        )
        argv = self._override_argv()
        self.assertIn("--metrics", argv)
        self.assertIn("--props", argv)
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "8192")
        self.assertNotIn("--log-verbosity", argv)
        self.assertNotIn("--verbosity", argv)
        self.assertEqual(sorted(result["dropped_capabilities"]),
                         ["timestamps", "verbose_logging"])

    def test_unknown_help_falls_back_to_default_flags(self):
        result = self._restart("insight-cache", help_flags=None)
        argv = self._override_argv()
        self.assertIn("--log-verbosity", argv)
        self.assertIn("--log-timestamps", argv)
        self.assertEqual(result["dropped_capabilities"], [])

    def test_supported_build_keeps_default_flags(self):
        result = self._restart(
            "insight",
            help_flags={"--metrics", "--props", "--log-verbosity",
                        "--log-timestamps", "--host"},
        )
        self.assertEqual(result["dropped_capabilities"], [])
        self.assertIn("--log-verbosity", self._override_argv())

    def test_resolve_preset_is_capability_aware(self):
        # Unit-level: ik-llama-style support set.
        supported = {"--metrics", "--cache-ram", "--verbosity"}
        tweaks, dropped = aipc_observer.resolve_preset("insight-debug", supported)
        self.assertIn(("--verbosity", "5"), tweaks)
        self.assertIn(("--metrics", None), tweaks)
        self.assertNotIn(("--log-verbosity", "5"), tweaks)
        self.assertEqual(sorted(dropped), ["props", "timestamps"])
        # Unknown support -> default (mainline) flags, nothing dropped.
        tweaks, dropped = aipc_observer.resolve_preset("insight-debug", None)
        self.assertIn(("--log-verbosity", "5"), tweaks)
        self.assertEqual(dropped, [])


class StopModelTests(unittest.TestCase):
    def test_stop_prefers_club3090_switch_down_when_available(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as repo:
            scripts = os.path.join(repo, "scripts")
            os.mkdir(scripts)
            open(os.path.join(scripts, "switch.sh"), "w").close()
            runner = FakeRunner()
            result = aipc_observer.stop_model(repo=repo, runner=runner)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["detail"], "club-3090 switch.sh --down ran")
        self.assertEqual(runner.calls[0]["cmd"][-3:], ["bash", "scripts/switch.sh", "--down"])

    def test_stop_uses_compose_project_and_all_config_files(self):
        runner = FakeRunner()
        mi = dict(MODEL_INFO)
        mi["compose_file"] = "/repo/base.yml,/tmp/override.yml"
        result = aipc_observer.stop_model(model_info=mi, repo="", runner=runner)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["container"], MODEL_INFO["container"])
        self.assertEqual(
            runner.calls[0]["cmd"],
            [
                "docker", "compose", "--project-name", "q",
                "-f", "/repo/base.yml", "-f", "/tmp/override.yml", "down",
            ],
        )
        self.assertEqual(runner.calls[0]["cwd"], MODEL_INFO["working_dir"])
        env = runner.calls[0]["env"]
        self.assertEqual(env["PORT"], "8020")
        self.assertEqual(env["ESTATE_CONTAINER"], MODEL_INFO["container"])

    def test_stop_is_noop_without_running_container(self):
        runner = FakeRunner()
        result = aipc_observer.stop_model(model_info={}, repo="", runner=runner)
        self.assertFalse(result["stopped"])
        self.assertEqual(runner.calls, [])


class UpdateRepoTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.origin = f"{self.tmp.name}/origin"
        self.clone = f"{self.tmp.name}/clone"
        self._git_in(self.tmp.name, "init", "-q", "-b", "main", self.origin)
        self._commit(self.origin, "first commit")
        self._git_in(self.tmp.name, "clone", "-q", self.origin, self.clone)

    def tearDown(self):
        self.tmp.cleanup()

    def _git_in(self, cwd, *args):
        import subprocess

        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )

    def _commit(self, repo, message):
        self._git_in(repo, "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-q", "--allow-empty", "-m", message)

    def test_pulls_when_behind(self):
        self._commit(self.origin, "upstream change")
        result = aipc_observer.update_repo(self.clone)
        self.assertTrue(result["updated"])
        self.assertNotEqual(result["from"], result["to"])
        self.assertEqual(len(result["commits"]), 1)
        info = aipc_observer.collect_repo_info(self.clone, fetch=False)
        self.assertEqual(info["behind"], 0)

    def test_noop_when_up_to_date(self):
        result = aipc_observer.update_repo(self.clone)
        self.assertFalse(result["updated"])

    def test_diverged_branch_fails_instead_of_merging(self):
        self._commit(self.clone, "local change")
        self._commit(self.origin, "upstream change")
        with self.assertRaises(RuntimeError):
            aipc_observer.update_repo(self.clone)


class RepoGitHardeningTests(unittest.TestCase):
    """A stalled git must fail fast, not wedge the control lock forever."""

    def test_timeout_raises_quickly_despite_surviving_child(self):
        import time

        # Stand in for the runuser -> git -> ssh tree: a shell that backgrounds
        # a grandchild inheriting the stdout pipe, then blocks. Killing only the
        # direct child would leave the grandchild holding the pipe so
        # communicate() hangs past the timeout; the process-group kill must reap
        # the whole tree.
        hang = ["sh", "-c", "sleep 30 & sleep 30"]
        orig = aipc_observer._repo_owner_cmd
        aipc_observer._repo_owner_cmd = lambda repo, cmd: hang
        try:
            start = time.monotonic()
            with self.assertRaises(RuntimeError) as ctx:
                aipc_observer.repo_git("/tmp", "fetch", timeout=1)
            elapsed = time.monotonic() - start
        finally:
            aipc_observer._repo_owner_cmd = orig
        self.assertIn("timed out", str(ctx.exception))
        # Must not block anywhere near the 30s the command would otherwise run.
        self.assertLess(elapsed, 8.0)


PROMETHEUS_SAMPLE = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 16185
llamacpp:prompt_seconds_total 13.636
llamacpp:tokens_predicted_total 394
llamacpp:tokens_predicted_seconds_total 5.75
llamacpp:n_decode_total 125
llamacpp:n_tokens_max 5016
llamacpp:prompt_tokens_seconds 1186.93
llamacpp:predicted_tokens_seconds 68.5217
llamacpp:requests_processing 1
llamacpp:requests_deferred 3
llamacpp:n_busy_slots_per_decode 1
"""


class MetricsTests(unittest.TestCase):
    def test_parse_skips_comments_and_parses_values(self):
        values = aipc_observer.parse_prometheus(PROMETHEUS_SAMPLE)
        self.assertEqual(values["llamacpp:prompt_tokens_total"], 16185.0)
        self.assertEqual(values["llamacpp:requests_deferred"], 3.0)
        self.assertNotIn("# HELP", str(values.keys()))

    def test_parse_tolerates_labels_and_junk(self):
        values = aipc_observer.parse_prometheus(
            'metric_with{label="x"} 7\nnot a metric\nbad_value abc\n'
        )
        self.assertEqual(values, {"metric_with": 7.0})

    def test_parse_handles_none_and_empty(self):
        self.assertEqual(aipc_observer.parse_prometheus(None), {})
        self.assertEqual(aipc_observer.parse_prometheus(""), {})

    def test_summarize_maps_queue_and_throughput(self):
        values = aipc_observer.parse_prometheus(PROMETHEUS_SAMPLE)
        m = aipc_observer.summarize_metrics(values)
        self.assertTrue(m["available"])
        self.assertEqual(m["queued"], 3)
        self.assertEqual(m["processing"], 1)
        self.assertIsInstance(m["queued"], int)
        self.assertAlmostEqual(m["gen_tps_avg"], 68.5217)
        self.assertEqual(m["prompt_tokens_total"], 16185)
        self.assertEqual(m["decode_calls_total"], 125)

    def test_summarize_omits_absent_metrics(self):
        m = aipc_observer.summarize_metrics({"llamacpp:requests_deferred": 0})
        self.assertEqual(m["queued"], 0)
        self.assertNotIn("kv_cache_usage_ratio", m)

    def test_snapshot_includes_metrics(self):
        state = aipc_observer.ObserverState()
        state.set_metrics({"available": True, "queued": 2})
        self.assertEqual(state.snapshot()["metrics"]["queued"], 2)


VLLM_METRICS_SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="qwen3.6-27b"} 2.0
vllm:num_requests_waiting{engine="0",model_name="qwen3.6-27b"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3.6-27b"} 0.031
vllm:prompt_tokens_total{engine="0",model_name="qwen3.6-27b"} 12947409.0
vllm:generation_tokens_total{engine="0",model_name="qwen3.6-27b"} 81664.0
vllm:prompt_tokens_cached_total{engine="0",model_name="qwen3.6-27b"} 900000.0
vllm:prefix_cache_queries_total{engine="0",model_name="qwen3.6-27b"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="qwen3.6-27b"} 826.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="qwen3.6-27b"} 3000.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="qwen3.6-27b"} 2400.0
vllm:num_preemptions_total{engine="0",model_name="qwen3.6-27b"} 4.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="qwen3.6-27b"} 600.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="qwen3.6-27b"} 58.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="qwen3.6-27b"} 0.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="qwen3.6-27b"} 12.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="qwen3.6-27b"} 100.0
vllm:request_time_per_output_token_seconds_sum{engine="0",model_name="qwen3.6-27b"} 5.0
vllm:request_time_per_output_token_seconds_count{engine="0",model_name="qwen3.6-27b"} 100.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="qwen3.6-27b"} 300.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="qwen3.6-27b"} 100.0
"""


class EngineDetectTests(unittest.TestCase):
    def test_detects_vllm_from_compose_path(self):
        info = {"compose_file": "models/qwen3.6-27b/vllm/compose/dual/fp8.yml"}
        self.assertEqual(aipc_observer.infer_engine(info), "vllm")

    def test_detects_vllm_from_image(self):
        self.assertEqual(
            aipc_observer.infer_engine({"image": "vllm/vllm-openai:v0.22.0"}),
            "vllm",
        )

    def test_detects_llamacpp_family(self):
        self.assertEqual(
            aipc_observer.infer_engine({"variant": "ik-llama/prism-pro-dq-dual"}),
            "llamacpp",
        )
        self.assertEqual(
            aipc_observer.infer_engine(
                {"compose_file": "models/m/llamacpp/compose/dual.yml"}),
            "llamacpp",
        )

    def test_unknown_when_unpopulated(self):
        self.assertIsNone(aipc_observer.infer_engine({}))
        self.assertIsNone(aipc_observer.infer_engine(None))


class VllmMetricsTests(unittest.TestCase):
    def test_maps_core_gauges_and_engine_marker(self):
        m = aipc_observer.summarize_vllm_metrics(VLLM_METRICS_SAMPLE)
        self.assertEqual(m["engine"], "vllm")
        self.assertTrue(m["available"])
        self.assertEqual(m["processing"], 2)
        self.assertEqual(m["queued"], 1)
        self.assertIsInstance(m["queued"], int)
        self.assertAlmostEqual(m["kv_cache_usage_ratio"], 0.031)
        self.assertEqual(m["prompt_tokens_total"], 12947409)
        self.assertEqual(m["gen_tokens_total"], 81664)
        self.assertEqual(m["preemptions_total"], 4)

    def test_derives_rates_and_latency_averages(self):
        m = aipc_observer.summarize_vllm_metrics(VLLM_METRICS_SAMPLE)
        self.assertAlmostEqual(m["prefix_cache_hit_pct"], 82.6)
        self.assertAlmostEqual(m["spec_accept_pct"], 80.0)
        self.assertAlmostEqual(m["avg_ttft_ms"], 120.0)
        self.assertAlmostEqual(m["avg_tpot_ms"], 50.0)
        self.assertAlmostEqual(m["avg_e2e_ms"], 3000.0)

    def test_success_total_breaks_down_by_reason(self):
        m = aipc_observer.summarize_vllm_metrics(VLLM_METRICS_SAMPLE)
        self.assertEqual(m["requests_total"], 658)
        self.assertEqual(m["success_by_reason"]["stop"], 600)
        self.assertEqual(m["success_by_reason"]["length"], 58)

    def test_throughput_from_counter_deltas(self):
        prev = {
            "scraped_at": aipc_observer.time.time() - 10,
            "prompt_tokens_total": 12947409 - 1000,
            "gen_tokens_total": 81664 - 200,
        }
        m = aipc_observer.summarize_vllm_metrics(VLLM_METRICS_SAMPLE, prev)
        self.assertGreater(m["prompt_tps_avg"], 0)
        self.assertGreater(m["gen_tps_avg"], 0)
        # No prior scrape → no throughput (avoids a bogus first-poll spike).
        first = aipc_observer.summarize_vllm_metrics(VLLM_METRICS_SAMPLE)
        self.assertNotIn("gen_tps_avg", first)

    def test_unavailable_when_empty(self):
        m = aipc_observer.summarize_vllm_metrics("")
        self.assertEqual(m["engine"], "vllm")
        self.assertNotIn("processing", m)

    def test_timeline_sample_is_compact(self):
        m = aipc_observer.summarize_vllm_metrics(VLLM_METRICS_SAMPLE)
        s = aipc_observer.vllm_timeline_sample(m)
        self.assertEqual(s["running"], 2)
        self.assertEqual(s["waiting"], 1)
        self.assertAlmostEqual(s["kv"], 3.1)
        self.assertEqual(s["spec"], m["spec_accept_pct"])
        self.assertIn("t", s)


class VllmLogTrackerTests(unittest.TestCase):
    RECV = ("(APIServer pid=1) INFO 06-14 17:27:48 [logger.py:39] Received "
            "request chatcmpl-abc: params: SamplingParams(temperature=0.7, "
            "max_tokens=512, top_p=0.9), lora_request: None.")
    RESP = ("(APIServer pid=1) INFO 06-14 17:27:50 [logger.py:71] Generated "
            "response chatcmpl-abc (streaming complete): output: 'hi there', "
            "output_token_ids: [1, 2, 3, 4], finish_reason: stop")

    def setUp(self):
        self.state = aipc_observer.ObserverState()
        self.tracker = aipc_observer.VllmLogTracker(self.state)

    def test_arrival_creates_active_row_with_params(self):
        self.tracker.process_line(self.RECV)
        snap = self.state.snapshot()
        self.assertEqual(len(snap["active_requests"]), 1)
        row = snap["active_requests"][0]
        self.assertEqual(row["task_id"], "chatcmpl-abc")
        self.assertEqual(row["status"], "processing")
        self.assertEqual(row["max_tokens"], 512)
        self.assertAlmostEqual(row["temperature"], 0.7)

    def test_response_finalizes_row(self):
        self.tracker.process_line(self.RECV)
        self.tracker.process_line(self.RESP)
        snap = self.state.snapshot()
        self.assertEqual(len(snap["active_requests"]), 0)
        self.assertEqual(len(snap["requests"]), 1)
        row = snap["requests"][0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["completion_tokens"], 4)
        self.assertEqual(row["finish_reason"], "stop")
        self.assertGreaterEqual(row["total_ms"], 0)

    def test_streaming_delta_lines_are_ignored(self):
        delta = self.RESP.replace("(streaming complete)", "(streaming delta)")
        self.tracker.process_line(self.RECV)
        self.tracker.process_line(delta)
        # Still in-flight: a delta must not finalize the row.
        self.assertEqual(len(self.state.snapshot()["active_requests"]), 1)
        self.assertEqual(len(self.state.snapshot()["requests"]), 0)

    def test_abort_marks_cancelled(self):
        self.tracker.process_line(self.RECV)
        self.tracker.process_line(self.RESP.replace("finish_reason: stop",
                                                    "finish_reason: abort"))
        self.assertEqual(self.state.snapshot()["requests"][0]["status"],
                         "cancelled")

    def test_response_without_arrival_still_logs(self):
        # Observer started mid-stream: a completion with no prior arrival.
        self.tracker.process_line(self.RESP)
        reqs = self.state.snapshot()["requests"]
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["completion_tokens"], 4)


class TraceLogTests(unittest.TestCase):
    """Lines that only appear at -lv 4 (the insight presets)."""

    def setUp(self):
        self.state = aipc_observer.ObserverState()
        self.tracker = aipc_observer.RequestTracker(self.state)

    def test_completion_post_status_is_counted(self):
        self.tracker.process_line(
            "0.06.1 I srv log_server_r: done request: "
            "POST /v1/chat/completions 172.18.0.1 200"
        )
        self.tracker.process_line(
            "0.07.1 I srv log_server_r: done request: "
            "POST /v1/chat/completions 172.18.0.1 500"
        )
        self.assertEqual(self.state.http_statuses, {"200": 1, "500": 1})
        snap = self.state.snapshot()
        self.assertEqual(snap["http_statuses"]["500"], 1)

    def test_observer_get_polling_is_ignored(self):
        self.tracker.process_line(
            "0.06.1 I srv log_server_r: done request: GET /slots 172.18.0.1 200"
        )
        self.assertEqual(self.state.http_statuses, {})

    def test_budget_hit_counts_only_non_natural_ends(self):
        self.tracker.process_line("I reasoning-budget: deactivated (natural end)")
        self.assertEqual(self.state.budget_hit_count, 0)
        self.tracker.process_line(
            "I reasoning-budget: deactivated (budget exhausted)"
        )
        self.assertEqual(self.state.budget_hit_count, 1)
        self.assertEqual(self.state.snapshot()["budget_hit_count"], 1)

    def test_adaptive_dm_and_graphs_reused_attach_to_task(self):
        self.tracker.process_line(
            "I slot launch_slot_: id 0 | task 100 | processing task"
        )
        self.tracker.process_line(
            "I slot print_timing: id 0 | task 100 | adaptive dm: fringe=0.44 n_max=24"
        )
        self.tracker.process_line(
            "I slot print_timing: id 0 | task 100 | graphs reused = 117"
        )
        req = self.tracker.active[100]
        self.assertEqual(req["dm_controller"], "fringe")
        self.assertAlmostEqual(req["dm_rate"], 0.44)
        self.assertEqual(req["draft_n_max"], 24)
        self.assertEqual(req["graphs_reused"], 117)

    def test_new_prompt_sets_early_prompt_tokens_on_live_row(self):
        self.tracker.process_line(
            "I slot launch_slot_: id 0 | task 100 | processing task"
        )
        self.tracker.process_line(
            "I slot update_slots: id 0 | task 100 | new prompt, n_ctx_slot = 102400,"
            " n_keep = 0, task.n_tokens = 5016"
        )
        self.assertEqual(self.tracker.active[100]["prompt_tokens"], 5016)
        self.assertEqual(self.state.active_requests[100]["prompt_tokens"], 5016)

    def test_new_prompt_does_not_overwrite_known_prompt_size(self):
        self.tracker.process_line(
            "I slot launch_slot_: id 0 | task 100 | processing task"
        )
        self.tracker.active[100]["prompt_tokens"] = 4000
        self.tracker.process_line(
            "I slot update_slots: id 0 | task 100 | new prompt, n_ctx_slot = 102400,"
            " n_keep = 0, task.n_tokens = 5016"
        )
        self.assertEqual(self.tracker.active[100]["prompt_tokens"], 4000)

    def test_cached_tokens_attach_to_live_row(self):
        self.tracker.process_line(
            "I slot launch_slot_: id 0 | task 100 | processing task"
        )
        self.tracker.process_line(
            "I slot update_slots: id 0 | task 100 | cached n_tokens = 4903,"
            " memory_seq_rm [4903, end)"
        )
        self.assertEqual(self.tracker.active[100]["cached_tokens"], 4903)


SWITCH_CATALOG = {
    "variants": {
        "eng/prod": {
            "status": "production",
            "model": "m1",
            "compose_path": "models/m1/eng/compose/dual/autoround-int4/fp8.yml",
        },
        "eng/cav": {"status": "caveats", "model": "m1"},
        "eng/exp": {"status": "experimental", "model": "m1"},
    },
    "defaults": {},
}


class ValidateSwitchTests(unittest.TestCase):
    def test_production_and_caveats_are_allowed(self):
        for key in ("eng/prod", "eng/cav"):
            entry = aipc_observer.validate_switch(key, SWITCH_CATALOG)
            self.assertEqual(entry["model"], "m1")

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            aipc_observer.validate_switch("eng/nope", SWITCH_CATALOG)

    def test_experimental_needs_force(self):
        with self.assertRaises(ValueError):
            aipc_observer.validate_switch("eng/exp", SWITCH_CATALOG)
        aipc_observer.validate_switch("eng/exp", SWITCH_CATALOG, force=True)

    def test_missing_catalog_is_a_runtime_error(self):
        with self.assertRaises(RuntimeError):
            aipc_observer.validate_switch("eng/prod", {})

    def test_normalizes_compose_derived_variant_to_slug(self):
        catalog = {
            "variants": {
                "beellama/dflash": {
                    "status": "caveats",
                    "compose_path": (
                        "models/qwen3.6-27b/beellama/compose/single/"
                        "beellama-q5ks-dflash/dflash.yml"
                    ),
                }
            }
        }
        self.assertEqual(
            aipc_observer.normalize_switch_variant(
                "qwen3.6-27b/beellama/single/beellama-q5ks-dflash/dflash",
                catalog,
            ),
            "beellama/dflash",
        )

    def test_normalize_leaves_unknown_variant_for_validator_error(self):
        self.assertEqual(
            aipc_observer.normalize_switch_variant("eng/nope", SWITCH_CATALOG),
            "eng/nope",
        )


class SwitchModelTests(unittest.TestCase):
    def test_parses_setup_hint_from_preflight(self):
        hint = aipc_observer.parse_setup_hint(
            "preflight: MODEL_DIR=/models WEIGHT_KEY=m1:autoround-int4 "
            "bash scripts/setup.sh m1"
        )
        self.assertEqual(hint["model"], "m1")
        self.assertEqual(hint["weight_key"], "m1:autoround-int4")
        self.assertEqual(hint["model_dir"], "/models")

    def test_infers_setup_from_variant_compose_path(self):
        hint = aipc_observer.infer_variant_setup(
            SWITCH_CATALOG["variants"]["eng/prod"]
        )
        self.assertEqual(hint["model"], "m1")
        self.assertEqual(hint["weight_key"], "m1:autoround-int4")

    def test_install_variant_assets_runs_setup_with_weight_key(self):
        runner = FakeRunner()
        result = aipc_observer.install_variant_assets(
            "/repo", "eng/prod", SWITCH_CATALOG, runner=runner)
        call = runner.calls[0]
        self.assertTrue(result["installed"])
        self.assertEqual(call["cmd"][-3:], ["bash", "scripts/setup.sh", "m1"])
        self.assertEqual(call["cwd"], "/repo")
        self.assertEqual(call["env"]["WEIGHT_KEY"], "m1:autoround-int4")

    def test_run_with_progress_reports_output_lines(self):
        lines = []
        output = aipc_observer._run_with_progress(
            [sys.executable, "-c", "print('setup one'); print('setup two')"],
            timeout=10,
            on_line=lines.append,
        )
        self.assertEqual(lines, ["setup one", "setup two"])
        self.assertIn("setup two", output)

    def test_run_with_progress_splits_on_cr(self):
        """Download tools use \\r for progress bars; each update should be a separate line."""
        lines = []
        # Simulate curl/wget style progress: updates separated by \r, final \n
        script = r"import sys; sys.stdout.write('10%\r50%\r100%\n'); sys.stdout.flush()"
        output = aipc_observer._run_with_progress(
            [sys.executable, "-c", script],
            timeout=10,
            on_line=lines.append,
        )
        self.assertEqual(lines, ["10%", "50%", "100%"])
        self.assertIn("100%", output)

    def test_runs_switch_sh_with_port_env(self):
        runner = FakeRunner()
        aipc_observer.switch_model("/repo", "eng/prod", 8020, runner=runner)
        call = runner.calls[0]
        self.assertEqual(call["cmd"][-3:], ["bash", "scripts/switch.sh", "eng/prod"])
        self.assertEqual(call["cwd"], "/repo")
        self.assertEqual(call["env"]["PORT"], "8020")
        self.assertEqual(call["env"]["READY_URL"],
                         "http://localhost:8020/v1/models")

    def test_force_flag_precedes_variant(self):
        runner = FakeRunner()
        aipc_observer.switch_model("/repo", "eng/exp", 8020, force=True,
                                   runner=runner)
        self.assertEqual(runner.calls[0]["cmd"][-2:], ["--force", "eng/exp"])


class SwitchWorkerTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        # The worker releases the module control lock; hold it like the
        # endpoint does before handing off.
        self.assertTrue(aipc_observer._control_lock.acquire(blocking=False))
        self.saved_status = aipc_observer.state.control_status
        self.saved_installed_assets = dict(aipc_observer.state.installed_assets)
        # Never touch the real OVERRIDE_FILE: on the deploy host it exists
        # root-owned (written by the daemon), so tests must use their own.
        tmp = tempfile.NamedTemporaryFile(suffix=".yml", delete=False)
        tmp.close()
        self.override_path = tmp.name

    def tearDown(self):
        import os

        if aipc_observer._control_lock.locked():
            aipc_observer._control_lock.release()
        aipc_observer.state.set_control_status(self.saved_status)
        aipc_observer.state.installed_assets = self.saved_installed_assets
        os.unlink(self.override_path)

    def test_baseline_switch_still_reups_for_log_rotation(self):
        runner = FakeRunner()
        aipc_observer._switch_worker(
            "/repo", "eng/prod", "baseline", 8020, False, runner=runner,
            info_getter=lambda port: dict(MODEL_INFO),
            override_path=self.override_path,
        )
        status = aipc_observer.state.control_status
        self.assertTrue(status["done"])
        self.assertTrue(status["ok"])
        self.assertFalse(aipc_observer._control_lock.locked())
        # switch.sh, then the baseline re-up (override = log rotation +
        # image pin only; no compose-config call for baseline).
        self.assertEqual(len(runner.calls), 2)
        self.assertIn("scripts/switch.sh", runner.calls[0]["cmd"])
        up_cmd = runner.calls[1]["cmd"]
        self.assertIn("up", up_cmd)
        self.assertIn(self.override_path, up_cmd)

    def test_preset_switch_resolves_config_then_reups(self):
        runner = FakeRunner(CONFIG_JSON)
        aipc_observer._switch_worker(
            "/repo", "eng/prod", "insight", 8020, False, runner=runner,
            info_getter=lambda port: dict(MODEL_INFO),
            override_path=self.override_path,
        )
        self.assertTrue(aipc_observer.state.control_status["ok"])
        # switch.sh + compose config + compose up.
        self.assertEqual(len(runner.calls), 3)
        self.assertIn("config", runner.calls[1]["cmd"])
        self.assertIn("up", runner.calls[2]["cmd"])

    def test_failed_switch_reports_error_and_releases_lock(self):
        def failing_runner(cmd, env=None, cwd=None, timeout=600, input_text=None):
            raise RuntimeError(
                "preflight: MODEL_DIR=/models WEIGHT_KEY=m1:autoround-int4 "
                "bash scripts/setup.sh m1"
            )

        aipc_observer._switch_worker("/repo", "eng/prod", "baseline", 8020,
                                     False, runner=failing_runner)
        status = aipc_observer.state.control_status
        self.assertTrue(status["done"])
        self.assertFalse(status["ok"])
        self.assertIn("setup.sh m1", status["detail"])
        self.assertEqual(status["install_hint"]["model"], "m1")
        self.assertEqual(status["install_hint"]["weight_key"], "m1:autoround-int4")
        self.assertEqual(status["install_hint"]["variant"], "eng/prod")
        self.assertFalse(aipc_observer._control_lock.locked())

    def test_install_worker_marks_variant_installed(self):
        runner = FakeRunner()
        aipc_observer.state.set_catalog(SWITCH_CATALOG)
        aipc_observer._install_worker(
            "/repo", "eng/prod", "baseline", 8020, False, False,
            {}, runner=runner,
        )
        status = aipc_observer.state.control_status
        self.assertTrue(status["ok"])
        self.assertEqual(status["installed_variant"], "eng/prod")
        self.assertIn("eng/prod", aipc_observer.state.installed_assets)
        self.assertEqual(
            aipc_observer.state.installed_assets["eng/prod"]["weight_key"],
            "m1:autoround-int4",
        )
        self.assertFalse(aipc_observer._control_lock.locked())

    def test_snapshot_reports_control_fields(self):
        st = aipc_observer.ObserverState()
        st.set_control_status({"action": "switch", "done": False})
        snap = st.snapshot()
        self.assertEqual(snap["control_status"]["action"], "switch")
        self.assertTrue(snap["control_busy"])


if __name__ == "__main__":
    unittest.main()
