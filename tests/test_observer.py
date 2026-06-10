#!/usr/bin/env python3
"""Tests for integrated aipc observer request parsing."""

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


if __name__ == "__main__":
    unittest.main()
