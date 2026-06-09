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

    def test_full_reprocess_marks_cache_defeated(self):
        self.tracker.process_line("I slot launch_slot_: id 0 | task 100 | processing task")
        self.tracker.process_line(
            "W slot update_slots: id 0 | task 100 | forcing full prompt re-processing due to lack of cache data"
        )
        self.assertTrue(self.tracker.active[100]["cache_defeated"])
        self.assertEqual(self.state.cache_defeated_count, 1)


if __name__ == "__main__":
    unittest.main()
