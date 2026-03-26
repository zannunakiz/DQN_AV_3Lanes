import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from main_train import (
    _discounted_step_value,
    _ensure_tester_stage_csv,
    _fmt_reward4,
    _increment_tester_stage_fail_count,
    run_tester_validation,
    should_stop_on_episode_target_valid,
)


class _DummyEnv:
    def __init__(self, *args, **kwargs):
        pass


def _read_stage_counts(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


class TestTrainingCore(unittest.TestCase):
    def test_episode_target_stop_rule(self):
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=1999,
                num_episodes=2000,
                episode_had_final_valid_success=True,
            )
        )
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=2000,
                num_episodes=2000,
                episode_had_final_valid_success=False,
            )
        )
        self.assertTrue(
            should_stop_on_episode_target_valid(
                episode=2000,
                num_episodes=2000,
                episode_had_final_valid_success=True,
            )
        )
        self.assertTrue(
            should_stop_on_episode_target_valid(
                episode=2009,
                num_episodes=2000,
                episode_had_final_valid_success=True,
            )
        )
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=5000,
                num_episodes=sys.maxsize,
                episode_had_final_valid_success=True,
            )
        )

    def test_discount_and_format_helpers(self):
        gamma = 0.99
        rewards = [-0.04, -0.03, -0.03, -5.0]
        expected = [-0.04, -0.0297, -0.029403, -4.851495]

        for power, (reward, expected_value) in enumerate(zip(rewards, expected)):
            self.assertAlmostEqual(
                _discounted_step_value(reward, gamma, power),
                expected_value,
                places=9,
            )

        self.assertEqual(_fmt_reward4(0), "0.0000")
        self.assertEqual(_fmt_reward4(1.2), "1.2000")
        self.assertEqual(_fmt_reward4(-0.0297), "-0.0297")

    def test_tester_stage_csv_create_and_increment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = _ensure_tester_stage_csv(tmp_dir, total_tester_stages=5)
            self.assertTrue(os.path.exists(csv_path))

            _increment_tester_stage_fail_count(
                tmp_dir,
                total_tester_stages=5,
                failed_stage=5,
            )
            _increment_tester_stage_fail_count(
                tmp_dir,
                total_tester_stages=5,
                failed_stage=5,
            )
            _increment_tester_stage_fail_count(
                tmp_dir,
                total_tester_stages=5,
                failed_stage=4,
            )

            rows = _read_stage_counts(csv_path)
            counts = {int(row["stage"]): int(row["fail_count"]) for row in rows}

            self.assertEqual(len(rows), 5)
            self.assertEqual(counts[5], 2)
            self.assertEqual(counts[4], 1)
            self.assertEqual(counts[1], 0)

    def test_run_tester_validation_updates_failed_stage_counter(self):
        test_obstacles = [[], [], [], []]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("main_train.TEST_OBSTACLES", test_obstacles):
                with patch("main_train.CarEnvironment", _DummyEnv):
                    with patch(
                        "main_train.run_independent_test",
                        side_effect=[True, False],
                    ):
                        result = run_tester_validation(
                            agent=object(),
                            max_steps=10,
                            verbose=False,
                            save_dir=tmp_dir,
                        )
                        self.assertEqual(result, (False, 2, 4))

                    with patch(
                        "main_train.run_independent_test",
                        side_effect=[True, False],
                    ):
                        result = run_tester_validation(
                            agent=object(),
                            max_steps=10,
                            verbose=False,
                            save_dir=tmp_dir,
                        )
                        self.assertEqual(result, (False, 2, 4))

                    with patch(
                        "main_train.run_independent_test",
                        side_effect=[True, True, True, False],
                    ):
                        result = run_tester_validation(
                            agent=object(),
                            max_steps=10,
                            verbose=False,
                            save_dir=tmp_dir,
                        )
                        self.assertEqual(result, (False, 4, 4))

            rows = _read_stage_counts(os.path.join(tmp_dir, "tester_stage.csv"))
            counts = {int(row["stage"]): int(row["fail_count"]) for row in rows}

            self.assertEqual(counts[2], 2)
            self.assertEqual(counts[4], 1)
            self.assertEqual(counts[1], 0)
            self.assertEqual(counts[3], 0)


if __name__ == "__main__":
    unittest.main()
