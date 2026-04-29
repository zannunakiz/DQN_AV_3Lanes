import os
import random
import tempfile
import unittest

from main_constant import (
    FINISH_DISTANCE,
    OBSTACLE_WARNING_DISTANCE_FRONT,
    OBSTACLE_WARNING_DISTANCE_SIDES,
)
from main_environment import CarEnvironment
from main_dqn_agent import DQNAgent
from main_visualize import (
    ExperimentObstaclePlanner,
    RandomObstacleGenerator,
    build_neuron_trace,
    build_visualize_episode_row,
    get_next_visualize_csv_path,
)


class TestVisualizeCore(unittest.TestCase):
    def test_next_visualize_csv_path_increments_existing_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            open(os.path.join(tmp_dir, "visualize-1.csv"), "w", encoding="utf-8").close()
            open(os.path.join(tmp_dir, "visualize-3.csv"), "w", encoding="utf-8").close()
            open(os.path.join(tmp_dir, "notes.txt"), "w", encoding="utf-8").close()

            next_path = get_next_visualize_csv_path(tmp_dir)

            self.assertTrue(next_path.endswith("visualize-4.csv"))

    def test_build_visualize_episode_row_format(self):
        row = build_visualize_episode_row(
            episode=7,
            close_distance=12,
            mse=0.0,
            reward=34.56789,
            avg_reward=20.1,
            time_ms=1234,
            timeframe=78,
            steps=56,
        )

        self.assertEqual(
            row,
            {
                "episode": 7,
                "close distance": 12,
                "MSE": "0.000",
                "Reward": "34.568",
                "Avg reward": "20.100",
                "time": 1234,
                "timeframe": 78,
                "steps": 56,
            },
        )

    def test_experiment_planner_builds_cumulative_spawn_plan(self):
        planner = ExperimentObstaclePlanner()

        planner.toggle_lane("left")
        planner.distance = 150
        self.assertTrue(planner.add_current_selection())

        planner.toggle_lane("center")
        planner.distance = 125
        self.assertTrue(planner.add_current_selection())

        plan = planner.build_spawn_plan(current_car_y=1000)
        configs = planner.build_obstacle_configs(current_car_y=1000)

        self.assertEqual(len(plan), 2)
        self.assertAlmostEqual(plan[0]["spawn_y"], 1350.0)
        self.assertAlmostEqual(plan[1]["spawn_y"], 1475.0)
        self.assertEqual(len(configs), 3)
        self.assertEqual(configs[0]["lane"], 0)
        self.assertAlmostEqual(configs[0]["y"], 1350.0)
        self.assertEqual(configs[1]["lane"], 0)
        self.assertAlmostEqual(configs[1]["y"], 1475.0)
        self.assertEqual(configs[2]["lane"], 1)
        self.assertAlmostEqual(configs[2]["y"], 1475.0)

    def test_random_obstacle_generator_appends_rows_with_configured_gap(self):
        generator = RandomObstacleGenerator(
            start_y=400,
            gap_y=125,
            lookahead_y=500,
            rng=random.Random(7),
        )
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        added_count = generator.append_due_obstacles(env)

        rows = sorted({obs["y"] for obs in env.obstacles})
        self.assertEqual(rows, [400.0, 525.0])
        self.assertEqual(generator.rows_spawned, 2)
        self.assertGreaterEqual(added_count, 2)
        self.assertLessEqual(added_count, 4)
        for y_value in rows:
            lanes_at_y = [obs["lane"] for obs in env.obstacles if obs["y"] == y_value]
            self.assertGreaterEqual(len(lanes_at_y), 1)
            self.assertLessEqual(len(lanes_at_y), 2)
            self.assertEqual(len(lanes_at_y), len(set(lanes_at_y)))

    def test_random_obstacle_generator_builds_fifty_row_stage(self):
        generator = RandomObstacleGenerator(
            start_y=400,
            gap_y=125,
            max_rows=50,
            rng=random.Random(11),
        )
        env = CarEnvironment(obstacles_config=[[]], disable_finish=False)
        env.reset()

        added_count = generator.append_all_obstacles(env)
        rows = sorted({obs["y"] for obs in env.obstacles})

        self.assertEqual(len(rows), 50)
        self.assertEqual(rows[0], 400.0)
        self.assertEqual(rows[-1], 400.0 + (125.0 * 49))
        self.assertGreaterEqual(added_count, 50)
        self.assertLessEqual(added_count, 100)
        self.assertEqual(env.finish_line_y, rows[-1] + FINISH_DISTANCE)

    def test_build_neuron_trace_reports_layer_shapes_and_q_values(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        state = env.reset()
        agent = DQNAgent(env.state_size, env.action_size, memory_size=10)

        trace = build_neuron_trace(
            agent,
            state,
            max_neurons_per_hidden_layer=2,
            max_inputs_per_neuron=3,
        )

        self.assertEqual(len(trace["input"]), env.state_size)
        self.assertEqual(len(trace["layers"]), 4)
        self.assertEqual(len(trace["q_values"]), env.action_size)
        self.assertEqual(trace["layers"][0]["input_size"], env.state_size)
        self.assertEqual(trace["layers"][-1]["output_size"], env.action_size)
        self.assertGreater(trace["total_params"], 0)
        for layer in trace["layers"][:-1]:
            self.assertLessEqual(len(layer["neurons"]), 2)
            for neuron in layer["neurons"]:
                self.assertLessEqual(len(neuron["contributions"]), 3)

    def test_warning_close_count_reports_front_plus_sides(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        call_index = {"value": 0}

        def fake_cast_ray(_angle, max_range):
            call_index["value"] += 1
            if call_index["value"] == 1:
                return OBSTACLE_WARNING_DISTANCE_FRONT - 1.0
            if call_index["value"] == 2:
                return OBSTACLE_WARNING_DISTANCE_SIDES - 1.0
            if call_index["value"] == 3:
                return OBSTACLE_WARNING_DISTANCE_SIDES + 1.0
            return float(max_range)

        env._cast_ray = fake_cast_ray
        _, _, _, info = env.step(4, apply_steering=True)

        self.assertTrue(info["warning_front"])
        self.assertTrue(info["warning_side_right"])
        self.assertFalse(info["warning_side_left"])
        self.assertEqual(info["warning_close_count"], 2)


if __name__ == "__main__":
    unittest.main()
