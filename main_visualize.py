"""
Pygame Visualization for DQN Car Navigation
Real-time rendering with scrolling road effect
Single-window display: Road + two-column Info panel
"""

import pygame
import math
import os
import csv
import re
import random
import numpy as np

from main_environment import CarEnvironment, get_num_stages
from main_dqn_agent import DQNAgent

try:
    from main_constant import (
        SCREEN_HEIGHT,
        CAR_STATIC_Y_POS,
        DEFAULT_SCALE,
        OBSTACLES,
        TEST_OBSTACLES,
        ALLSTAGE_CONSECUTIVE_REQ,
        KEYONE_MULTIPLIER,
        USE_PNG,
        LANE_CENTER_REWARD_WIDTH,
        SHOW_CENTERLANE_REWARD_INDICATOR,
        CENTERLANE_REWARD_INDICATOR_COLOR,
        LEFT_LR_OFFSETX,
        RIGHT_LR_OFFSETX,
        CENTER_LR_OFFSETX,
        DECISION_INTERVAL,
        MEMORY_SIZE,
        CAR_MAX_SPEED,
        EPSILON_DECAY,
        LEARNING_RATE,
        GAMMA,
        BATCH_SIZE,
        TARGET_UPDATE_FREQ,
        TRAIN_MAX_EPSILON,
        TRAIN_MIN_EPSILON,
        startRandom,
        gapRandom,
        maxRandom,
    )
except ImportError:
    SCREEN_HEIGHT = 600
    CAR_STATIC_Y_POS = 150
    DEFAULT_SCALE = 1.5
    OBSTACLES = [[]]
    TEST_OBSTACLES = [[]]
    ALLSTAGE_CONSECUTIVE_REQ = 2
    KEYONE_MULTIPLIER = 5
    USE_PNG = False
    DECISION_INTERVAL = 10
    MEMORY_SIZE = 100000
    CAR_MAX_SPEED = 3.2444444444444445
    EPSILON_DECAY = 0.998
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10
    TRAIN_MAX_EPSILON = 1.0
    TRAIN_MIN_EPSILON = 0.001
    startRandom = 400
    gapRandom = 125
    maxRandom = 50


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)

VISUALIZE_LOG_DIR = "visualize_logs"
VISUALIZE_LOG_BASENAME = "visualize"
VISUALIZE_LOG_PATTERN = re.compile(r"^visualize-(\d+)\.csv$")
NEURON_INPUT_NAMES = ["R2", "R1", "F", "L1", "L2", "SR", "SL", "Speed"]
NEURON_ACTION_NAMES = ["SlowL", "SlowS", "SlowR", "FastL", "FastS", "FastR"]


def build_neuron_trace(
    agent,
    state,
    max_neurons_per_hidden_layer=6,
    max_inputs_per_neuron=8,
):
    """Build a compact forward-pass trace from normalized input to Q-values."""
    state_array = np.asarray(state, dtype=np.float64).reshape(-1)
    activations = state_array
    linear_layers = [
        module
        for module in agent.policy_net.network
        if hasattr(module, "weight") and hasattr(module, "bias")
    ]

    trace_layers = []
    total_params = 0
    for layer_index, module in enumerate(linear_layers, start=1):
        weights = module.weight.detach().cpu().numpy().astype(np.float64, copy=False)
        bias = module.bias.detach().cpu().numpy().astype(np.float64, copy=False)
        total_params += int(weights.size + bias.size)

        pre_activation = weights.dot(activations) + bias
        is_output_layer = layer_index == len(linear_layers)
        post_activation = (
            pre_activation
            if is_output_layer
            else np.maximum(pre_activation, 0.0)
        )

        if is_output_layer:
            selected_indices = list(range(len(pre_activation)))
        else:
            ranked_indices = np.argsort(np.abs(post_activation))[::-1]
            selected_indices = [
                int(idx) for idx in ranked_indices[: int(max_neurons_per_hidden_layer)]
            ]

        neuron_rows = []
        for neuron_index in selected_indices:
            products = weights[neuron_index] * activations
            ranked_inputs = np.argsort(np.abs(products))[::-1]
            contribution_rows = []
            for input_index in ranked_inputs[: int(max_inputs_per_neuron)]:
                input_index = int(input_index)
                input_name = (
                    NEURON_INPUT_NAMES[input_index]
                    if layer_index == 1 and input_index < len(NEURON_INPUT_NAMES)
                    else f"a{layer_index - 1}[{input_index}]"
                )
                contribution_rows.append(
                    {
                        "input_index": input_index,
                        "input_name": input_name,
                        "weight": float(weights[neuron_index, input_index]),
                        "input": float(activations[input_index]),
                        "product": float(products[input_index]),
                    }
                )

            output_label = (
                NEURON_ACTION_NAMES[neuron_index]
                if is_output_layer and neuron_index < len(NEURON_ACTION_NAMES)
                else f"n{neuron_index}"
            )
            neuron_rows.append(
                {
                    "index": int(neuron_index),
                    "label": output_label,
                    "bias": float(bias[neuron_index]),
                    "z": float(pre_activation[neuron_index]),
                    "activation": float(post_activation[neuron_index]),
                    "contributions": contribution_rows,
                }
            )

        trace_layers.append(
            {
                "index": int(layer_index),
                "type": "output" if is_output_layer else "hidden",
                "input_size": int(weights.shape[1]),
                "output_size": int(weights.shape[0]),
                "weight_shape": tuple(int(v) for v in weights.shape),
                "bias_shape": tuple(int(v) for v in bias.shape),
                "activation_min": float(np.min(post_activation)),
                "activation_max": float(np.max(post_activation)),
                "activation_mean": float(np.mean(post_activation)),
                "active_count": int(np.sum(post_activation > 0.0))
                if not is_output_layer
                else None,
                "neurons": neuron_rows,
                "q_values": post_activation.tolist() if is_output_layer else None,
            }
        )

        activations = post_activation

    q_values = trace_layers[-1]["q_values"] if trace_layers else []
    return {
        "input": state_array.tolist(),
        "input_names": NEURON_INPUT_NAMES[: len(state_array)],
        "layers": trace_layers,
        "q_values": q_values,
        "total_params": int(total_params),
    }


class RandomObstacleGenerator:
    """Generate finite random obstacle rows for --random visualization."""

    def __init__(
        self,
        start_y=startRandom,
        gap_y=gapRandom,
        min_vehicles_per_row=1,
        max_vehicles_per_row=2,
        max_rows=maxRandom,
        lookahead_y=1500,
        cleanup_behind_y=300,
        rng=None,
    ):
        self.start_y = float(start_y)
        self.gap_y = float(gap_y)
        self.min_vehicles_per_row = int(min_vehicles_per_row)
        self.max_vehicles_per_row = int(max_vehicles_per_row)
        self.max_rows = int(max(1, max_rows))
        self.lookahead_y = float(lookahead_y)
        self.cleanup_behind_y = float(cleanup_behind_y)
        self.rng = rng if rng is not None else random
        self.next_spawn_y = self.start_y
        self.rows_spawned = 0
        self.total_obstacles_spawned = 0

    def reset(self):
        self.next_spawn_y = self.start_y
        self.rows_spawned = 0
        self.total_obstacles_spawned = 0

    def _choose_row_lanes(self, lane_count):
        lane_count = int(max(1, lane_count))
        min_count = max(1, min(self.min_vehicles_per_row, lane_count))
        max_count = max(min_count, min(self.max_vehicles_per_row, lane_count))
        vehicle_count = self.rng.randint(min_count, max_count)
        return sorted(self.rng.sample(range(lane_count), vehicle_count))

    def build_next_row_configs(self, lane_count):
        lanes = self._choose_row_lanes(lane_count)
        return [{"lane": lane, "y": self.next_spawn_y} for lane in lanes]

    def build_all_configs(self, lane_count):
        configs = []
        self.reset()
        while self.rows_spawned < self.max_rows:
            row_configs = self.build_next_row_configs(lane_count)
            configs.extend(row_configs)
            self.rows_spawned += 1
            self.total_obstacles_spawned += len(row_configs)
            self.next_spawn_y += self.gap_y
        return configs

    def append_all_obstacles(self, env):
        configs = self.build_all_configs(env.lane_count)
        return env.append_obstacles(configs)

    def append_due_obstacles(self, env):
        added = 0
        while (
            self.rows_spawned < self.max_rows
            and self.next_spawn_y <= float(env.car_y) + self.lookahead_y
        ):
            row_configs = self.build_next_row_configs(env.lane_count)
            added += env.append_obstacles(row_configs)
            self.rows_spawned += 1
            self.total_obstacles_spawned += len(row_configs)
            self.next_spawn_y += self.gap_y

        self.cleanup_obstacles(env)
        return added

    def cleanup_obstacles(self, env):
        cutoff_y = float(env.car_y) - self.cleanup_behind_y
        before_count = len(env.obstacles)
        env.obstacles = [
            obs for obs in env.obstacles if float(obs.get("y", 0.0)) >= cutoff_y
        ]
        return before_count - len(env.obstacles)


def get_next_visualize_csv_path(log_dir=VISUALIZE_LOG_DIR):
    """Create log directory if needed and return next visualize-<N>.csv path."""
    abs_log_dir = os.path.abspath(log_dir)
    os.makedirs(abs_log_dir, exist_ok=True)

    max_index = 0
    for entry in os.listdir(abs_log_dir):
        match = VISUALIZE_LOG_PATTERN.match(entry)
        if not match:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except Exception:
            continue

    next_index = max_index + 1
    filename = f"{VISUALIZE_LOG_BASENAME}-{next_index}.csv"
    return os.path.join(abs_log_dir, filename)


def build_visualize_episode_row(
    episode: int,
    close_distance: int,
    mse: float,
    reward: float,
    avg_reward: float,
    time_ms: int,
    timeframe: int,
    steps: int,
):
    """Build one visualize CSV row with fixed formatting."""
    return {
        "episode": int(episode),
        "close distance": int(close_distance),
        "MSE": f"{float(mse):.3f}",
        "Reward": f"{float(reward):.3f}",
        "Avg reward": f"{float(avg_reward):.3f}",
        "time": int(time_ms),
        "timeframe": int(timeframe),
        "steps": int(steps),
    }


class ExperimentObstaclePlanner:
    """State container for experiment obstacle controls."""

    LANE_LABEL_TO_INDEX = {"left": 0, "center": 1, "right": 2}
    LANE_INDEX_TO_LABEL = {0: "left", 1: "center", 2: "right"}

    def __init__(self):
        self.selected_lanes = set()
        self.distance = 125
        self.to_spawn_list = []

    def toggle_lane(self, lane_label):
        lane = str(lane_label).strip().lower()
        if lane not in self.LANE_LABEL_TO_INDEX:
            return False
        if lane in self.selected_lanes:
            self.selected_lanes.remove(lane)
        else:
            self.selected_lanes.add(lane)
        return True

    def increment_distance(self):
        self.distance += 5
        return self.distance

    def decrement_distance(self):
        self.distance = max(5, self.distance - 5)
        return self.distance

    def can_add(self):
        return len(self.selected_lanes) > 0

    def add_current_selection(self):
        if not self.can_add():
            return False
        ordered_lanes = sorted(
            self.selected_lanes, key=lambda name: self.LANE_LABEL_TO_INDEX[name]
        )
        self.to_spawn_list.append({"lanes": ordered_lanes, "distance": self.distance})
        return True

    def clear_spawn_list(self):
        self.to_spawn_list = []

    def build_spawn_plan(self, current_car_y):
        """Build cumulative spawn Y values from current car position."""
        if not self.to_spawn_list:
            return []

        plan = []
        previous_y = None
        for entry in self.to_spawn_list:
            distance = float(entry["distance"])
            if previous_y is None:
                spawn_y = float(current_car_y) + 200.0 + distance
            else:
                spawn_y = previous_y + distance
            lanes = list(entry["lanes"])
            plan.append({"lanes": lanes, "distance": distance, "spawn_y": spawn_y})
            previous_y = spawn_y
        return plan

    def build_obstacle_configs(self, current_car_y):
        configs = []
        for row in self.build_spawn_plan(current_car_y):
            for lane_label in row["lanes"]:
                lane_index = self.LANE_LABEL_TO_INDEX.get(lane_label)
                if lane_index is not None:
                    configs.append({"lane": lane_index, "y": row["spawn_y"]})
        return configs

    def snapshot(self):
        return {
            "selected_lanes": sorted(
                self.selected_lanes, key=lambda name: self.LANE_LABEL_TO_INDEX[name]
            ),
            "distance": self.distance,
            "to_spawn_list": [
                {"lanes": list(entry["lanes"]), "distance": int(entry["distance"])}
                for entry in self.to_spawn_list
            ],
        }


class GameRenderer:
    """Pygame renderer for car environment (single window with 3 sections)."""

    def __init__(
        self,
        env,
        scale=DEFAULT_SCALE,
        experiment_mode=False,
        neuron_mode=False,
    ):
        pygame.init()

        self.env = env
        self.scale = scale
        self.experiment_mode = bool(experiment_mode)
        self.neuron_mode = bool(neuron_mode)


        self.road_display_width = int(env.road_width * scale)
        self.road_display_height = SCREEN_HEIGHT


        self.car_screen_y = CAR_STATIC_Y_POS


        self.info_left_width = 320
        self.info_right_width = 320
        self.info_panel_width = self.info_left_width + self.info_right_width
        self.experiment_panel_width = 360 if self.experiment_mode else 0
        self.neuron_panel_width = 760 if self.neuron_mode else 0
        self.window_width = (
            self.road_display_width
            + self.info_panel_width
            + self.experiment_panel_width
            + self.neuron_panel_width
        )
        self.window_height = self.road_display_height
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        if self.experiment_mode and self.neuron_mode:
            pygame.display.set_caption(
                "DQN Car Navigation - Road + Info + Obstacle + Neuron Trace"
            )
        elif self.experiment_mode:
            pygame.display.set_caption(
                "DQN Car Navigation - Road + Info + Obstacle Controls (Experiment)"
            )
        elif self.neuron_mode:
            pygame.display.set_caption("DQN Car Navigation - Road + Info + Neuron Trace")
        else:
            pygame.display.set_caption("DQN Car Navigation - Road + Info")

        self.clock = pygame.time.Clock()
        self.base_fps = 60
        self.speed_multiplier = 1
        self.render_fps_multiplier = 1.0
        self.speed_mode = "normal"
        self.show_sensor_tip_labels = True
        self.font = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 24)


        self.car_img = None
        self.obstacle_img = None
        if USE_PNG:
            try:
                self.car_img = pygame.image.load("car.png").convert_alpha()
                self.obstacle_img = self.car_img.copy()
            except (FileNotFoundError, pygame.error) as e:
                print(
                    f"Warning: Could not load car.png ({e}). Falling back to rectangle rendering."
                )
                self.car_img = None
                self.obstacle_img = None


        self.last_render_info = None
        self.last_episode = 0
        self.last_total_reward = 0
        self.last_epsilon = 0
        self.last_timeframe = 0
        self.last_paused = False
        self.last_experiment_data = None
        self.last_neuron_data = None
        self.experiment_button_rects = {}

    def _set_speed_mode(self, mode):
        mode_str = str(mode).strip().lower()
        if mode_str not in {"normal", "fast", "slow"}:
            mode_str = "normal"

        self.speed_mode = mode_str
        if self.speed_mode == "fast":
            self.speed_multiplier = int(max(1, KEYONE_MULTIPLIER))
            self.render_fps_multiplier = 1.0
        elif self.speed_mode == "slow":
            self.speed_multiplier = 1
            self.render_fps_multiplier = 0.25
        else:
            self.speed_multiplier = 1
            self.render_fps_multiplier = 1.0

    def toggle_speed(self):
        """Toggle fast simulation mode (key '1')."""
        if self.speed_mode == "fast":
            self._set_speed_mode("normal")
        else:
            self._set_speed_mode("fast")
        return self.speed_multiplier

    def toggle_slow_motion(self):
        """Toggle slow-motion mode (key '2')."""
        if self.speed_mode == "slow":
            self._set_speed_mode("normal")
        else:
            self._set_speed_mode("slow")
        return self.get_speed_mode_label()

    def get_speed_mode_label(self):
        if self.speed_mode == "fast":
            return f"FAST x{int(self.speed_multiplier)}"
        if self.speed_mode == "slow":
            return "SLOW x0.25"
        return "NORMAL x1"

    def get_steps_per_frame(self):
        return int(max(1, self.speed_multiplier))

    def get_effective_render_fps(self):
        return int(max(1, round(float(self.base_fps) * float(self.render_fps_multiplier))))

    def get_camera_offset(self, car_y):
        """Calculate camera offset for scrolling effect"""
        return car_y - (CAR_STATIC_Y_POS / self.scale)

    def world_to_screen(self, x, y, camera_offset):
        """Convert world coordinates to screen coordinates with camera offset"""
        screen_x = int(x * self.scale)
        adjusted_y = y - camera_offset
        screen_y = int(self.road_display_height - adjusted_y * self.scale)
        return screen_x, screen_y

    def draw_road(self, camera_offset, finish_line_y):
        """Draw the 3-lane road with scrolling effect"""
        pygame.draw.rect(
            self.screen,
            DARK_GRAY,
            (0, 0, self.road_display_width, self.road_display_height),
        )


        try:
            if SHOW_CENTERLANE_REWARD_INDICATOR:

                for i in range(self.env.lane_count):
                    lane_center_world = (i * self.env.lane_width) + (
                        self.env.lane_width / 2
                    )

                    try:
                        if i == 0:
                            lane_center_world = float(lane_center_world) + float(
                                LEFT_LR_OFFSETX
                            )
                        elif i == 1:
                            lane_center_world = float(lane_center_world) + float(
                                CENTER_LR_OFFSETX
                            )
                        elif i == 2:
                            lane_center_world = float(lane_center_world) + float(
                                RIGHT_LR_OFFSETX
                            )
                    except Exception:
                        lane_center_world = (i * self.env.lane_width) + (
                            self.env.lane_width / 2
                        )
                    rect_width_px = int(LANE_CENTER_REWARD_WIDTH * self.scale)
                    rect_height_px = self.road_display_height
                    rect_x = int(lane_center_world * self.scale) - rect_width_px // 2
                    rect_y = 0

                    overlay = pygame.Surface(
                        (rect_width_px, rect_height_px), pygame.SRCALPHA
                    )
                    overlay.fill(CENTERLANE_REWARD_INDICATOR_COLOR)
                    self.screen.blit(overlay, (rect_x, rect_y))
        except NameError:

            pass

        lane_positions = self.env.get_lane_positions()
        for lane_x in lane_positions:
            screen_x = int(lane_x * self.scale)
            dash_length = 30
            gap_length = 20
            world_y_start = camera_offset
            world_y_end = camera_offset + (self.road_display_height / self.scale)

            y_world = int(world_y_start / (dash_length + gap_length)) * (
                dash_length + gap_length
            )
            while y_world < world_y_end + (dash_length + gap_length):
                _, screen_y_start = self.world_to_screen(0, y_world, camera_offset)
                _, screen_y_end = self.world_to_screen(
                    0, y_world + dash_length, camera_offset
                )

                if -10 < screen_y_start < self.road_display_height + 10:
                    pygame.draw.line(
                        self.screen,
                        WHITE,
                        (screen_x, max(0, screen_y_start)),
                        (screen_x, min(self.road_display_height, screen_y_end)),
                        2,
                    )
                y_world += dash_length + gap_length

        pygame.draw.line(self.screen, YELLOW, (0, 0), (0, self.road_display_height), 4)
        pygame.draw.line(
            self.screen,
            YELLOW,
            (self.road_display_width - 2, 0),
            (self.road_display_width - 2, self.road_display_height),
            4,
        )

        if not self.experiment_mode and math.isfinite(float(finish_line_y)):
            _, finish_screen_y = self.world_to_screen(0, finish_line_y, camera_offset)
            if -20 < finish_screen_y < self.road_display_height + 20:
                for i in range(0, self.road_display_width, 20):
                    color = WHITE if (i // 20) % 2 == 0 else BLACK
                    pygame.draw.rect(
                        self.screen, color, (i, finish_screen_y - 10, 20, 20)
                    )

        _, start_screen_y = self.world_to_screen(0, 50, camera_offset)
        if -10 < start_screen_y < self.road_display_height + 10:
            pygame.draw.line(
                self.screen,
                GREEN,
                (0, start_screen_y),
                (self.road_display_width, start_screen_y),
                2,
            )

    def draw_obstacle_car(self, obs_x, obs_y, obs_width, obs_height, camera_offset):
        """Draw an obstacle car (PNG with color masking or rectangle fallback)"""
        screen_x, screen_y = self.world_to_screen(obs_x, obs_y, camera_offset)

        if not (-100 < screen_y < self.road_display_height + 100):
            return

        if USE_PNG and self.obstacle_img is not None:

            scaled_width = int(obs_width * self.scale)
            scaled_height = int(obs_height * self.scale)
            obs_img_scaled = pygame.transform.scale(
                self.obstacle_img, (scaled_width, scaled_height)
            )


            mask_surface = pygame.Surface(
                (scaled_width, scaled_height), pygame.SRCALPHA
            )
            mask_surface.fill(RED)


            mask_surface.blit(
                obs_img_scaled, (0, 0), special_flags=pygame.BLEND_RGBA_MIN
            )


            obs_rect = mask_surface.get_rect(center=(screen_x, screen_y))
            self.screen.blit(mask_surface, obs_rect)

        else:

            obs_surface = pygame.Surface(
                (obs_width * self.scale, obs_height * self.scale), pygame.SRCALPHA
            )
            pygame.draw.rect(
                obs_surface,
                RED,
                (0, 0, obs_width * self.scale, obs_height * self.scale),
            )
            pygame.draw.rect(
                obs_surface,
                (180, 0, 0),
                (0, 0, obs_width * self.scale, obs_height * self.scale),
                2,
            )

            obs_rect = obs_surface.get_rect(center=(screen_x, screen_y))
            self.screen.blit(obs_surface, obs_rect)

    def draw_car(self, car_x, car_y, car_angle, car_width, car_height, camera_offset):
        """Draw the car (PNG with color masking or rectangle fallback)"""
        screen_x, screen_y = self.world_to_screen(car_x, car_y, camera_offset)

        if USE_PNG and self.car_img is not None:


            scaled_width = int(car_width * self.scale)
            scaled_height = int(car_height * self.scale)
            car_img_scaled = pygame.transform.scale(
                self.car_img, (scaled_width, scaled_height)
            )


            mask_surface = pygame.Surface(
                (scaled_width, scaled_height), pygame.SRCALPHA
            )
            mask_surface.fill(BLUE)


            mask_surface.blit(
                car_img_scaled, (0, 0), special_flags=pygame.BLEND_RGBA_MIN
            )


            center_x = scaled_width / 2
            pygame.draw.polygon(
                mask_surface,
                YELLOW,
                [(center_x, 5), (center_x - 8, 20), (center_x + 8, 20)],
            )


            rotation_angle = car_angle - 90
            rotated_car = pygame.transform.rotate(mask_surface, rotation_angle)
            rotated_rect = rotated_car.get_rect(center=(screen_x, screen_y))
            self.screen.blit(rotated_car, rotated_rect)

        else:

            car_surface = pygame.Surface(
                (car_width * self.scale, car_height * self.scale), pygame.SRCALPHA
            )
            pygame.draw.rect(
                car_surface,
                BLUE,
                (0, 0, car_width * self.scale, car_height * self.scale),
            )
            pygame.draw.rect(
                car_surface,
                LIGHT_BLUE,
                (0, 0, car_width * self.scale, car_height * self.scale),
                2,
            )

            center_x = car_width * self.scale / 2
            pygame.draw.polygon(
                car_surface,
                YELLOW,
                [(center_x, 5), (center_x - 8, 20), (center_x + 8, 20)],
            )

            rotation_angle = car_angle - 90
            rotated_car = pygame.transform.rotate(car_surface, rotation_angle)
            rotated_rect = rotated_car.get_rect(center=(screen_x, screen_y))

            self.screen.blit(rotated_car, rotated_rect)

    def draw_sensors(self, car_x, car_y, sensors, camera_offset, show_labels=False):
        """Draw sensor rays"""
        screen_x, screen_y = self.world_to_screen(car_x, car_y, camera_offset)
        sensor_tip_labels = ["R2", "R1", "F", "L1", "L2", "SR", "SL"]

        for i, sensor in enumerate(sensors):
            angle = sensor["angle"]
            distance = sensor["distance"]
            normalized = sensor.get("normalized", 1.0)

            rad = math.radians(angle)
            end_x = car_x + distance * math.cos(rad)
            end_y = car_y + distance * math.sin(rad)
            end_screen_x, end_screen_y = self.world_to_screen(
                end_x, end_y, camera_offset
            )

            if i == 2:
                from main_constant import OBSTACLE_WARNING_DISTANCE_FRONT

                if distance < OBSTACLE_WARNING_DISTANCE_FRONT:
                    color = RED
                elif normalized is None:
                    color = GREEN
                elif normalized > 0.5:
                    color = GREEN
                else:
                    color = GREEN
            elif i == 5 or i == 6:
                from main_constant import OBSTACLE_WARNING_DISTANCE_SIDES

                if distance < OBSTACLE_WARNING_DISTANCE_SIDES:
                    color = RED
                elif normalized is None:
                    color = GREEN
                elif normalized > 0.5:
                    color = GREEN
                else:
                    color = GREEN
            else:
                if normalized is None:
                    color = GREEN
                elif normalized > 0.5:
                    color = GREEN
                else:
                    color = GREEN

            pygame.draw.line(
                self.screen,
                color,
                (screen_x, screen_y),
                (end_screen_x, end_screen_y),
                2,
            )
            pygame.draw.circle(self.screen, color, (end_screen_x, end_screen_y), 3)
            if show_labels and i < len(sensor_tip_labels):
                label = self.font.render(sensor_tip_labels[i], True, WHITE)
                self.screen.blit(label, (end_screen_x + 4, end_screen_y - 10))

    def draw_info_panels(
        self,
        info,
        episode=0,
        total_reward=0,
        epsilon=0,
        fps=0,
        timeframe=0,
        paused=False,
    ):
        """Draw the two-column info area (left: stats/state/controls, right: sensors/NN/actions)."""
        left_x = self.road_display_width
        right_x = left_x + self.info_left_width

        pygame.draw.rect(
            self.screen,
            (40, 40, 40),
            (left_x, 0, self.info_left_width, self.window_height),
        )
        pygame.draw.rect(
            self.screen,
            (40, 40, 40),
            (right_x, 0, self.info_right_width, self.window_height),
        )


        pygame.draw.line(
            self.screen, WHITE, (right_x, 0), (right_x, self.window_height), 2
        )


        title = self.font_large.render("DQN Car Info", True, WHITE)
        self.screen.blit(title, (left_x + 20, 20))

        if paused:
            pause_text = self.font_large.render("PAUSED", True, RED)
            self.screen.blit(pause_text, (left_x + 20, 50))
            y_left = 80
        else:
            y_left = 60

        line_height = 22
        target_speed = float(info.get("target_speed", info["speed"]))
        speed_delta = float(info.get("speed_delta", 0.0))
        speed_kmh = (
            (float(info["speed"]) / float(CAR_MAX_SPEED)) * 75.0
            if float(CAR_MAX_SPEED) > 0
            else 0.0
        )
        in_center_zone = info.get("in_lane_center_zone", None)
        info_timeframe = info.get("timeframe", timeframe)
        world_step = info.get("world_step", None)
        if world_step is None:
            world_distance = info.get("world_distance", 0.0)
            try:
                world_step = int(round(float(world_distance)))
            except Exception:
                world_step = 0
        try:
            info_timeframe = int(info_timeframe)
        except Exception:
            info_timeframe = int(timeframe)

        center_zone_text = "Center Zone: N/A"
        if in_center_zone is not None:
            center_zone_text = f"Center Zone: {'IN' if in_center_zone else 'OUT'}"

        left_lines = [
            f"Episode: {episode}",
            f"Timeframe: {info_timeframe}",
            f"Step: {int(world_step)}",
            f"Total Reward: {total_reward:.2f}",
            f"Epsilon: {epsilon:.3f}",
            f"FPS: {fps:.0f}",
            f"Speed Mode: {self.get_speed_mode_label()}",
            "",
            "--- Car State ---",
            f"Position X: {info['car_x']:.1f}",
            f"Position Y: {info['car_y']:.1f}",
            f"Angle: {info['car_angle']:.1f} deg",
            f"Speed: {info['speed']:.2f}",
            f"Km/h: {speed_kmh:.1f}",
            f"Target Speed: {target_speed:.2f}",
            f"Delta/Timeframe: {speed_delta:+.4f}",
            center_zone_text,
            "",
            "--- Controls ---",
            "P: Pause/Resume",
            "R: Reset Episode",
            "Q: Quit",
            "I: Toggle Sensor Labels",
            f"1: Speed x{KEYONE_MULTIPLIER}",
            "2: Slow Motion",
        ]

        for line in left_lines:
            color = (
                YELLOW
                if (
                    line.startswith("--- Controls")
                    or ":" in line
                    and line[0].isalpha()
                    and line[1] == ":"
                )
                else WHITE
            )
            text = self.font.render(line, True, color)
            self.screen.blit(text, (left_x + 20, y_left))
            y_left += line_height


        right_title = self.font_large.render("Sensors / Networks", True, WHITE)
        self.screen.blit(right_title, (right_x + 20, 20))
        y_right = 60

        text = self.font.render("--- Sensors ---", True, WHITE)
        self.screen.blit(text, (right_x + 20, y_right))
        y_right += line_height

        sensor_names = [
            "Front-R2",
            "Front-R1",
            "Front",
            "Front-L1",
            "Front-L2",
            "Side-R",
            "Side-L",
        ]
        for i, sensor in enumerate(info["sensors"]):
            normalized = sensor.get("normalized", None)
            is_front = i == 2
            is_side = i == 5 or i == 6
            text = self.font.render(
                f"{sensor_names[i]}: {sensor['distance']:.1f}",
                True,
                self._get_sensor_color(
                    sensor["distance"],
                    normalized,
                    is_front_sensor=is_front,
                    is_side_sensor=is_side,
                ),
            )
            self.screen.blit(text, (right_x + 20, y_right))
            y_right += 20

        nn = info.get("nn_input", None)
        nn_out = info.get("nn_output", None)
        if nn is not None or nn_out is not None:
            y_right += 8
            text = self.font.render("--- Networks ---", True, WHITE)
            self.screen.blit(text, (right_x + 20, y_right))
            y_right += 22

        if nn is not None:
            text = self.font.render("NN Input:", True, WHITE)
            self.screen.blit(text, (right_x + 20, y_right))
            y_right += 18
            line1 = ", ".join(f"{v:.2f}" for v in nn[:4])
            line2 = ", ".join(f"{v:.2f}" for v in nn[4:8])
            text = self.font.render(line1, True, WHITE)
            self.screen.blit(text, (right_x + 20, y_right))
            y_right += 18
            text = self.font.render(line2, True, WHITE)
            self.screen.blit(text, (right_x + 20, y_right))
            y_right += 30

        if nn_out is not None:
            try:
                outputs = [float(v) for v in nn_out]
            except TypeError:
                outputs = None

            if outputs is not None:
                text = self.font.render("NN Output (Q):", True, WHITE)
                self.screen.blit(text, (right_x + 20, y_right))
                y_right += 18

                action_names = ["SlowL", "SlowS", "SlowR", "FastL", "FastS", "FastR"]
                for i, q in enumerate(outputs):
                    name = action_names[i] if i < len(action_names) else f"A{i}"
                    line = f"Q[{name}]: {q:.2f}"
                    text = self.font.render(line, True, WHITE)
                    self.screen.blit(text, (right_x + 20, y_right))
                    y_right += 18

        last_action = info.get("last_action", None)
        y_right += 12
        actions_header = self.font.render("-- Actions --", True, WHITE)
        self.screen.blit(actions_header, (right_x + 20, y_right))
        y_right += 22
        action_names = [
            "Slow Left",
            "Slow Straight",
            "Slow Right",
            "Fast Left",
            "Fast Straight",
            "Fast Right",
        ]
        for i, action_name in enumerate(action_names):
            action_text = self.font.render(f"{i + 1} = {action_name}", True, WHITE)
            self.screen.blit(action_text, (right_x + 20, y_right))
            y_right += 18

        if last_action is not None and last_action in range(6):
            current_name = action_names[last_action]
            current_text = self.font.render(
                f"Current Action: {last_action + 1} ({current_name})", True, ORANGE
            )
            self.screen.blit(current_text, (right_x + 20, y_right))
            y_right += 18

    def _draw_experiment_button(self, rect, label, enabled=True, active=False):
        if not enabled:
            bg_color = (65, 65, 65)
            border_color = (120, 120, 120)
            text_color = (170, 170, 170)
        elif active:
            bg_color = (0, 155, 0)
            border_color = (190, 255, 190)
            text_color = WHITE
        else:
            bg_color = (40, 90, 170)
            border_color = (130, 180, 255)
            text_color = WHITE

        pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
        pygame.draw.rect(self.screen, border_color, rect, width=2, border_radius=5)
        text = self.font.render(label, True, text_color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_experiment_panel(self, experiment_data):
        """Draw obstacle control panel used in --experiment mode."""
        if not self.experiment_mode:
            return

        if experiment_data is None:
            experiment_data = {"selected_lanes": [], "distance": 125, "to_spawn_list": []}

        selected_lanes = set(experiment_data.get("selected_lanes", []))
        distance = int(experiment_data.get("distance", 125))
        to_spawn_list = list(experiment_data.get("to_spawn_list", []))

        panel_x = self.road_display_width + self.info_panel_width
        panel_width = self.experiment_panel_width
        panel_rect = pygame.Rect(panel_x, 0, panel_width, self.window_height)

        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.line(
            self.screen, WHITE, (panel_x, 0), (panel_x, self.window_height), 2
        )

        self.experiment_button_rects = {}

        title = self.font_large.render("Obstacle Controls", True, WHITE)
        self.screen.blit(title, (panel_x + 20, 20))

        y = 60
        subtitle = self.font.render("3 Path", True, YELLOW)
        self.screen.blit(subtitle, (panel_x + 20, y))
        y += 24

        lane_labels = ["left", "center", "right"]
        lane_width = (panel_width - 50) // 3
        lane_gap = 5
        lane_x = panel_x + 20
        for lane_name in lane_labels:
            rect = pygame.Rect(lane_x, y, lane_width, 36)
            self.experiment_button_rects[f"lane_{lane_name}"] = rect
            self._draw_experiment_button(
                rect, lane_name.upper(), enabled=True, active=lane_name in selected_lanes
            )
            lane_x += lane_width + lane_gap

        y += 52
        distance_title = self.font.render("Distance", True, YELLOW)
        self.screen.blit(distance_title, (panel_x + 20, y))
        y += 24

        minus_rect = pygame.Rect(panel_x + 20, y, 60, 36)
        plus_rect = pygame.Rect(panel_x + panel_width - 80, y, 60, 36)
        value_rect = pygame.Rect(panel_x + 90, y, panel_width - 180, 36)

        self.experiment_button_rects["distance_minus"] = minus_rect
        self.experiment_button_rects["distance_plus"] = plus_rect

        self._draw_experiment_button(minus_rect, "-", enabled=True, active=False)
        self._draw_experiment_button(plus_rect, "+", enabled=True, active=False)

        pygame.draw.rect(self.screen, (50, 50, 50), value_rect, border_radius=5)
        pygame.draw.rect(self.screen, (140, 140, 140), value_rect, width=2, border_radius=5)
        value_text = self.font_large.render(str(distance), True, WHITE)
        value_text_rect = value_text.get_rect(center=value_rect.center)
        self.screen.blit(value_text, value_text_rect)

        y += 50
        can_add = len(selected_lanes) > 0
        add_rect = pygame.Rect(panel_x + 20, y, panel_width - 40, 38)
        self.experiment_button_rects["add_list"] = add_rect
        self._draw_experiment_button(
            add_rect, "ADD LIST", enabled=can_add, active=False
        )

        y += 50
        spawn_rect = pygame.Rect(panel_x + 20, y, (panel_width - 50) // 2, 38)
        clear_rect = pygame.Rect(
            spawn_rect.right + 10, y, (panel_width - 50) // 2, 38
        )
        self.experiment_button_rects["spawn"] = spawn_rect
        self.experiment_button_rects["clear_spawn"] = clear_rect

        self._draw_experiment_button(spawn_rect, "SPAWN", enabled=True, active=False)
        self._draw_experiment_button(
            clear_rect, "CLEAR SPAWN", enabled=True, active=False
        )

        y += 54
        box_label = self.font.render("To Spawn Lists", True, YELLOW)
        self.screen.blit(box_label, (panel_x + 20, y))
        y += 22

        list_rect = pygame.Rect(panel_x + 20, y, panel_width - 40, self.window_height - y - 20)
        pygame.draw.rect(self.screen, (20, 20, 20), list_rect)
        pygame.draw.rect(self.screen, (120, 120, 120), list_rect, width=2)

        row_height = 20
        max_rows = max(1, (list_rect.height - 10) // row_height)
        start_index = max(0, len(to_spawn_list) - max_rows)
        visible_rows = to_spawn_list[start_index:]

        row_y = list_rect.y + 6
        for idx, row in enumerate(visible_rows, start=start_index + 1):
            lanes = row.get("lanes", [])
            distance_val = int(row.get("distance", 0))
            row_text = f"{idx}. {'+'.join(lanes)} | d={distance_val}"
            text = self.font.render(row_text, True, WHITE)
            self.screen.blit(text, (list_rect.x + 8, row_y))
            row_y += row_height

    def draw_neuron_panel(self, neuron_data):
        """Draw forward-pass neuron details used in --neuron mode."""
        if not self.neuron_mode:
            return

        panel_x = (
            self.road_display_width
            + self.info_panel_width
            + self.experiment_panel_width
        )
        panel_width = self.neuron_panel_width
        panel_rect = pygame.Rect(panel_x, 0, panel_width, self.window_height)

        pygame.draw.rect(self.screen, (24, 26, 30), panel_rect)
        pygame.draw.line(
            self.screen, WHITE, (panel_x, 0), (panel_x, self.window_height), 2
        )

        title = self.font_large.render("Neuron Trace", True, WHITE)
        self.screen.blit(title, (panel_x + 18, 18))

        y = 52
        line_height = 16
        max_y = self.window_height - 18

        def _clip_text(text, width, indent=0):
            max_chars = max(20, (int(width) - 12 - int(indent)) // 7)
            clipped = str(text)
            if len(clipped) > max_chars:
                clipped = clipped[: max_chars - 3] + "..."
            return clipped

        def draw_line(text, color=WHITE, indent=0):
            nonlocal y
            if y > max_y:
                return
            clipped = _clip_text(text, panel_width - 30, indent)
            surface = self.font.render(clipped, True, color)
            self.screen.blit(surface, (panel_x + 18 + indent, y))
            y += line_height

        def draw_column_line(text, x, y_pos, width, color=WHITE, indent=0):
            if y_pos > max_y:
                return y_pos
            clipped = _clip_text(text, width, indent)
            surface = self.font.render(clipped, True, color)
            self.screen.blit(surface, (x + indent, y_pos))
            return y_pos + line_height

        if not neuron_data:
            draw_line("Waiting for first network pass...", YELLOW)
            return
        if neuron_data.get("error"):
            draw_line("Neuron trace error:", RED)
            draw_line(neuron_data.get("error"), WHITE, 8)
            return

        total_params = int(neuron_data.get("total_params", 0))
        draw_line(f"Policy net params: {total_params}", YELLOW)
        draw_line("Forward: y = W*x + b, hidden uses ReLU", LIGHT_BLUE)
        y += 4

        layers = neuron_data.get("layers", [])
        hidden_layers = [
            layer for layer in layers if str(layer.get("type", "hidden")) != "output"
        ]
        output_layers = [
            layer for layer in layers if str(layer.get("type", "hidden")) == "output"
        ]

        draw_line("-- Layers (top active neurons) --", YELLOW)
        column_top = y
        column_gap = 18
        column_width = (panel_width - 36 - column_gap) // 2
        left_x = panel_x + 18
        right_x = left_x + column_width + column_gap
        pygame.draw.line(
            self.screen,
            (80, 80, 80),
            (right_x - 9, column_top),
            (right_x - 9, self.window_height - 18),
            1,
        )

        def draw_layer_column(column_layers, x, start_y, width, title):
            y_col = start_y
            y_col = draw_column_line(title, x, y_col, width, YELLOW)
            for layer in column_layers:
                layer_index = int(layer.get("index", 0))
                layer_kind = layer.get("type", "hidden")
                in_size = int(layer.get("input_size", 0))
                out_size = int(layer.get("output_size", 0))
                active_count = layer.get("active_count", None)
                active_text = (
                    f", active={int(active_count)}/{out_size}"
                    if active_count is not None
                    else ""
                )
                y_col = draw_column_line(
                    f"L{layer_index} {layer_kind}: {in_size}->{out_size}{active_text}",
                    x,
                    y_col,
                    width,
                    LIGHT_BLUE,
                )
                y_col = draw_column_line(
                    "a min/mean/max = "
                    f"{float(layer.get('activation_min', 0.0)):+.4f} / "
                    f"{float(layer.get('activation_mean', 0.0)):+.4f} / "
                    f"{float(layer.get('activation_max', 0.0)):+.4f}",
                    x,
                    y_col,
                    width,
                    WHITE,
                    8,
                )

                neurons = layer.get("neurons", [])
                max_neurons_to_draw = 6 if layer_kind == "output" else 2
                for neuron in neurons[:max_neurons_to_draw]:
                    y_col = draw_column_line(
                        f"{neuron.get('label')} "
                        f"b={float(neuron.get('bias', 0.0)):+.4f} "
                        f"z={float(neuron.get('z', 0.0)):+.4f} "
                        f"a={float(neuron.get('activation', 0.0)):+.4f}",
                        x,
                        y_col,
                        width,
                        WHITE,
                        8,
                    )
                    for contrib in neuron.get("contributions", [])[:2]:
                        y_col = draw_column_line(
                            f"{contrib.get('input_name')} "
                            f"w={float(contrib.get('weight', 0.0)):+.4f} "
                            f"in={float(contrib.get('input', 0.0)):+.4f} "
                            f"prod={float(contrib.get('product', 0.0)):+.4f}",
                            x,
                            y_col,
                            width,
                            GRAY,
                            20,
                        )
                y_col += 4
                if y_col > max_y:
                    break
            return y_col

        draw_layer_column(hidden_layers, left_x, column_top, column_width, "Hidden")
        draw_layer_column(output_layers, right_x, column_top, column_width, "Output")

    def _get_sensor_color(
        self, distance, normalized=None, is_front_sensor=False, is_side_sensor=False
    ):
        """Get color based on sensor normalized value"""
        if is_front_sensor:
            from main_constant import OBSTACLE_WARNING_DISTANCE_FRONT

            if distance < OBSTACLE_WARNING_DISTANCE_FRONT:
                return RED

        if is_side_sensor:
            from main_constant import OBSTACLE_WARNING_DISTANCE_SIDES

            if distance < OBSTACLE_WARNING_DISTANCE_SIDES:
                return RED

        return GREEN

    def render(
        self,
        info,
        episode=0,
        total_reward=0,
        epsilon=0,
        timeframe=0,
        paused=False,
        experiment_data=None,
        neuron_data=None,
    ):
        """Render the entire scene"""

        self.last_render_info = info
        self.last_episode = episode
        self.last_total_reward = total_reward
        self.last_epsilon = epsilon
        self.last_timeframe = timeframe
        self.last_paused = paused
        self.last_experiment_data = experiment_data
        self.last_neuron_data = neuron_data

        self.screen.fill(BLACK)

        camera_offset = self.get_camera_offset(info["car_y"])
        finish_line_y = info.get("finish_line_y", self.env.finish_line_y)
        self.draw_road(camera_offset, finish_line_y)

        for obstacle in info["obstacles"]:
            self.draw_obstacle_car(
                obstacle["x"],
                obstacle["y"],
                obstacle["width"],
                obstacle["height"],
                camera_offset,
            )

        self.draw_sensors(
            info["car_x"],
            info["car_y"],
            info["sensors"],
            camera_offset,
            show_labels=self.show_sensor_tip_labels,
        )
        self.draw_car(
            info["car_x"],
            info["car_y"],
            info["car_angle"],
            info["car_width"],
            info["car_height"],
            camera_offset,
        )

        fps = self.clock.get_fps()
        self.draw_info_panels(
            info, episode, total_reward, epsilon, fps, timeframe, paused
        )
        self.draw_experiment_panel(experiment_data)
        self.draw_neuron_panel(neuron_data)

        pygame.display.flip()


        self.clock.tick(self.get_effective_render_fps())

    def render_cached(self, paused=None):
        """Re-render using cached state (for pause)."""
        if self.last_render_info is not None:
            self.render(
                self.last_render_info,
                self.last_episode,
                self.last_total_reward,
                self.last_epsilon,
                self.last_timeframe,
                self.last_paused if paused is None else paused,
                self.last_experiment_data,
                self.last_neuron_data,
            )

    def handle_events(self, paused=False):
        """Handle pygame events"""
        running = True
        reset = False
        manual_action = None
        pause_toggle = False
        experiment_actions = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    reset = True
                elif event.key == pygame.K_p:
                    pause_toggle = True
                elif event.key == pygame.K_1:
                    self.toggle_speed()
                    print(f"Speed Mode: {self.get_speed_mode_label()}")
                elif event.key == pygame.K_2:
                    self.toggle_slow_motion()
                    print(f"Speed Mode: {self.get_speed_mode_label()}")
                elif event.key == pygame.K_i:
                    self.show_sensor_tip_labels = not self.show_sensor_tip_labels
                    print(
                        "Sensor tip labels: "
                        f"{'ON' if self.show_sensor_tip_labels else 'OFF'}"
                    )
                elif event.key == pygame.K_LEFT:
                    manual_action = 3
                elif event.key == pygame.K_UP:
                    manual_action = 4
                elif event.key == pygame.K_DOWN:
                    manual_action = 1
                elif event.key == pygame.K_RIGHT:
                    manual_action = 5
            elif (
                self.experiment_mode
                and event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                mouse_pos = event.pos
                for button_id, rect in self.experiment_button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        if button_id.startswith("lane_"):
                            lane_name = button_id.replace("lane_", "", 1)
                            experiment_actions.append(
                                {"type": "toggle_lane", "lane": lane_name}
                            )
                        elif button_id == "distance_minus":
                            experiment_actions.append({"type": "distance_minus"})
                        elif button_id == "distance_plus":
                            experiment_actions.append({"type": "distance_plus"})
                        elif button_id == "add_list":
                            experiment_actions.append({"type": "add_list"})
                        elif button_id == "spawn":
                            experiment_actions.append({"type": "spawn"})
                        elif button_id == "clear_spawn":
                            experiment_actions.append({"type": "clear_spawn"})
                        break

        return running, reset, manual_action, pause_toggle, experiment_actions

    def close(self):
        """Close pygame window"""
        pygame.quit()


def run_visualization(
    model_path=None,
    episodes=10,
    manual_mode=False,
    allstage=False,
    tester=False,
    experiment=False,
    random_mode=False,
    neuron_mode=False,
):
    """Run visualization with trained model or manual control"""
    experiment_mode = bool(experiment)
    random_mode = bool(random_mode)
    if random_mode and experiment_mode:
        print("Random mode ignores --experiment.")
        experiment_mode = False
    if experiment_mode and allstage:
        print("Experiment mode ignores --allstage.")
        allstage = False
    if random_mode and allstage:
        print("Random mode ignores --allstage.")
        allstage = False
    if random_mode and tester:
        print("Random mode ignores --tester.")
        tester = False
    if experiment_mode:

        obstacles_cfg = [[]]
    elif random_mode:
        obstacles_cfg = [[]]
    else:
        obstacles_cfg = TEST_OBSTACLES if tester else OBSTACLES

    if experiment_mode:
        env = CarEnvironment(obstacles_config=obstacles_cfg, disable_finish=True)
    elif random_mode:
        env = CarEnvironment(obstacles_config=obstacles_cfg, disable_finish=False)
    elif allstage:
        env = CarEnvironment(curriculum_stage=0, obstacles_config=obstacles_cfg)
    else:
        env = CarEnvironment(obstacles_config=obstacles_cfg)

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=TRAIN_MAX_EPSILON,
        epsilon_min=TRAIN_MIN_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_size=MEMORY_SIZE,
    )

    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0.0
        print(f"Loaded model from {model_path}")
    else:
        print("No model loaded, using random actions or manual control")
        agent.epsilon = 0.0

    renderer = GameRenderer(
        env,
        scale=DEFAULT_SCALE,
        experiment_mode=experiment_mode,
        neuron_mode=neuron_mode,
    )
    planner = ExperimentObstaclePlanner() if experiment_mode else None
    random_generator = RandomObstacleGenerator() if random_mode else None
    visualize_csv_path = get_next_visualize_csv_path()
    visualize_rewards = []
    print(f"Visualize CSV: {visualize_csv_path}")

    print("\n=== Visualization Mode ===")
    print("Controls:")
    print("  Arrow Keys: Manual control (Left/Up/Right/Down)")
    print("  P: Pause/Resume")
    print("  R: Reset episode")
    print(f"  1: Toggle speed x{KEYONE_MULTIPLIER}")
    print("  2: Toggle slow motion")
    print("  Q: Quit")
    print("  I: Toggle Sensor Labels")
    if experiment_mode:
        print("  Mouse: Use Obstacle Controls panel")
        print("Obstacle source: EXPERIMENT (empty start)")
    elif random_mode:
        print(
            f"Obstacle source: RANDOM finite (start={int(startRandom)}, gap={int(gapRandom)}, "
            f"rows={int(maxRandom)}, 1-2 vehicle(s) per row)"
        )
    else:
        print(f"Obstacle source: {'TEST_OBSTACLES' if tester else 'OBSTACLES'}")
    if neuron_mode:
        print("Neuron trace: ON (normalized input, hidden activations, weights/bias, Q)")
    print("=" * 30)

    episode = 0
    running = True
    paused = False


    num_stages = get_num_stages(obstacles_cfg)
    current_stage = 0
    consecutive_success = 0
    csv_headers = [
        "episode",
        "close distance",
        "MSE",
        "Reward",
        "Avg reward",
        "time",
        "timeframe",
        "steps",
    ]
    csv_file = open(visualize_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    csv_writer.writeheader()

    while running and (experiment_mode or episode < episodes):
        state = env.reset()
        if random_generator is not None:
            random_generator.append_all_obstacles(env)
            state = env._get_state()
        total_reward = 0
        done = False
        close_distance_total = 0
        episode += 1

        if allstage:
            print(
                f"\nStarting Episode {episode} (Stage {current_stage + 1}/{num_stages}, streak {consecutive_success}/{ALLSTAGE_CONSECUTIVE_REQ})"
            )
        else:
            print(f"\nStarting Episode {episode}")

        start_ticks = pygame.time.get_ticks()
        step_count = 0
        action = 4 if manual_mode else 1
        nn_output = None
        neuron_data = None


        current_vis_action = action

        while running and not done:
            running, reset, manual_action, pause_toggle, experiment_actions = (
                renderer.handle_events(paused)
            )

            if planner is not None and experiment_actions:
                for experiment_action in experiment_actions:
                    action_type = experiment_action.get("type")
                    if action_type == "toggle_lane":
                        planner.toggle_lane(experiment_action.get("lane"))
                    elif action_type == "distance_minus":
                        planner.decrement_distance()
                    elif action_type == "distance_plus":
                        planner.increment_distance()
                    elif action_type == "add_list":
                        if planner.add_current_selection():
                            latest = planner.to_spawn_list[-1]
                            lanes_text = "+".join(latest["lanes"])
                            print(
                                f"[Experiment] Added: lanes={lanes_text}, distance={latest['distance']}"
                            )
                        else:
                            print("[Experiment] Select at least one path before ADD LIST.")
                    elif action_type == "spawn":
                        spawn_plan = planner.build_spawn_plan(env.car_y)
                        obstacle_configs = planner.build_obstacle_configs(env.car_y)
                        if obstacle_configs:
                            added_count = env.append_obstacles(obstacle_configs)
                            print(
                                f"[Experiment] Spawned {added_count} obstacle(s) from {len(spawn_plan)} list item(s)."
                            )
                        else:
                            print("[Experiment] To Spawn Lists is empty.")
                    elif action_type == "clear_spawn":
                        planner.clear_spawn_list()
                        print("[Experiment] Cleared To Spawn Lists.")

            if pause_toggle:
                paused = not paused
                print(f"{'PAUSED' if paused else 'RESUMED'}")

            if paused:
                paused_info = env.render_info()
                if nn_output is not None:
                    paused_info["nn_output"] = nn_output
                paused_info["last_action"] = action
                renderer.render(
                    paused_info,
                    episode,
                    total_reward,
                    agent.epsilon,
                    step_count,
                    paused=True,
                    experiment_data=planner.snapshot() if planner else None,
                    neuron_data=neuron_data,
                )
                continue

            if reset:
                break


            steps_this_frame = renderer.get_steps_per_frame()
            last_action = action

            for _ in range(steps_this_frame):
                if manual_mode:
                    if manual_action is not None:
                        last_action = manual_action
                    if last_action is None:
                        last_action = 4
                else:

                    if step_count % DECISION_INTERVAL == 0:
                        current_vis_action = agent.select_action(state, training=False)
                    last_action = current_vis_action


                is_decision_step = step_count % DECISION_INTERVAL == 0
                next_state, reward, done, info = env.step(
                    last_action, apply_steering=is_decision_step
                )
                if random_generator is not None:
                    random_generator.append_due_obstacles(env)
                    next_state = env._get_state()

                total_reward += reward
                close_distance_total += int(info.get("warning_close_count", 0))
                state = next_state
                step_count += 1

                if done:
                    break

            action = last_action


            try:
                nn_output = agent.get_q_values(state)
            except Exception:
                nn_output = None
            try:
                neuron_data = build_neuron_trace(agent, state) if neuron_mode else None
            except Exception as e:
                neuron_data = {"error": str(e)}

            render_info = env.render_info()
            if nn_output is not None:
                render_info["nn_output"] = nn_output
            render_info["last_action"] = action
            renderer.render(
                render_info,
                episode,
                total_reward,
                agent.epsilon,
                step_count,
                paused,
                planner.snapshot() if planner else None,
                neuron_data,
            )

        if done:
            end_ticks = pygame.time.get_ticks()
            duration_ms = end_ticks - start_ticks
            seconds = int(duration_ms / 1000)
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"

            finish_y = info.get("finish_line_y", getattr(env, "finish_line_y", 2000))
            raw_prog = (info["car_y"] / finish_y) * 100.0 if finish_y != 0 else 0.0
            progress_pct = (
                100.0 if info.get("reached_finish") else min(95.0, max(0.0, raw_prog))
            )
            mse_value = 0.0
            visualize_rewards.append(float(total_reward))
            avg_reward_all = (
                float(sum(visualize_rewards) / len(visualize_rewards))
                if visualize_rewards
                else 0.0
            )
            world_step = int(
                round(float(info.get("world_distance", getattr(env, "world_distance", 0.0))))
            )

            csv_writer.writerow(
                build_visualize_episode_row(
                    episode=episode,
                    close_distance=close_distance_total,
                    mse=mse_value,
                    reward=total_reward,
                    avg_reward=avg_reward_all,
                    time_ms=duration_ms,
                    timeframe=step_count,
                    steps=world_step,
                )
            )
            csv_file.flush()

            print(f"Episode {episode} finished!")
            print(f"  Total Reward: {total_reward:.2f}")
            print("  MSE: 0.000 (Inference)")
            print(f"  Close Distance: {close_distance_total}")
            print(f"  Avg Reward: {avg_reward_all:.2f}")
            print(f"  Time: {time_str}")
            print(f"  Timeframe: {step_count}")
            print(f"  Step: {world_step}")
            print(f"  Progress: {progress_pct:.1f}%")
            print(f"  Final Position: ({info['car_x']:.1f}, {info['car_y']:.1f})")
            print(
                f"  CSV Row -> episode={episode}, close distance={close_distance_total}, "
                f"MSE={mse_value:.3f}, Reward={float(total_reward):.3f}, Avg reward={avg_reward_all:.3f}, "
                f"time(ms)={int(duration_ms)}, timeframe={int(step_count)}, steps={int(world_step)}"
            )

            if allstage:
                if info.get("reached_finish"):
                    consecutive_success += 1
                else:
                    consecutive_success = 0


                if (
                    current_stage < (num_stages - 1)
                    and consecutive_success >= ALLSTAGE_CONSECUTIVE_REQ
                ):
                    current_stage += 1
                    consecutive_success = 0
                    try:
                        env.set_curriculum_stage(current_stage)
                        renderer.env = env
                    except Exception as e:
                        print(f"Warning: could not advance stage: {e}")
                    else:
                        print(f"[OK] ADVANCED TO STAGE {current_stage + 1}/{num_stages}")


            end_pause_ms = int(max(50, 1000 / max(1, renderer.get_steps_per_frame())))
            pygame.time.wait(end_pause_ms)

    csv_file.close()
    renderer.close()
    print("\nVisualization ended.")


def run_speedtest_visualization():
    """
    Run endless-road speed test mode.

    Controls:
      - Up Arrow: queue FAST straight decision (action 4)
      - Down Arrow: queue SLOW straight decision (action 1)
      - Queued decision is applied only at the next decision boundary
        (step % DECISION_INTERVAL == 0).
    """
    env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
    env.max_steps = 10**12
    renderer = GameRenderer(env, scale=DEFAULT_SCALE, experiment_mode=False)

    print("\n=== Speed Test Mode ===")
    print("Controls:")
    print("  UP: Queue FAST straight decision")
    print("  DOWN: Queue SLOW straight decision")
    print("  P: Pause/Resume")
    print("  R: Reset")
    print(f"  1: Toggle speed x{KEYONE_MULTIPLIER}")
    print("  2: Toggle slow motion")
    print("  Q: Quit")
    print("=" * 30)

    state = env.reset()
    total_reward = 0.0
    step_count = 0
    episode = 1
    running = True
    paused = False
    nn_output = None


    current_decision_action = 4
    pending_decision_action = 4

    def _decision_label(action_id):
        return "FAST" if int(action_id) == 4 else "SLOW"

    while running:
        running, reset, manual_action, pause_toggle, _experiment_actions = (
            renderer.handle_events(paused)
        )

        if manual_action in [1, 4]:
            pending_decision_action = int(manual_action)
            print(
                f"[SpeedTest] Queued {_decision_label(pending_decision_action)} decision "
                f"(applies at next decision step)."
            )

        if pause_toggle:
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")

        if reset:
            state = env.reset()
            total_reward = 0.0
            step_count = 0
            current_decision_action = pending_decision_action
            nn_output = None
            print("[SpeedTest] Reset.")
            continue

        if paused:
            paused_info = env.render_info()
            if nn_output is not None:
                paused_info["nn_output"] = nn_output
            paused_info["last_action"] = current_decision_action
            renderer.render(
                paused_info,
                episode,
                total_reward,
                0.0,
                step_count,
                paused=True,
                experiment_data=None,
            )
            continue

        steps_this_frame = renderer.get_steps_per_frame()
        for _ in range(steps_this_frame):
            is_decision_step = step_count % DECISION_INTERVAL == 0
            if is_decision_step:
                prev_action = current_decision_action
                current_decision_action = pending_decision_action
                if current_decision_action != prev_action:
                    print(
                        f"[SpeedTest] Timeframe {step_count}: applied "
                        f"{_decision_label(current_decision_action)} decision."
                    )

            next_state, reward, done, _info = env.step(
                current_decision_action, apply_steering=is_decision_step
            )
            state = next_state
            total_reward += reward
            step_count += 1

            if done:

                state = env.reset()
                total_reward = 0.0
                step_count = 0
                current_decision_action = pending_decision_action
                print("[SpeedTest] Auto-reset after terminal state.")
                break

        render_info = env.render_info()
        if nn_output is not None:
            render_info["nn_output"] = nn_output
        render_info["last_action"] = current_decision_action
        renderer.render(
            render_info,
            episode,
            total_reward,
            0.0,
            step_count,
            paused=False,
            experiment_data=None,
        )

    renderer.close()
    print("\nSpeed test ended.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN Car Navigation Visualization")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--manual", action="store_true", help="Use manual keyboard control"
    )
    parser.add_argument(
        "--allstage",
        action="store_true",
        help="Run visualization across all curriculum stages (advance after consecutive finishes)",
    )
    parser.add_argument(
        "--tester",
        action="store_true",
        help="Use TEST_OBSTACLES from main_constant.py instead of OBSTACLES",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Enable obstacle experiment mode (no finish line, custom spawn controls)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Enable finite random obstacles (startRandom/gapRandom/maxRandom, 1-2 vehicles per row)",
    )
    parser.add_argument(
        "--neuron",
        action="store_true",
        help="Show detailed neural-network forward-pass panel (input, weights, bias, hidden activations, Q-values)",
    )
    parser.add_argument(
        "--speedtest",
        action="store_true",
        help="Run endless speed test (Up/Down queue fast/slow straight decisions at interval boundaries)",
    )

    args = parser.parse_args()

    if args.speedtest:
        run_speedtest_visualization()
    else:
        run_visualization(
            model_path=args.model,
            episodes=args.episodes,
            manual_mode=args.manual,
            allstage=args.allstage,
            tester=args.tester,
            experiment=args.experiment,
            random_mode=args.random,
            neuron_mode=args.neuron,
        )
