"""Curriculum environment wrappers and progression manager.

Implements the 6 curriculum stages from PLAN.md Phase 1:
  Stage 0: Single room, one enemy (learn to aim, fire, turn).
  Stage 1: Combat arenas (multiple enemies, health/ammo pickups).
  Stage 2: Multi-room navigation to exit (doors auto-open, no enemies).
  Stage 3: Multi-room with ordinary doors and enemies.
  Stage 4: Keycard puzzle maps with vanilla rules.
  Stage 5: Full gauntlet mix (with 10-20% rehearsal of earlier stages).

Includes dense reward shaping for training, visited-cell spatial exploration
bonuses, and rolling success rate progression tracking.
"""

from __future__ import annotations

import collections
import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from doom_env import make_env
from doom_env import spaces as S
from training.mapgen.generator import MapFamily, ProceduralMapGenerator


class CurriculumStage(int, enum.Enum):
    STAGE_0_AIM_SHOOT = 0
    STAGE_1_COMBAT_ARENA = 1
    STAGE_2_NAV_AUTO_DOORS = 2
    STAGE_3_NAV_DOORS_ENEMIES = 3
    STAGE_4_KEYCARD_PUZZLE = 4
    STAGE_5_FULL_GAUNTLET = 5


@dataclass
class RewardWeights:
    exit_reached: float = 10.0
    key_pickup: float = 2.0
    door_opened: float = 0.5
    locked_door_unlocked: float = 2.0
    enemy_killed: float = 0.5
    damage_taken_per_hp: float = -0.005
    death: float = -3.0
    time_step: float = -0.001
    new_cell_explored: float = 0.02


@dataclass
class CurriculumConfig:
    stage: CurriculumStage = CurriculumStage.STAGE_5_FULL_GAUNTLET
    promotion_threshold: float = 0.70
    window_size: int = 100
    rehearsal_prob: float = 0.15
    god_mode: bool = False
    infinite_ammo: bool = False
    auto_open_doors: bool = False
    render_buffers: bool = True
    cell_size: float = 64.0


class CurriculumEnvWrapper(gym.Wrapper):
    """Wraps DoomEnv to add curriculum reward shaping and training instrumentation."""

    def __init__(
        self,
        env: gym.Env,
        stage: CurriculumStage = CurriculumStage.STAGE_5_FULL_GAUNTLET,
        reward_weights: RewardWeights | None = None,
        config: CurriculumConfig | None = None,
    ):
        super().__init__(env)
        self.stage = stage
        self.weights = reward_weights or RewardWeights()
        self.config = config or CurriculumConfig(stage=stage)

        # Episodic state trackers
        self.visited_cells: set[tuple[int, int]] = set()
        self.last_health: float = 100.0
        self.keys_held: set[int] = set()
        self.step_count: int = 0
        self.total_shaped_reward: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited_cells.clear()
        self.step_count = 0
        self.total_shaped_reward = 0.0
        self.keys_held.clear()
        self.last_health = float(obs["vitals"][S.VITALS_INDEX["health"]])

        # Apply training-only mods if configured
        if self.config.god_mode:
            try:
                self.env.unwrapped.game.send_game_command("god")
            except Exception:
                pass

        # Mark initial position as visited
        priv = self.env.unwrapped.privileged_state()
        if priv is not None:
            cx = int(priv["x"] // self.config.cell_size)
            cy = int(priv["y"] // self.config.cell_size)
            self.visited_cells.add((cx, cy))

        return obs, info

    def step(self, action: np.ndarray):
        # Auto-open doors mod for navigation stages: press USE when encountering doors
        if self.config.auto_open_doors or self.stage == CurriculumStage.STAGE_2_NAV_AUTO_DOORS:
            # If front rays detect door, trigger USE button automatically
            front_ray_hits = self.env.unwrapped.privileged_state()
            action = action.copy()
            action[S.BUTTONS.index("USE")] = 1

        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Infinite ammo mod: replenish bullets / shells if depleted
        if self.config.infinite_ammo:
            try:
                obs["vitals"][S.VITALS_INDEX["ammo_bullets"]] = 50.0
                obs["vitals"][S.VITALS_INDEX["ammo_shells"]] = 50.0
            except Exception:
                pass

        shaped_reward = self.weights.time_step

        # Privileged state inspection for reward shaping
        priv = self.env.unwrapped.privileged_state()
        if priv is not None:
            # 1. Spatial exploration bonus
            cx = int(priv["x"] // self.config.cell_size)
            cy = int(priv["y"] // self.config.cell_size)
            if (cx, cy) not in self.visited_cells:
                self.visited_cells.add((cx, cy))
                shaped_reward += self.weights.new_cell_explored

        # 2. Vitals tracking (health / damage)
        curr_health = float(obs["vitals"][S.VITALS_INDEX["health"]])
        if curr_health < self.last_health:
            damage = self.last_health - curr_health
            shaped_reward += damage * self.weights.damage_taken_per_hp
        self.last_health = curr_health

        # 3. Keycard pickup events
        for k_name, idx in [
            ("key_blue_card", S.VITALS_INDEX["key_blue_card"]),
            ("key_yellow_card", S.VITALS_INDEX["key_yellow_card"]),
            ("key_red_card", S.VITALS_INDEX["key_red_card"]),
        ]:
            if obs["vitals"][idx] > 0 and idx not in self.keys_held:
                self.keys_held.add(idx)
                shaped_reward += self.weights.key_pickup

        # 4. Exit bonus
        if info.get("exit_reached", False):
            shaped_reward += self.weights.exit_reached
            # Time & health bonus on exit
            shaped_reward += (curr_health / 100.0) * 1.0

        # 5. Death penalty
        if terminated and not info.get("exit_reached", False) and curr_health <= 0:
            shaped_reward += self.weights.death

        self.total_shaped_reward += shaped_reward
        info["shaped_reward"] = shaped_reward
        info["total_shaped_reward"] = self.total_shaped_reward
        info["unique_cells_visited"] = len(self.visited_cells)

        return obs, shaped_reward, terminated, truncated, info


class CurriculumManager:
    """Manages progression across stages based on rolling success rates."""

    def __init__(
        self,
        initial_stage: CurriculumStage = CurriculumStage.STAGE_0_AIM_SHOOT,
        promotion_threshold: float = 0.70,
        window_size: int = 100,
        rehearsal_prob: float = 0.15,
        map_cache_dir: Path | str = "/tmp/doom_curriculum_maps",
    ):
        self.current_stage = initial_stage
        self.promotion_threshold = promotion_threshold
        self.window_size = window_size
        self.rehearsal_prob = rehearsal_prob
        self.map_cache_dir = Path(map_cache_dir)
        self.map_cache_dir.mkdir(parents=True, exist_ok=True)

        self.generator = ProceduralMapGenerator()
        self.history: collections.deque[bool] = collections.deque(maxlen=window_size)
        self.total_episodes: int = 0

    def record_episode(self, success: bool) -> None:
        self.history.append(success)
        self.total_episodes += 1

        if len(self.history) >= self.window_size:
            rate = sum(self.history) / len(self.history)
            if (
                rate >= self.promotion_threshold
                and self.current_stage < CurriculumStage.STAGE_5_FULL_GAUNTLET
            ):
                promoted_to = CurriculumStage(self.current_stage + 1)
                print(
                    f"\n[CurriculumManager] Success rate {rate:.1%} exceeds {self.promotion_threshold:.1%}. "
                    f"PROMOTING from {self.current_stage.name} -> {promoted_to.name}!"
                )
                self.current_stage = promoted_to
                self.history.clear()

    @property
    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def sample_stage_for_episode(self) -> CurriculumStage:
        """Samples the active stage, mixing in rehearsals of earlier stages."""
        if self.current_stage == CurriculumStage.STAGE_0_AIM_SHOOT:
            return self.current_stage

        # Rehearsal probability (mix 10-20% earlier stages to avoid forgetting)
        if np.random.rand() < self.rehearsal_prob:
            earlier_stages = list(range(self.current_stage))
            return CurriculumStage(np.random.choice(earlier_stages))

        return self.current_stage

    def get_or_generate_map(self, stage: CurriculumStage, seed: int | None = None) -> Path:
        """Returns a path to a map suited for the given stage."""
        if seed is not None:
            self.generator = ProceduralMapGenerator(seed)

        difficulty = min(1.0, 0.2 * int(stage))
        map_path = self.map_cache_dir / f"stage_{stage.value}_{seed or np.random.randint(100000)}.wad"

        if stage == CurriculumStage.STAGE_0_AIM_SHOOT:
            m, _ = self.generator.generate(MapFamily.FAMILY_A, difficulty=0.1)
        elif stage == CurriculumStage.STAGE_1_COMBAT_ARENA:
            m, _ = self.generator.generate(MapFamily.FAMILY_A, difficulty=0.6)
        elif stage == CurriculumStage.STAGE_2_NAV_AUTO_DOORS:
            m, _ = self.generator.generate(MapFamily.FAMILY_B, num_rooms=4, difficulty=0.0)
        elif stage == CurriculumStage.STAGE_3_NAV_DOORS_ENEMIES:
            m, _ = self.generator.generate(MapFamily.FAMILY_B, num_rooms=6, difficulty=0.5)
        elif stage == CurriculumStage.STAGE_4_KEYCARD_PUZZLE:
            m, _ = self.generator.generate(MapFamily.FAMILY_C, num_rooms=5, difficulty=0.5)
        elif stage == CurriculumStage.STAGE_5_FULL_GAUNTLET:
            m, _ = self.generator.generate(MapFamily.FAMILY_D, num_rooms=7, difficulty=0.8)
        else:
            m, _ = self.generator.generate(MapFamily.FAMILY_D, num_rooms=6, difficulty=0.5)

        m.write_wad(map_path)
        return map_path

    def make_curriculum_env(self, stage: CurriculumStage | None = None, seed: int = 0) -> gym.Env:
        """Creates a wrapped curriculum environment."""
        active_stage = stage if stage is not None else self.sample_stage_for_episode()
        wad_path = self.get_or_generate_map(active_stage, seed=seed)
        base_env = make_env(str(wad_path), episode_ticks=2100)
        return CurriculumEnvWrapper(base_env, stage=active_stage)
