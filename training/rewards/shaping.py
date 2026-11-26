"""Extrinsic Reward Shaping, Potential-Based Shaping, and Anti-Stuck Detection.

Implements all extrinsic reward events from PLAN.md Phase 4:
  - Exit reached: +10.0 (+ health and time bonus)
  - Keycard pickup: +2.0
  - Door opened: +0.5
  - Locked door unlocked: +2.0
  - New area (unique cell visited): +0.02 (with episode cap)
  - Enemy killed: +0.5
  - Damage taken: -0.005 per HP
  - Death: -3.0
  - Step penalty: -0.001
  - Stuck penalty: -0.05 when agent displacement < threshold over N steps
  - Potential-based shaping: gamma * Phi(s') - Phi(s) for key/exit navigation
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class RewardBreakdown:
    exit: float = 0.0
    key: float = 0.0
    door: float = 0.0
    locked_door: float = 0.0
    exploration: float = 0.0
    kill: float = 0.0
    damage: float = 0.0
    death: float = 0.0
    time: float = 0.0
    stuck: float = 0.0
    potential: float = 0.0
    curiosity: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.exit
            + self.key
            + self.door
            + self.locked_door
            + self.exploration
            + self.kill
            + self.damage
            + self.death
            + self.time
            + self.stuck
            + self.potential
            + self.curiosity
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "exit": round(self.exit, 4),
            "key": round(self.key, 4),
            "door": round(self.door, 4),
            "locked_door": round(self.locked_door, 4),
            "exploration": round(self.exploration, 4),
            "kill": round(self.kill, 4),
            "damage": round(self.damage, 4),
            "death": round(self.death, 4),
            "time": round(self.time, 4),
            "stuck": round(self.stuck, 4),
            "potential": round(self.potential, 4),
            "curiosity": round(self.curiosity, 4),
            "total": round(self.total, 4),
        }


@dataclass
class RewardConfig:
    exit_reward: float = 10.0
    key_reward: float = 2.0
    door_reward: float = 0.5
    locked_door_reward: float = 2.0
    cell_reward: float = 0.02
    max_cell_reward_per_episode: float = 5.0
    kill_reward: float = 0.5
    damage_penalty_per_hp: float = -0.005
    death_penalty: float = -3.0
    step_penalty: float = -0.001
    stuck_penalty: float = -0.05
    stuck_window: int = 16
    stuck_min_dist: float = 16.0
    cell_size: float = 64.0
    gamma: float = 0.995


class ExtrinsicRewardShaper:
    """Computes dense extrinsic rewards from privileged state and game events."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        self.visited_cells: set[tuple[int, int]] = set()
        self.cell_reward_accum: float = 0.0
        self.last_kills: int = 0
        self.last_health: float = 100.0
        self.last_damage_taken: float = 0.0
        self.keys_held: set[str] = set()
        self.open_doors: set[int] = set()
        self.unlocked_doors: set[int] = set()

        # Stuck detector
        self.position_history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=self.config.stuck_window
        )
        self.last_potential: float | None = None
        self.last_target_coords: tuple[float, float] | None = None

    def compute(
        self,
        obs: Mapping[str, Any],
        privileged_state: Mapping[str, Any] | None,
        base_reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        target_coords: tuple[float, float] | None = None,
    ) -> tuple[float, RewardBreakdown]:
        """Calculates shaped reward and per-component breakdown."""
        breakdown = RewardBreakdown()
        breakdown.time = self.config.step_penalty

        if privileged_state is None:
            return breakdown.total, breakdown

        curr_x = float(privileged_state.get("x", 0.0))
        curr_y = float(privileged_state.get("y", 0.0))
        curr_health = float(privileged_state.get("health", 100.0))
        curr_kills = int(privileged_state.get("kills", privileged_state.get("killcount", 0)))

        # 1. Spatial cell exploration
        cx = int(curr_x // self.config.cell_size)
        cy = int(curr_y // self.config.cell_size)
        if (cx, cy) not in self.visited_cells:
            self.visited_cells.add((cx, cy))
            if self.cell_reward_accum < self.config.max_cell_reward_per_episode:
                breakdown.exploration = self.config.cell_reward
                self.cell_reward_accum += self.config.cell_reward

        # 2. Combat: Kills & Damage
        if curr_kills > self.last_kills:
            breakdown.kill = (curr_kills - self.last_kills) * self.config.kill_reward
            self.last_kills = curr_kills

        if curr_health < self.last_health:
            damage = self.last_health - curr_health
            breakdown.damage = damage * self.config.damage_penalty_per_hp
        self.last_health = curr_health

        # 3. Keys: array of 6 flags or dict
        keys_priv = privileged_state.get("keys")
        if isinstance(keys_priv, np.ndarray):
            for kidx, val in enumerate(keys_priv):
                if val > 0 and kidx not in self.keys_held:
                    self.keys_held.add(kidx)
                    breakdown.key += self.config.key_reward
        elif isinstance(keys_priv, dict):
            for k, val in keys_priv.items():
                if val and k not in self.keys_held:
                    self.keys_held.add(k)
                    breakdown.key += self.config.key_reward

        # 4. Doors & Locked Doors: checked via sector_ceiling - sector_floor and door_lock
        door_locks = privileged_state.get("door_lock")
        sector_ceilings = privileged_state.get("sector_ceiling")
        sector_floors = privileged_state.get("sector_floor")
        if isinstance(door_locks, np.ndarray) and isinstance(sector_ceilings, np.ndarray):
            for sec_id, lock_val in enumerate(door_locks):
                if lock_val >= 0:  # is a door
                    floor_h = (
                        float(sector_floors[sec_id])
                        if isinstance(sector_floors, np.ndarray)
                        else 0.0
                    )
                    opening = float(sector_ceilings[sec_id]) - floor_h
                    if opening > 16.0:  # door is raised / opened
                        if sec_id not in self.open_doors:
                            self.open_doors.add(sec_id)
                            breakdown.door += self.config.door_reward
                        if lock_val > 0 and sec_id not in self.unlocked_doors:
                            self.unlocked_doors.add(sec_id)
                            breakdown.locked_door += self.config.locked_door_reward
        elif "doors" in privileged_state:
            doors_state = privileged_state["doors"]
            for sec_id, dinfo in doors_state.items():
                is_open = dinfo.get("is_open", False)
                lock = dinfo.get("lock", 0)
                if is_open:
                    if sec_id not in self.open_doors:
                        self.open_doors.add(sec_id)
                        breakdown.door += self.config.door_reward
                    if lock > 0 and sec_id not in self.unlocked_doors:
                        self.unlocked_doors.add(sec_id)
                        breakdown.locked_door += self.config.locked_door_reward

        # 5. Exit Reached
        if info.get("exit_reached", False):
            time_bonus = max(0.0, 1.0 - (len(self.visited_cells) / 2000.0))
            health_bonus = (curr_health / 100.0) * 1.0
            breakdown.exit = self.config.exit_reward + time_bonus + health_bonus

        # 6. Death
        if terminated and not info.get("exit_reached", False) and curr_health <= 0:
            breakdown.death = self.config.death_penalty

        # 7. Anti-Stuck Detector
        self.position_history.append((curr_x, curr_y))
        if len(self.position_history) == self.config.stuck_window:
            p0 = self.position_history[0]
            disp = math.hypot(curr_x - p0[0], curr_y - p0[1])
            if disp < self.config.stuck_min_dist:
                breakdown.stuck = self.config.stuck_penalty

        # 8. Potential-Based Shaping
        if target_coords is not None:
            dist_to_target = math.hypot(curr_x - target_coords[0], curr_y - target_coords[1])
            curr_potential = -dist_to_target / 1000.0
            if self.last_potential is not None and self.last_target_coords == target_coords:
                breakdown.potential = self.config.gamma * curr_potential - self.last_potential
            self.last_potential = curr_potential
            self.last_target_coords = target_coords
        else:
            self.last_potential = None
            self.last_target_coords = None

        return breakdown.total, breakdown
