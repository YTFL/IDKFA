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
    ammo_waste: float = 0.0
    wall_hug: float = 0.0
    action_hygiene: float = 0.0
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
            + self.ammo_waste
            + self.wall_hug
            + self.action_hygiene
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
            "ammo_waste": round(self.ammo_waste, 4),
            "wall_hug": round(self.wall_hug, 4),
            "action_hygiene": round(self.action_hygiene, 4),
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
    heal_reward_per_hp: float = 0.0
    death_penalty: float = -3.0
    step_penalty: float = -0.001
    stuck_penalty: float = -0.05
    stuck_window: int = 16
    stuck_min_dist: float = 16.0
    long_stuck_window: int = 48
    long_stuck_min_dist: float = 96.0
    spin_window: int = 32
    spin_min_deg: float = 300.0
    spin_max_disp: float = 80.0
    dry_fire_penalty: float = -0.08
    wasted_ammo_penalty: float = -0.03
    wall_hug_penalty: float = -0.03
    wall_hug_ray_threshold: float = 0.045
    held_use_penalty: float = -0.04
    conflict_action_penalty: float = -0.02
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

        # Stuck & spin detectors
        self.position_history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=self.config.stuck_window
        )
        self.long_position_history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=max(1, self.config.long_stuck_window)
        )
        self.yaw_delta_history: collections.deque[float] = collections.deque(
            maxlen=max(1, self.config.spin_window)
        )
        self.last_angle: float | None = None
        self.last_xy: tuple[float, float] | None = None
        self.last_ammo: float | None = None
        self.prev_use_pressed: int = 0
        self.last_potential: float | None = None
        self.last_target_coords: tuple[float, float] | None = None

    @staticmethod
    def _has_front_enemy_or_barrel(obs: Mapping[str, Any]) -> tuple[bool, bool]:
        """Return `(target_in_front, melee_contact)` from raw obs dict."""
        if not obs:
            return False, False
        rays = obs.get("rays") or {}
        hits = rays.get("hit_type")
        enemy_ray = False
        if hits is not None and len(hits) >= 64:
            for idx in (0, 1, 2, 3, 4, 60, 61, 62, 63):
                if int(hits[idx]) == 4:
                    enemy_ray = True
                    break

        ent = obs.get("visible_entities") or {}
        feats = ent.get("features")
        etypes = ent.get("entity_type")
        emask = ent.get("mask")
        target_in_front = enemy_ray
        melee_contact = False
        if feats is not None and etypes is not None and emask is not None:
            for i in range(len(emask)):
                if int(emask[i]) == 0:
                    continue
                et = int(etypes[i])
                d = float(feats[i][0])
                c = float(feats[i][2])
                if et in (1, 2, 3) and c > 0.35:
                    target_in_front = True
                elif et == 10 and d <= 0.35 and c > 0.5:
                    target_in_front = True
                if et in (1, 2) and d <= 0.11 and c > 0.5:
                    melee_contact = True
        return target_in_front, melee_contact

    def compute(
        self,
        obs: Mapping[str, Any],
        privileged_state: Mapping[str, Any] | None,
        base_reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        target_coords: tuple[float, float] | None = None,
        action: np.ndarray | None = None,
        pre_obs: Mapping[str, Any] | None = None,
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
        got_kill = curr_kills > self.last_kills
        if got_kill:
            breakdown.kill = (curr_kills - self.last_kills) * self.config.kill_reward
            self.last_kills = curr_kills

        if curr_health < self.last_health:
            damage = self.last_health - curr_health
            breakdown.damage = damage * self.config.damage_penalty_per_hp
        elif curr_health > self.last_health and self.config.heal_reward_per_hp > 0.0:
            healed = curr_health - self.last_health
            breakdown.damage = healed * self.config.heal_reward_per_hp
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

        # 7. Anti-Stuck, Spin-Loop, and Wall-Hug Detection
        step_disp = (
            math.hypot(curr_x - self.last_xy[0], curr_y - self.last_xy[1])
            if self.last_xy is not None
            else 0.0
        )
        self.last_xy = (curr_x, curr_y)

        self.position_history.append((curr_x, curr_y))
        if len(self.position_history) == self.config.stuck_window:
            p0 = self.position_history[0]
            disp = math.hypot(curr_x - p0[0], curr_y - p0[1])
            if disp < self.config.stuck_min_dist:
                breakdown.stuck = self.config.stuck_penalty

        self.long_position_history.append((curr_x, curr_y))
        if (
            breakdown.stuck == 0.0
            and len(self.long_position_history) == self.config.long_stuck_window
        ):
            p_long = self.long_position_history[0]
            long_disp = math.hypot(curr_x - p_long[0], curr_y - p_long[1])
            if long_disp < self.config.long_stuck_min_dist:
                breakdown.stuck = self.config.stuck_penalty

        if "angle" in privileged_state:
            curr_angle = float(privileged_state["angle"])
            if self.last_angle is not None:
                dyaw = ((curr_angle - self.last_angle + 180.0) % 360.0) - 180.0
                self.yaw_delta_history.append(dyaw)
            self.last_angle = curr_angle
            if (
                breakdown.stuck == 0.0
                and len(self.yaw_delta_history) == self.config.spin_window
                and len(self.long_position_history) >= self.config.spin_window
            ):
                net_yaw = abs(sum(self.yaw_delta_history))
                idx_spin = len(self.long_position_history) - self.config.spin_window
                p_spin = self.long_position_history[idx_spin]
                spin_disp = math.hypot(curr_x - p_spin[0], curr_y - p_spin[1])
                if net_yaw >= self.config.spin_min_deg and spin_disp < self.config.spin_max_disp:
                    breakdown.stuck = self.config.stuck_penalty

        # Close front-wall grinding penalty (front 3 rays < threshold with near-zero progress)
        ref_obs = pre_obs if pre_obs is not None else obs
        if ref_obs and not got_kill:
            rays_d = (ref_obs.get("rays") or {}).get("distance")
            if rays_d is not None and len(rays_d) >= 64:
                min_front = min(float(rays_d[0]), float(rays_d[1]), float(rays_d[63]))
                if min_front < self.config.wall_hug_ray_threshold and step_disp < 4.0:
                    breakdown.wall_hug = self.config.wall_hug_penalty

        # 8. Ammo-Waste & Action-Hygiene Penalties
        vitals_curr = (obs.get("vitals") if obs else None)
        curr_ammo = float(vitals_curr[3]) if vitals_curr is not None and len(vitals_curr) > 3 else None
        vitals_pre = (ref_obs.get("vitals") if ref_obs else None)
        pre_ammo = float(vitals_pre[3]) if vitals_pre is not None and len(vitals_pre) > 3 else self.last_ammo

        has_target, melee_contact = self._has_front_enemy_or_barrel(ref_obs)
        if action is not None and len(action) >= 9:
            act_i = np.asarray(action, dtype=np.int64)
            # Dry-fire: pressing ATTACK when active weapon has 0 ammo and no close melee enemy
            if act_i[6] == 1 and pre_ammo is not None and pre_ammo <= 0.0 and not melee_contact:
                breakdown.ammo_waste += self.config.dry_fire_penalty
            # Held edge-triggered USE
            if act_i[7] == 1 and self.prev_use_pressed == 1:
                breakdown.action_hygiene += self.config.held_use_penalty
            self.prev_use_pressed = int(act_i[7])
            # Simultaneous conflicting axes
            if (act_i[0] and act_i[1]) or (act_i[2] and act_i[3]) or (act_i[4] and act_i[5]):
                breakdown.action_hygiene += self.config.conflict_action_penalty

        # Wasted shot: ammo consumed without kill and without any front enemy/barrel target
        if (
            self.last_ammo is not None
            and curr_ammo is not None
            and curr_ammo < self.last_ammo - 1e-6
            and not got_kill
            and not has_target
        ):
            breakdown.ammo_waste += self.config.wasted_ammo_penalty
        if curr_ammo is not None:
            self.last_ammo = curr_ammo

        # 9. Potential-Based Shaping
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
