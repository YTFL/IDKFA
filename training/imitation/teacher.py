"""Training-only BFS + ray-steering teacher oracle for Behavioral Cloning data collection.

Uses `env.privileged_state()`, procedural map room connectivity metadata (`room_centers`,
`room_bounds`, `nav_edges`, `key_xy`, `exit_xy`), and ego-centric rays/entities.
Strictly prohibited inside `submission/`; lives solely in `training/`.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Mapping

import numpy as np

from submission.action_mask import compute_obs_action_mask_np_arrays
from submission.features import ACTION_DIM

_FRONT_DOOR_RAYS = (0, 1, 2, 62, 63)
_FRONT_WALL_RAYS = (0, 1, 2, 3, 61, 62, 63)
_LEFT_CLEAR_RAYS = tuple(range(4, 15))
_RIGHT_CLEAR_RAYS = tuple(range(50, 61))


def _wrap_angle_rad(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


class PrivilegedTeacherOracle:
    """Solves procedural combat, exploration, door, and keycard maps using privileged metadata."""

    def __init__(self, map_meta: Mapping[str, Any] | None = None, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.map_meta: dict[str, Any] = {}
        self.room_centers: dict[int, tuple[float, float]] = {}
        self.room_bounds: dict[int, tuple[float, float, float, float]] = {}
        self.nav_edges: list[tuple[int, int, int]] = []
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.int8)
        self.use_cooldown = 0
        self.door_pause_ticks = 0
        self.unstuck_ticks = 0
        self.unstuck_strafe_left = True
        self.pos_hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=20)
        self.patrol_angle = 0.0
        self.reset(map_meta)

    def reset(self, map_meta: Mapping[str, Any] | None = None) -> None:
        self.map_meta = dict(map_meta or {})
        raw_centers = self.map_meta.get("room_centers") or {}
        raw_bounds = self.map_meta.get("room_bounds") or {}
        self.room_centers = {int(k): (float(v[0]), float(v[1])) for k, v in raw_centers.items()}
        self.room_bounds = {
            int(k): (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            for k, v in raw_bounds.items()
        }
        self.nav_edges = [(int(u), int(v), int(lock)) for (u, v, lock) in (self.map_meta.get("nav_edges") or [])]
        self.prev_action.fill(0)
        self.use_cooldown = 0
        self.door_pause_ticks = 0
        self.unstuck_ticks = 0
        self.unstuck_strafe_left = bool(self.rng.integers(0, 2))
        self.pos_hist.clear()
        self.patrol_angle = float(self.rng.uniform(-math.pi, math.pi))

    def _find_room(self, x: float, y: float) -> int | None:
        if not self.room_centers:
            return None
        for rid, (rx1, ry1, rx2, ry2) in self.room_bounds.items():
            if rx1 - 8.0 <= x <= rx2 + 8.0 and ry1 - 8.0 <= y <= ry2 + 8.0:
                return rid
        return min(
            self.room_centers,
            key=lambda rid: math.hypot(x - self.room_centers[rid][0], y - self.room_centers[rid][1]),
        )

    def _bfs_next_room(self, start_rid: int, goal_rid: int, has_key: bool) -> int | None:
        if start_rid == goal_rid:
            return None
        adj: dict[int, list[int]] = collections.defaultdict(list)
        for u, v, lock in self.nav_edges:
            if lock <= 0 or has_key:
                adj[u].append(v)
                adj[v].append(u)
        q: collections.deque[list[int]] = collections.deque([[start_rid]])
        seen = {start_rid}
        while q:
            path = q.popleft()
            cur = path[-1]
            if cur == goal_rid:
                return path[1] if len(path) >= 2 else None
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(path + [nxt])
        return None

    def _select_nav_waypoint(
        self, px: float, py: float, target_xy: tuple[float, float] | None, has_key: bool
    ) -> tuple[float, float] | None:
        if target_xy is None:
            return None
        if not self.room_centers or not self.nav_edges:
            return target_xy

        curr_rid = self._find_room(px, py)
        goal_rid = self._find_room(target_xy[0], target_xy[1])
        if curr_rid is None or goal_rid is None or curr_rid == goal_rid:
            return target_xy

        next_rid = self._bfs_next_room(curr_rid, goal_rid, has_key)
        if next_rid is None:
            return target_xy

        cx_u, cy_u = self.room_centers[curr_rid]
        cx_v, cy_v = self.room_centers[next_rid]
        mid_x = 0.5 * (cx_u + cx_v)
        mid_y = 0.5 * (cy_u + cy_v)

        # Align perpendicular to corridor axis before crossing so doorframes are never clipped
        if abs(cx_v - cx_u) > abs(cy_v - cy_u):
            sign = 1.0 if cx_v > cx_u else -1.0
            pre_x = cx_u + sign * 88.0
            pre_y = mid_y
            dist_pre = math.hypot(px - pre_x, py - pre_y)
            in_corridor_band = abs(py - mid_y) < 34.0 and (px - cx_u) * sign > 30.0
            if dist_pre > 44.0 and not in_corridor_band:
                return (pre_x, pre_y)
            return (mid_x + sign * 64.0, mid_y)
        else:
            sign = 1.0 if cy_v > cy_u else -1.0
            pre_x = mid_x
            pre_y = cy_u + sign * 88.0
            dist_pre = math.hypot(px - pre_x, py - pre_y)
            in_corridor_band = abs(px - mid_x) < 34.0 and (py - cy_u) * sign > 30.0
            if dist_pre > 44.0 and not in_corridor_band:
                return (pre_x, pre_y)
            return (mid_x, mid_y + sign * 64.0)

    def _find_visible_targets(
        self, raw_obs: Mapping[str, Any], health: float, ammo: float
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Return `(nearest_enemy, useful_pickup)` as `(dist_units, rel_angle, cos_a)` or None."""
        ent = raw_obs.get("visible_entities") or {}
        feats = ent.get("features")
        etypes = ent.get("entity_type")
        emask = ent.get("mask")
        if feats is None or etypes is None or emask is None:
            return None, None

        best_enemy: tuple[float, float, float] | None = None
        best_item: tuple[float, float, float] | None = None
        for i in range(len(emask)):
            if int(emask[i]) == 0:
                continue
            et = int(etypes[i])
            dist = float(feats[i][0]) * 1024.0
            rel_a = math.atan2(float(feats[i][1]), float(feats[i][2]))
            cos_a = float(feats[i][2])
            if et in (1, 2):  # melee / hitscan enemy
                if best_enemy is None or dist < best_enemy[0]:
                    best_enemy = (dist, rel_a, cos_a)
            elif et == 9:  # keycard always high priority
                if best_item is None or dist < best_item[0]:
                    best_item = (dist, rel_a, cos_a)
            elif et in (7, 8) and ammo < 35.0 and dist < 240.0:
                if best_item is None:
                    best_item = (dist, rel_a, cos_a)
            elif et in (5, 6) and health < 65.0 and dist < 220.0:
                if best_item is None:
                    best_item = (dist, rel_a, cos_a)
        return best_enemy, best_item

    def act(self, raw_obs: Mapping[str, Any], priv: Mapping[str, Any]) -> np.ndarray:
        """Compute clean masked teacher MultiBinary(9) action for the current step."""
        act = np.zeros(ACTION_DIM, dtype=np.int8)

        px = float(priv.get("x", 0.0))
        py = float(priv.get("y", 0.0))
        heading_rad = math.radians(float(priv.get("angle", 0.0)))
        health = float(priv.get("health", 100.0))
        keys = priv.get("keys")
        has_key = bool(np.sum(keys) > 0) if keys is not None else False

        vitals = np.asarray(raw_obs["vitals"], dtype=np.float32)
        ammo = float(vitals[3])
        reserves = float(np.sum(vitals[4:8]))

        rays_dist = np.asarray(raw_obs["rays"]["distance"], dtype=np.float32)
        rays_hit = np.asarray(raw_obs["rays"]["hit_type"], dtype=np.int64)

        if self.use_cooldown > 0:
            self.use_cooldown -= 1
        if self.door_pause_ticks > 0:
            self.door_pause_ticks -= 1

        self.pos_hist.append((px, py))
        if self.unstuck_ticks > 0:
            self.unstuck_ticks -= 1
        elif len(self.pos_hist) == self.pos_hist.maxlen:
            p0 = self.pos_hist[0]
            if math.hypot(px - p0[0], py - p0[1]) < 20.0 and self.door_pause_ticks == 0:
                self.unstuck_ticks = 8
                left_clear = float(np.mean(rays_dist[list(_LEFT_CLEAR_RAYS)]))
                right_clear = float(np.mean(rays_dist[list(_RIGHT_CLEAR_RAYS)]))
                self.unstuck_strafe_left = left_clear >= right_clear
                self.pos_hist.clear()

        # 1. Determine objective waypoint (keycard before locked exit; arena roam if no exit)
        key_xy = self.map_meta.get("key_xy")
        exit_xy = self.map_meta.get("exit_xy")
        goal_xy: tuple[float, float] | None = None
        if key_xy is not None and not has_key:
            goal_xy = (float(key_xy[0]), float(key_xy[1]))
        elif exit_xy is not None:
            goal_xy = (float(exit_xy[0]), float(exit_xy[1]))

        waypoint = self._select_nav_waypoint(px, py, goal_xy, has_key)
        nearest_enemy, nearest_item = self._find_visible_targets(raw_obs, health, ammo)

        # 2. Switch weapon if active clip empty and reserve ammo exists
        if ammo <= 0.0 and reserves > 0.0 and self.prev_action[8] == 0:
            act[8] = 1

        # 3. Single-tap USE on closed front door / locked door within reach
        door_close = any(
            int(rays_hit[idx]) in (2, 3) and float(rays_dist[idx]) * 1024.0 <= 128.0
            for idx in _FRONT_DOOR_RAYS
        )
        if door_close and self.use_cooldown == 0 and self.prev_action[7] == 0:
            act[7] = 1
            self.use_cooldown = 10
            self.door_pause_ticks = 3

        # 4. Steering & Combat arbitration
        in_combat = nearest_enemy is not None and (
            goal_xy is None or nearest_enemy[0] <= 520.0
        )

        if in_combat and nearest_enemy is not None:
            dist_e, rel_a, cos_a = nearest_enemy
            if rel_a > 0.07:
                act[4] = 1  # TURN_LEFT
            elif rel_a < -0.07:
                act[5] = 1  # TURN_RIGHT

            # Attack only when aimed in tight cone and ammo available (or melee range)
            if abs(rel_a) <= 0.16 and (ammo > 0.0 or dist_e <= 105.0):
                act[6] = 1

            # Maintain spacing / weave during combat without backing into walls
            back_ray = float(min(rays_dist[30], rays_dist[31], rays_dist[32], rays_dist[33]))
            front_ray = float(min(rays_dist[idx] for idx in _FRONT_WALL_RAYS))
            if dist_e > 220.0 and front_ray > 0.07:
                act[0] = 1
            elif dist_e < 110.0 and back_ray > 0.08:
                act[1] = 1
            else:
                left_c = float(np.mean(rays_dist[list(_LEFT_CLEAR_RAYS)]))
                right_c = float(np.mean(rays_dist[list(_RIGHT_CLEAR_RAYS)]))
                if left_c > right_c + 0.03:
                    act[2] = 1
                elif right_c > left_c + 0.03:
                    act[3] = 1
        elif nearest_item is not None and abs(nearest_item[1]) < 0.7:
            # Detour slightly to grab key/ammo/health in immediate FOV
            rel_a = nearest_item[1]
            if rel_a > 0.09:
                act[4] = 1
            elif rel_a < -0.09:
                act[5] = 1
            if abs(rel_a) < 0.75:
                act[0] = 1
        elif waypoint is not None:
            dx = waypoint[0] - px
            dy = waypoint[1] - py
            desired = math.atan2(dy, dx)
            rel_a = _wrap_angle_rad(desired - heading_rad)
            if rel_a > 0.09:
                act[4] = 1
            elif rel_a < -0.09:
                act[5] = 1

            front_min = float(min(rays_dist[idx] for idx in _FRONT_WALL_RAYS))
            # Move forward if generally facing waypoint and not jammed in pause/door
            if self.door_pause_ticks == 0 and abs(rel_a) < 0.95:
                if front_min > 0.045 or door_close:
                    act[0] = 1
            # Slight ray clearance strafe to avoid catching corridor corners
            if front_min < 0.085 and not door_close:
                left_c = float(np.mean(rays_dist[list(_LEFT_CLEAR_RAYS)]))
                right_c = float(np.mean(rays_dist[list(_RIGHT_CLEAR_RAYS)]))
                if left_c >= right_c:
                    act[2] = 1
                else:
                    act[3] = 1
        else:
            # Combat arena search sweep when no enemy currently in FOV
            act[4] = 1
            front_min = float(min(rays_dist[idx] for idx in _FRONT_WALL_RAYS))
            if front_min > 0.14:
                act[0] = 1

        # 5. Unstuck override when wedged against a corner
        if self.unstuck_ticks > 0 and not in_combat:
            act[0] = 0
            act[1] = 1 if self.unstuck_ticks >= 5 else 0
            act[2] = 1 if self.unstuck_strafe_left else 0
            act[3] = 0 if self.unstuck_strafe_left else 1

        # 6. Guarantee teacher action satisfies the exact observation mask used at train/eval
        vitals_norm = vitals.copy()
        vitals_norm[3] = np.clip(vitals_norm[3] / 100.0, 0.0, 5.0)
        vitals_norm[4:8] = np.clip(vitals_norm[4:8] / 100.0, 0.0, 5.0)
        ent = raw_obs.get("visible_entities") or {}
        obs_mask = compute_obs_action_mask_np_arrays(
            vitals=vitals_norm,
            rays_dist=rays_dist,
            rays_hit_type=rays_hit,
            entity_features=np.asarray(ent.get("features", np.zeros((32, 5))), dtype=np.float32),
            entity_types=np.asarray(ent.get("entity_type", np.zeros(32)), dtype=np.int64),
            entity_mask=np.asarray(ent.get("mask", np.zeros(32)), dtype=np.float32),
            prev_action=self.prev_action.astype(np.float32),
        )
        act = (act * (obs_mask > 0.5).astype(np.int8)).astype(np.int8)

        # Guarantee no opposing axes
        for pos, neg in ((0, 1), (2, 3), (4, 5)):
            if act[pos] and act[neg]:
                act[neg] = 0

        self.prev_action[:] = act
        return act
