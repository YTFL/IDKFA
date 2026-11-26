"""Shared Observation Preprocessing and Feature Extraction Pipeline.

Shared between training and submission (zero train/serve skew).
Runs in pure NumPy with optional PyTorch tensor conversion.

Features:
- Rays: distance normalized to [0, 1] + log scale + 6-class one-hot hit types -> (64, 8)
- Depth: downsampled (e.g. 8x32 or 1x32 average-pooled) -> (32,)
- Vitals: normalized health, armor, ammo, binary key flags, velocities -> (17,)
- Entities: (32, 17) with masked relative coordinates and 12-class one-hot entity types
- Action history: one-hot / binary vector of previous action -> (9,)
- Sanitization: handles NaNs, infs, empty masks, and unexpected dtypes.
- RunningMeanStd: optional running normalization statistics frozen at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

# Constant definitions matching env/doom_env/spaces.py
NUM_RAYS = 64
RAY_RANGE = 1024.0
NUM_RAY_HIT_TYPES = 6  # none, wall, door, locked_door, enemy, item

MAX_ENTITIES = 32
NUM_ENTITY_RAW_FEATURES = 5  # dist/1024, sin(ang), cos(ang), rel_vel_fwd, rel_vel_right
NUM_ENTITY_TYPES = 12  # pad, enemy_melee, enemy_hitscan, projectile, etc.
ENTITY_FEAT_DIM = NUM_ENTITY_RAW_FEATURES + NUM_ENTITY_TYPES  # 17

VITALS_DIM = 17
ACTION_DIM = 9

DEPTH_DOWN_WIDTH = 32


class RunningMeanStd:
    """Tracks running mean and variance with Welford's algorithm."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.mean.shape) else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


@dataclass
class ProcessedObservation:
    rays: np.ndarray  # (64, 8) float32
    depth: np.ndarray  # (32,) float32
    vitals: np.ndarray  # (17,) float32
    entities: np.ndarray  # (32, 17) float32
    entity_mask: np.ndarray  # (32,) bool/float32
    prev_action: np.ndarray  # (9,) float32

    def to_flat_vector(self) -> np.ndarray:
        """Flattens all features into a single 1D vector (useful for MLP baselines/RND)."""
        return np.concatenate(
            [
                self.rays.reshape(-1),
                self.depth,
                self.vitals,
                (self.entities * self.entity_mask[:, None]).reshape(-1),
                self.entity_mask,
                self.prev_action,
            ],
            axis=0,
        )


class ObservationPreprocessor:
    """End-to-end observation preprocessor."""

    def __init__(self, normalize_vitals: bool = True):
        self.normalize_vitals = normalize_vitals
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self) -> None:
        """Reset episodic state (e.g. previous action buffer)."""
        self.prev_action.fill(0.0)

    def process(
        self,
        obs: Mapping[str, Any],
        prev_action: np.ndarray | None = None,
    ) -> ProcessedObservation:
        """Sanitizes and featurizes an ego-centric observation dictionary."""
        if prev_action is not None:
            self.prev_action = np.asarray(prev_action, dtype=np.float32).copy()

        # 1. Process Rays: (64, 8)
        # distance in [0, 1], log(1 + 9*dist), and 6 one-hot classes
        rays_dist = np.nan_to_num(obs["rays"]["distance"], nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)
        rays_dist = np.clip(rays_dist, 0.0, 1.0)
        rays_log = np.log1p(9.0 * rays_dist) / np.log(10.0)  # maps [0, 1] smoothly

        hit_types = np.nan_to_num(obs["rays"]["hit_type"], nan=0).astype(np.int64)
        hit_types = np.clip(hit_types, 0, NUM_RAY_HIT_TYPES - 1)
        hit_one_hot = np.eye(NUM_RAY_HIT_TYPES, dtype=np.float32)[hit_types]  # (64, 6)

        # rays: (64, 8) -> [distance, log_distance, one_hot_hit_types(6)]
        rays_feat = np.column_stack([rays_dist, rays_log, hit_one_hot]).astype(np.float32)

        # 2. Process Depth Buffer: downsample to (32,)
        depth_raw = obs.get("depth")
        if depth_raw is not None:
            depth_arr = np.nan_to_num(depth_raw, nan=0.0)
            if depth_arr.ndim == 2 and depth_arr.shape[0] >= 90 and depth_arr.shape[1] >= DEPTH_DOWN_WIDTH:
                depth_f = depth_arr.astype(np.float32) / 255.0
                # Mean over vertical band (central 60 rows where geometry lives)
                depth_band = depth_f[30:90, :].mean(axis=0)  # (160,)
                cols = (depth_band.shape[0] // DEPTH_DOWN_WIDTH) * DEPTH_DOWN_WIDTH
                depth_feat = depth_band[:cols].reshape(DEPTH_DOWN_WIDTH, -1).mean(axis=1)
            else:
                depth_feat = np.zeros(DEPTH_DOWN_WIDTH, dtype=np.float32)
        else:
            depth_feat = np.zeros(DEPTH_DOWN_WIDTH, dtype=np.float32)

        # 3. Process Vitals: (17,)
        # Order: health(0), armor(1), selected_weapon(2), selected_weapon_ammo(3),
        # ammo_bullets(4), ammo_shells(5), ammo_rockets(6), ammo_cells(7),
        # keys(8..13), vel_forward(14), vel_right(15), vel_up(16)
        vitals_raw = np.nan_to_num(obs["vitals"], nan=0.0).astype(np.float32).copy()
        if self.normalize_vitals:
            vitals_raw[0] = np.clip(vitals_raw[0] / 100.0, 0.0, 2.0)  # Health
            vitals_raw[1] = np.clip(vitals_raw[1] / 100.0, 0.0, 2.0)  # Armor
            vitals_raw[2] = np.clip(vitals_raw[2] / 7.0, 0.0, 1.0)    # Selected weapon
            vitals_raw[3] = np.clip(vitals_raw[3] / 100.0, 0.0, 5.0)  # Current ammo
            vitals_raw[4:8] = np.clip(vitals_raw[4:8] / 100.0, 0.0, 5.0)  # Ammo reserves
            # 8:14 are binary key flags, keep [0, 1]
            vitals_raw[14:17] = np.clip(vitals_raw[14:17] / 20.0, -2.0, 2.0)  # Velocities

        # 4. Process Visible Entities: (32, 17) and mask (32,)
        ent_dict = obs.get("visible_entities", {})
        raw_feats = np.nan_to_num(ent_dict.get("features", np.zeros((MAX_ENTITIES, NUM_ENTITY_RAW_FEATURES))), nan=0.0)
        ent_types = np.nan_to_num(ent_dict.get("entity_type", np.zeros(MAX_ENTITIES)), nan=0).astype(np.int64)
        ent_mask = np.nan_to_num(ent_dict.get("mask", np.zeros(MAX_ENTITIES)), nan=0).astype(bool)

        ent_types = np.clip(ent_types, 0, NUM_ENTITY_TYPES - 1)
        type_one_hot = np.eye(NUM_ENTITY_TYPES, dtype=np.float32)[ent_types]  # (32, 12)

        # Concatenate features and type one-hot: (32, 5 + 12) = (32, 17)
        entities_feat = np.column_stack([raw_feats.astype(np.float32), type_one_hot]).astype(np.float32)
        # Zero out padding slots
        entities_feat[~ent_mask] = 0.0

        return ProcessedObservation(
            rays=rays_feat,
            depth=depth_feat,
            vitals=vitals_raw,
            entities=entities_feat,
            entity_mask=ent_mask.astype(np.float32),
            prev_action=self.prev_action.copy(),
        )
