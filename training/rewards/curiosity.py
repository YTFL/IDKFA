"""Intrinsic Curiosity Modules: Random Network Distillation (RND) and Episodic Novelty.

1. RND:
   - Target network: randomly initialized, fixed weights.
   - Predictor network: trained online to match the target.
   - Input: ray features and depth buffer (spatial layout only, ignoring moving enemies).
   - Intrinsic reward: MSE(predictor(x), target(x)) normalized by running standard deviation.

2. Episodic Novelty (NGU style):
   - Keeps a buffer of visited spatial embeddings during the current episode.
   - Computes distance to k-nearest neighbors in episodic memory.
   - Rewards discovering novel rooms and penalizes looping inside the same room.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from submission.features import ProcessedObservation, RunningMeanStd


class RNDTargetNetwork(nn.Module):
    """Fixed random target network embedding spatial features."""

    def __init__(self, in_dim: int = 64 * 8 + 32, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, out_dim),
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Trained predictor network trying to match target network outputs."""

    def __init__(self, in_dim: int = 64 * 8 + 32, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDCuriosity:
    """Manages RND intrinsic exploration bonus."""

    def __init__(
        self,
        in_dim: int = 64 * 8 + 32,
        emb_dim: int = 128,
        lr: float = 1e-4,
        scale: float = 0.1,
        device: str = "cpu",
    ):
        self.scale = scale
        self.device = torch.device(device)
        self.target = RNDTargetNetwork(in_dim=in_dim, out_dim=emb_dim).to(self.device)
        self.predictor = RNDPredictorNetwork(in_dim=in_dim, out_dim=emb_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.reward_rms = RunningMeanStd()

    def _extract_spatial_vector(self, processed_obs: ProcessedObservation) -> np.ndarray:
        # Spatial only: rays (64 * 8) + depth (32) = 544 dims
        return np.concatenate([processed_obs.rays.reshape(-1), processed_obs.depth], axis=0)

    def compute_intrinsic_reward(
        self,
        processed_obs: ProcessedObservation,
        update_predictor: bool = False,
    ) -> float:
        spatial_vec = self._extract_spatial_vector(processed_obs)
        x = torch.from_numpy(spatial_vec).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            target_out = self.target(x)
            pred_out = self.predictor(x)
            raw_err = F.mse_loss(pred_out, target_out, reduction="none").sum(dim=-1).item()

        # Update running reward normalizer
        self.reward_rms.update(np.array([raw_err]))
        norm_reward = raw_err / (math.sqrt(self.reward_rms.var.item()) + 1e-8)
        intrinsic_reward = float(norm_reward * self.scale)

        if update_predictor:
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.predictor(x), target_out)
            loss.backward()
            self.optimizer.step()

        return intrinsic_reward


class EpisodicNoveltyBuffer:
    """Tracks state novelty within a single episode using nearest-neighbor distances."""

    def __init__(
        self,
        emb_dim: int = 16,
        capacity: int = 1000,
        k_neighbors: int = 5,
        c_novelty: float = 0.05,
    ):
        self.capacity = capacity
        self.k_neighbors = k_neighbors
        self.c_novelty = c_novelty
        # Random projection matrix for fast low-dim hashing of spatial vectors (544 -> 16)
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((544, emb_dim)).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-8
        self.memory: list[np.ndarray] = []

    def reset(self) -> None:
        self.memory.clear()

    def compute_novelty_bonus(self, processed_obs: ProcessedObservation) -> float:
        spatial_vec = np.concatenate([processed_obs.rays.reshape(-1), processed_obs.depth], axis=0)
        emb = (spatial_vec @ self.proj).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8

        if len(self.memory) < self.k_neighbors:
            self.memory.append(emb)
            return self.c_novelty

        # Compute cosine distances to past steps in this episode
        mem_matrix = np.array(self.memory)  # (N, 16)
        sims = mem_matrix @ emb  # cosine similarities
        dists = np.maximum(0.0, 1.0 - sims)

        # k smallest distances
        k_count = min(self.k_neighbors, len(dists))
        k_smallest = np.sort(dists)[:k_count]
        avg_dist = float(np.mean(k_smallest))

        if len(self.memory) < self.capacity:
            self.memory.append(emb)

        # Reward is proportional to average distance to nearest states in episodic memory
        return float(self.c_novelty * avg_dist)
