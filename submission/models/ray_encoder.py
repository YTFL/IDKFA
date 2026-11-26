"""Circular 1D CNN ray encoder for 360-degree ego-centric raycasts."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .config import EMLRPConfig


class RayEncoder(nn.Module):
    """Encode 64 circular rays (distance + hit_type) into global and per-sector features."""

    def __init__(self, config: EMLRPConfig) -> None:
        super().__init__()
        self.config = config
        self.hit_embed = nn.Embedding(config.ray_hit_types, config.ray_hit_embed_dim)
        in_ch = 1 + config.ray_hit_embed_dim

        # Per-sector stem preserves full 64-bin angular resolution for spatial map writing.
        self.sector_stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.GroupNorm(4, 32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.GroupNorm(8, 64),
        )
        self.sector_proj = nn.Conv1d(64, config.ray_sector_dim, kernel_size=1)

        # Downsampling trunk summarizes 360-degree geometry into a compact ray embedding.
        self.trunk = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.GroupNorm(8, 128),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.GroupNorm(8, 128),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.GroupNorm(8, 128),
        )
        pooled_bins = config.ray_bins // 8  # 64 -> 8
        self.global_head = nn.Sequential(
            nn.Linear(128 * pooled_bins, config.ray_embed_dim),
            nn.LayerNorm(config.ray_embed_dim),
            nn.GELU(),
        )

    def forward(self, ray_distance: Tensor, ray_hit_type: Tensor) -> tuple[Tensor, Tensor]:
        """Encode rays.

        Args:
            ray_distance: `(B, 64)` normalized distances in `[0, 1]`.
            ray_hit_type: `(B, 64)` integer hit class IDs in `[0, 5]`.

        Returns:
            ray_embed: `(B, ray_embed_dim)` global embedding.
            ray_sector_feats: `(B, 64, ray_sector_dim)` per-ray features.
        """
        hit_emb = self.hit_embed(ray_hit_type.long()).permute(0, 2, 1)  # (B, E, 64)
        x = torch.cat([ray_distance.unsqueeze(1), hit_emb], dim=1)  # (B, 1+E, 64)

        sector_h = self.sector_stem(x)  # (B, 64, 64)
        ray_sector_feats = self.sector_proj(sector_h).permute(0, 2, 1)  # (B, 64, S)

        trunk_h = self.trunk(sector_h).flatten(start_dim=1)  # (B, 128 * 8)
        ray_embed = self.global_head(trunk_h)
        return ray_embed, ray_sector_feats
