"""Egocentric neural spatial map and learned egomotion estimator."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import EMLRPConfig


class EgomotionEstimator(nn.Module):
    """Estimate per-step egocentric translation (normalized grid units) and yaw delta."""

    def __init__(self, config: EMLRPConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.action_dim + 2  # prev_action + (vel_fwd, vel_right)
        self.correction = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        # Initialize residual correction near zero so dead-reckoning prior dominates early.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, prev_action: Tensor, vitals: Tensor) -> tuple[Tensor, Tensor]:
        """Compute `(delta_xy, delta_theta)` in normalized `[-1, 1]` map coordinates.

        Convention:
            +x = forward, +y = left (CCW), +theta = counter-clockwise yaw.
        """
        cfg = self.config
        vel_fwd = vitals[:, cfg.vel_fwd_idx] * cfg.nominal_vel_scale
        vel_right = vitals[:, cfg.vel_right_idx] * cfg.nominal_vel_scale
        vel_left = -vel_right

        # Half-span of the 32x32 map in DOOM map units: (32 * 64) / 2 = 1024.0
        # Velocity in vitals is map units per tic -> multiply by frame_skip for per-step displacement
        half_span = 0.5 * cfg.map_size * cfg.map_cell_units
        prior_dx = (vel_fwd * cfg.frame_skip) / half_span
        prior_dy = (vel_left * cfg.frame_skip) / half_span

        turn_left = prev_action[:, cfg.turn_left_idx]
        turn_right = prev_action[:, cfg.turn_right_idx]
        prior_dtheta = (turn_left - turn_right) * cfg.nominal_turn_rad_per_step

        corr_in = torch.cat([prev_action, vel_fwd.unsqueeze(-1), vel_right.unsqueeze(-1)], dim=-1)
        corr = 0.1 * torch.tanh(self.correction(corr_in))

        delta_xy = torch.stack([prior_dx, prior_dy], dim=-1) + corr[:, :2]
        delta_theta = prior_dtheta + corr[:, 2]
        return delta_xy, delta_theta


class EgocentricSpatialMap(nn.Module):
    """Differentiable 32x32x16 egocentric feature map with egomotion warp, GRU write, and readout."""

    def __init__(self, config: EMLRPConfig) -> None:
        super().__init__()
        self.config = config
        S = config.map_size
        C = config.map_channels

        # Precompute normalized cell coordinates (x=forward, y=left) in [-1, 1].
        coords_1d = torch.linspace(-1.0 + 1.0 / S, 1.0 - 1.0 / S, S)
        grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        # cell_xy: (S*S, 2) with [x_fwd, y_left]
        cell_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        self.register_buffer("cell_xy", cell_xy, persistent=False)

        # Ray unit directions (0 = straight ahead, counter-clockwise)
        angles = torch.arange(config.ray_bins, dtype=torch.float32) * (2.0 * math.pi / config.ray_bins)
        ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (64, 2)
        self.register_buffer("ray_dirs", ray_dirs, persistent=False)

        # Write path: splatted ray sector features + free-space evidence + optional depth context
        write_in_ch = config.ray_sector_dim + 1 + (16 if config.depth_feat_dim > 0 else 0)
        if config.depth_feat_dim > 0:
            self.depth_to_map = nn.Linear(config.depth_feat_dim, 16)
        else:
            self.depth_to_map = None

        self.write_net = nn.Sequential(
            nn.Conv2d(write_in_ch, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, C, kernel_size=3, padding=1),
        )
        # ConvGRU-style update and candidate gates over [warped_map, write_feats]
        self.gate_conv = nn.Conv2d(C * 2, C * 2, kernel_size=3, padding=1)

        # Read path: strided CNN down to 8x8 + GRU-query attention readout
        self.read_cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.GroupNorm(8, 64),
        )
        self.gru_query = nn.Linear(config.gru_hidden_dim, 64)
        self.read_proj = nn.Sequential(
            nn.Linear(64 * 2, config.map_readout_dim),
            nn.LayerNorm(config.map_readout_dim),
            nn.GELU(),
        )

    def warp_map(self, spatial_map: Tensor, delta_xy: Tensor, delta_theta: Tensor) -> Tensor:
        """Transform previous egocentric map into the new agent frame via `grid_sample`."""
        B, _, S, _ = spatial_map.shape
        cos_t = torch.cos(delta_theta)
        sin_t = torch.sin(delta_theta)
        # Inverse transform (from new ego frame back to old ego frame):
        # p_old = R(+dtheta) @ p_new + delta_xy
        theta = torch.zeros(B, 2, 3, device=spatial_map.device, dtype=spatial_map.dtype)
        theta[:, 0, 0] = cos_t
        theta[:, 0, 1] = -sin_t
        theta[:, 0, 2] = delta_xy[:, 0]
        theta[:, 1, 0] = sin_t
        theta[:, 1, 1] = cos_t
        theta[:, 1, 2] = delta_xy[:, 1]

        grid = F.affine_grid(theta, size=spatial_map.shape, align_corners=False)
        return F.grid_sample(spatial_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _splat_rays(self, ray_distance: Tensor, ray_sector_feats: Tensor) -> Tensor:
        """Project per-ray features and free-space evidence onto the 32x32 grid."""
        B = ray_distance.shape[0]
        S = self.config.map_size
        # hit_xy: (B, 64, 2) in [-1, 1]
        hit_xy = ray_distance.unsqueeze(-1) * self.ray_dirs.unsqueeze(0)

        # Smooth Gaussian splat onto 32x32 cells (sigma = 1.5 cells)
        sigma2 = (3.0 / S) ** 2
        diff = self.cell_xy.unsqueeze(0).unsqueeze(2) - hit_xy.unsqueeze(1)  # (B, 1024, 64, 2)
        dist2 = (diff * diff).sum(dim=-1)  # (B, 1024, 64)
        weights = torch.exp(-dist2 / (2.0 * sigma2))  # (B, 1024, 64)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-4)

        # Weighted ray features at hit locations: (B, 1024, ray_sector_dim)
        splatted = torch.bmm(weights, ray_sector_feats)

        # Free-space evidence: projection of cell onto ray direction inside [0, ray_distance]
        proj = torch.matmul(self.cell_xy, self.ray_dirs.t()).unsqueeze(0)  # (1, 1024, 64)
        perp2 = (self.cell_xy.pow(2).sum(dim=-1, keepdim=True).unsqueeze(0)) - proj.pow(2)
        ang_w = torch.exp(-perp2.clamp(min=0.0) / (2.0 * sigma2))
        in_seg = torch.sigmoid((ray_distance.unsqueeze(1) - proj) * 16.0) * torch.sigmoid(proj * 16.0)
        free_space = (ang_w * in_seg).max(dim=-1, keepdim=True).values  # (B, 1024, 1)

        out = torch.cat([splatted, free_space], dim=-1)  # (B, 1024, S_dim+1)
        return out.permute(0, 2, 1).view(B, -1, S, S)

    def forward(
        self,
        prev_map: Tensor,
        delta_xy: Tensor,
        delta_theta: Tensor,
        ray_distance: Tensor,
        ray_sector_feats: Tensor,
        gru_h: Tensor,
        depth_feat: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update the spatial map and return `(next_map, map_readout)`."""
        B, _, S, _ = prev_map.shape
        warped = self.warp_map(prev_map, delta_xy, delta_theta)

        splat = self._splat_rays(ray_distance, ray_sector_feats)
        if self.depth_to_map is not None:
            if depth_feat is None:
                d_map = torch.zeros(B, 16, S, S, device=prev_map.device, dtype=prev_map.dtype)
            else:
                d_map = self.depth_to_map(depth_feat).view(B, 16, 1, 1).expand(B, 16, S, S)
            write_in = torch.cat([splat, d_map], dim=1)
        else:
            write_in = splat

        obs_map = self.write_net(write_in)
        gates = self.gate_conv(torch.cat([warped, obs_map], dim=1))
        z_gate, cand = gates.chunk(2, dim=1)
        z = torch.sigmoid(z_gate)
        next_map = (1.0 - z) * warped + z * torch.tanh(cand)

        # Readout: strided CNN + GRU attention over 8x8 cells
        read_grid = self.read_cnn(next_map)  # (B, 64, 8, 8)
        cells = read_grid.flatten(start_dim=2).permute(0, 2, 1)  # (B, 64_cells, 64)
        mean_pool = cells.mean(dim=1)  # (B, 64)

        q = self.gru_query(gru_h).unsqueeze(1)  # (B, 1, 64)
        attn_logits = torch.bmm(q, cells.transpose(1, 2)).squeeze(1) * (64 ** -0.5)
        attn_w = F.softmax(attn_logits, dim=-1).unsqueeze(1)
        attn_pool = torch.bmm(attn_w, cells).squeeze(1)  # (B, 64)

        map_readout = self.read_proj(torch.cat([mean_pool, attn_pool], dim=-1))
        return next_map, map_readout
