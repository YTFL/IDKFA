"""Configuration and state dataclasses for the EMLRP neural architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EMLRPConfig:
    """Hyperparameters for EgoMap-Landmark Recurrent Policy (EMLRP)."""

    # Observation contract dimensions (aligned with env/doom_env/spaces.py)
    ray_bins: int = 64
    ray_hit_types: int = 6
    ray_hit_embed_dim: int = 8
    ray_embed_dim: int = 128
    ray_sector_dim: int = 16

    entity_max: int = 32
    entity_feat_dim: int = 5
    entity_type_vocab: int = 12
    entity_type_embed_dim: int = 16
    entity_hidden_dim: int = 128
    entity_embed_dim: int = 64
    entity_heads: int = 4
    entity_layers: int = 2

    vitals_dim: int = 17
    vitals_embed_dim: int = 64
    key_flags_start: int = 8
    key_flags_count: int = 6
    vel_fwd_idx: int = 14
    vel_right_idx: int = 15

    action_dim: int = 9
    prev_action_embed_dim: int = 32
    turn_left_idx: int = 4
    turn_right_idx: int = 5

    # Optional visual/depth feature input from Phase 2 encoder (DEPTH_DOWN_WIDTH = 32)
    depth_feat_dim: int = 32

    # Recurrent core
    fusion_dim: int = 256
    gru_hidden_dim: int = 256

    # Egocentric neural spatial map
    map_size: int = 32
    map_channels: int = 16
    map_cell_units: float = 64.0
    map_readout_dim: int = 128

    # Landmark slot memory
    landmark_slots: int = 16
    landmark_dim: int = 64
    landmark_readout_dim: int = 64

    # Hierarchical latent mode controller & heads
    num_latent_modes: int = 5
    mode_embed_dim: int = 32
    head_hidden_dim: int = 256

    # Motion scaling for dead-reckoning prior (per FRAME_SKIP=4 step; vitals vel is /20.0)
    frame_skip: int = 4
    nominal_turn_rad_per_step: float = 0.2454369260617026  # 14.0625 deg steady-state TURN_LEFT/RIGHT per 4 tics
    nominal_vel_scale: float = 20.0


@dataclass
class EMLRPState:
    """Recurrent, spatial map, and landmark slot memory state across steps."""

    gru_h: Tensor
    spatial_map: Tensor
    landmark_keys: Tensor
    landmark_pos: Tensor
    landmark_valid: Tensor

    def reset_mask(self, done: Tensor) -> EMLRPState:
        """Zero out state entries where `done` (shape `(B,)`, bool or float) is truthy."""
        keep = (1.0 - done.float()).to(self.gru_h.dtype)
        keep_2d = keep.unsqueeze(-1)
        keep_3d = keep.view(-1, 1, 1)
        keep_4d = keep.view(-1, 1, 1, 1)
        return EMLRPState(
            gru_h=self.gru_h * keep_2d,
            spatial_map=self.spatial_map * keep_4d,
            landmark_keys=self.landmark_keys * keep_3d,
            landmark_pos=self.landmark_pos * keep_3d,
            landmark_valid=self.landmark_valid * keep_2d,
        )

    def detach(self) -> EMLRPState:
        """Detach all tensors for truncated BPTT boundaries."""
        return EMLRPState(
            gru_h=self.gru_h.detach(),
            spatial_map=self.spatial_map.detach(),
            landmark_keys=self.landmark_keys.detach(),
            landmark_pos=self.landmark_pos.detach(),
            landmark_valid=self.landmark_valid.detach(),
        )


def init_emlrp_state(
    batch_size: int,
    config: EMLRPConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EMLRPState:
    """Allocate zero-initialized EMLRP state tensors."""
    return EMLRPState(
        gru_h=torch.zeros(batch_size, config.gru_hidden_dim, device=device, dtype=dtype),
        spatial_map=torch.zeros(
            batch_size, config.map_channels, config.map_size, config.map_size, device=device, dtype=dtype
        ),
        landmark_keys=torch.zeros(
            batch_size, config.landmark_slots, config.landmark_dim, device=device, dtype=dtype
        ),
        landmark_pos=torch.zeros(batch_size, config.landmark_slots, 2, device=device, dtype=dtype),
        landmark_valid=torch.zeros(batch_size, config.landmark_slots, device=device, dtype=dtype),
    )
