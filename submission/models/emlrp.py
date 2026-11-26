"""Top-level EgoMap-Landmark Recurrent Policy (EMLRP) module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import EMLRPConfig, EMLRPState, init_emlrp_state
from .entity_encoder import EntitySetEncoder
from .landmark_memory import LandmarkSlotMemory
from .ray_encoder import RayEncoder
from .spatial_map import EgomotionEstimator, EgocentricSpatialMap


@dataclass
class EMLRPStepOutput:
    """Outputs produced by a single step of EMLRPPolicy."""

    action_logits: Tensor  # (B, 9)
    value_ext: Tensor  # (B, 1)
    value_int: Tensor  # (B, 1)
    mode_logits: Tensor  # (B, num_latent_modes)
    state: EMLRPState
    aux: dict[str, Tensor]


class EMLRPPolicy(nn.Module):
    """EgoMap-Landmark Recurrent Policy for zero-shot DOOM navigation and combat."""

    def __init__(self, config: EMLRPConfig | None = None) -> None:
        super().__init__()
        self.config = config or EMLRPConfig()
        cfg = self.config

        # Modules 1 & 2: Perception encoders
        self.ray_encoder = RayEncoder(cfg)
        self.entity_encoder = EntitySetEncoder(cfg)

        self.vitals_encoder = nn.Sequential(
            nn.Linear(cfg.vitals_dim, cfg.vitals_embed_dim),
            nn.LayerNorm(cfg.vitals_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.vitals_embed_dim, cfg.vitals_embed_dim),
            nn.LayerNorm(cfg.vitals_embed_dim),
            nn.GELU(),
        )
        self.prev_action_encoder = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.prev_action_embed_dim),
            nn.LayerNorm(cfg.prev_action_embed_dim),
            nn.GELU(),
        )

        # Modules 4 & 5: Egocentric neural spatial map + landmark slot memory
        self.egomotion = EgomotionEstimator(cfg)
        self.spatial_map = EgocentricSpatialMap(cfg)
        self.landmark_memory = LandmarkSlotMemory(cfg)

        # Module 3: Recurrent core
        fusion_in = (
            cfg.ray_embed_dim
            + cfg.entity_embed_dim
            + cfg.vitals_embed_dim
            + cfg.prev_action_embed_dim
            + cfg.map_readout_dim
            + cfg.landmark_readout_dim
            + cfg.depth_feat_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_dim),
            nn.LayerNorm(cfg.fusion_dim),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(cfg.fusion_dim, cfg.gru_hidden_dim)
        self.gru_norm = nn.LayerNorm(cfg.gru_hidden_dim)

        # Module 6: Latent hierarchical mode controller
        self.mode_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.num_latent_modes),
        )
        self.mode_embeddings = nn.Parameter(
            torch.randn(cfg.num_latent_modes, cfg.mode_embed_dim) * 0.02
        )

        # Module 7: Policy & Value heads
        policy_in = cfg.gru_hidden_dim + cfg.mode_embed_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_in, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim // 2, cfg.action_dim),
        )
        self.value_ext_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim // 2, 1),
        )
        self.value_int_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim // 2, 1),
        )

        # Training-only auxiliary heads (skipped when compute_aux=False at inference)
        self.aux_next_ray = nn.Sequential(
            nn.Linear(policy_in, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, cfg.ray_bins),
        )
        self.aux_landmark_dir = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        self.aux_inverse_dyn = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, cfg.action_dim),
        )
        self.aux_combat = nn.Sequential(
            nn.Linear(policy_in, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # [damage_taken, enemy_kill]
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> EMLRPState:
        """Create a fresh zero-initialized recurrent and spatial memory state."""
        return init_emlrp_state(batch_size, self.config, device=device, dtype=dtype)

    def step(
        self,
        obs: dict[str, Tensor],
        state: EMLRPState,
        prev_action: Tensor,
        done: Tensor | None = None,
        compute_aux: bool = False,
    ) -> EMLRPStepOutput:
        """Run one decision step.

        Expected keys in `obs`:
            - `ray_distance`: `(B, 64)` float32
            - `ray_hit_type`: `(B, 64)` int64
            - `entity_features`: `(B, 32, 5)` float32
            - `entity_types`: `(B, 32)` int64
            - `entity_mask`: `(B, 32)` float32/bool
            - `vitals`: `(B, 17)` float32
            - `depth_feat` (optional): `(B, depth_feat_dim)` float32
        """
        if done is not None:
            state = state.reset_mask(done)

        prev_gru_h = state.gru_h
        ray_dist = obs["ray_distance"]
        ray_hit = obs["ray_hit_type"]
        ent_feat = obs["entity_features"]
        ent_type = obs["entity_types"]
        ent_mask = obs["entity_mask"]
        vitals = obs["vitals"]
        depth_feat = obs.get("depth_feat")

        B = ray_dist.shape[0]
        if self.config.depth_feat_dim > 0 and depth_feat is None:
            depth_feat = torch.zeros(
                B, self.config.depth_feat_dim, device=ray_dist.device, dtype=ray_dist.dtype
            )

        # 1. Encode current sensory streams
        ray_embed, ray_sector_feats = self.ray_encoder(ray_dist, ray_hit)
        entity_summary, entity_tokens, entity_importance = self.entity_encoder(
            ent_feat, ent_type, ent_mask, ray_embed
        )
        vitals_embed = self.vitals_encoder(vitals)
        prev_act_embed = self.prev_action_encoder(prev_action.float())

        # 2. Estimate egomotion and update spatial + landmark memories
        delta_xy, delta_theta = self.egomotion(prev_action.float(), vitals)
        next_map, map_readout = self.spatial_map(
            state.spatial_map,
            delta_xy,
            delta_theta,
            ray_dist,
            ray_sector_feats,
            prev_gru_h,
            depth_feat=depth_feat,
        )
        next_keys, next_pos, next_valid, landmark_readout = self.landmark_memory(
            state.landmark_keys,
            state.landmark_pos,
            state.landmark_valid,
            delta_xy,
            delta_theta,
            entity_tokens,
            ent_feat,
            ent_mask,
            ray_embed,
            prev_gru_h,
            vitals,
        )

        # 3. Fuse modalities and update GRU core
        parts = [
            ray_embed,
            entity_summary,
            vitals_embed,
            prev_act_embed,
            map_readout,
            landmark_readout,
        ]
        if self.config.depth_feat_dim > 0 and depth_feat is not None:
            parts.append(depth_feat)

        fused = self.fusion(torch.cat(parts, dim=-1))
        next_gru_h = self.gru_norm(self.gru(fused, prev_gru_h))

        # 4. Latent mode conditioning and policy/value heads
        mode_logits = self.mode_head(next_gru_h)
        mode_probs = F.softmax(mode_logits, dim=-1)
        mode_embed = torch.matmul(mode_probs, self.mode_embeddings)  # (B, mode_embed_dim)

        policy_in = torch.cat([next_gru_h, mode_embed], dim=-1)
        action_logits = self.policy_head(policy_in)
        value_ext = self.value_ext_head(next_gru_h)
        value_int = self.value_int_head(next_gru_h)

        aux: dict[str, Tensor] = {}
        if compute_aux:
            aux = {
                "entity_importance": entity_importance,
                "next_ray": self.aux_next_ray(policy_in),
                "landmark_dir": self.aux_landmark_dir(next_gru_h),
                "inverse_dyn": self.aux_inverse_dyn(torch.cat([prev_gru_h, next_gru_h], dim=-1)),
                "combat": self.aux_combat(policy_in),
            }

        next_state = EMLRPState(
            gru_h=next_gru_h,
            spatial_map=next_map,
            landmark_keys=next_keys,
            landmark_pos=next_pos,
            landmark_valid=next_valid,
        )
        return EMLRPStepOutput(
            action_logits=action_logits,
            value_ext=value_ext,
            value_int=value_int,
            mode_logits=mode_logits,
            state=next_state,
            aux=aux,
        )

    def forward_sequence(
        self,
        obs_seq: dict[str, Tensor],
        init_state: EMLRPState,
        prev_action_seq: Tensor,
        done_seq: Tensor | None = None,
        compute_aux: bool = True,
    ) -> tuple[dict[str, Tensor], EMLRPState]:
        """Unroll policy over a `(B, T, ...)` sequence for truncated BPTT training."""
        T = obs_seq["ray_distance"].shape[1]
        state = init_state

        logits_list: list[Tensor] = []
        vext_list: list[Tensor] = []
        vint_list: list[Tensor] = []
        mode_list: list[Tensor] = []
        aux_lists: dict[str, list[Tensor]] = {}

        for t in range(T):
            step_obs = {k: v[:, t] for k, v in obs_seq.items()}
            step_done = done_seq[:, t] if done_seq is not None else None
            out = self.step(
                step_obs,
                state,
                prev_action_seq[:, t],
                done=step_done,
                compute_aux=compute_aux,
            )
            state = out.state
            logits_list.append(out.action_logits)
            vext_list.append(out.value_ext)
            vint_list.append(out.value_int)
            mode_list.append(out.mode_logits)
            for ak, av in out.aux.items():
                aux_lists.setdefault(ak, []).append(av)

        outputs: dict[str, Tensor] = {
            "action_logits": torch.stack(logits_list, dim=1),
            "value_ext": torch.stack(vext_list, dim=1),
            "value_int": torch.stack(vint_list, dim=1),
            "mode_logits": torch.stack(mode_list, dim=1),
        }
        for ak, av_list in aux_lists.items():
            outputs[f"aux_{ak}"] = torch.stack(av_list, dim=1)
        return outputs, state
