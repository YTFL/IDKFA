"""Egocentric landmark slot memory for keycards, locked doors, and objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import EMLRPConfig


class LandmarkSlotMemory(nn.Module):
    """Fixed-capacity key-value slot memory maintained in the agent's egocentric frame."""

    def __init__(self, config: EMLRPConfig) -> None:
        super().__init__()
        self.config = config
        D = config.landmark_dim

        # Candidate generator from visible entities
        self.entity_to_key = nn.Sequential(
            nn.Linear(config.entity_hidden_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )
        self.entity_salience = nn.Linear(config.entity_hidden_dim, 1)

        # Candidate generator from rays (captures doors/locked doors/items not in entity list)
        self.ray_to_key = nn.Sequential(
            nn.Linear(config.ray_embed_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )
        self.ray_to_pos = nn.Linear(config.ray_embed_dim, 2)
        self.ray_write_gate = nn.Linear(config.ray_embed_dim, 1)

        # Readout cross-attention conditioned on GRU state + binary keycard flags
        query_in = config.gru_hidden_dim + config.key_flags_count
        self.read_query = nn.Linear(query_in, D)
        self.pos_enc = nn.Linear(2, 16)
        self.slot_val_proj = nn.Linear(D + 16, D)
        self.read_out = nn.Sequential(
            nn.Linear(D + config.key_flags_count, config.landmark_readout_dim),
            nn.LayerNorm(config.landmark_readout_dim),
            nn.GELU(),
        )

    def transform_positions(
        self, landmark_pos: Tensor, delta_xy: Tensor, delta_theta: Tensor
    ) -> Tensor:
        """Rotate and translate stored landmark positions into the current egocentric frame."""
        # p_new = R(-dtheta) @ (p_old - delta_xy)
        shifted = landmark_pos - delta_xy.unsqueeze(1)
        cos_t = torch.cos(delta_theta).unsqueeze(1)
        sin_t = torch.sin(delta_theta).unsqueeze(1)
        x_new = cos_t * shifted[..., 0] + sin_t * shifted[..., 1]
        y_new = -sin_t * shifted[..., 0] + cos_t * shifted[..., 1]
        return torch.stack([x_new, y_new], dim=-1).clamp(-4.0, 4.0)

    def _write_candidate(
        self,
        keys: Tensor,
        pos: Tensor,
        valid: Tensor,
        cand_key: Tensor,
        cand_pos: Tensor,
        write_strength: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Softly match or allocate a slot for one candidate per batch element."""
        B, M, D = keys.shape
        # Similarity combines key cosine similarity and spatial proximity
        key_sim = F.cosine_similarity(keys, cand_key.unsqueeze(1), dim=-1)  # (B, M)
        pos_dist2 = ((pos - cand_pos.unsqueeze(1)) ** 2).sum(dim=-1)  # (B, M)
        match_score = (key_sim - 0.5 * pos_dist2) * valid + (-10.0) * (1.0 - valid)

        best_match_val, best_match_idx = match_score.max(dim=-1)  # (B,)
        # Empty/lowest-validity slot for allocation
        alloc_idx = valid.argmin(dim=-1)  # (B,)
        use_match = (best_match_val > 0.25) & (valid.max(dim=-1).values > 0.1)
        target_idx = torch.where(use_match, best_match_idx, alloc_idx)  # (B,)

        slot_onehot = F.one_hot(target_idx, num_classes=M).to(keys.dtype)  # (B, M)
        alpha = (write_strength.unsqueeze(-1) * slot_onehot).clamp(0.0, 1.0)  # (B, M)

        alpha_k = alpha.unsqueeze(-1)
        new_keys = (1.0 - alpha_k) * keys + alpha_k * cand_key.unsqueeze(1)
        new_pos = (1.0 - alpha_k) * pos + alpha_k * cand_pos.unsqueeze(1)
        new_valid = torch.maximum(valid, alpha)
        return new_keys, new_pos, new_valid

    def forward(
        self,
        prev_keys: Tensor,
        prev_pos: Tensor,
        prev_valid: Tensor,
        delta_xy: Tensor,
        delta_theta: Tensor,
        entity_tokens: Tensor,
        entity_features: Tensor,
        entity_mask: Tensor,
        ray_embed: Tensor,
        gru_h: Tensor,
        vitals: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Update landmark slots and compute cross-attention readout."""
        pos = self.transform_positions(prev_pos, delta_xy, delta_theta)
        keys = prev_keys
        valid = prev_valid

        # 1) Most salient visible entity candidate
        mask_bool = entity_mask > 0.5
        has_ent = mask_bool.any(dim=1)
        salience_logits = self.entity_salience(entity_tokens).squeeze(-1)  # (B, N)
        salience_logits = salience_logits.masked_fill(~mask_bool, -1e4)
        best_ent_idx = salience_logits.argmax(dim=-1)  # (B,)

        batch_idx = torch.arange(keys.shape[0], device=keys.device)
        best_token = entity_tokens[batch_idx, best_ent_idx]
        best_feat = entity_features[batch_idx, best_ent_idx]
        # Entity features: [dist/1024, sin_rel, cos_rel, ...]
        ent_pos = torch.stack(
            [best_feat[:, 0] * best_feat[:, 2], best_feat[:, 0] * best_feat[:, 1]], dim=-1
        )
        ent_key = self.entity_to_key(best_token)
        ent_strength = torch.sigmoid(salience_logits[batch_idx, best_ent_idx]) * has_ent.to(keys.dtype)

        keys, pos, valid = self._write_candidate(keys, pos, valid, ent_key, ent_pos, ent_strength)

        # 2) Global ray landmark candidate (e.g. locked doors / exits)
        ray_key = self.ray_to_key(ray_embed)
        ray_pos = torch.tanh(self.ray_to_pos(ray_embed))
        ray_strength = torch.sigmoid(self.ray_write_gate(ray_embed)).squeeze(-1)
        keys, pos, valid = self._write_candidate(keys, pos, valid, ray_key, ray_pos, ray_strength)

        # 3) Cross-attention readout conditioned on GRU hidden state + held keycards
        cfg = self.config
        key_flags = vitals[:, cfg.key_flags_start : cfg.key_flags_start + cfg.key_flags_count]
        q = self.read_query(torch.cat([gru_h, key_flags], dim=-1)).unsqueeze(1)  # (B, 1, D)

        attn_logits = torch.bmm(q, keys.transpose(1, 2)).squeeze(1) * (cfg.landmark_dim ** -0.5)
        valid_mask = valid > 0.05
        has_slot = valid_mask.any(dim=1)
        attn_logits = attn_logits.masked_fill(~valid_mask, -1e4)

        attn_w = F.softmax(attn_logits, dim=-1) * valid_mask.to(keys.dtype)
        slot_vals = self.slot_val_proj(torch.cat([keys, F.gelu(self.pos_enc(pos))], dim=-1))
        pooled = torch.bmm(attn_w.unsqueeze(1), slot_vals).squeeze(1)
        pooled = pooled * has_slot.unsqueeze(-1).to(pooled.dtype)

        landmark_readout = self.read_out(torch.cat([pooled, key_flags], dim=-1))
        return keys, pos, valid, landmark_readout
