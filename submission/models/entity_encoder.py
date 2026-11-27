"""Permutation-invariant set transformer for variable visible entities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import EMLRPConfig


class EntitySetEncoder(nn.Module):
    """Encode a variable-size masked set of visible entities via self-attention and pooling."""

    def __init__(self, config: EMLRPConfig) -> None:
        super().__init__()
        self.config = config
        self.type_embed = nn.Embedding(
            config.entity_type_vocab, config.entity_type_embed_dim, padding_idx=0
        )
        in_dim = config.entity_feat_dim + config.entity_type_embed_dim
        h_dim = config.entity_hidden_dim

        self.entity_mlp = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=config.entity_heads,
            dim_feedforward=h_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.entity_layers, enable_nested_tensor=False
        )

        self.ray_to_query = nn.Linear(config.ray_embed_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(h_dim, config.entity_embed_dim),
            nn.LayerNorm(config.entity_embed_dim),
            nn.GELU(),
        )

        # Auxiliary per-entity threat/importance logits (used during training).
        self.importance_head = nn.Linear(h_dim, 1)

    def forward(
        self,
        entity_features: Tensor,
        entity_types: Tensor,
        entity_mask: Tensor,
        ray_embed: Tensor,
        ablate_attention: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode visible entities.

        Args:
            entity_features: `(B, N, 5)` float features.
            entity_types: `(B, N)` integer class IDs in `[0, 11]`.
            entity_mask: `(B, N)` binary mask (`1` valid, `0` padded).
            ray_embed: `(B, ray_embed_dim)` conditioning context for attention pooling.
            ablate_attention: If True, bypass self-attention and use masked mean pooling.

        Returns:
            entity_summary: `(B, entity_embed_dim)` permutation-invariant summary.
            entity_tokens: `(B, N, entity_hidden_dim)` contextualized per-entity tokens (zeroed on pad).
            entity_importance: `(B, N)` auxiliary importance logits.
        """
        mask_bool = entity_mask > 0.5  # (B, N)
        has_any = mask_bool.any(dim=1)  # (B,)
        mask_f = mask_bool.unsqueeze(-1).to(entity_features.dtype)

        type_emb = self.type_embed(entity_types.long())
        x = torch.cat([entity_features, type_emb], dim=-1)
        tokens = self.entity_mlp(x) * mask_f

        if ablate_attention:
            v = self.v_proj(tokens) * mask_f
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = (v.sum(dim=1) / denom) * has_any.unsqueeze(-1).to(v.dtype)
        else:
            # Guard against all-masked batches so MultiheadAttention never divides by zero.
            safe_key_padding_mask = ~mask_bool
            if not bool(has_any.all()):
                safe_key_padding_mask = safe_key_padding_mask.clone()
                safe_key_padding_mask[~has_any, 0] = False

            tokens = self.transformer(tokens, src_key_padding_mask=safe_key_padding_mask)
            tokens = tokens * mask_f

            # Ray-conditioned cross-attention pooling over valid entities.
            q = self.ray_to_query(ray_embed).unsqueeze(1)  # (B, 1, H)
            k = self.k_proj(tokens)  # (B, N, H)
            v = self.v_proj(tokens)  # (B, N, H)
            attn_logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * (tokens.shape[-1] ** -0.5)
            attn_logits = attn_logits.masked_fill(~mask_bool, -1e4)

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = attn_weights * mask_bool.to(attn_weights.dtype)
            pooled = torch.bmm(attn_weights.unsqueeze(1), v).squeeze(1)  # (B, H)
            pooled = pooled * has_any.unsqueeze(-1).to(pooled.dtype)

        entity_summary = self.out_proj(pooled) * has_any.unsqueeze(-1).to(pooled.dtype)
        entity_importance = self.importance_head(tokens).squeeze(-1)
        return entity_summary, tokens, entity_importance
