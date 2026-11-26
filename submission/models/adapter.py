"""Adapters bridging Phase 2 ProcessedObservation with Phase 3 EMLRPPolicy tensors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from submission.features import ProcessedObservation


def processed_to_tensors(
    batch: ProcessedObservation | Sequence[ProcessedObservation],
    device: torch.device | str = "cpu",
    include_depth: bool = True,
) -> tuple[dict[str, Tensor], Tensor]:
    """Convert a single `ProcessedObservation` or batch into `(obs_dict, prev_action)` tensors."""
    items = [batch] if isinstance(batch, ProcessedObservation) else list(batch)
    if not items:
        raise ValueError("Cannot convert an empty batch of ProcessedObservation")

    rays_np = np.stack([x.rays for x in items], axis=0)  # (B, 64, 8)
    vitals_np = np.stack([x.vitals for x in items], axis=0)  # (B, 17)
    entities_np = np.stack([x.entities for x in items], axis=0)  # (B, 32, 17)
    mask_np = np.stack([x.entity_mask for x in items], axis=0)  # (B, 32)
    prev_act_np = np.stack([x.prev_action for x in items], axis=0)  # (B, 9)

    rays_t = torch.as_tensor(rays_np, dtype=torch.float32, device=device)
    entities_t = torch.as_tensor(entities_np, dtype=torch.float32, device=device)

    obs_dict: dict[str, Tensor] = {
        "ray_distance": rays_t[..., 0],
        "ray_hit_type": rays_t[..., 2:8].argmax(dim=-1),
        "entity_features": entities_t[..., :5],
        "entity_types": entities_t[..., 5:17].argmax(dim=-1),
        "entity_mask": torch.as_tensor(mask_np, dtype=torch.float32, device=device),
        "vitals": torch.as_tensor(vitals_np, dtype=torch.float32, device=device),
    }
    if include_depth:
        depth_np = np.stack([x.depth for x in items], axis=0)  # (B, 32)
        obs_dict["depth_feat"] = torch.as_tensor(depth_np, dtype=torch.float32, device=device)
    prev_action_t = torch.as_tensor(prev_act_np, dtype=torch.float32, device=device)
    return obs_dict, prev_action_t


def processed_sequence_to_tensors(
    sequences: Sequence[Sequence[ProcessedObservation]],
    device: torch.device | str = "cpu",
    include_depth: bool = True,
) -> tuple[dict[str, Tensor], Tensor]:
    """Convert `[B][T]` nested `ProcessedObservation` sequences into `(B, T, ...)` tensors."""
    if not sequences or not sequences[0]:
        raise ValueError("Cannot convert an empty sequence batch of ProcessedObservation")

    B = len(sequences)
    T = len(sequences[0])
    flat = [item for seq in sequences for item in seq]
    flat_obs, flat_act = processed_to_tensors(flat, device=device, include_depth=include_depth)

    obs_seq = {k: v.view(B, T, *v.shape[1:]) for k, v in flat_obs.items()}
    act_seq = flat_act.view(B, T, *flat_act.shape[1:])
    return obs_seq, act_seq
