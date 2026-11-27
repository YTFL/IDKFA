"""Adapters bridging Phase 2 ProcessedObservation with Phase 3 EMLRPPolicy tensors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from submission.features import ProcessedObservation


def single_obs_to_tensors(
    obs: ProcessedObservation,
    device: torch.device | str = "cpu",
    include_depth: bool = True,
) -> tuple[dict[str, Tensor], Tensor]:
    """Fast B=1 conversion without np.stack or per-step pin_memory allocation."""
    dev = torch.device(device)

    def _to_dev_1(arr: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=dev, dtype=torch.float32).unsqueeze(0)

    rays_t = _to_dev_1(obs.rays)
    entities_t = _to_dev_1(obs.entities)

    obs_dict: dict[str, Tensor] = {
        "ray_distance": rays_t[..., 0],
        "ray_hit_type": rays_t[..., 2:8].argmax(dim=-1),
        "entity_features": entities_t[..., :5],
        "entity_types": entities_t[..., 5:17].argmax(dim=-1),
        "entity_mask": _to_dev_1(obs.entity_mask),
        "vitals": _to_dev_1(obs.vitals),
    }
    if include_depth:
        obs_dict["depth_feat"] = _to_dev_1(obs.depth)
    prev_action_t = _to_dev_1(obs.prev_action)
    return obs_dict, prev_action_t


def processed_to_tensors(
    batch: ProcessedObservation | Sequence[ProcessedObservation],
    device: torch.device | str = "cpu",
    include_depth: bool = True,
) -> tuple[dict[str, Tensor], Tensor]:
    """Convert a single `ProcessedObservation` or batch into `(obs_dict, prev_action)` tensors."""
    if isinstance(batch, ProcessedObservation):
        return single_obs_to_tensors(batch, device=device, include_depth=include_depth)
    items = list(batch)
    if not items:
        raise ValueError("Cannot convert an empty batch of ProcessedObservation")
    if len(items) == 1:
        return single_obs_to_tensors(items[0], device=device, include_depth=include_depth)

    rays_np = np.stack([x.rays for x in items], axis=0)  # (B, 64, 8)
    vitals_np = np.stack([x.vitals for x in items], axis=0)  # (B, 17)
    entities_np = np.stack([x.entities for x in items], axis=0)  # (B, 32, 17)
    mask_np = np.stack([x.entity_mask for x in items], axis=0)  # (B, 32)
    prev_act_np = np.stack([x.prev_action for x in items], axis=0)  # (B, 9)

    dev = torch.device(device)
    if dev.type == "cuda":
        def _to_dev(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype)
            return t.pin_memory().to(dev, non_blocking=True)
    else:
        def _to_dev(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
            return torch.as_tensor(arr, dtype=dtype, device=dev)

    rays_t = _to_dev(rays_np)
    entities_t = _to_dev(entities_np)

    obs_dict: dict[str, Tensor] = {
        "ray_distance": rays_t[..., 0],
        "ray_hit_type": rays_t[..., 2:8].argmax(dim=-1),
        "entity_features": entities_t[..., :5],
        "entity_types": entities_t[..., 5:17].argmax(dim=-1),
        "entity_mask": _to_dev(mask_np),
        "vitals": _to_dev(vitals_np),
    }
    if include_depth:
        depth_np = np.stack([x.depth for x in items], axis=0)  # (B, 32)
        obs_dict["depth_feat"] = _to_dev(depth_np)
    prev_action_t = _to_dev(prev_act_np)
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
