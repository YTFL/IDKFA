"""Shared observation-based action masking for DoomAgent and RecurrentPPOTrainer.

Strictly operates on ego-centric observation fields (allowed by docs/RULES.md §1)
to prevent:
1. Empty-air punching / wall-spraying (`ATTACK` requires enemy/close-barrel or melee contact).
2. Held edge-trigger spam on `USE` and `SELECT_NEXT_WEAPON`.
3. Weapon switching when ammo remains in the active weapon or no reserves exist.
4. Simultaneous opposing movement/turning inputs (`FWD/BWD`, `LEFT/RIGHT`, `TURN_L/TURN_R`),
   using factored conditional masking so PPO log-probs and entropies stay exact.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .features import ACTION_DIM, ProcessedObservation

# Center-cone ray index subsets (64 rays, counter-clockwise from 0 = straight ahead, 5.625 deg/ray)
FRONT_CONE_22_RAYS: tuple[int, ...] = (0, 1, 2, 3, 4, 60, 61, 62, 63)
FRONT_CONE_45_RAYS: tuple[int, ...] = tuple(range(0, 9)) + tuple(range(56, 64))

# Opposing binary axis pairs: (positive_idx, negative_idx)
OPPOSING_AXIS_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),  # MOVE_FORWARD vs MOVE_BACKWARD
    (2, 3),  # MOVE_LEFT vs MOVE_RIGHT
    (4, 5),  # TURN_LEFT vs TURN_RIGHT
)


def compute_obs_action_mask_np(obs: ProcessedObservation | Sequence[ProcessedObservation]) -> np.ndarray:
    """Compute `(ACTION_DIM,)` or `(B, ACTION_DIM)` float32 validity mask from ProcessedObservation."""
    single = isinstance(obs, ProcessedObservation)
    items = [obs] if single else list(obs)
    if not items:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)

    rays_np = np.stack([x.rays for x in items], axis=0)
    ents_np = np.stack([x.entities for x in items], axis=0)
    ent_mask = np.stack([x.entity_mask for x in items], axis=0)
    vitals = np.stack([x.vitals for x in items], axis=0)
    prev_act = np.stack([x.prev_action for x in items], axis=0)

    rays_dist = rays_np[..., 0]
    rays_hit = rays_np[..., 2:8].argmax(axis=-1)
    ent_feat = ents_np[..., :5]
    ent_type = ents_np[..., 5:17].argmax(axis=-1)

    mask = compute_obs_action_mask_np_arrays(
        vitals=vitals,
        rays_dist=rays_dist,
        rays_hit_type=rays_hit,
        entity_features=ent_feat,
        entity_types=ent_type,
        entity_mask=ent_mask,
        prev_action=prev_act,
    )
    return mask[0] if single else mask


def compute_obs_action_mask_np_arrays(
    vitals: np.ndarray,
    rays_dist: np.ndarray,
    rays_hit_type: np.ndarray,
    entity_features: np.ndarray,
    entity_types: np.ndarray,
    entity_mask: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Vectorized NumPy observation action mask supporting arbitrary leading batch dims `(..., ...)`."""
    lead_shape = vitals.shape[:-1]
    mask = np.ones((*lead_shape, ACTION_DIM), dtype=np.float32)

    has_ammo = vitals[..., 3] > 0.005
    has_reserve = np.sum(vitals[..., 4:8], axis=-1) > 0.005

    valid_ent = entity_mask > 0.5
    is_enemy_ent = valid_ent & ((entity_types == 1) | (entity_types == 2) | (entity_types == 3))
    ent_cos = entity_features[..., 2]
    ent_dist = entity_features[..., 0]

    enemy_in_cone = np.any(is_enemy_ent & (ent_cos > 0.35), axis=-1)
    barrel_close = np.any(valid_ent & (entity_types == 10) & (ent_dist <= 0.35) & (ent_cos > 0.5), axis=-1)

    front22_hit = np.take(rays_hit_type, FRONT_CONE_22_RAYS, axis=-1)
    enemy_ray = np.any(front22_hit == 4, axis=-1)

    melee_ent = valid_ent & ((entity_types == 1) | (entity_types == 2))
    melee_contact = np.any(melee_ent & (ent_dist <= 0.11) & (ent_cos > 0.5), axis=-1)

    attack_ok = np.where(has_ammo, enemy_in_cone | barrel_close | enemy_ray, melee_contact)
    mask[..., 6] = attack_ok.astype(np.float32)

    # USE (idx 7): enforce 0 -> 1 edge release and require front-cone door, locked door, or close wall/switch
    front45_hit = np.take(rays_hit_type, FRONT_CONE_45_RAYS, axis=-1)
    front45_dist = np.take(rays_dist, FRONT_CONE_45_RAYS, axis=-1)
    near_door = np.any(((front45_hit == 2) | (front45_hit == 3)) & (front45_dist <= 0.25), axis=-1)
    near_switch_wall = np.any((front45_hit == 1) & (front45_dist <= 0.11), axis=-1)
    use_released = prev_action[..., 7] <= 0.5
    mask[..., 7] = (use_released & (near_door | near_switch_wall)).astype(np.float32)

    # SELECT_NEXT_WEAPON (idx 8): edge release, current clip empty, reserve ammo exists
    swap_released = prev_action[..., 8] <= 0.5
    mask[..., 8] = (swap_released & (~has_ammo) & has_reserve).astype(np.float32)

    return mask


def compute_obs_action_mask_torch(
    vitals: Tensor,
    rays_dist: Tensor,
    rays_hit_type: Tensor,
    entity_features: Tensor,
    entity_types: Tensor,
    entity_mask: Tensor,
    prev_action: Tensor,
) -> Tensor:
    """PyTorch equivalent of `compute_obs_action_mask_np_arrays` with arbitrary leading dims."""
    device = vitals.device
    lead_shape = vitals.shape[:-1]
    mask = torch.ones((*lead_shape, ACTION_DIM), dtype=torch.float32, device=device)

    has_ammo = vitals[..., 3] > 0.005
    has_reserve = vitals[..., 4:8].sum(dim=-1) > 0.005

    valid_ent = entity_mask > 0.5
    is_enemy_ent = valid_ent & ((entity_types == 1) | (entity_types == 2) | (entity_types == 3))
    ent_cos = entity_features[..., 2]
    ent_dist = entity_features[..., 0]

    enemy_in_cone = (is_enemy_ent & (ent_cos > 0.35)).any(dim=-1)
    barrel_close = (valid_ent & (entity_types == 10) & (ent_dist <= 0.35) & (ent_cos > 0.5)).any(dim=-1)

    front22_idx = torch.tensor(FRONT_CONE_22_RAYS, dtype=torch.long, device=device)
    front22_hit = rays_hit_type.index_select(dim=-1, index=front22_idx)
    enemy_ray = (front22_hit == 4).any(dim=-1)

    melee_ent = valid_ent & ((entity_types == 1) | (entity_types == 2))
    melee_contact = (melee_ent & (ent_dist <= 0.11) & (ent_cos > 0.5)).any(dim=-1)

    attack_ok = torch.where(has_ammo, enemy_in_cone | barrel_close | enemy_ray, melee_contact)
    mask[..., 6] = attack_ok.float()

    front45_idx = torch.tensor(FRONT_CONE_45_RAYS, dtype=torch.long, device=device)
    front45_hit = rays_hit_type.index_select(dim=-1, index=front45_idx)
    front45_dist = rays_dist.index_select(dim=-1, index=front45_idx)
    near_door = (((front45_hit == 2) | (front45_hit == 3)) & (front45_dist <= 0.25)).any(dim=-1)
    near_switch_wall = ((front45_hit == 1) & (front45_dist <= 0.11)).any(dim=-1)
    use_released = prev_action[..., 7] <= 0.5
    mask[..., 7] = (use_released & (near_door | near_switch_wall)).float()

    swap_released = prev_action[..., 8] <= 0.5
    mask[..., 8] = (swap_released & (~has_ammo) & has_reserve).float()

    return mask


def sample_masked_bernoulli(
    logits: Tensor,
    base_mask: Tensor,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample factored masked MultiBinary actions and exact joint log-probs.

    For each opposing pair `(pos, neg)`, `pos` is sampled using `base_mask[..., pos]`
    and `neg` is masked to 0 whenever `a_pos == 1`, keeping the joint distribution normalized
    and identical between rollout collection and PPO updates.
    """
    logits_f = logits.float()
    probs = torch.sigmoid(logits_f)
    if deterministic:
        actions = (logits_f > 0.0).float() * base_mask
        # For deterministic mode, if both opposing logits are positive, pick stronger one
        for pos, neg in OPPOSING_AXIS_PAIRS:
            both_pos = (actions[..., pos] > 0.5) & (actions[..., neg] > 0.5)
            prefer_neg = logits_f[..., neg] > logits_f[..., pos]
            drop_pos = both_pos & prefer_neg
            actions[..., pos] = torch.where(drop_pos, torch.zeros_like(actions[..., pos]), actions[..., pos])
            actions[..., neg] = torch.where(
                actions[..., pos] > 0.5, torch.zeros_like(actions[..., neg]), actions[..., neg]
            )
    else:
        u = torch.rand(probs.shape, dtype=probs.dtype, device=probs.device, generator=generator)
        actions = (u < probs).float() * base_mask
        for pos, neg in OPPOSING_AXIS_PAIRS:
            actions[..., neg] = actions[..., neg] * (1.0 - actions[..., pos])

    log_probs, _ = eval_masked_bernoulli(logits_f, actions, base_mask)
    return actions, log_probs


def eval_masked_bernoulli(
    logits: Tensor,
    actions: Tensor,
    base_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate exact joint log-probability `(...,)` and entropy `(...,)` under factored mask."""
    logits_f = logits.float()
    act_f = actions.float()
    mask_f = base_mask.float()

    # Exact negative-axis conditional effective mask during realized action
    eff_mask = mask_f.clone()
    for pos, neg in OPPOSING_AXIS_PAIRS:
        eff_mask[..., neg] = mask_f[..., neg] * (1.0 - act_f[..., pos])

    # Binary cross-entropy with logits gives -log P(a_d | logits_d) stably
    neg_logp = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_f, act_f, reduction="none"
    )
    joint_logp = -(neg_logp * eff_mask).sum(dim=-1)

    # Analytical entropy of factored masked Bernoulli distribution:
    # For independent dims: mask * H(d)
    # For opposing pair (pos, neg): mask_pos * H(pos) + (1 - mask_pos * p_pos) * mask_neg * H(neg)
    probs = torch.sigmoid(logits_f)
    elem_entropy = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits_f, probs, reduction="none")
    )
    ent_weights = mask_f.clone()
    for pos, neg in OPPOSING_AXIS_PAIRS:
        p_pos_active = mask_f[..., pos] * probs[..., pos]
        ent_weights[..., neg] = mask_f[..., neg] * (1.0 - p_pos_active)
    joint_entropy = (elem_entropy * ent_weights).sum(dim=-1)

    return joint_logp, joint_entropy


def compute_masked_action_probs(logits: Tensor, base_mask: Tensor) -> Tensor:
    """Return effective marginal Bernoulli probabilities zeroed on masked bits (for telemetry)."""
    logits_f = logits.float()
    probs = torch.sigmoid(logits_f) * base_mask.float()
    return probs
