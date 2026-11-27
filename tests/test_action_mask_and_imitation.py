"""Unit & contract tests for shared observation action masking, new reward penalties, and BC->PPO."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from submission.action_mask import (
    OPPOSING_AXIS_PAIRS,
    compute_obs_action_mask_np,
    compute_obs_action_mask_torch,
    eval_masked_bernoulli,
    sample_masked_bernoulli,
)
from submission.features import ACTION_DIM, ObservationPreprocessor
from submission.models import EMLRPConfig, EMLRPPolicy
from training.algo.ppo_recurrent import RecurrentPPOConfig, RecurrentPPOTrainer, RecurrentRolloutBuffer
from training.imitation.teacher import PrivilegedTeacherOracle
from training.rewards.shaping import ExtrinsicRewardShaper, RewardConfig


def _make_raw_obs(
    ammo: float = 50.0,
    reserves: tuple[float, float, float, float] = (50.0, 0.0, 0.0, 0.0),
    ray_dist_fill: float = 0.5,
    front_hit: int = 1,
    front_dist: float | None = None,
    enemy: tuple[float, float, float, int] | None = None,  # (dist_norm, sin, cos, etype)
) -> dict:
    dists = np.full(64, ray_dist_fill, dtype=np.float32)
    hits = np.ones(64, dtype=np.int64)
    hits[0] = front_hit
    if front_dist is not None:
        dists[0] = front_dist

    vitals = np.zeros(17, dtype=np.float32)
    vitals[0] = 100.0
    vitals[3] = ammo
    vitals[4:8] = reserves

    ent_f = np.zeros((32, 5), dtype=np.float32)
    ent_t = np.zeros(32, dtype=np.int64)
    ent_m = np.zeros(32, dtype=np.int8)
    if enemy is not None:
        ent_f[0, :3] = (enemy[0], enemy[1], enemy[2])
        ent_t[0] = enemy[3]
        ent_m[0] = 1

    return {
        "rays": {"distance": dists, "hit_type": hits},
        "depth": np.full((120, 160), 120, dtype=np.uint8),
        "vitals": vitals,
        "visible_entities": {"features": ent_f, "entity_type": ent_t, "mask": ent_m},
    }


def test_action_mask_no_privileged_or_training_imports():
    src = Path("submission/action_mask.py").read_text(encoding="utf-8")
    assert "vizdoom" not in src
    assert "privileged_state" not in src
    assert "training." not in src


def test_obs_action_mask_rules_and_numpy_torch_parity():
    prep = ObservationPreprocessor(normalize_vitals=True, use_depth=True)

    # 1. Empty wall, zero ammo -> ATTACK=0, USE=0 (far wall), SELECT_NEXT_WEAPON=1 (has reserve)
    p1 = prep.process(_make_raw_obs(ammo=0.0, reserves=(20.0, 0, 0, 0), front_dist=0.5), prev_action=np.zeros(9))
    # 2. Empty close wall, ammo=50, no enemy -> ATTACK=0, USE=1 (close wall/switch <=0.11), SWAP=0
    p2 = prep.process(_make_raw_obs(ammo=50.0, front_hit=1, front_dist=0.08), prev_action=np.zeros(9))
    # 3. Enemy ahead with ammo=50 -> ATTACK=1
    p3 = prep.process(_make_raw_obs(ammo=50.0, enemy=(0.3, 0.0, 1.0, 2)), prev_action=np.zeros(9))
    # 4. Melee contact at ammo=0 -> ATTACK=1
    p4 = prep.process(_make_raw_obs(ammo=0.0, reserves=(0, 0, 0, 0), enemy=(0.09, 0.0, 1.0, 1)), prev_action=np.zeros(9))
    # 5. Door ahead, prev_use=1 -> USE masked to 0
    prev_use = np.zeros(9, dtype=np.float32)
    prev_use[7] = 1.0
    p5 = prep.process(_make_raw_obs(ammo=50.0, front_hit=2, front_dist=0.15), prev_action=prev_use)

    m_np = compute_obs_action_mask_np([p1, p2, p3, p4, p5])
    assert m_np.shape == (5, ACTION_DIM)
    assert m_np[0, 6] == 0.0 and m_np[0, 7] == 0.0 and m_np[0, 8] == 1.0
    assert m_np[1, 6] == 0.0 and m_np[1, 7] == 1.0 and m_np[1, 8] == 0.0
    assert m_np[2, 6] == 1.0
    assert m_np[3, 6] == 1.0 and m_np[3, 8] == 0.0
    assert m_np[4, 7] == 0.0

    # Exact parity with torch helper
    rays_np = np.stack([x.rays for x in (p1, p2, p3, p4, p5)], axis=0)
    ents_np = np.stack([x.entities for x in (p1, p2, p3, p4, p5)], axis=0)
    m_torch = compute_obs_action_mask_torch(
        vitals=torch.as_tensor(np.stack([x.vitals for x in (p1, p2, p3, p4, p5)], axis=0)),
        rays_dist=torch.as_tensor(rays_np[..., 0]),
        rays_hit_type=torch.as_tensor(rays_np[..., 2:8].argmax(axis=-1)),
        entity_features=torch.as_tensor(ents_np[..., :5]),
        entity_types=torch.as_tensor(ents_np[..., 5:17].argmax(axis=-1)),
        entity_mask=torch.as_tensor(np.stack([x.entity_mask for x in (p1, p2, p3, p4, p5)], axis=0)),
        prev_action=torch.as_tensor(np.stack([x.prev_action for x in (p1, p2, p3, p4, p5)], axis=0)),
    ).numpy()
    np.testing.assert_allclose(m_np, m_torch, atol=1e-6)


def test_factored_masked_bernoulli_is_normalized_and_conflict_free():
    torch.manual_seed(7)
    logits = torch.randn(ACTION_DIM)
    base_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1, 0], dtype=torch.float32)

    # Check sampling never outputs opposing axes or masked bits
    for _ in range(50):
        act, lp = sample_masked_bernoulli(logits.unsqueeze(0), base_mask.unsqueeze(0), deterministic=False)
        a = act[0]
        assert a[6] == 0.0 and a[8] == 0.0
        for pos, neg in OPPOSING_AXIS_PAIRS:
            assert not (a[pos] > 0.5 and a[neg] > 0.5)
        lp_eval, _ = eval_masked_bernoulli(logits.unsqueeze(0), act, base_mask.unsqueeze(0))
        assert torch.allclose(lp, lp_eval, atol=1e-6)

    # Sum probabilities over ALL valid binary vectors must equal 1.0 exactly
    total_prob = 0.0
    for bits in itertools.product((0.0, 1.0), repeat=ACTION_DIM):
        a_t = torch.tensor(bits, dtype=torch.float32)
        if torch.any(a_t * (1.0 - base_mask) > 0.5):
            continue
        if any(a_t[p] > 0.5 and a_t[n] > 0.5 for p, n in OPPOSING_AXIS_PAIRS):
            continue
        lp, _ = eval_masked_bernoulli(logits.unsqueeze(0), a_t.unsqueeze(0), base_mask.unsqueeze(0))
        total_prob += float(torch.exp(lp).item())
    assert total_prob == pytest.approx(1.0, abs=1e-5)


def test_reward_shaper_anti_exploit_penalties():
    shaper = ExtrinsicRewardShaper(RewardConfig(spin_window=8, spin_min_deg=240.0, spin_max_disp=60.0))

    # Dry-fire at empty wall with ammo=0
    raw_empty = _make_raw_obs(ammo=0.0, front_dist=0.03)
    act_bad = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0], dtype=np.int8)
    _, bd1 = shaper.compute(
        obs=raw_empty,
        privileged_state={"x": 100.0, "y": 100.0, "angle": 0.0, "health": 100.0, "killcount": 0},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        action=act_bad,
        pre_obs=raw_empty,
    )
    assert bd1.ammo_waste == pytest.approx(-0.08)
    assert bd1.action_hygiene == pytest.approx(-0.02)  # FWD+BWD conflict
    assert bd1.wall_hug == pytest.approx(-0.03)

    # Consecutive held USE + wasted shot into wall (ammo 50 -> 49, no target, no kill)
    raw_shot0 = _make_raw_obs(ammo=50.0, front_dist=0.5)
    raw_shot1 = _make_raw_obs(ammo=49.0, front_dist=0.5)
    shaper.compute(
        obs=raw_shot0,
        privileged_state={"x": 110.0, "y": 100.0, "angle": 10.0, "health": 100.0, "killcount": 0},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        action=np.array([1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8),
        pre_obs=raw_shot0,
    )
    _, bd2 = shaper.compute(
        obs=raw_shot1,
        privileged_state={"x": 120.0, "y": 100.0, "angle": 20.0, "health": 100.0, "killcount": 0},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
        action=np.array([1, 0, 0, 0, 0, 0, 1, 1, 0], dtype=np.int8),
        pre_obs=raw_shot0,
    )
    assert bd2.ammo_waste == pytest.approx(-0.03)
    assert bd2.action_hygiene == pytest.approx(-0.04)

    # Spin-loop detection: 8 steps turning +35 deg/step in small circle (disp < 60, net yaw = 280 >= 240)
    shaper_spin = ExtrinsicRewardShaper(
        RewardConfig(stuck_window=16, long_stuck_window=48, spin_window=8, spin_min_deg=240.0, spin_max_disp=60.0)
    )
    last_stuck = 0.0
    for step in range(9):
        ang = (step * 35.0) % 360.0
        _, bd_s = shaper_spin.compute(
            obs={},
            privileged_state={"x": 200.0 + 18.0 * np.cos(np.radians(ang)), "y": 200.0 + 18.0 * np.sin(np.radians(ang)), "angle": ang, "health": 100.0},
            base_reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )
        last_stuck = bd_s.stuck
    assert last_stuck == pytest.approx(-0.05)


def test_ppo_trainer_masked_update_and_critic_warmup():
    cfg = EMLRPConfig(depth_feat_dim=32)
    policy = EMLRPPolicy(cfg)
    ppo_cfg = RecurrentPPOConfig(
        rollout_len=8, seq_len=4, minibatch_seqs=2, ppo_epochs=1,
        use_action_mask=True, critic_warmup_updates=1, bc_kl_coef=0.05, use_bf16=False,
    )
    trainer = RecurrentPPOTrainer(policy, config=ppo_cfg, device="cpu", seed=1)
    prep = ObservationPreprocessor(normalize_vitals=True, use_depth=True)
    buf = RecurrentRolloutBuffer(rollout_len=8, num_envs=2, seq_len=4, include_depth=True, use_action_mask=True)

    state = policy.init_state(2, device="cpu")
    reset_mask = np.ones(2, dtype=bool)
    obs_list = [prep.process(_make_raw_obs(ammo=0.0, front_dist=0.5)) for _ in range(2)]
    for _ in range(8):
        act, lp, ve, vi, next_state = trainer.select_actions(obs_list, state, reset_mask)
        assert np.all(act[:, 6] == 0)  # ATTACK masked out cleanly
        buf.add(
            obs=obs_list,
            reset_before_step=reset_mask,
            state_before_step=state,
            action=act.astype(np.float32),
            log_prob=lp,
            value_ext=ve,
            value_int=vi,
            reward_ext=np.zeros(2, dtype=np.float32),
            reward_int=np.zeros(2, dtype=np.float32),
            done=np.zeros(2, dtype=bool),
            next_ray_distance=np.full((2, 64), 0.5, dtype=np.float32),
            aux_combat=np.zeros((2, 2), dtype=np.float32),
            aux_landmark_dir=np.zeros((2, 2), dtype=np.float32),
            aux_landmark_mask=np.zeros(2, dtype=np.float32),
        )
        state = next_state
        reset_mask = np.zeros(2, dtype=bool)

    buf.compute_returns_and_advantages(np.zeros(2, np.float32), np.zeros(2, np.float32), 0.995, 0.99, 0.95, 0.1)
    w_before = policy.policy_head[-1].weight.detach().clone()
    trainer.update(buf, progress_frac=0.0, update_idx=1, bc_ref_policy=policy)
    w_after = policy.policy_head[-1].weight.detach()
    # Critic warmup update_idx=1 <= 1 must freeze actor weights strictly!
    assert torch.allclose(w_before, w_after, atol=1e-7)


def test_teacher_oracle_respects_action_mask():
    oracle = PrivilegedTeacherOracle(
        map_meta={
            "room_centers": {0: (128.0, 128.0), 1: (448.0, 128.0)},
            "room_bounds": {0: (0.0, 0.0, 256.0, 256.0), 1: (320.0, 0.0, 576.0, 256.0)},
            "nav_edges": [(0, 1, 0)],
            "exit_xy": (448.0, 128.0),
        }
    )
    raw = _make_raw_obs(ammo=0.0, front_hit=2, front_dist=0.09)
    act = oracle.act(raw, {"x": 220.0, "y": 128.0, "angle": 0.0, "health": 100.0, "keys": np.zeros(6)})
    assert act[6] == 0  # never fires empty
    assert act[7] == 1  # single-tap USE on closed door
    act2 = oracle.act(raw, {"x": 220.0, "y": 128.0, "angle": 0.0, "health": 100.0, "keys": np.zeros(6)})
    assert act2[7] == 0  # released on next step
