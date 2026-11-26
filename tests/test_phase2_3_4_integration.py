"""End-to-end integration tests across Phase 2 (features), Phase 3 (EMLRP), and Phase 4 (rewards)."""

from __future__ import annotations

import numpy as np
import torch

from submission.features import ObservationPreprocessor
from submission.models import (
    EMLRPConfig,
    EMLRPPolicy,
    processed_sequence_to_tensors,
    processed_to_tensors,
)
from training.rewards.curiosity import EpisodicNoveltyBuffer, RNDCuriosity
from training.rewards.shaping import ExtrinsicRewardShaper


def _make_raw_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mask = np.zeros(32, dtype=np.int8)
    mask[:5] = 1
    ent_types = np.zeros(32, dtype=np.int64)
    ent_types[:5] = rng.integers(1, 12, size=5)
    return {
        "rgb": rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8),
        "depth": rng.integers(0, 256, size=(120, 160), dtype=np.uint8),
        "vitals": rng.uniform(0.0, 100.0, size=(17,)).astype(np.float32),
        "rays": {
            "distance": rng.uniform(0.05, 1.0, size=(64,)).astype(np.float32),
            "hit_type": rng.integers(0, 6, size=(64,), dtype=np.int64),
        },
        "visible_entities": {
            "features": rng.normal(0.0, 1.0, size=(32, 5)).astype(np.float32),
            "entity_type": ent_types,
            "mask": mask,
        },
    }


def test_phase2_to_phase3_step_and_sequence() -> None:
    prep = ObservationPreprocessor()
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg)

    # Single-step flow
    raw0 = _make_raw_obs(seed=1)
    act0 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    p0 = prep.process(raw0, prev_action=act0)
    obs_t, prev_act_t = processed_to_tensors(p0)

    state = model.init_state(1)
    out = model.step(obs_t, state, prev_act_t, compute_aux=False)
    assert out.action_logits.shape == (1, cfg.action_dim)
    assert torch.isfinite(out.action_logits).all()

    # Batched sequence flow for BPTT
    seqs = []
    for b in range(2):
        prep.reset()
        seq = []
        for t in range(4):
            a = np.random.default_rng(10 * b + t).integers(0, 2, size=9, dtype=np.int8)
            seq.append(prep.process(_make_raw_obs(seed=100 * b + t), prev_action=a))
        seqs.append(seq)

    obs_seq, act_seq = processed_sequence_to_tensors(seqs)
    seq_out, _ = model.forward_sequence(obs_seq, model.init_state(2), act_seq, compute_aux=True)
    assert seq_out["action_logits"].shape == (2, 4, cfg.action_dim)
    loss = seq_out["action_logits"].pow(2).mean() + seq_out["value_ext"].pow(2).mean()
    loss.backward()


def test_phase2_phase3_phase4_combined_loop() -> None:
    prep = ObservationPreprocessor()
    model = EMLRPPolicy(EMLRPConfig()).eval()
    shaper = ExtrinsicRewardShaper()
    rnd = RNDCuriosity(scale=0.1)
    novelty = EpisodicNoveltyBuffer(c_novelty=0.05)

    prep.reset()
    shaper.reset()
    novelty.reset()
    state = model.init_state(1)
    prev_act = np.zeros(9, dtype=np.int8)

    for step_idx in range(5):
        raw_obs = _make_raw_obs(seed=50 + step_idx)
        p_obs = prep.process(raw_obs, prev_action=prev_act)
        obs_t, prev_act_t = processed_to_tensors(p_obs)

        with torch.inference_mode():
            out = model.step(obs_t, state, prev_act_t, compute_aux=False)
        state = out.state
        prev_act = (torch.sigmoid(out.action_logits[0]) > 0.5).to(torch.int8).numpy()

        priv = {
            "x": float(step_idx * 10.0),
            "y": 0.0,
            "health": 100.0,
            "armor": 0.0,
            "kills": step_idx,
            "keys": 0,
            "visited_cells": step_idx + 1,
        }
        r_ext, bd = shaper.compute(
            obs=raw_obs,
            privileged_state=priv,
            base_reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )
        r_rnd = rnd.compute_intrinsic_reward(p_obs, update_predictor=True)
        r_nov = novelty.compute_novelty_bonus(p_obs)
        total_r = r_ext + r_rnd + r_nov
        assert np.isfinite(total_r)
        assert bd.kill >= 0.0
