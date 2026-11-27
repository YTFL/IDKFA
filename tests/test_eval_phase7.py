"""Tests for Phase 7: spatial/landmark memory, robustness, ablations, generalization, and failure taxonomy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from doom_env import make_env
from doom_env import spaces as S
from submission.agent import DoomAgent
from submission.models import AblationFlags, EMLRPConfig, EMLRPPolicy
from training.eval import evaluate_on_manifest, score_agent_episode
from training.tools.analyze_failures import classify_failure


def _synthetic_tensor_obs(
    batch: int = 1,
    ray_hit: int = 1,
    ray_dist: float = 0.6,
    has_key_ent: bool = False,
    key_flag: float = 0.0,
) -> dict[str, torch.Tensor]:
    hit_t = torch.full((batch, 64), ray_hit, dtype=torch.long)
    dist_t = torch.full((batch, 64), ray_dist, dtype=torch.float32)
    ent_feat = torch.zeros(batch, 32, 5, dtype=torch.float32)
    ent_type = torch.zeros(batch, 32, dtype=torch.long)
    ent_mask = torch.zeros(batch, 32, dtype=torch.float32)
    if has_key_ent:
        ent_feat[:, 0] = torch.tensor([0.3, 0.0, 1.0, 0.0, 0.0])
        ent_type[:, 0] = 9  # key
        ent_mask[:, 0] = 1.0
    vitals = torch.zeros(batch, 17, dtype=torch.float32)
    vitals[:, 0] = 1.0
    vitals[:, 8] = key_flag
    vitals[:, 14] = 0.5
    return {
        "ray_distance": dist_t,
        "ray_hit_type": hit_t,
        "entity_features": ent_feat,
        "entity_types": ent_type,
        "entity_mask": ent_mask,
        "vitals": vitals,
        "depth_feat": torch.full((batch, 32), 0.4, dtype=torch.float32),
    }


def _raw_obs(seed: int = 0, locked_door_ahead: bool = False, health: float = 100.0) -> dict:
    rng = np.random.default_rng(seed)
    hit_type = rng.integers(0, 3, size=(64,), dtype=np.int64)
    ray_dist = rng.uniform(0.2, 0.9, size=(64,)).astype(np.float32)
    if locked_door_ahead:
        hit_type[0] = 3  # locked_door straight ahead
        ray_dist[0] = 0.25
    vitals = np.zeros(17, dtype=np.float32)
    vitals[0] = health
    return {
        "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": np.full((120, 160), 100, dtype=np.uint8),
        "vitals": vitals,
        "rays": {"distance": ray_dist, "hit_type": hit_type},
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros(32, dtype=np.int64),
            "mask": np.zeros(32, dtype=np.int8),
        },
    }


def test_spatial_map_warp_exact_translation_and_rotation() -> None:
    cfg = EMLRPConfig()
    policy = EMLRPPolicy(cfg).eval()
    smap_mod = policy.spatial_map
    S_dim = cfg.map_size

    # cell_xy holds [x_fwd, y_left] indexed as [H=y_left, W=x_fwd]
    coords_1d = torch.linspace(-1.0 + 1.0 / S_dim, 1.0 - 1.0 / S_dim, S_dim)
    grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    blob = torch.exp(-((grid_x - 0.5) ** 2 + (grid_y - 0.0) ** 2) / (2 * 0.08**2))
    smap = torch.zeros(1, cfg.map_channels, S_dim, S_dim)
    smap[0, 0] = blob

    def _centroid(m: torch.Tensor) -> tuple[float, float]:
        w = m[0, 0].clamp(min=0.0)
        tot = w.sum().item()
        cx = (w * grid_x).sum().item() / tot
        cy = (w * grid_y).sum().item() / tot
        return cx, cy

    # Forward motion +0.25 -> peak shifts from x=+0.50 to x=+0.25
    shifted = smap_mod.warp_map(smap, torch.tensor([[0.25, 0.0]]), torch.tensor([0.0]))
    sx, sy = _centroid(shifted)
    assert sx == pytest.approx(0.25, abs=0.08)
    assert sy == pytest.approx(0.0, abs=0.08)

    # 90 deg CCW turn -> peak at (0.25, 0.0) rotates to right (0.0, -0.25)
    turned = smap_mod.warp_map(shifted, torch.tensor([[0.0, 0.0]]), torch.tensor([math.pi / 2]))
    tx, ty = _centroid(turned)
    assert tx == pytest.approx(0.0, abs=0.08)
    assert ty == pytest.approx(-0.25, abs=0.08)


def test_landmark_slot_write_track_and_readout() -> None:
    cfg = EMLRPConfig()
    mem = EMLRPPolicy(cfg).landmark_memory.eval()
    keys = torch.zeros(1, cfg.landmark_slots, cfg.landmark_dim)
    pos = torch.zeros(1, cfg.landmark_slots, 2)
    valid = torch.zeros(1, cfg.landmark_slots)

    cand_key = torch.randn(1, cfg.landmark_dim)
    cand_pos = torch.tensor([[0.6, 0.0]])
    keys, pos, valid = mem._write_candidate(keys, pos, valid, cand_key, cand_pos, torch.tensor([0.95]))
    assert float(valid.max()) == pytest.approx(0.95, abs=1e-5)

    # Track over 5 forward steps (delta_x = 0.05) with zero-strength distractors
    gru_h = torch.zeros(1, cfg.gru_hidden_dim)
    vitals_nokey = torch.zeros(1, cfg.vitals_dim)
    vitals_key = torch.zeros(1, cfg.vitals_dim)
    vitals_key[0, cfg.key_flags_start] = 1.0

    for _ in range(5):
        keys, pos, valid, _ = mem.step_with_candidates(
            keys, pos, valid,
            delta_xy=torch.tensor([[0.05, 0.0]]),
            delta_theta=torch.tensor([0.0]),
            ent_key=torch.zeros(1, cfg.landmark_dim),
            ent_pos=torch.zeros(1, 2),
            ent_strength=torch.tensor([0.0]),
            ray_key=torch.randn(1, cfg.landmark_dim),
            ray_pos=torch.tensor([[-0.5, 0.5]]),
            ray_strength=torch.tensor([0.0]),
            gru_h=gru_h,
            vitals=vitals_nokey,
        )

    slot_idx = int(valid[0].argmax())
    assert float(pos[0, slot_idx, 0]) == pytest.approx(0.95 * 0.6 - 0.25, abs=1e-4)

    _, _, _, read_no_key = mem.step_with_candidates(
        keys, pos, valid, torch.zeros(1, 2), torch.zeros(1),
        torch.zeros(1, cfg.landmark_dim), torch.zeros(1, 2), torch.zeros(1),
        torch.zeros(1, cfg.landmark_dim), torch.zeros(1, 2), torch.zeros(1),
        gru_h, vitals_nokey,
    )
    _, _, _, read_with_key = mem.step_with_candidates(
        keys, pos, valid, torch.zeros(1, 2), torch.zeros(1),
        torch.zeros(1, cfg.landmark_dim), torch.zeros(1, 2), torch.zeros(1),
        torch.zeros(1, cfg.landmark_dim), torch.zeros(1, 2), torch.zeros(1),
        gru_h, vitals_key,
    )
    assert not torch.allclose(read_no_key, read_with_key, atol=1e-4)


def test_synthetic_key_seen_wander_and_return_memory() -> None:
    torch.manual_seed(0)
    policy = EMLRPPolicy(EMLRPConfig()).eval()

    def _run_sequence(see_locked_door_early: bool):
        st = policy.init_state(1)
        prev_act = torch.zeros(1, 9)
        with torch.inference_mode():
            for t in range(26):
                if t < 3:
                    obs = _synthetic_tensor_obs(
                        ray_hit=3 if see_locked_door_early else 1,
                        ray_dist=0.25 if see_locked_door_early else 0.9,
                        has_key_ent=see_locked_door_early,
                        key_flag=0.0,
                    )
                    act_cmd = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
                else:
                    # 20+ wandering steps with empty entities + key pickup at t >= 23
                    obs = _synthetic_tensor_obs(ray_hit=1, ray_dist=0.8, has_key_ent=False, key_flag=1.0 if t >= 23 else 0.0)
                    act_cmd = torch.tensor([[1, 0, 0, 0, 1 if t < 15 else 0, 1 if t >= 15 else 0, 0, 0, 0]], dtype=torch.float32)
                out = policy.step(obs, st, prev_act, compute_aux=False)
                st = out.state
                prev_act = act_cmd
        return out.action_logits, st

    logits_door, state_door = _run_sequence(see_locked_door_early=True)
    logits_ctrl, _ = _run_sequence(see_locked_door_early=False)
    assert float(state_door.landmark_valid.max()) > 0.05
    assert float(state_door.spatial_map.norm()) > 0.0
    assert not torch.allclose(logits_door, logits_ctrl, atol=1e-4)


def test_robustness_early_death_no_key_and_locked_door_first() -> None:
    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.reset()

    # Locked door visible before any key pickup + early death vitals (health = 0)
    for step_obs in (
        _raw_obs(seed=1, locked_door_ahead=True, health=100.0),
        _raw_obs(seed=2, locked_door_ahead=True, health=0.0),
    ):
        act = agent.act(step_obs)
        assert act.shape == (9,) and act.dtype == np.int8 and np.all((act == 0) | (act == 1))

    # Combat arena map with no keycards via full DoomAgent wrapper
    val_combat = sorted(Path("maps/dataset/val").glob("combat_arena_*.wad"))[0]
    ep = score_agent_episode(agent, val_combat, episode_ticks=150, seed=0)
    assert ep["keys_found"] == 0
    assert ep["mean_ms"] < 40.0


def test_multi_seed_eval_report_and_ablation_hooks() -> None:
    cfg = EMLRPConfig()
    policy = EMLRPPolicy(cfg).eval()
    obs = _synthetic_tensor_obs(batch=1, ray_hit=3, has_key_ent=True, key_flag=1.0)
    st = policy.init_state(1)
    prev = torch.zeros(1, 9)

    with torch.inference_mode():
        base_out = policy.step(obs, st, prev, compute_aux=False, ablations=AblationFlags())
        for flag_name in ("ablate_map", "ablate_landmarks", "ablate_entity_attn", "ablate_mode_head", "ablate_depth"):
            abl_out = policy.step(obs, st, prev, compute_aux=False, ablations=AblationFlags(**{flag_name: True}))
            assert torch.isfinite(abl_out.action_logits).all()
            assert not torch.allclose(base_out.action_logits, abl_out.action_logits, atol=1e-6), flag_name

    from submission.features import ObservationPreprocessor

    prep = ObservationPreprocessor()
    report = evaluate_on_manifest(
        policy=policy,
        preprocessor=prep,
        manifest_path="maps/dataset/manifest.json",
        max_maps_per_tier=1,
        episode_ticks=120,
        seeds=[0, 1],
        device="cpu",
    )
    assert "weighted_score_std" in report
    assert "p99_ms_per_step" in report
    for tier_stats in report["tiers"].values():
        assert tier_stats["count"] == 2.0
        assert "survival_rate" in tier_stats
        assert "locked_door_rate" in tier_stats


@pytest.mark.parametrize(
    ("wad_rel", "map_name"),
    [
        ("maps/two_rooms.wad", "MAP01"),
        ("freedoom-0.13.0/freedoom1.wad", "E1M1"),
        ("freedoom-0.13.0/freedoom2.wad", "MAP01"),
    ],
)
def test_freedoom_classic_generalization_smoke(wad_rel: str, map_name: str) -> None:
    wad_path = Path(wad_rel)
    if not wad_path.is_file():
        pytest.skip(f"Missing WAD: {wad_path}")
    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    res = score_agent_episode(agent, wad_path, map_name=map_name, episode_ticks=200, seed=0)
    assert res["steps"] > 0
    assert res["mean_ms"] < 40.0


def test_failure_classifier_taxonomy() -> None:
    assert classify_failure({"exit_reached": True, "dead": False}) == "success"
    assert classify_failure({"exit_reached": False, "dead": True, "unique_cells": 30}) == "died_early"
    assert classify_failure({"exit_reached": False, "dead": False, "unique_cells": 6, "max_displacement": 300.0}) == "stuck_loop"
    assert classify_failure({"exit_reached": False, "dead": False, "unique_cells": 30, "max_displacement": 400.0, "family": "keycard_puzzle", "keys_found": 0}) == "missed_key"
    assert classify_failure({"exit_reached": False, "dead": False, "unique_cells": 30, "max_displacement": 400.0, "family": "keycard_puzzle", "keys_found": 1, "locked_doors_opened": 0}) == "missed_locked_door"
    assert classify_failure({"exit_reached": False, "dead": False, "unique_cells": 40, "max_displacement": 500.0, "family": "multi_room_exploration", "keys_found": 0, "locked_doors_opened": 0}) == "exit_not_reached"
