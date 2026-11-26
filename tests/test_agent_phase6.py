"""Tests for Phase 6: DoomAgent inference, weight export, robustness, latency, and ViZDoom rollout."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from doom_env import make_env
from doom_env import spaces as S
from submission.agent import DoomAgent
from submission.models import EMLRPConfig, EMLRPPolicy
from training.eval import load_policy_from_checkpoint
from training.tools.export_weights import export_inference_weights


def _make_obs(seed: int = 0, num_entities: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    mask = np.zeros(32, dtype=np.int8)
    mask[:num_entities] = 1
    etypes = np.zeros(32, dtype=np.int64)
    if num_entities > 0:
        etypes[:num_entities] = rng.integers(1, 12, size=num_entities)
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
            "entity_type": etypes,
            "mask": mask,
        },
    }


def _make_temp_weights(tmp_path: Path, temperature: float = 0.0) -> Path:
    cfg = EMLRPConfig()
    policy = EMLRPPolicy(cfg)
    raw_ckpt = tmp_path / "raw_ckpt.pt"
    out_pt = tmp_path / "agent.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "model_config": {
                "depth_feat_dim": cfg.depth_feat_dim,
            },
            "preprocess_config": {"normalize_vitals": True, "use_depth": True},
            "optimizer_state_dict": {"dummy": torch.zeros(1024)},
            "update_idx": 12,
            "global_step": 1234,
            "curriculum_stage": 1,
            "best_val_score": 3.5,
        },
        raw_ckpt,
    )
    export_inference_weights(raw_ckpt, out_pt, temperature=temperature)
    return out_pt


def test_export_weights_roundtrip_and_size(tmp_path: Path) -> None:
    src = Path("submission/weights/emlrp_best.pt")
    if not src.is_file() or src.stat().st_size < 1024:
        src = _make_temp_weights(tmp_path)
        return

    out = tmp_path / "agent_exported.pt"
    info = export_inference_weights(src, out, temperature=0.0)
    assert out.is_file()
    assert info["output_bytes"] < 10 * 1024 * 1024
    assert info["output_bytes"] < info["source_bytes"]

    reloaded = torch.load(out, map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" not in reloaded
    assert "rnd_predictor_state_dict" not in reloaded

    p_src, _, _ = load_policy_from_checkpoint(src, device="cpu")
    p_dst, _, _ = load_policy_from_checkpoint(out, device="cpu")
    for k, v in p_src.state_dict().items():
        assert torch.allclose(v, p_dst.state_dict()[k])


def test_agent_contract_reset_and_reproducibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights_pt = _make_temp_weights(tmp_path, temperature=0.0)
    monkeypatch.setenv("NORDLICHT_WEIGHTS_PATH", str(weights_pt))

    obs_space = S.observation_space()
    act_space = S.action_space()
    agent_a = DoomAgent(obs_space, act_space, "cpu")
    agent_b = DoomAgent(obs_space, act_space, "cpu")

    seq = [_make_obs(seed=100 + i) for i in range(15)]

    # Pollute agent_a state then reset()
    agent_a.reset()
    for obs in seq:
        act = agent_a.act(obs)
        assert isinstance(act, np.ndarray)
        assert act.shape == (9,)
        assert act.dtype == np.int8
        assert np.all((act == 0) | (act == 1))

    agent_a.reset()
    agent_b.reset()
    for obs in seq:
        act_a = agent_a.act(obs)
        act_b = agent_b.act(obs)
        assert np.array_equal(act_a, act_b)


def test_agent_temperature_reproducibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights_pt = _make_temp_weights(tmp_path, temperature=0.7)
    monkeypatch.setenv("NORDLICHT_WEIGHTS_PATH", str(weights_pt))

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    seq = [_make_obs(seed=200 + i) for i in range(10)]

    agent.reset()
    traj1 = [agent.act(obs) for obs in seq]
    agent.reset()
    traj2 = [agent.act(obs) for obs in seq]
    assert all(np.array_equal(a, b) for a, b in zip(traj1, traj2))


def test_agent_sanitization_edge_cases_and_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights_pt = _make_temp_weights(tmp_path, temperature=0.0)
    monkeypatch.setenv("NORDLICHT_WEIGHTS_PATH", str(weights_pt))

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.reset()

    # Empty entities, max entities, NaNs/Infs, missing depth, truncated arrays, invalid non-dict
    nan_obs = _make_obs(seed=5, num_entities=0)
    nan_obs["rays"]["distance"][:10] = np.nan
    nan_obs["rays"]["distance"][10:20] = np.inf
    nan_obs["vitals"][:] = np.nan
    nan_obs["depth"] = None
    nan_obs["visible_entities"]["features"][:] = np.nan

    full_obs = _make_obs(seed=6, num_entities=32)
    partial_obs = {"rays": {"distance": [0.5, 0.2]}, "vitals": [100.0]}

    for sample in (nan_obs, full_obs, partial_obs, {}):
        act = agent.act(sample)
        assert act.shape == (9,) and act.dtype == np.int8 and np.all((act == 0) | (act == 1))

    # Simulate positive USE/SELECT_NEXT_WEAPON logits -> _last_safe_action must clear edge triggers [7, 8]
    def _all_ones(*args, **kwargs):
        from submission.models.emlrp import EMLRPStepOutput

        return EMLRPStepOutput(
            action_logits=torch.full((1, 9), 5.0),
            value_ext=torch.zeros(1, 1),
            value_int=torch.zeros(1, 1),
            mode_logits=torch.zeros(1, 5),
            state=agent.state,
            aux={},
        )

    monkeypatch.setattr(agent.policy, "step", _all_ones)
    ones_act = agent.act(_make_obs(seed=7))
    assert ones_act[7] == 1 and ones_act[8] == 1
    assert agent._last_safe_action[7] == 0 and agent._last_safe_action[8] == 0

    # Simulate an internal failure inside policy.step -> must recover cleanly with edge triggers cleared
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated inference error")

    monkeypatch.setattr(agent.policy, "step", _boom)
    fallback = agent.act(_make_obs(seed=8))
    assert fallback.shape == (9,) and fallback.dtype == np.int8
    assert np.all((fallback == 0) | (fallback == 1))
    assert fallback[7] == 0 and fallback[8] == 0


def test_agent_long_horizon_latency_1_thread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights_pt = _make_temp_weights(tmp_path, temperature=0.0)
    monkeypatch.setenv("NORDLICHT_WEIGHTS_PATH", str(weights_pt))

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    assert torch.get_num_threads() == 1
    agent.reset()

    obs = _make_obs(seed=42, num_entities=8)
    t0 = time.perf_counter()
    steps = 1050  # Full 4200-tick map budget
    for _ in range(steps):
        act = agent.act(obs)
        assert act.shape == (9,)
    mean_ms = (time.perf_counter() - t0) * 1000.0 / steps
    print(f"\n[DoomAgent] 1050-step 1-thread CPU latency: {mean_ms:.2f} ms/step")
    assert mean_ms < 40.0


def test_agent_live_two_rooms_rollout() -> None:
    env = make_env("maps/two_rooms.wad", "MAP01", episode_ticks=400)
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DoomAgent(env.observation_space, env.action_space, dev)
        obs, _ = env.reset(seed=0)
        agent.reset()
        for _ in range(100):
            act = agent.act(obs)
            assert act.shape == (9,) and act.dtype == np.int8 and np.all((act == 0) | (act == 1))
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    finally:
        env.close()
