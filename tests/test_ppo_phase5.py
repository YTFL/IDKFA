"""Tests for Phase 5: Recurrent PPO, SubprocVecEnv, train/eval parity, and smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from submission.features import ObservationPreprocessor
from submission.models import EMLRPConfig, EMLRPPolicy, processed_to_tensors
from training.algo import RecurrentPPOConfig, RecurrentPPOTrainer, RecurrentRolloutBuffer
from training.envs.curriculum import load_curriculum_yaml
from training.envs.vec_env import DummyVecEnv, SubprocVecEnv, WorkerConfig
from training.eval import evaluate_on_manifest, load_policy_from_checkpoint
from training.train import train


def _dummy_raw_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mask = np.zeros(32, dtype=np.int8)
    mask[:3] = 1
    etypes = np.zeros(32, dtype=np.int64)
    etypes[:3] = [1, 2, 9]
    return {
        "rgb": rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8),
        "depth": rng.integers(10, 250, size=(120, 160), dtype=np.uint8),
        "vitals": rng.uniform(0.0, 100.0, size=(17,)).astype(np.float32),
        "rays": {
            "distance": rng.uniform(0.1, 1.0, size=(64,)).astype(np.float32),
            "hit_type": rng.integers(0, 6, size=(64,), dtype=np.int64),
        },
        "visible_entities": {
            "features": rng.normal(0.0, 1.0, size=(32, 5)).astype(np.float32),
            "entity_type": etypes,
            "mask": mask,
        },
    }


def test_depth_parity_guard() -> None:
    prep = ObservationPreprocessor(use_depth=False)
    p_obs = prep.process(_dummy_raw_obs(0), prev_action=np.zeros(9, dtype=np.float32))
    assert np.all(p_obs.depth == 0.0)

    obs_t, prev_act = processed_to_tensors(p_obs, include_depth=False)
    assert "depth_feat" not in obs_t

    policy = EMLRPPolicy(EMLRPConfig(depth_feat_dim=0))
    state = policy.init_state(1)
    out = policy.step(obs_t, state, prev_act, compute_aux=True)
    assert torch.isfinite(out.action_logits).all()
    out.action_logits.sum().backward()


def test_rollout_buffer_dual_gae_and_bptt_chunks() -> None:
    T, B, seq_len = 8, 2, 4
    buf = RecurrentRolloutBuffer(rollout_len=T, num_envs=B, seq_len=seq_len, include_depth=True)
    prep = ObservationPreprocessor()
    cfg = EMLRPConfig()
    policy = EMLRPPolicy(cfg)
    state = policy.init_state(B)

    for t in range(T):
        obs_list = [prep.process(_dummy_raw_obs(10 * t + b), prev_action=np.zeros(9)) for b in range(B)]
        done = np.array([t == 3, False], dtype=bool)  # env 0 terminates at t=3
        reset_before = np.array([t == 0 or t == 4, t == 0], dtype=bool)
        buf.add(
            obs=obs_list,
            reset_before_step=reset_before,
            state_before_step=state,
            action=np.ones((B, 9), dtype=np.float32),
            log_prob=np.zeros(B, dtype=np.float32),
            value_ext=np.ones(B, dtype=np.float32),
            value_int=np.ones(B, dtype=np.float32),
            reward_ext=np.ones(B, dtype=np.float32),
            reward_int=np.ones(B, dtype=np.float32),
            done=done,
            next_ray_distance=np.zeros((B, 64), dtype=np.float32),
            aux_combat=np.zeros((B, 2), dtype=np.float32),
            aux_landmark_dir=np.zeros((B, 2), dtype=np.float32),
            aux_landmark_mask=np.ones(B, dtype=np.float32),
        )

    buf.compute_returns_and_advantages(
        last_value_ext=np.ones(B, dtype=np.float32),
        last_value_int=np.ones(B, dtype=np.float32),
        gamma=0.99,
        gamma_int=0.99,
        gae_lambda=0.95,
        intrinsic_coef=0.2,
    )
    # At t=3 for env 0 (done=True), ret_ext[3, 0] must equal reward_ext[3, 0] (no bootstrap)
    assert buf.ret_ext[3, 0] == pytest.approx(1.0, abs=1e-5)
    # Non-episodic intrinsic return must still bootstrap past done=True
    assert buf.ret_int[3, 0] > 1.5

    rng = np.random.default_rng(0)
    batches = list(buf.iter_minibatches(minibatch_seqs=2, device=torch.device("cpu"), rng=rng))
    assert len(batches) == (T // seq_len * B) // 2
    assert batches[0]["obs_seq"]["ray_distance"].shape == (2, seq_len, 64)


def test_ppo_trainer_cuda_and_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EMLRPConfig()
    policy = EMLRPPolicy(cfg)
    ppo_cfg = RecurrentPPOConfig(rollout_len=8, seq_len=4, minibatch_seqs=2, ppo_epochs=1)
    trainer = RecurrentPPOTrainer(policy, config=ppo_cfg, device=device)

    buf = RecurrentRolloutBuffer(rollout_len=8, num_envs=2, seq_len=4, include_depth=True)
    prep = ObservationPreprocessor()
    state = policy.init_state(2, device=device)
    reset_mask = np.ones(2, dtype=bool)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for t in range(8):
        obs_list = [prep.process(_dummy_raw_obs(t * 2 + b)) for b in range(2)]
        actions, logp, vext, vint, next_state = trainer.select_actions(obs_list, state, reset_mask)
        buf.add(
            obs=obs_list,
            reset_before_step=reset_mask,
            state_before_step=state,
            action=actions.astype(np.float32),
            log_prob=logp,
            value_ext=vext,
            value_int=vint,
            reward_ext=np.ones(2, dtype=np.float32) * 0.1,
            reward_int=np.ones(2, dtype=np.float32) * 0.05,
            done=np.zeros(2, dtype=bool),
            next_ray_distance=np.full((2, 64), 0.5, dtype=np.float32),
            aux_combat=np.zeros((2, 2), dtype=np.float32),
            aux_landmark_dir=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            aux_landmark_mask=np.ones(2, dtype=np.float32),
        )
        state = next_state
        reset_mask = np.zeros(2, dtype=bool)

    lv_ext, lv_int = trainer.evaluate_values(obs_list, state, reset_mask)
    buf.compute_returns_and_advantages(lv_ext, lv_int, 0.995, 0.99, 0.95, 0.2)
    stats = trainer.update(buf, progress_frac=0.1)

    for val in stats.values():
        assert np.isfinite(val)
    for prefix in ("ray_encoder.", "entity_encoder.", "spatial_map.", "landmark_memory.", "gru.", "policy_head."):
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for n, p in policy.named_parameters()
            if n.startswith(prefix)
        ), f"Missing gradient in {prefix}"

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        assert peak_mb < 2048.0, f"Peak VRAM {peak_mb:.1f} MB exceeds 2048 MB budget"


def test_dummy_and_subproc_vec_env(tmp_path: Path) -> None:
    d_cfg = WorkerConfig(
        initial_stage=0,
        episode_ticks=60,
        fixed_wad_path="maps/two_rooms.wad",
        cache_root=str(tmp_path / "dummy_cache"),
    )
    d_env = DummyVecEnv(2, d_cfg)
    obs0 = d_env.reset()
    assert len(obs0) == 2
    res = d_env.step(np.zeros((2, 9), dtype=np.int8))
    assert res.spatial_vec.shape == (2, 64 * 8 + 32)
    assert res.aux_combat.shape == (2, 2)
    d_env.close()

    s_cfg = WorkerConfig(
        initial_stage=2,
        episode_ticks=40,
        cache_root=str(tmp_path / "subproc_cache"),
    )
    s_env = SubprocVecEnv(2, s_cfg)
    obs_s = s_env.reset()
    assert len(obs_s) == 2
    s_env.set_stage(3)
    for _ in range(15):
        s_res = s_env.step(np.ones((2, 9), dtype=np.int8))
        assert len(s_res.obs) == 2
    s_env.close()


def test_smoke_train_resume_and_eval(tmp_path: Path) -> None:
    cfg = load_curriculum_yaml("training/configs/ppo_smoke.yaml")
    run_name = "pytest_smoke_run"
    # Direct smoke run artifacts under repo runs/checkpoints or verify end-to-end
    out = train(
        config=cfg,
        num_envs_override=2,
        total_steps_override=128,
        run_name_override=run_name,
        force_dummy_vec=True,
    )
    run_dir = Path(out["run_dir"])
    ckpt_latest = Path(out["latest_ckpt"])
    ckpt_best = Path(out["best_ckpt"])

    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert ckpt_latest.is_file()
    assert ckpt_best.is_file()

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) >= 2
    assert all(np.isfinite(r["policy_loss"]) and np.isfinite(r["aux_loss"]) for r in rows)

    # Resume for 1 more update
    out2 = train(
        config=cfg,
        num_envs_override=2,
        total_steps_override=192,
        run_name_override=run_name,
        resume_path=ckpt_latest,
        force_dummy_vec=True,
    )
    assert out2["global_step"] >= 192

    # Evaluate saved best checkpoint on held-out val subset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, prep, _ = load_policy_from_checkpoint(ckpt_best, device=dev)
    val_report = evaluate_on_manifest(
        policy=policy,
        preprocessor=prep,
        manifest_path="maps/dataset/manifest.json",
        max_maps_per_tier=1,
        episode_ticks=200,
        device=dev,
    )
    assert "weighted_score" in val_report
    assert val_report["mean_ms_per_step"] < 40.0
