"""Config-driven Recurrent PPO training entry point for EMLRPPolicy.

Usage:
    python -m training.train --config training/configs/ppo_smoke.yaml
    python -m training.train --config training/configs/ppo_default.yaml --device cuda
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.features import ObservationPreprocessor
from submission.models import EMLRPConfig, EMLRPPolicy
from training.algo import RecurrentPPOConfig, RecurrentPPOTrainer, RecurrentRolloutBuffer
from training.envs.curriculum import CurriculumManager, CurriculumStage, load_curriculum_yaml
from training.envs.vec_env import DummyVecEnv, SubprocVecEnv, WorkerConfig
from training.eval import evaluate_on_manifest
from training.rewards.curiosity import RNDCuriosity


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _format_duration(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {mins:02d}m"
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def _load_dataset_train_maps(manifest_path: str | Path | None) -> list[str]:
    if manifest_path is None:
        return []
    m_path = Path(manifest_path)
    if not m_path.is_file():
        return []
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    root = m_path.parent
    return [str(root / entry["file"]) for entry in manifest.get("train", [])]


def _save_checkpoint(
    path: Path,
    policy: EMLRPPolicy,
    prep: ObservationPreprocessor,
    trainer: RecurrentPPOTrainer,
    rnd: RNDCuriosity,
    curriculum: CurriculumManager,
    global_step: int,
    update_idx: int,
    best_val_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "model_config": dataclasses.asdict(policy.config),
            "preprocess_config": prep.to_config(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "rnd_predictor_state_dict": rnd.predictor.state_dict(),
            "rnd_target_state_dict": rnd.target.state_dict(),
            "rnd_optimizer_state_dict": rnd.optimizer.state_dict(),
            "rnd_rms": {
                "mean": rnd.reward_rms.mean,
                "var": rnd.reward_rms.var,
                "count": rnd.reward_rms.count,
            },
            "curriculum_stage": int(curriculum.current_stage),
            "global_step": int(global_step),
            "update_idx": int(update_idx),
            "best_val_score": float(best_val_score),
        },
        path,
    )


def train(
    config: dict[str, Any],
    device_override: str | None = None,
    num_envs_override: int | None = None,
    total_steps_override: int | None = None,
    run_name_override: str | None = None,
    resume_path: str | Path | None = None,
    force_dummy_vec: bool = False,
) -> dict[str, Any]:
    train_cfg = dict(config.get("train", {}))
    env_cfg = dict(config.get("env", {}))
    ppo_cfg_dict = dict(config.get("ppo", {}))
    rnd_cfg = dict(config.get("rnd", {}))

    run_name = run_name_override or str(train_cfg.get("run_name", "emlrp_ppo"))
    device = _resolve_device(device_override or str(train_cfg.get("device", "auto")))
    seed = int(train_cfg.get("seed", 42))
    total_steps = int(total_steps_override or train_cfg.get("total_steps", 100000))
    eval_interval = int(train_cfg.get("eval_interval_updates", 25))
    save_interval = int(train_cfg.get("save_interval_updates", 25))
    eval_maps_per_tier = train_cfg.get("eval_maps_per_tier", 4)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Structural train/serve depth parity guard
    render_buffers = bool(env_cfg.get("render_buffers", True))
    use_depth = render_buffers and bool(env_cfg.get("use_depth", True))
    depth_feat_dim = 32 if use_depth else 0

    prep = ObservationPreprocessor(normalize_vitals=True, use_depth=use_depth)
    model_cfg = EMLRPConfig(depth_feat_dim=depth_feat_dim)
    policy = EMLRPPolicy(model_cfg).to(device)

    ppo_cfg = RecurrentPPOConfig(**ppo_cfg_dict)
    trainer = RecurrentPPOTrainer(policy, config=ppo_cfg, device=device, seed=seed)

    rnd = RNDCuriosity(
        emb_dim=int(rnd_cfg.get("emb_dim", 128)),
        lr=float(rnd_cfg.get("lr", 1e-4)),
        scale=float(rnd_cfg.get("scale", 0.1)),
        device=str(device),
    )

    curriculum = CurriculumManager(
        initial_stage=CurriculumStage(int(env_cfg.get("initial_stage", 0)))
    )

    global_step = 0
    start_update = 0
    best_val_score = -float("inf")

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        rnd.predictor.load_state_dict(ckpt["rnd_predictor_state_dict"])
        rnd.target.load_state_dict(ckpt["rnd_target_state_dict"])
        rnd.optimizer.load_state_dict(ckpt["rnd_optimizer_state_dict"])
        rms = ckpt.get("rnd_rms")
        if rms:
            rnd.reward_rms.mean = rms["mean"]
            rnd.reward_rms.var = rms["var"]
            rnd.reward_rms.count = rms["count"]
        curriculum.current_stage = CurriculumStage(int(ckpt.get("curriculum_stage", 0)))
        global_step = int(ckpt.get("global_step", 0))
        start_update = int(ckpt.get("update_idx", 0))
        best_val_score = float(ckpt.get("best_val_score", -float("inf")))

    num_envs = int(num_envs_override or env_cfg.get("num_envs", 8))
    use_subproc = (not force_dummy_vec) and bool(env_cfg.get("use_subproc", True)) and num_envs > 1
    manifest_path = env_cfg.get("dataset_manifest")
    dataset_train_maps = _load_dataset_train_maps(manifest_path)

    worker_cfg = WorkerConfig(
        initial_stage=int(curriculum.current_stage),
        preprocess_config=prep.to_config(),
        episode_ticks=int(env_cfg.get("episode_ticks", 2100)),
        render_buffers=render_buffers,
        use_episodic_novelty=bool(env_cfg.get("use_episodic_novelty", True)),
        sticky_action_prob=float(env_cfg.get("sticky_action_prob", 0.05)),
        potential_scale=float(env_cfg.get("potential_scale", 1.0)),
        dataset_train_maps=dataset_train_maps,
        dataset_mix_prob=float(env_cfg.get("dataset_mix_prob", 0.25)),
        map_reuse_episodes=int(env_cfg.get("map_reuse_episodes", 3)),
        seed=seed,
    )

    vec_env = SubprocVecEnv(num_envs, worker_cfg) if use_subproc else DummyVecEnv(num_envs, worker_cfg)

    run_dir = ROOT / "runs" / run_name
    ckpt_dir = ROOT / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "metrics.jsonl"
    csv_path = run_dir / "metrics.csv"

    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]

        tb_writer = SummaryWriter(log_dir=str(run_dir))
    except Exception:
        pass

    buffer = RecurrentRolloutBuffer(
        rollout_len=ppo_cfg.rollout_len,
        num_envs=num_envs,
        seq_len=ppo_cfg.seq_len,
        include_depth=use_depth,
    )

    steps_per_update = ppo_cfg.rollout_len * num_envs
    total_updates = max(1, (total_steps + steps_per_update - 1) // steps_per_update)

    obs_list = vec_env.reset()
    state = policy.init_state(num_envs, device=device)
    reset_mask = np.ones(num_envs, dtype=bool)
    recent_episodes: list[dict[str, Any]] = []
    csv_header_written = csv_path.exists()
    csv_fieldnames: list[str] | None = None
    if csv_header_written:
        first_line = csv_path.read_text(encoding="utf-8").splitlines()[:1]
        if first_line:
            csv_fieldnames = first_line[0].split(",")
    last_metrics: dict[str, Any] = {}

    session_start_step = global_step
    t_session_start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for update_idx in range(start_update + 1, total_updates + 1):
            t_up = time.perf_counter()
            buffer.reset()
            progress = min(1.0, global_step / max(1, total_steps))
            intrinsic_coef = ppo_cfg.intrinsic_coef_start + progress * (
                ppo_cfg.intrinsic_coef_end - ppo_cfg.intrinsic_coef_start
            )

            for _t in range(ppo_cfg.rollout_len):
                state_before = state
                reset_before = reset_mask.copy()

                actions, log_probs, v_ext, v_int, next_state = trainer.select_actions(
                    obs_list, state_before, reset_before, deterministic=False
                )
                step_res = vec_env.step(actions)

                # Centralized RND over post-step spatial layout
                r_rnd = rnd.compute_intrinsic_rewards_batch(
                    torch.from_numpy(step_res.spatial_vec), update_predictor=True
                )
                r_int = r_rnd + step_res.reward_episodic

                # Next ray distance for non-terminal steps (from step_res.spatial_vec first 64*8=512 -> (B,64,8)[...,0])
                next_rays = step_res.spatial_vec[:, : 64 * 8].reshape(num_envs, 64, 8)[..., 0]

                buffer.add(
                    obs=obs_list,
                    reset_before_step=reset_before,
                    state_before_step=state_before,
                    action=actions.astype(np.float32),
                    log_prob=log_probs,
                    value_ext=v_ext,
                    value_int=v_int,
                    reward_ext=step_res.reward_ext,
                    reward_int=r_int,
                    done=step_res.done,
                    next_ray_distance=next_rays,
                    aux_combat=step_res.aux_combat,
                    aux_landmark_dir=step_res.aux_landmark_dir,
                    aux_landmark_mask=step_res.aux_landmark_mask,
                )

                obs_list = step_res.obs
                state = next_state
                reset_mask = step_res.done.copy()
                global_step += num_envs

                for summ in step_res.episode_summaries:
                    recent_episodes.append(summ)
                    prev_stage = curriculum.current_stage
                    curriculum.record_episode(bool(summ["success"]))
                    if curriculum.current_stage != prev_stage:
                        vec_env.set_stage(int(curriculum.current_stage))

            last_vext, last_vint = trainer.evaluate_values(obs_list, state, reset_mask)
            buffer.compute_returns_and_advantages(
                last_value_ext=last_vext,
                last_value_int=last_vint,
                gamma=ppo_cfg.gamma,
                gamma_int=ppo_cfg.gamma_int,
                gae_lambda=ppo_cfg.gae_lambda,
                intrinsic_coef=intrinsic_coef,
            )

            ppo_stats = trainer.update(buffer, progress_frac=progress)
            now = time.perf_counter()
            dt_up = now - t_up
            fps = steps_per_update / max(1e-6, dt_up)
            avg_fps = (global_step - session_start_step) / max(1e-6, now - t_session_start)
            progress_pct = min(100.0, 100.0 * global_step / max(1, total_steps))
            eta_s = max(0, total_steps - global_step) / max(1e-6, avg_fps)

            ep_slice = recent_episodes[-100:]
            mean_ep_ext = float(np.mean([e["total_ext_reward"] for e in ep_slice])) if ep_slice else 0.0
            mean_ep_int = float(np.mean([e["total_episodic_reward"] for e in ep_slice])) if ep_slice else 0.0
            mean_kills = float(np.mean([e["kills"] for e in ep_slice])) if ep_slice else 0.0
            mean_cells = float(np.mean([e["unique_cells"] for e in ep_slice])) if ep_slice else 0.0
            ep_success = float(np.mean([float(e["success"]) for e in ep_slice])) if ep_slice else 0.0
            peak_vram_mb = (
                float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
                if device.type == "cuda"
                else 0.0
            )

            row: dict[str, Any] = {
                "update": update_idx,
                "global_step": global_step,
                "progress_pct": round(progress_pct, 2),
                "eta_s": round(eta_s, 1),
                "fps": round(fps, 1),
                "avg_fps": round(avg_fps, 1),
                "stage": int(curriculum.current_stage),
                "curriculum_success_rate": round(curriculum.success_rate, 4),
                "ep_success_rate": round(ep_success, 4),
                "ep_ext_reward": round(mean_ep_ext, 4),
                "ep_int_reward": round(mean_ep_int, 4),
                "ep_kills": round(mean_kills, 3),
                "ep_cells": round(mean_cells, 2),
                "intrinsic_coef": round(intrinsic_coef, 4),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "val_weighted_score": None,
                "val_ms_per_step": None,
                **{k: round(v, 5) for k, v in ppo_stats.items()},
            }

            if (
                eval_interval > 0
                and update_idx % eval_interval == 0
                and manifest_path
                and Path(manifest_path).is_file()
            ):
                val_report = evaluate_on_manifest(
                    policy=policy,
                    preprocessor=prep,
                    manifest_path=manifest_path,
                    max_maps_per_tier=eval_maps_per_tier,
                    episode_ticks=int(env_cfg.get("episode_ticks", 2100)),
                    seed=seed,
                    device=device,
                    deterministic=True,
                )
                val_score = float(val_report["weighted_score"])
                row["val_weighted_score"] = round(val_score, 4)
                row["val_ms_per_step"] = round(float(val_report["mean_ms_per_step"]), 3)
                if val_score > best_val_score:
                    best_val_score = val_score
                    _save_checkpoint(
                        ckpt_dir / "best.pt",
                        policy, prep, trainer, rnd, curriculum, global_step, update_idx, best_val_score,
                    )

            if save_interval > 0 and (update_idx % save_interval == 0 or update_idx == total_updates):
                _save_checkpoint(
                    ckpt_dir / "latest.pt",
                    policy, prep, trainer, rnd, curriculum, global_step, update_idx, best_val_score,
                )

            with jsonl_path.open("a", encoding="utf-8") as fj:
                fj.write(json.dumps(row) + "\n")

            with csv_path.open("a", newline="", encoding="utf-8") as fc:
                fieldnames = csv_fieldnames or list(row.keys())
                writer = csv.DictWriter(fc, fieldnames=fieldnames, extrasaction="ignore")
                if not csv_header_written:
                    writer.writeheader()
                    csv_header_written = True
                    csv_fieldnames = fieldnames
                writer.writerow(row)

            if tb_writer is not None:
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f"train/{k}", v, global_step)

            last_metrics = row
            eta_str = _format_duration(eta_s)
            print(
                f"[Upd {update_idx}/{total_updates} ({progress_pct:.2f}%) | "
                f"Step {global_step}/{total_steps} | ETA {eta_str}] "
                f"FPS={row['fps']:.0f} (avg {row['avg_fps']:.0f}) Stage={row['stage']} "
                f"Succ={row['ep_success_rate']:.1%} ExtR={row['ep_ext_reward']:.2f} "
                f"Ent={row['entropy']:.2f} KL={row['approx_kl']:.4f} "
                f"Aux={row['aux_loss']:.3f} VRAM={row['peak_vram_mb']:.0f}MB",
                flush=True,
            )
    finally:
        vec_env.close()
        if tb_writer is not None:
            tb_writer.close()

    # Always write final latest.pt
    _save_checkpoint(
        ckpt_dir / "latest.pt",
        policy, prep, trainer, rnd, curriculum, global_step, total_updates, best_val_score,
    )
    return {
        "run_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "latest_ckpt": str(ckpt_dir / "latest.pt"),
        "best_ckpt": str(ckpt_dir / "best.pt"),
        "last_metrics": last_metrics,
        "global_step": global_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EMLRPPolicy with Recurrent PPO")
    parser.add_argument("--config", type=str, default="training/configs/ppo_default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dummy-vec", action="store_true")
    args = parser.parse_args()

    cfg = load_curriculum_yaml(args.config)
    train(
        config=cfg,
        device_override=args.device,
        num_envs_override=args.num_envs,
        total_steps_override=args.total_steps,
        run_name_override=args.run_name,
        resume_path=args.resume,
        force_dummy_vec=args.dummy_vec,
    )


if __name__ == "__main__":
    main()
