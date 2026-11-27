"""Collect verified teacher demonstrations across Curriculum Stages 0-5 for Behavioral Cloning.

Runs `PrivilegedTeacherOracle` against procedurally generated maps (strictly disjoint from
`maps/dataset/val`) with realistic evaluation rules (`god_mode=False`, `auto_open_doors=False`,
`infinite_ammo=False`), saving exact `ObservationPreprocessor` features + auxiliary targets +
discounted Monte-Carlo returns for truncated-BPTT BC pretraining.

Usage:
    python -m training.imitation.collect_demos --episodes-per-stage 60 --out-dir data/bc_demos
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.action_mask import compute_obs_action_mask_np
from submission.features import ACTION_DIM, DEPTH_DOWN_WIDTH, MAX_ENTITIES, NUM_RAYS, VITALS_DIM, ObservationPreprocessor
from training.envs.curriculum import CurriculumEnvWrapper, CurriculumManager, CurriculumStage
from training.envs.vec_env import _compute_aux_landmark_dir, _compute_objective_coords
from training.imitation.teacher import PrivilegedTeacherOracle


def _compute_mc_returns(rewards: list[float], gamma: float = 0.995) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = float(rewards[i]) + gamma * g
        out[i] = g
    return out


def _collect_single_episode(
    args_tuple: tuple[int, int, int, int, int, float, bool, int, bool, str],
) -> tuple[int, int, dict[str, Any]]:
    (
        st_id,
        ep_idx,
        ep_seed,
        episode_ticks,
        base_seed,
        gamma,
        min_success_only,
        keep_fallback_cells,
        use_depth,
        tmp_dir,
    ) = args_tuple
    st = CurriculumStage(st_id)
    cm = CurriculumManager(map_cache_dir=tmp_dir)
    prep = ObservationPreprocessor(normalize_vitals=True, use_depth=use_depth)
    oracle = PrivilegedTeacherOracle(seed=ep_seed)

    wad_path, st_cfg, meta = cm.sample_map_spec(
        stage=st, seed=ep_seed, filename=f"bc_s{st_id}_{ep_idx}.wad"
    )
    base_env = make_env(str(wad_path), episode_ticks=episode_ticks, render_buffers=use_depth)
    st_cfg.god_mode = False
    st_cfg.auto_open_doors = False
    st_cfg.infinite_ammo = False
    env = CurriculumEnvWrapper(base_env, stage=st, config=st_cfg)

    try:
        raw_obs, _ = env.reset()
        prep.reset()
        oracle.reset(meta)

        ep_ray_d: list[np.ndarray] = []
        ep_ray_h: list[np.ndarray] = []
        ep_ent_f: list[np.ndarray] = []
        ep_ent_t: list[np.ndarray] = []
        ep_ent_m: list[np.ndarray] = []
        ep_vit: list[np.ndarray] = []
        ep_dep: list[np.ndarray] = []
        ep_prev_a: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        ep_mask: list[np.ndarray] = []
        ep_next_ray: list[np.ndarray] = []
        ep_aux_combat: list[np.ndarray] = []
        ep_aux_dir: list[np.ndarray] = []
        ep_aux_mask: list[float] = []
        ep_rew: list[float] = []
        ep_done: list[bool] = []

        last_kills = 0
        last_dmg = 0.0
        success = False
        last_info: dict[str, Any] = {}

        while True:
            priv_before = env.unwrapped.privileged_state()
            p_obs = prep.process(raw_obs)
            act_mask = compute_obs_action_mask_np(p_obs)
            action = oracle.act(raw_obs, priv_before)

            next_obs, r_ext, terminated, truncated, info = env.step(action)
            priv_after = env.unwrapped.privileged_state()
            last_info = info

            arena_cleared = False
            num_enemies = int(meta.get("num_enemies", 0))
            if (
                not terminated
                and meta.get("exit_xy") is None
                and num_enemies > 0
                and priv_after is not None
                and int(priv_after.get("kills", 0)) >= num_enemies
                and float(priv_after.get("health", 0.0)) > 0.0
            ):
                arena_cleared = True
                terminated = True
                r_ext += 5.0 + (float(priv_after["health"]) / 100.0)

            target_xy = _compute_objective_coords(meta, priv_before)
            if target_xy is not None and priv_after is not None:
                shaper = env.reward_shaper
                dist = float(np.hypot(float(priv_after["x"]) - target_xy[0], float(priv_after["y"]) - target_xy[1]))
                pot = -dist / 1000.0
                if shaper.last_potential is not None and shaper.last_target_coords == target_xy:
                    r_ext += (shaper.config.gamma * pot - shaper.last_potential)
                shaper.last_potential = pot
                shaper.last_target_coords = target_xy

            p_next = prep.process(next_obs, prev_action=action)
            curr_k = int(priv_after.get("kills", 0)) if priv_after else last_kills
            curr_d = float(priv_after.get("damage_taken", 0.0)) if priv_after else last_dmg
            aux_c = np.array([float(curr_d > last_dmg), float(curr_k > last_kills)], dtype=np.float32)
            last_kills, last_dmg = curr_k, curr_d

            post_target = _compute_objective_coords(meta, priv_after)
            aux_d, aux_m = _compute_aux_landmark_dir(priv_after, post_target)

            done = bool(terminated or truncated)
            ep_ray_d.append(p_obs.rays[..., 0])
            ep_ray_h.append(p_obs.rays[..., 2:8].argmax(axis=-1))
            ep_ent_f.append(p_obs.entities[..., :5])
            ep_ent_t.append(p_obs.entities[..., 5:17].argmax(axis=-1))
            ep_ent_m.append(p_obs.entity_mask)
            ep_vit.append(p_obs.vitals)
            ep_dep.append(p_obs.depth)
            ep_prev_a.append(p_obs.prev_action)
            ep_act.append(action.astype(np.float32))
            ep_mask.append(act_mask)
            ep_next_ray.append(p_next.rays[..., 0])
            ep_aux_combat.append(aux_c)
            ep_aux_dir.append(aux_d)
            ep_aux_mask.append(float(aux_m))
            ep_rew.append(float(r_ext))
            ep_done.append(done)

            raw_obs = next_obs
            if done:
                success = bool(info.get("exit_reached", False) or arena_cleared)
                break

        cells = int(last_info.get("unique_cells_visited", 0))
        keep_ep = success or ((not min_success_only) and cells >= keep_fallback_cells)
        ep_dict: dict[str, np.ndarray] | None = None
        if keep_ep and len(ep_act) >= 4:
            ep_dict = {
                "ray_distance": np.stack(ep_ray_d, axis=0).astype(np.float32),
                "ray_hit_type": np.stack(ep_ray_h, axis=0).astype(np.int64),
                "entity_features": np.stack(ep_ent_f, axis=0).astype(np.float32),
                "entity_types": np.stack(ep_ent_t, axis=0).astype(np.int64),
                "entity_mask": np.stack(ep_ent_m, axis=0).astype(np.float32),
                "vitals": np.stack(ep_vit, axis=0).astype(np.float32),
                "depth_feat": np.stack(ep_dep, axis=0).astype(np.float32),
                "prev_action": np.stack(ep_prev_a, axis=0).astype(np.float32),
                "action": np.stack(ep_act, axis=0).astype(np.float32),
                "action_mask": np.stack(ep_mask, axis=0).astype(np.float32),
                "next_ray_distance": np.stack(ep_next_ray, axis=0).astype(np.float32),
                "aux_combat": np.stack(ep_aux_combat, axis=0).astype(np.float32),
                "aux_landmark_dir": np.stack(ep_aux_dir, axis=0).astype(np.float32),
                "aux_landmark_mask": np.asarray(ep_aux_mask, dtype=np.float32),
                "mc_return": _compute_mc_returns(ep_rew, gamma=gamma),
                "done": np.asarray(ep_done, dtype=bool),
            }
        return st_id, ep_idx, {"success": success, "ep": ep_dict}
    finally:
        env.close()


def collect_demos(
    out_dir: str | Path = "data/bc_demos",
    stages: list[int] | None = None,
    episodes_per_stage: int = 40,
    episode_ticks: int = 2100,
    seed: int = 1000,
    gamma: float = 0.995,
    min_success_only: bool = True,
    keep_fallback_cells: int = 120,
    use_depth: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stage_list = stages if stages is not None else [0, 1, 2, 3, 4, 5]

    with tempfile.TemporaryDirectory(prefix="bc_maps_") as tmp_dir:
        tasks: list[tuple[int, int, int, int, int, float, bool, int, bool, str]] = []
        seed_cursor = seed
        for st_id in stage_list:
            for ep_idx in range(episodes_per_stage):
                seed_cursor += 1
                tasks.append(
                    (
                        st_id,
                        ep_idx,
                        seed_cursor,
                        episode_ticks,
                        seed,
                        gamma,
                        min_success_only,
                        keep_fallback_cells,
                        use_depth,
                        tmp_dir,
                    )
                )

        total_tasks = len(tasks)
        if workers is None:
            workers = os.cpu_count() or 1
        n_workers = max(1, min(int(workers), total_tasks)) if total_tasks > 0 else 1

        results_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        if n_workers > 1 and total_tasks >= 4:
            print(f"[collect_demos] Collecting {total_tasks} episodes using {n_workers} workers...", flush=True)
            done_count = 0
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(_collect_single_episode, t) for t in tasks]
                for fut in as_completed(futs):
                    st_id, ep_idx, res = fut.result()
                    results_by_key[(st_id, ep_idx)] = res
                    done_count += 1
                    if done_count % 25 == 0 or done_count == total_tasks:
                        print(f"[collect_demos] Progress: {done_count}/{total_tasks} episodes", flush=True)
        else:
            for t in tasks:
                st_id, ep_idx, res = _collect_single_episode(t)
                results_by_key[(st_id, ep_idx)] = res

        episodes_data: list[dict[str, np.ndarray]] = []
        stage_stats: dict[int, dict[str, Any]] = {}
        for st_id in stage_list:
            st = CurriculumStage(st_id)
            attempted = episodes_per_stage
            solved = 0
            kept = 0
            total_steps_stage = 0
            for ep_idx in range(episodes_per_stage):
                res = results_by_key[(st_id, ep_idx)]
                if res["success"]:
                    solved += 1
                if res["ep"] is not None:
                    kept += 1
                    total_steps_stage += len(res["ep"]["action"])
                    episodes_data.append(res["ep"])

            stage_stats[st_id] = {
                "stage_name": st.name,
                "attempted": attempted,
                "solved": solved,
                "solve_rate": round(solved / max(1, attempted), 4),
                "kept_episodes": kept,
                "frames": total_steps_stage,
            }
            print(
                f"[collect_demos] Stage {st_id} ({st.name}): "
                f"solved {solved}/{attempted} ({100.0 * solved / max(1, attempted):.1f}%), "
                f"kept={kept}, frames={total_steps_stage}",
                flush=True,
            )

    if not episodes_data:
        raise RuntimeError("No usable demonstration episodes collected!")

    # Write flat concatenated shards with explicit episode boundaries
    shard_size_eps = 100
    shards: list[dict[str, Any]] = []
    total_frames = 0
    for s_idx, start_ep in enumerate(range(0, len(episodes_data), shard_size_eps)):
        chunk_eps = episodes_data[start_ep : start_ep + shard_size_eps]
        lengths = np.array([len(e["action"]) for e in chunk_eps], dtype=np.int64)
        starts = np.zeros_like(lengths)
        if len(lengths) > 1:
            starts[1:] = np.cumsum(lengths[:-1])

        merged: dict[str, np.ndarray] = {
            k: np.concatenate([e[k] for e in chunk_eps], axis=0)
            for k in chunk_eps[0]
        }
        merged["ep_starts"] = starts
        merged["ep_lengths"] = lengths

        shard_name = f"shard_{s_idx:03d}.npz"
        np.savez_compressed(out_path / shard_name, **merged)
        n_frames = int(lengths.sum())
        total_frames += n_frames
        shards.append({"file": shard_name, "episodes": len(chunk_eps), "frames": n_frames})

    manifest = {
        "total_episodes": len(episodes_data),
        "total_frames": total_frames,
        "use_depth": use_depth,
        "shards": shards,
        "stage_stats": stage_stats,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[collect_demos] Saved {len(episodes_data)} episodes ({total_frames} frames) to {out_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BC teacher demonstrations")
    parser.add_argument("--out-dir", type=str, default="data/bc_demos")
    parser.add_argument("--stages", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--episodes-per-stage", type=int, default=60)
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--keep-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(",") if s.strip()]
    collect_demos(
        out_dir=args.out_dir,
        stages=stages,
        episodes_per_stage=args.episodes_per_stage,
        episode_ticks=args.episode_ticks,
        seed=args.seed,
        min_success_only=(not args.keep_partial),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
