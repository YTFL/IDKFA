"""Validation evaluation harness reimplementing the tier-weighted DOOM scoring rubric.

Usage:
    python -m training.eval --checkpoint checkpoints/emlrp_ppo_v1/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from submission.features import ObservationPreprocessor  # noqa: E402
from submission.models import AblationFlags, EMLRPConfig, EMLRPPolicy, processed_to_tensors  # noqa: E402
from training.rewards.shaping import ExtrinsicRewardShaper, RewardConfig  # noqa: E402

TIER_WEIGHTS: dict[str, float] = {
    "combat_arena": 1.0,
    "multi_room_exploration": 1.5,
    "keycard_puzzle": 2.5,
    "full_gauntlet": 3.5,
}

DEFAULT_TIER_TICKS: dict[str, int] = {
    "combat_arena": 2100,
    "multi_room_exploration": 2100,
    "keycard_puzzle": 4200,
    "full_gauntlet": 4200,
}

_WORKER_POLICY: EMLRPPolicy | None = None
_WORKER_PREP: ObservationPreprocessor | None = None
_WORKER_DEVICE: str = "cpu"


def _init_eval_worker(
    model_cfg_dict: dict[str, Any],
    state_dict_cpu: dict[str, Any],
    prep_cfg_dict: dict[str, Any],
    device_str: str,
) -> None:
    global _WORKER_POLICY, _WORKER_PREP, _WORKER_DEVICE
    torch.set_num_threads(1)
    _WORKER_DEVICE = device_str
    policy = EMLRPPolicy(EMLRPConfig(**model_cfg_dict)).to(device_str)
    policy.load_state_dict(state_dict_cpu)
    policy.eval()
    _WORKER_POLICY = policy
    _WORKER_PREP = ObservationPreprocessor.from_config(prep_cfg_dict)


def _run_eval_task(task: tuple[int, str, str, int, int, bool, float, AblationFlags | None]) -> tuple[int, str, dict[str, Any]]:
    task_idx, family, wad_path_str, ep_ticks, ep_seed, deterministic, temperature, ablations = task
    assert _WORKER_POLICY is not None and _WORKER_PREP is not None
    res = score_episode(
        policy=_WORKER_POLICY,
        preprocessor=_WORKER_PREP,
        wad_path=wad_path_str,
        episode_ticks=ep_ticks,
        seed=ep_seed,
        device=_WORKER_DEVICE,
        deterministic=deterministic,
        temperature=temperature,
        ablations=ablations,
    )
    res["family"] = family
    return task_idx, family, res


def score_episode(
    policy: EMLRPPolicy,
    preprocessor: ObservationPreprocessor,
    wad_path: str | Path,
    map_name: str = "MAP01",
    episode_ticks: int = 2100,
    seed: int = 0,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    temperature: float = 0.0,
    ablations: AblationFlags | None = None,
) -> dict[str, Any]:
    """Run one deterministic or temperature-sampled evaluation episode under vanilla rules."""
    include_depth = policy.config.depth_feat_dim > 0
    env = make_env(str(wad_path), map_name, episode_ticks=episode_ticks, render_buffers=include_depth)
    shaper = ExtrinsicRewardShaper(RewardConfig(step_penalty=0.0, stuck_penalty=0.0))

    try:
        raw_obs, _ = env.reset(seed=seed)
        preprocessor.reset()
        shaper.reset()
        priv0 = env.unwrapped.privileged_state()
        shaper.last_health = float(priv0.get("health", 100.0))

        state = policy.init_state(1, device=device)
        prev_action = np.zeros(9, dtype=np.int8)
        done_t = torch.zeros(1, dtype=torch.bool, device=device)
        rng: torch.Generator | None = None
        if not deterministic and temperature > 0.0:
            rng = torch.Generator(device=device)
            rng.manual_seed(int(seed))

        latencies_ms: list[float] = []
        steps = 0
        total_score = 0.0
        exit_reached = False
        dead = False
        kills = 0
        damage_taken = 0.0
        keys_found = 0
        doors_opened = 0
        locked_doors_opened = 0

        policy.eval()
        with torch.inference_mode():
            while True:
                t0 = time.perf_counter()
                p_obs = preprocessor.process(raw_obs, prev_action=prev_action)
                obs_t, prev_act_t = processed_to_tensors(
                    p_obs, device=device, include_depth=include_depth
                )
                out = policy.step(
                    obs_t,
                    state,
                    prev_act_t,
                    done=done_t,
                    compute_aux=False,
                    ablations=ablations,
                )
                state = out.state
                if deterministic or temperature <= 0.0:
                    act_t = (out.action_logits[0] > 0.0).to(torch.int8)
                else:
                    probs = torch.sigmoid(out.action_logits[0] / temperature)
                    act_t = torch.bernoulli(probs, generator=rng).to(torch.int8)
                action = act_t.cpu().numpy()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                raw_obs, base_r, term, trunc, info = env.step(action)
                priv = env.unwrapped.privileged_state()
                r_step, _bd = shaper.compute(
                    obs=raw_obs,
                    privileged_state=priv,
                    base_reward=base_r,
                    terminated=term,
                    truncated=trunc,
                    info=info,
                )
                total_score += float(r_step)
                prev_action = action
                steps += 1
                kills = int(priv.get("kills", kills))
                damage_taken = float(priv.get("damage_taken", damage_taken))
                dead = bool(priv.get("dead", False))
                keys_found = len(shaper.keys_held)
                doors_opened = len(shaper.open_doors)
                locked_doors_opened = len(shaper.unlocked_doors)

                if term or trunc:
                    exit_reached = bool(info.get("exit_reached", False))
                    break

        lat_arr = np.asarray(latencies_ms, dtype=np.float64)
        return {
            "wad": Path(wad_path).name,
            "map_name": map_name,
            "seed": int(seed),
            "episode_ticks": int(episode_ticks),
            "score": float(total_score),
            "exit_reached": exit_reached,
            "survived": not dead,
            "dead": dead,
            "steps": steps,
            "kills": kills,
            "damage_taken": damage_taken,
            "keys_found": keys_found,
            "doors_opened": doors_opened,
            "locked_doors_opened": locked_doors_opened,
            "unique_cells": len(shaper.visited_cells),
            "mean_ms": float(lat_arr.mean()) if lat_arr.size else 0.0,
            "p50_ms": float(np.percentile(lat_arr, 50)) if lat_arr.size else 0.0,
            "p99_ms": float(np.percentile(lat_arr, 99)) if lat_arr.size else 0.0,
            "max_ms": float(lat_arr.max()) if lat_arr.size else 0.0,
        }
    finally:
        env.close()


def score_agent_episode(
    agent: Any,
    wad_path: str | Path,
    map_name: str = "MAP01",
    episode_ticks: int = 2100,
    seed: int = 0,
    render_buffers: bool = True,
) -> dict[str, Any]:
    """Run one evaluation episode through `submission.agent.DoomAgent` end-to-end."""
    env = make_env(str(wad_path), map_name, episode_ticks=episode_ticks, render_buffers=render_buffers)
    shaper = ExtrinsicRewardShaper(RewardConfig(step_penalty=0.0, stuck_penalty=0.0))
    try:
        raw_obs, _ = env.reset(seed=seed)
        agent.reset()
        shaper.reset()
        priv0 = env.unwrapped.privileged_state()
        shaper.last_health = float(priv0.get("health", 100.0))

        latencies_ms: list[float] = []
        steps = 0
        total_score = 0.0
        exit_reached = False
        dead = False
        kills = 0
        damage_taken = 0.0

        while True:
            t0 = time.perf_counter()
            action = agent.act(raw_obs)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            raw_obs, base_r, term, trunc, info = env.step(action)
            priv = env.unwrapped.privileged_state()
            r_step, _bd = shaper.compute(
                obs=raw_obs,
                privileged_state=priv,
                base_reward=base_r,
                terminated=term,
                truncated=trunc,
                info=info,
            )
            total_score += float(r_step)
            steps += 1
            kills = int(priv.get("kills", kills))
            damage_taken = float(priv.get("damage_taken", damage_taken))
            dead = bool(priv.get("dead", False))
            if term or trunc:
                exit_reached = bool(info.get("exit_reached", False))
                break

        lat_arr = np.asarray(latencies_ms, dtype=np.float64)
        return {
            "wad": Path(wad_path).name,
            "map_name": map_name,
            "seed": int(seed),
            "episode_ticks": int(episode_ticks),
            "score": float(total_score),
            "exit_reached": exit_reached,
            "survived": not dead,
            "dead": dead,
            "steps": steps,
            "kills": kills,
            "damage_taken": damage_taken,
            "keys_found": len(shaper.keys_held),
            "doors_opened": len(shaper.open_doors),
            "locked_doors_opened": len(shaper.unlocked_doors),
            "unique_cells": len(shaper.visited_cells),
            "mean_ms": float(lat_arr.mean()) if lat_arr.size else 0.0,
            "p50_ms": float(np.percentile(lat_arr, 50)) if lat_arr.size else 0.0,
            "p99_ms": float(np.percentile(lat_arr, 99)) if lat_arr.size else 0.0,
            "max_ms": float(lat_arr.max()) if lat_arr.size else 0.0,
        }
    finally:
        env.close()


def evaluate_on_manifest(
    policy: EMLRPPolicy,
    preprocessor: ObservationPreprocessor,
    manifest_path: str | Path = "maps/dataset/manifest.json",
    max_maps_per_tier: int | None = None,
    episode_ticks: int = 2100,
    seed: int = 0,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    temperature: float = 0.0,
    seeds: Sequence[int] | None = None,
    tier_ticks: Mapping[str, int] | None = None,
    ablations: AblationFlags | None = None,
    verbose: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    """Evaluate across held-out validation maps (optionally multi-seed) and compute tier-weighted score."""
    m_path = Path(manifest_path)
    dataset_root = m_path.parent
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    val_entries = manifest.get("val", [])

    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in val_entries:
        by_tier[entry["family"]].append(entry)

    seed_list = [int(s) for s in seeds] if seeds is not None else None
    tasks: list[tuple[int, str, str, int, int, bool, float, AblationFlags | None]] = []
    tier_ep_ticks: dict[str, int] = {}
    for family, entries in sorted(by_tier.items()):
        subset = entries[:max_maps_per_tier] if max_maps_per_tier is not None else entries
        ep_ticks = int(tier_ticks.get(family, episode_ticks)) if tier_ticks is not None else int(episode_ticks)
        tier_ep_ticks[family] = ep_ticks
        for idx, entry in enumerate(subset):
            wad_path = dataset_root / entry["file"]
            ep_seeds = [s + idx for s in seed_list] if seed_list is not None else [seed + idx]
            for ep_seed in ep_seeds:
                tasks.append(
                    (
                        len(tasks),
                        family,
                        str(wad_path),
                        ep_ticks,
                        ep_seed,
                        deterministic,
                        temperature,
                        ablations,
                    )
                )

    total_episodes = len(tasks)
    if workers is None:
        workers = int(os.environ.get("IDKFA_EVAL_WORKERS", str(os.cpu_count() or 1)))
    n_workers = max(1, min(int(workers), total_episodes)) if total_episodes > 0 else 1

    ordered_results: list[tuple[str, dict[str, Any]] | None] = [None] * total_episodes
    if n_workers > 1 and total_episodes >= 8:
        model_cfg_dict = dict(vars(policy.config))
        state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        prep_cfg_dict = preprocessor.to_config() if hasattr(preprocessor, "to_config") else {}
        device_str = str(device)
        completed_episodes = 0
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_eval_worker,
            initargs=(model_cfg_dict, state_dict_cpu, prep_cfg_dict, device_str),
        ) as pool:
            futures = [pool.submit(_run_eval_task, t) for t in tasks]
            for fut in as_completed(futures):
                task_idx, family, res = fut.result()
                ordered_results[task_idx] = (family, res)
                completed_episodes += 1
                if verbose:
                    print(
                        f"    [{completed_episodes:3d}/{total_episodes:3d}] {family} | {res['wad']} "
                        f"| seed={res['seed']} | score={res['score']:6.2f} | exit={int(res['exit_reached'])} "
                        f"| kills={int(res['kills'])} | {res['mean_ms']:.2f}ms/step",
                        flush=True,
                    )
    else:
        for completed_episodes, task in enumerate(tasks, start=1):
            task_idx, family, wad_path_str, ep_ticks, ep_seed, det, temp, abl = task
            res = score_episode(
                policy=policy,
                preprocessor=preprocessor,
                wad_path=wad_path_str,
                episode_ticks=ep_ticks,
                seed=ep_seed,
                device=device,
                deterministic=det,
                temperature=temp,
                ablations=abl,
            )
            res["family"] = family
            ordered_results[task_idx] = (family, res)
            if verbose:
                print(
                    f"    [{completed_episodes:3d}/{total_episodes:3d}] {family} | {res['wad']} "
                    f"| seed={ep_seed} | score={res['score']:6.2f} | exit={int(res['exit_reached'])} "
                    f"| kills={int(res['kills'])} | {res['mean_ms']:.2f}ms/step",
                    flush=True,
                )

    by_tier_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_means: list[float] = []
    all_p50s: list[float] = []
    all_p99s: list[float] = []
    all_maxs: list[float] = []
    all_episodes: list[dict[str, Any]] = []
    for item in ordered_results:
        assert item is not None
        family, res = item
        by_tier_metrics[family].append(res)
        all_episodes.append(res)
        all_means.append(res["mean_ms"])
        all_p50s.append(res["p50_ms"])
        all_p99s.append(res["p99_ms"])
        all_maxs.append(res["max_ms"])

    tier_results: dict[str, dict[str, float]] = {}
    weighted_total = 0.0
    weighted_var = 0.0
    for family in sorted(by_tier.keys()):
        ep_metrics = by_tier_metrics[family]
        ep_ticks = tier_ep_ticks[family]

        scores = [m["score"] for m in ep_metrics]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0
        tier_w = TIER_WEIGHTS.get(family, 1.0)
        weighted_total += tier_w * mean_score
        weighted_var += (tier_w * std_score) ** 2

        tier_results[family] = {
            "count": float(len(ep_metrics)),
            "episode_ticks": float(ep_ticks),
            "mean_score": mean_score,
            "std_score": std_score,
            "exit_rate": float(np.mean([float(m["exit_reached"]) for m in ep_metrics])) if ep_metrics else 0.0,
            "survival_rate": float(np.mean([float(m["survived"]) for m in ep_metrics])) if ep_metrics else 0.0,
            "mean_kills": float(np.mean([float(m["kills"]) for m in ep_metrics])) if ep_metrics else 0.0,
            "mean_damage": float(np.mean([float(m["damage_taken"]) for m in ep_metrics])) if ep_metrics else 0.0,
            "mean_cells": float(np.mean([float(m["unique_cells"]) for m in ep_metrics])) if ep_metrics else 0.0,
            "key_rate": float(np.mean([float(m["keys_found"] > 0) for m in ep_metrics])) if ep_metrics else 0.0,
            "door_rate": float(np.mean([float(m["doors_opened"] > 0) for m in ep_metrics])) if ep_metrics else 0.0,
            "locked_door_rate": (
                float(np.mean([float(m["locked_doors_opened"] > 0) for m in ep_metrics])) if ep_metrics else 0.0
            ),
            "p50_ms": float(np.mean([m["p50_ms"] for m in ep_metrics])) if ep_metrics else 0.0,
            "p99_ms": float(np.max([m["p99_ms"] for m in ep_metrics])) if ep_metrics else 0.0,
            "weight": tier_w,
        }

    return {
        "weighted_score": float(weighted_total),
        "weighted_score_std": float(np.sqrt(weighted_var)),
        "mean_ms_per_step": float(np.mean(all_means)) if all_means else 0.0,
        "p50_ms_per_step": float(np.mean(all_p50s)) if all_p50s else 0.0,
        "p99_ms_per_step": float(np.max(all_p99s)) if all_p99s else 0.0,
        "max_ms_per_step": float(np.max(all_maxs)) if all_maxs else 0.0,
        "tiers": tier_results,
        "episodes": all_episodes,
    }


def load_policy_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[EMLRPPolicy, ObservationPreprocessor, dict[str, Any]]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = EMLRPConfig(**ckpt.get("model_config", {}))
    prep = ObservationPreprocessor.from_config(ckpt.get("preprocess_config", {}))
    policy = EMLRPPolicy(model_cfg).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    return policy, prep, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EMLRP checkpoint on validation maps")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", type=str, default="maps/dataset/manifest.json")
    parser.add_argument("--max-maps-per-tier", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    policy, prep, _ = load_policy_from_checkpoint(args.checkpoint, device=dev)
    det = (not args.stochastic) and (args.temperature <= 0.0)
    report = evaluate_on_manifest(
        policy=policy,
        preprocessor=prep,
        manifest_path=args.manifest,
        max_maps_per_tier=args.max_maps_per_tier,
        episode_ticks=args.ticks,
        seed=args.seed,
        seeds=args.seeds,
        device=dev,
        deterministic=det,
        temperature=args.temperature,
    )

    print(f"\n=== Validation Report ({dev}) ===")
    print(
        f"Weighted Score: {report['weighted_score']:.3f} +/- {report['weighted_score_std']:.3f} | "
        f"Mean Latency: {report['mean_ms_per_step']:.2f} ms/step (p99={report['p99_ms_per_step']:.2f} ms)"
    )
    for fam, stats in report["tiers"].items():
        print(
            f"  {fam:24s} | n={int(stats['count']):2d} | w={stats['weight']:.1f} | "
            f"score={stats['mean_score']:.2f}+/-{stats['std_score']:.2f} | "
            f"exit={stats['exit_rate']:.1%} | surv={stats['survival_rate']:.1%} | "
            f"kills={stats['mean_kills']:.2f} | cells={stats['mean_cells']:.1f}"
        )


if __name__ == "__main__":
    main()
