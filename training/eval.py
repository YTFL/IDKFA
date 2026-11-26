"""Validation evaluation harness reimplementing the tier-weighted DOOM scoring rubric.

Usage:
    python -m training.eval --checkpoint checkpoints/emlrp_ppo_v1/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Bernoulli

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.features import ObservationPreprocessor
from submission.models import EMLRPConfig, EMLRPPolicy, processed_to_tensors
from training.rewards.shaping import ExtrinsicRewardShaper, RewardConfig

TIER_WEIGHTS: dict[str, float] = {
    "combat_arena": 1.0,
    "multi_room_exploration": 1.5,
    "keycard_puzzle": 2.5,
    "full_gauntlet": 3.5,
}


def score_episode(
    policy: EMLRPPolicy,
    preprocessor: ObservationPreprocessor,
    wad_path: str | Path,
    map_name: str = "MAP01",
    episode_ticks: int = 2100,
    seed: int = 0,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run one deterministic/stochastic evaluation episode under vanilla rules."""
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

        latencies_ms: list[float] = []
        steps = 0
        total_score = 0.0
        exit_reached = False
        kills = 0
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
                out = policy.step(obs_t, state, prev_act_t, done=done_t, compute_aux=False)
                state = out.state
                if deterministic:
                    act_t = (out.action_logits[0] > 0.0).to(torch.int8)
                else:
                    act_t = Bernoulli(logits=out.action_logits[0]).sample().to(torch.int8)
                action = act_t.cpu().numpy()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                raw_obs, base_r, term, trunc, info = env.step(action)
                priv = env.unwrapped.privileged_state()
                r_step, bd = shaper.compute(
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
                keys_found = len(shaper.keys_held)
                doors_opened = len(shaper.open_doors)
                locked_doors_opened = len(shaper.unlocked_doors)

                if term or trunc:
                    exit_reached = bool(info.get("exit_reached", False))
                    break

        return {
            "score": float(total_score),
            "exit_reached": exit_reached,
            "steps": steps,
            "kills": kills,
            "keys_found": keys_found,
            "doors_opened": doors_opened,
            "locked_doors_opened": locked_doors_opened,
            "unique_cells": len(shaper.visited_cells),
            "mean_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
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
) -> dict[str, Any]:
    """Evaluate across held-out validation maps and compute tier-weighted score."""
    m_path = Path(manifest_path)
    dataset_root = m_path.parent
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    val_entries = manifest.get("val", [])

    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in val_entries:
        by_tier[entry["family"]].append(entry)

    tier_results: dict[str, dict[str, float]] = {}
    weighted_total = 0.0
    all_latencies: list[float] = []

    for family, entries in sorted(by_tier.items()):
        subset = entries[:max_maps_per_tier] if max_maps_per_tier is not None else entries
        ep_metrics = []
        for idx, entry in enumerate(subset):
            wad_path = dataset_root / entry["file"]
            res = score_episode(
                policy=policy,
                preprocessor=preprocessor,
                wad_path=wad_path,
                episode_ticks=episode_ticks,
                seed=seed + idx,
                device=device,
                deterministic=deterministic,
            )
            ep_metrics.append(res)
            all_latencies.append(res["mean_ms"])

        scores = [m["score"] for m in ep_metrics]
        exits = [float(m["exit_reached"]) for m in ep_metrics]
        kills = [float(m["kills"]) for m in ep_metrics]
        cells = [float(m["unique_cells"]) for m in ep_metrics]
        mean_score = float(np.mean(scores)) if scores else 0.0
        tier_w = TIER_WEIGHTS.get(family, 1.0)
        weighted_total += tier_w * mean_score

        tier_results[family] = {
            "count": float(len(ep_metrics)),
            "mean_score": mean_score,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "exit_rate": float(np.mean(exits)) if exits else 0.0,
            "mean_kills": float(np.mean(kills)) if kills else 0.0,
            "mean_cells": float(np.mean(cells)) if cells else 0.0,
            "weight": tier_w,
        }

    return {
        "weighted_score": float(weighted_total),
        "mean_ms_per_step": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "tiers": tier_results,
    }


def load_policy_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[EMLRPPolicy, ObservationPreprocessor, dict[str, Any]]:
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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    policy, prep, _ = load_policy_from_checkpoint(args.checkpoint, device=dev)
    report = evaluate_on_manifest(
        policy=policy,
        preprocessor=prep,
        manifest_path=args.manifest,
        max_maps_per_tier=args.max_maps_per_tier,
        episode_ticks=args.ticks,
        seed=args.seed,
        device=dev,
        deterministic=not args.stochastic,
    )

    print(f"\n=== Validation Report ({dev}) ===")
    print(f"Weighted Score: {report['weighted_score']:.3f} | Mean Latency: {report['mean_ms_per_step']:.2f} ms/step")
    for fam, stats in report["tiers"].items():
        print(
            f"  {fam:24s} | n={int(stats['count']):2d} | w={stats['weight']:.1f} | "
            f"score={stats['mean_score']:.2f}+/-{stats['std_score']:.2f} | "
            f"exit={stats['exit_rate']:.1%} | kills={stats['mean_kills']:.2f} | cells={stats['mean_cells']:.1f}"
        )


if __name__ == "__main__":
    main()
