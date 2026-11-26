"""Failure-mode classification and spatial/landmark diagnostic tracer for Phase 7.

Usage:
    python training/tools/analyze_failures.py --checkpoint submission/weights/agent.pt --max-maps-per-tier 2 --ticks 600
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from submission.models import processed_to_tensors  # noqa: E402
from training.eval import load_policy_from_checkpoint  # noqa: E402
from training.rewards.shaping import ExtrinsicRewardShaper, RewardConfig  # noqa: E402


def classify_failure(summary: Mapping[str, Any]) -> str:
    """Classify an evaluation episode outcome into a concrete failure category."""
    if bool(summary.get("exit_reached", False)):
        return "success"
    if bool(summary.get("dead", False)):
        return "died_early"
    unique_cells = int(summary.get("unique_cells", 0))
    max_disp = float(summary.get("max_displacement", math.inf))
    if unique_cells <= 12 or max_disp < 128.0:
        return "stuck_loop"
    family = str(summary.get("family", ""))
    keys_found = int(summary.get("keys_found", 0))
    locked_opened = int(summary.get("locked_doors_opened", 0))
    if family in {"keycard_puzzle", "full_gauntlet"} and keys_found == 0:
        return "missed_key"
    if keys_found > 0 and locked_opened == 0:
        return "missed_locked_door"
    return "exit_not_reached"


def trace_episode(
    policy: Any,
    preprocessor: Any,
    wad_path: str | Path,
    family: str,
    episode_ticks: int = 2100,
    seed: int = 0,
    device: str = "cpu",
    temperature: float = 0.0,
) -> dict[str, Any]:
    include_depth = policy.config.depth_feat_dim > 0
    env = make_env(str(wad_path), "MAP01", episode_ticks=episode_ticks, render_buffers=include_depth)
    shaper = ExtrinsicRewardShaper(RewardConfig(step_penalty=0.0, stuck_penalty=0.0))
    try:
        raw_obs, _ = env.reset(seed=seed)
        preprocessor.reset()
        shaper.reset()
        priv0 = env.unwrapped.privileged_state()
        shaper.last_health = float(priv0.get("health", 100.0))
        x0, y0 = float(priv0.get("x", 0.0)), float(priv0.get("y", 0.0))

        state = policy.init_state(1, device=device)
        prev_action = np.zeros(9, dtype=np.int8)
        done_t = torch.zeros(1, dtype=torch.bool, device=device)
        rng = torch.Generator(device=device).manual_seed(int(seed)) if temperature > 0.0 else None

        trajectory: list[dict[str, Any]] = []
        max_disp = 0.0
        steps = 0
        exit_reached = False
        dead = False
        last_priv = priv0

        policy.eval()
        with torch.inference_mode():
            while True:
                p_obs = preprocessor.process(raw_obs, prev_action=prev_action)
                obs_t, prev_act_t = processed_to_tensors(p_obs, device=device, include_depth=include_depth)
                out = policy.step(obs_t, state, prev_act_t, done=done_t, compute_aux=False)
                state = out.state
                if temperature > 0.0:
                    action = torch.bernoulli(torch.sigmoid(out.action_logits[0] / temperature), generator=rng).to(torch.int8).cpu().numpy()
                else:
                    action = (out.action_logits[0] > 0.0).to(torch.int8).cpu().numpy()

                raw_obs, base_r, term, trunc, info = env.step(action)
                last_priv = env.unwrapped.privileged_state()
                shaper.compute(raw_obs, last_priv, base_r, term, trunc, info)
                dx = float(last_priv.get("x", x0)) - x0
                dy = float(last_priv.get("y", y0)) - y0
                max_disp = max(max_disp, math.hypot(dx, dy))

                if steps % 10 == 0 or term or trunc:
                    mode_probs = torch.softmax(out.mode_logits[0], dim=-1).cpu().tolist()
                    trajectory.append(
                        {
                            "step": steps,
                            "x": round(float(last_priv.get("x", 0.0)), 1),
                            "y": round(float(last_priv.get("y", 0.0)), 1),
                            "yaw": round(float(last_priv.get("angle", 0.0)), 1),
                            "health": round(float(last_priv.get("health", 0.0)), 1),
                            "active_slots": int((state.landmark_valid[0] > 0.05).sum().item()),
                            "map_norm": round(float(state.spatial_map[0].norm().item()), 3),
                            "mode_probs": [round(float(p), 3) for p in mode_probs],
                        }
                    )

                prev_action = action
                steps += 1
                dead = bool(last_priv.get("dead", False))
                if term or trunc:
                    exit_reached = bool(info.get("exit_reached", False))
                    break

        summary = {
            "wad": Path(wad_path).name,
            "family": family,
            "seed": int(seed),
            "steps": steps,
            "exit_reached": exit_reached,
            "dead": dead,
            "kills": int(last_priv.get("kills", 0)),
            "unique_cells": len(shaper.visited_cells),
            "max_displacement": round(max_disp, 1),
            "keys_found": len(shaper.keys_held),
            "doors_opened": len(shaper.open_doors),
            "locked_doors_opened": len(shaper.unlocked_doors),
        }
        summary["classification"] = classify_failure(summary)
        summary["trajectory"] = trajectory
        summary["final_map_energy"] = np.round(state.spatial_map[0].pow(2).mean(dim=0).cpu().numpy(), 3).tolist()
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and classify agent failures on validation maps")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/agent.pt")
    parser.add_argument("--manifest", type=str, default="maps/dataset/manifest.json")
    parser.add_argument("--max-maps-per-tier", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="runs/diagnostics/failures")
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    policy, prep, _ = load_policy_from_checkpoint(args.checkpoint, device=dev)

    m_path = Path(args.manifest)
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest.get("val", []):
        by_tier[entry["family"]].append(entry)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    all_summaries: list[dict[str, Any]] = []

    for family, entries in sorted(by_tier.items()):
        for idx, entry in enumerate(entries[: args.max_maps_per_tier]):
            wad_path = m_path.parent / entry["file"]
            res = trace_episode(
                policy=policy,
                preprocessor=prep,
                wad_path=wad_path,
                family=family,
                episode_ticks=args.ticks,
                seed=args.seed + idx,
                device=dev,
                temperature=args.temperature,
            )
            cls = res["classification"]
            counts[cls] += 1
            compact = {k: v for k, v in res.items() if k not in {"trajectory", "final_map_energy"}}
            all_summaries.append(compact)
            if cls != "success":
                trace_file = out_dir / f"{family}_{wad_path.stem}_s{args.seed + idx}.json"
                trace_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

    report = {"checkpoint": args.checkpoint, "counts": dict(counts), "episodes": all_summaries}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Failure breakdown:", dict(counts))
    for item in all_summaries:
        print(
            f"  {item['family']:22s} | {item['wad']:32s} | {item['classification']:18s} | "
            f"cells={item['unique_cells']:3d} | disp={item['max_displacement']:6.1f} | "
            f"kills={item['kills']} | keys={item['keys_found']} | locked={item['locked_doors_opened']}"
        )


if __name__ == "__main__":
    main()
