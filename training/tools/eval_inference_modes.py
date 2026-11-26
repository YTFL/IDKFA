"""Compare deterministic vs low-temperature action decoding on held-out validation maps.

Usage:
    python training/tools/eval_inference_modes.py --checkpoint submission/weights/agent.pt --max-maps-per-tier 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.eval import evaluate_on_manifest, load_policy_from_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic vs temperature-sampled decoding on val maps")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/agent.pt")
    parser.add_argument("--manifest", type=str, default="maps/dataset/manifest.json")
    parser.add_argument("--max-maps-per-tier", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.8])
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    policy, prep, _ = load_policy_from_checkpoint(args.checkpoint, device=dev)

    print(f"Evaluating {args.checkpoint} on {args.manifest} (device={dev}, max_maps_per_tier={args.max_maps_per_tier})")
    for temp in args.temperatures:
        deterministic = temp <= 0.0
        report = evaluate_on_manifest(
            policy=policy,
            preprocessor=prep,
            manifest_path=args.manifest,
            max_maps_per_tier=args.max_maps_per_tier,
            episode_ticks=args.ticks,
            seed=args.seed,
            device=dev,
            deterministic=deterministic,
            temperature=max(0.0, float(temp)),
        )
        mode_str = "deterministic" if deterministic else f"temp={temp:.2f}"
        tier_bits = " | ".join(
            f"{fam}:{st['mean_score']:.2f}(exit {st['exit_rate']:.0%})"
            for fam, st in sorted(report["tiers"].items())
        )
        print(
            f"  [{mode_str:13s}] weighted={report['weighted_score']:7.3f} | "
            f"ms/step={report['mean_ms_per_step']:5.2f} | {tier_bits}"
        )


if __name__ == "__main__":
    main()
