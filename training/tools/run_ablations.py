"""Run EMLRP module ablations (inference knockout or retrained configs) on held-out validation maps.

Usage:
    python training/tools/run_ablations.py --checkpoint submission/weights/agent.pt --max-maps-per-tier 2 --ticks 400
    python training/tools/run_ablations.py --mode train --variant no_rnd --total-steps 200000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.models import AblationFlags  # noqa: E402
from training.envs.curriculum import load_curriculum_yaml  # noqa: E402
from training.eval import evaluate_on_manifest, load_policy_from_checkpoint  # noqa: E402

ABLATION_CASES: dict[str, AblationFlags] = {
    "baseline": AblationFlags(),
    "no_map": AblationFlags(ablate_map=True),
    "no_landmarks": AblationFlags(ablate_landmarks=True),
    "no_entity_attn": AblationFlags(ablate_entity_attn=True),
    "no_mode": AblationFlags(ablate_mode_head=True),
    "no_depth": AblationFlags(ablate_depth=True),
}

TRAIN_CONFIGS: dict[str, str] = {
    "no_rnd": "training/configs/ablation_no_rnd.yaml",
    "no_depth": "training/configs/ablation_no_depth.yaml",
}


def run_inference_ablations(
    checkpoint: str | Path,
    manifest: str | Path,
    max_maps_per_tier: int | None,
    episode_ticks: int,
    seeds: list[int],
    temperature: float,
    device: str,
) -> dict[str, Any]:
    policy, prep, _ = load_policy_from_checkpoint(checkpoint, device=device)
    rows: dict[str, dict[str, Any]] = {}
    base_score: float | None = None

    for name, flags in ABLATION_CASES.items():
        rep = evaluate_on_manifest(
            policy=policy,
            preprocessor=prep,
            manifest_path=manifest,
            max_maps_per_tier=max_maps_per_tier,
            episode_ticks=episode_ticks,
            seeds=seeds,
            device=device,
            deterministic=temperature <= 0.0,
            temperature=temperature,
            ablations=flags,
        )
        if base_score is None:
            base_score = rep["weighted_score"]
        rows[name] = {
            "weighted_score": rep["weighted_score"],
            "weighted_score_std": rep["weighted_score_std"],
            "delta_weighted": rep["weighted_score"] - base_score,
            "mean_ms_per_step": rep["mean_ms_per_step"],
            "tiers": {
                fam: {
                    "mean_score": st["mean_score"],
                    "exit_rate": st["exit_rate"],
                    "mean_kills": st["mean_kills"],
                    "mean_cells": st["mean_cells"],
                }
                for fam, st in rep["tiers"].items()
            },
        }
    return {"checkpoint": str(checkpoint), "seeds": seeds, "episode_ticks": episode_ticks, "variants": rows}


def _format_markdown(results: dict[str, Any]) -> str:
    variants = results["variants"]
    base = variants["baseline"]
    lines = [
        "# Phase 7 Ablation Study (Inference Module Knockout)",
        "",
        f"- **Checkpoint:** `{results['checkpoint']}` | **Seeds:** `{results['seeds']}` | **Ticks:** `{results['episode_ticks']}`",
        "",
        "| Variant | Weighted Score | Delta vs Baseline | Combat | Explore | Keycard | Gauntlet | ms/step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, info in variants.items():
        tiers = info["tiers"]
        lines.append(
            f"| `{name}` | {info['weighted_score']:.3f} | {info['delta_weighted']:+.3f} | "
            f"{tiers.get('combat_arena', {}).get('mean_score', 0.0):.2f} | "
            f"{tiers.get('multi_room_exploration', {}).get('mean_score', 0.0):.2f} | "
            f"{tiers.get('keycard_puzzle', {}).get('mean_score', 0.0):.2f} | "
            f"{tiers.get('full_gauntlet', {}).get('mean_score', 0.0):.2f} | "
            f"{info['mean_ms_per_step']:.2f} |"
        )
    lines.append(f"\nBaseline score: `{base['weighted_score']:.3f} +/- {base['weighted_score_std']:.3f}`\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EMLRP module ablations")
    parser.add_argument("--mode", choices=["inference", "train"], default="inference")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/agent.pt")
    parser.add_argument("--manifest", type=str, default="maps/dataset/manifest.json")
    parser.add_argument("--max-maps-per-tier", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="runs/diagnostics")
    parser.add_argument("--variant", choices=sorted(TRAIN_CONFIGS), default="no_rnd")
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        from training.train import train

        cfg = load_curriculum_yaml(ROOT / TRAIN_CONFIGS[args.variant])
        train(config=cfg, device_override=args.device, total_steps_override=args.total_steps)
        return

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    results = run_inference_ablations(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        max_maps_per_tier=args.max_maps_per_tier,
        episode_ticks=args.ticks,
        seeds=args.seeds,
        temperature=args.temperature,
        device=dev,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = _format_markdown(results)
    (out_dir / "ablation_report.md").write_text(md, encoding="utf-8")
    (out_dir / "ablation_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
