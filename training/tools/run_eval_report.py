"""Full held-out validation and classic Freedoom generalization report generator.

Usage:
    python training/tools/run_eval_report.py --checkpoint submission/weights/agent.pt --quick
    python training/tools/run_eval_report.py --checkpoint submission/weights/agent.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.agent import DoomAgent  # noqa: E402
from training.eval import (  # noqa: E402
    DEFAULT_TIER_TICKS,
    evaluate_on_manifest,
    load_policy_from_checkpoint,
    score_agent_episode,
)

DEFAULT_GENERALIZATION_MAPS: tuple[tuple[str, str, int], ...] = (
    ("maps/two_rooms.wad", "MAP01", 2100),
    ("freedoom-0.13.0/freedoom1.wad", "E1M1", 2100),
    ("freedoom-0.13.0/freedoom1.wad", "E1M2", 2100),
    ("freedoom-0.13.0/freedoom2.wad", "MAP01", 2100),
    ("freedoom-0.13.0/freedoom2.wad", "MAP02", 2100),
)


def _format_markdown(
    checkpoint: str,
    val_report: dict[str, Any],
    gen_rows: list[dict[str, Any]],
    seeds: list[int],
    temperature: float,
) -> str:
    lines = [
        "# Phase 7 Evaluation Report",
        "",
        f"- **Checkpoint:** `{checkpoint}`",
        f"- **Seeds:** `{seeds}` | **Temperature:** `{temperature}`",
        f"- **Weighted Score:** `{val_report['weighted_score']:.3f} +/- {val_report['weighted_score_std']:.3f}`",
        (
            f"- **Latency:** mean `{val_report['mean_ms_per_step']:.2f} ms/step`, "
            f"p50 `{val_report['p50_ms_per_step']:.2f} ms`, p99 `{val_report['p99_ms_per_step']:.2f} ms`"
        ),
        "",
        "## Held-Out Validation Tiers",
        "",
        "| Tier | Weight | Ticks | N | Score (mean +/- std) | Exit % | Survival % | Key % | Locked Door % | Kills | Cells | p99 ms |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam, st in sorted(val_report["tiers"].items()):
        lines.append(
            f"| `{fam}` | {st['weight']:.1f} | {int(st['episode_ticks'])} | {int(st['count'])} | "
            f"{st['mean_score']:.2f} +/- {st['std_score']:.2f} | {st['exit_rate']:.1%} | {st['survival_rate']:.1%} | "
            f"{st['key_rate']:.1%} | {st['locked_door_rate']:.1%} | {st['mean_kills']:.2f} | {st['mean_cells']:.1f} | {st['p99_ms']:.2f} |"
        )
    if gen_rows:
        lines.extend(
            [
                "",
                "## Unseen Classic Map Generalization (`DoomAgent`)",
                "",
                "| Map | Outcome | Score | Kills | Cells | Mean ms | p99 ms |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in gen_rows:
            outcome = "exit" if row["exit_reached"] else ("died" if row["dead"] else "tick_cap")
            lines.append(
                f"| `{row['wad']}:{row['map_name']}` | {outcome} | {row['score']:.2f} | "
                f"{row['kills']} | {row['unique_cells']} | {row['mean_ms']:.2f} | {row['p99_ms']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed held-out evaluation and generalization report")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/agent.pt")
    parser.add_argument("--manifest", type=str, default="maps/dataset/manifest.json")
    parser.add_argument("--max-maps-per-tier", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ticks", type=int, default=None, help="Override uniform tick cap (default uses 2100/4200 per tier)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="runs/diagnostics")
    parser.add_argument("--quick", action="store_true", help="Fast smoke preset (4 maps/tier, seed 0, 400 ticks)")
    parser.add_argument("--skip-generalization", action="store_true")
    args = parser.parse_args()

    if args.quick:
        max_maps = 4 if args.max_maps_per_tier is None else args.max_maps_per_tier
        seeds = [0]
        tier_ticks = {k: (args.ticks or 400) for k in DEFAULT_TIER_TICKS}
        gen_ticks = args.ticks or 400
    else:
        max_maps = args.max_maps_per_tier
        seeds = args.seeds
        tier_ticks = {k: (args.ticks if args.ticks is not None else v) for k, v in DEFAULT_TIER_TICKS.items()}
        gen_ticks = args.ticks or 2100

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    policy, prep, _ = load_policy_from_checkpoint(args.checkpoint, device=dev)

    val_report = evaluate_on_manifest(
        policy=policy,
        preprocessor=prep,
        manifest_path=args.manifest,
        max_maps_per_tier=max_maps,
        seeds=seeds,
        tier_ticks=tier_ticks,
        device=dev,
        deterministic=args.temperature <= 0.0,
        temperature=args.temperature,
        verbose=True,
    )

    gen_rows: list[dict[str, Any]] = []
    if not args.skip_generalization:
        from doom_env import spaces as S

        prev_w = os.environ.get("NORDLICHT_WEIGHTS_PATH")
        prev_t = os.environ.get("NORDLICHT_TEMPERATURE")
        os.environ["NORDLICHT_WEIGHTS_PATH"] = str(Path(args.checkpoint).resolve())
        os.environ["NORDLICHT_TEMPERATURE"] = str(args.temperature)
        try:
            agent = DoomAgent(S.observation_space(), S.action_space(), dev)
            specs = DEFAULT_GENERALIZATION_MAPS[:2] if args.quick else DEFAULT_GENERALIZATION_MAPS
            for rel_wad, map_name, default_t in specs:
                wad_p = ROOT / rel_wad
                if not wad_p.is_file():
                    continue
                gen_rows.append(
                    score_agent_episode(
                        agent,
                        wad_p,
                        map_name=map_name,
                        episode_ticks=min(gen_ticks, default_t) if args.quick else default_t,
                        seed=seeds[0],
                    )
                )
        finally:
            if prev_w is None:
                os.environ.pop("NORDLICHT_WEIGHTS_PATH", None)
            else:
                os.environ["NORDLICHT_WEIGHTS_PATH"] = prev_w
            if prev_t is None:
                os.environ.pop("NORDLICHT_TEMPERATURE", None)
            else:
                os.environ["NORDLICHT_TEMPERATURE"] = prev_t

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_text = _format_markdown(args.checkpoint, val_report, gen_rows, seeds, args.temperature)
    (out_dir / "eval_report.md").write_text(md_text, encoding="utf-8")
    (out_dir / "eval_report.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "validation": val_report, "generalization": gen_rows}, indent=2),
        encoding="utf-8",
    )
    print(md_text)


if __name__ == "__main__":
    main()
