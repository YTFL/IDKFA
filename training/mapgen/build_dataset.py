"""Batch generator for training and validation map datasets.

Generates verified-solvable UDMF maps across all four families:
  Family A: Combat Arena
  Family B: Multi-room Exploration
  Family C: Keycard Puzzle
  Family D: Full Gauntlet

Partitions them into train (85%) and validation (15%) splits, outputting
maps to maps/dataset/train and maps/dataset/val along with a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.mapgen.generator import MapFamily, ProceduralMapGenerator


def generate_dataset(
    output_dir: str | Path = "maps/dataset",
    total_maps: int = 600,
    val_split: float = 0.15,
    seed: int = 42,
) -> dict:
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    families = [
        MapFamily.FAMILY_A,
        MapFamily.FAMILY_B,
        MapFamily.FAMILY_C,
        MapFamily.FAMILY_D,
    ]

    manifest = {"train": [], "val": []}
    val_count = int(total_maps * val_split)
    train_count = total_maps - val_count

    print(f"Generating {total_maps} maps ({train_count} train, {val_count} val)...")

    generated = 0
    cur_seed = seed

    for split, count in [("train", train_count), ("val", val_count)]:
        target_dir = train_dir if split == "train" else val_dir
        for i in range(count):
            family = families[i % len(families)]
            gen = ProceduralMapGenerator(seed=cur_seed)
            cur_seed += 1

            # Domain randomization parameters
            difficulty = (i / count) * 0.9 + 0.1
            num_rooms = 4 + (i % 5)

            # Generate and verify solvability
            m, solvable = gen.generate(
                family=family,
                num_rooms=num_rooms,
                difficulty=difficulty,
            )

            if not solvable:
                # Retry with new seed if ever unsolvable
                retries = 0
                while not solvable and retries < 25:
                    cur_seed += 1
                    gen = ProceduralMapGenerator(seed=cur_seed)
                    m, solvable = gen.generate(
                        family=family,
                        num_rooms=num_rooms,
                        difficulty=difficulty,
                    )
                    retries += 1
            if not solvable:
                raise RuntimeError(f"Failed to generate solvable map for {family} after retries")

            filename = f"{family.value}_{i:04d}.wad"
            filepath = target_dir / filename
            m.write_wad(filepath)

            manifest[split].append(
                {
                    "file": str(filepath.relative_to(output_path)),
                    "family": family.value,
                    "difficulty": round(difficulty, 3),
                    "num_rooms": num_rooms,
                    "num_linedefs": len(m.linedefs),
                    "num_things": len(m.things),
                    "seed": cur_seed,
                }
            )
            generated += 1
            if generated % 25 == 0 or generated == total_maps:
                print(f"  Generated {generated}/{total_maps} maps...")

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDataset generation complete! Manifest written to {manifest_path}")
    print(f"Train maps: {len(manifest['train'])}, Validation maps: {len(manifest['val'])}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate procedural Doom map dataset")
    parser.add_argument("--output", type=str, default="maps/dataset", help="Output directory")
    parser.add_argument("--count", type=int, default=600, help="Total maps to generate")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    args = parser.parse_args()

    generate_dataset(output_dir=args.output, total_maps=args.count, val_split=args.val_split, seed=args.seed)
