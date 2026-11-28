# Training Pipeline (`training/`)

Modular training, curriculum, evaluation, and diagnostic pipeline for **IDKFA (EMLRP)**.

## Structure

- `algo/ppo_recurrent.py`: Custom Recurrent PPO trainer with truncated BPTT (`seq_len=32`), chunk-boundary `EMLRPState` snapshots, dual extrinsic/intrinsic GAE, auxiliary heads, KL early stopping, and `bf16` CUDA mixed precision.
- `mapgen/`: Procedural UDMF WAD generator (`combat_arena`, `multi_room_exploration`, `keycard_puzzle`, `full_gauntlet`), BFS solvability checker, ZDBSP/node support, and `dataset_builder.py` generating train/val splits (`maps/dataset/manifest.json`).
- `envs/`: Curriculum manager (`curriculum.py` stages `0–5` with domain randomization and rehearsal) and `SubprocVecEnv` / `DummyVecEnv` (`vec_env.py`) supporting split-half asynchronous stepping.
- `rewards/`: `ExtrinsicRewardShaper` (exit, keycards, doors, unlocked doors, exploration cells, kills, annealed shortest-path potential shaping) + centralized CUDA `RNDCuriosity` and `EpisodicNoveltyBuffer`.
- `configs/`: `curriculum.yaml`, `ppo_default.yaml`, `ppo_smoke.yaml`, `ablation_no_rnd.yaml`, `ablation_no_depth.yaml`.
- `train.py`: Config-driven training entry point with atomic checkpointing, two-stage `Ctrl+C` protection, and `--resume` support.
- `eval.py`: Multi-seed validation harness scoring held-out `val/` maps and `DoomAgent` episodes.
- `tools/`: `export_weights.py`, `eval_inference_modes.py`, `run_eval_report.py`, `run_ablations.py`, `analyze_failures.py`, `demo_replay.py`, `build_brief.py`, `pre_submit_check.py`.

## Usage

```bash
python -m training.train --config training/configs/ppo_default.yaml --device cuda
python -m training.eval --checkpoint submission/weights/agent.pt --max-maps-per-tier 4
```
