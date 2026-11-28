# IDKFA — Zero-Shot DOOM Agent (EMLRP)

Submission for **Yugaantar 2026 · DevForge Track 1: "Can It Play DOOM?"**.
A `1.9M`-parameter neural agent (**EgoMap-Landmark Recurrent Policy — EMLRP**) built to play unseen `freedoom2.wad` levels zero-shot across combat, multi-room exploration, keycard puzzles, and full gauntlets using strictly ego-centric observations (`rays`, `depth`, `vitals`, `visible_entities`) and trained neural parameters.

---

## Architecture Highlights (`submission/`)

- **Circular 1D Ray Encoder** ([ray_encoder.py](submission/models/ray_encoder.py)): Circular convolutions over 64 ego-centric rays (linear + log distance and 6 hit types).
- **Entity Set Transformer** ([entity_encoder.py](submission/models/entity_encoder.py)): Permutation-invariant 2-layer self-attention + ray-conditioned attention pooling over up to 32 visible entities.
- **Egocentric Neural Spatial Map** ([spatial_map.py](submission/models/spatial_map.py)): Differentiable `F.grid_sample` affine-warped `32x32x16` feature grid driven by learned egomotion (`delta_xy, delta_theta`) + ConvGRU spatial writes.
- **Landmark Slot Memory** ([landmark_memory.py](submission/models/landmark_memory.py)): `16`-slot egocentric key-value memory tracking high-salience doors/items/entities and conditioning cross-attention on binary keycard vitals.
- **GRU + Latent Hierarchical Mode Head** ([emlrp.py](submission/models/emlrp.py)): `256`-dim GRU fused with a learned `5`-mode controller, `MultiBinary(9)` actor head, dual extrinsic/intrinsic critics, and training-only auxiliary heads.
- **Inference Agent** ([agent.py](submission/agent.py)): Runs single-step inference under `torch.inference_mode()` (`8.09 ms/step` over 1,050 decisions on 1 CPU thread, compact `7.32 MB` weights via Git LFS, full sanitization and edge-safe exception recovery).

## Tech & Rewards

- **Algorithm & Framework:** Custom PyTorch **Recurrent PPO** ([ppo_recurrent.py](training/algo/ppo_recurrent.py)) with truncated BPTT (`seq_len: 32`), dual extrinsic/intrinsic GAE critics, and `bf16` AMP running on **ViZDoom 1.3.3** (`freedoom2.wad`)—not Stable-Baselines3.
- **Parameters vs. Timesteps:** `1.9M` (`1,904,053`) refers to the **actual neural network size** (trainable parameters in `EMLRPPolicy`, `7.32 MB` exported weights), while training runs for `2.5M` (`2,500,000`) environment steps.
- **Rewards & Penalties:** Positive shaping for reaching the exit (`+10.0` + health/time bonus), keycards (`+2.0`), unlocking locked doors (`+2.0`), opening doors (`+0.5`), kills (`+0.5`), new-area exploration (`+0.02`), and potential-based navigation; penalties for death (`-3.0`), damage (`-0.005/HP`), getting stuck (`-0.05`), and step time (`-0.001`), plus RND + episodic novelty curiosity.

---

## Setup

```bash
uv sync
git lfs pull
```

---

## Reproducing Training & Evaluation

### Train from Config (Curriculum Stages 0–5)

```bash
python -m training.train --config training/configs/ppo_default.yaml --device cuda
```

Resume from a saved checkpoint:

```bash
python -m training.train --config training/configs/ppo_default.yaml --resume submission/weights/emlrp_latest.pt --device cuda
```

### Export Inference Weights & Evaluate

```bash
python training/tools/export_weights.py --checkpoint checkpoints/emlrp_ppo_v1/best.pt --output submission/weights/agent.pt
python training/tools/eval_inference_modes.py --checkpoint submission/weights/agent.pt --max-maps-per-tier 4
python training/tools/run_eval_report.py --checkpoint submission/weights/agent.pt
python training/tools/run_ablations.py --checkpoint submission/weights/agent.pt
python training/tools/analyze_failures.py --checkpoint submission/weights/agent.pt
```

### Live Q&A Memory Demo & 2-Page Brief

```bash
python training/tools/demo_replay.py --checkpoint submission/weights/agent.pt --terminal --save-html runs/diagnostics/demo_replay.html
python training/tools/build_brief.py --eval-json runs/diagnostics/eval_report.json --ablation-json runs/diagnostics/ablation_report.json --output docs/brief.pdf
```

### Validate Before Submission

```bash
python training/tools/pre_submit_check.py
python validate.py
python validate.py --docker
```

*(Note: `validate.py` uses Unix domain sockets for `dry_run()`; on Windows hosts run `validate.py --docker` via Docker/WSL.)*

---

## Repository Layout

```text
submission/
  agent.py                  # DoomAgent evaluation entry point
  features.py               # Shared preprocessing (zero train/serve skew)
  models/                   # EMLRP modules & B=1 / sequence adapters
  weights/agent.pt          # Compact Git LFS inference weights (~7.3 MB)
training/
  train.py                  # Config-driven Recurrent PPO training CLI
  eval.py                   # Multi-seed tier-weighted validation harness
  algo/ppo_recurrent.py     # Truncated BPTT Recurrent PPO + dual GAE + bf16 AMP
  configs/                  # ppo_default.yaml, curriculum.yaml, ablations
  envs/                     # Curriculum wrappers & double-buffered SubprocVecEnv
  mapgen/                   # Procedural UDMF map factory & BFS solvability checker
  rewards/                  # Extrinsic shaping + RND + episodic novelty
  tools/                    # Weight export, eval/ablation reports, failure analyzer, demo, brief
tests/                      # Unit, memory, robustness, interrupt & submission tests
docs/
  CONTRACT.md               # Official observation/action/interface contract
  EVALUATION.md             # Scoring & sandbox hardware limits
  brief.pdf                 # 2-page technical project brief
```

---

## Licences & Rules Compliance

- Decisions come strictly from trained neural network weights (`EMLRPPolicy`). No rule trees, `A*`/Dijkstra pathfinding, or `atan2` aimbots exist in `submission/`.
- Evaluation uses the unmodified ViZDoom `1.3.3` engine and `freedoom2.wad`. Redistributed Freedoom assets in `freedoom-0.13.0/` remain under the Freedoom BSD-style licence ([freedoom-0.13.0/COPYING.txt](freedoom-0.13.0/COPYING.txt)).
