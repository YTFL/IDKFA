# DOOM Zero-Shot Agent: End-to-End Technical Project Plan

Prepared for the DevForge DOOM challenge. Covers rules, setup, data and maps, architecture, training, inference, testing and submission.

## 1. Mission, scoring and strategy

**Goal:** a neural agent (class `DoomAgent`) that plays unseen DOOM maps zero-shot using only the wrapper's observations: 64 ego-centric raycasts with hit types, a depth buffer, a vitals vector, a visible-entity list and a low-res RGB frame.

**Score = tier-weighted sum over hidden maps.** Per map: exit (largest reward, small bonus for health and time), progression (keycards, doors, locked doors), exploration (unique areas), combat (kills minus small penalties for damage and early death).

**Strategy implications**

1. Partial credit is everywhere. A robust explorer that survives beats a brittle agent that sometimes exits.
2. The training-map distribution is your real advantage. The hidden maps are unseen, so diversity of training maps decides generalization more than any single architectural trick.
3. Tier 3 (keycard, locked door, exit) needs memory of where a locked door was seen. Design memory explicitly for this.
4. A crash means standing still for the rest of the map. Latency safety and exception handling are worth points.
5. The innovation criterion rewards bespoke modules: entity set-encoder, egocentric neural map memory, landmark memory, curiosity, hierarchical policy. Build a clear, defensible USP.

## 2. Rules and limits cheat sheet

| Item | Constraint |
| --- | --- |
| Repo | Public GitHub, created from official template during the hackathon, at most 1 GB including Git LFS; weights via LFS |
| Entry point | `submission/agent.py` with `DoomAgent.__init__(observation_space, action_space, device)`, `reset()`, `act(observation) -> action` (one decision per 4 ticks) |
| Dependencies | No Dockerfile. Base image: Python 3.13, vizdoom 1.3.3, gymnasium, torch, jax/flax, numpy, scipy, opencv. Optional `submission/requirements.txt` with pinned extras |
| Hardware at eval | RTX 3090 sandbox, 2 GB VRAM, 8 GB RAM, 1 CPU thread, no network, repo mounted read-only |
| Timeouts | 60 s init, 10 s reset, 10 s first act per map, 5 s other acts, about 40 ms average per decision per map |
| Failure mode | Crash, timeout or invalid action: player stands still for the rest of the map |
| Maps | Hidden, four tiers, tick cap about 2,100 to 4,200, fixed seed, Freedoom 2 IWAD, skill 3, vanilla engine |
| Training freedom | You may mod the engine, ACS, game variables and maps for curriculum. Evaluation is unmodified |
| Discouraged | Hardcoded if/else trees, A\*/Dijkstra, atan2 aimbots. Decisions must come from trained neural parameters |
| Fair play | Use only given observations. No access to engine, eval maps, network or host. No resource exhaustion |
| Judging | Auto-score, Top 10 get code verification and desk review with live demo and Q&A, Top 2 go to final judging |
| Deliverables | Agent, training pipeline in repo, validator pass, 2-page PDF brief |

**Exact values live in** `env/doom_env/limits.py` and `docs/CONTRACT.md`. Treat those files as the source of truth over this plan.

## 3. Prerequisites and tooling

**Accounts and hardware**

- GitHub account, Git and Git LFS installed (`git lfs install`).
- Training machine: as many CPU cores as possible (env throughput is CPU-bound) plus one GPU for learning. A single 3090-class GPU is plenty because the model is small.
- Disk: 20 GB for maps, checkpoints and logs (keep these out of the repo).

**Software**

- Python 3.13 environment matching the base image where possible (otherwise 3.11/3.12 for training, but test inference in 3.13 before submitting).
- `vizdoom==1.3.3`, `gymnasium`, `torch`, `numpy`, `scipy`, `opencv-python`, `tensorboard` or `wandb` (offline mode is fine).
- WAD tooling: SLADE or Ultimate Doom Builder (inspect and edit maps), `omgifol` (Python WAD I/O) or direct UDMF text generation, Freedoom 2 IWAD (bundled with the wrapper), optional Obsidian level generator for rapid map variety.
- Profiling: `py-spy`, `torch.profiler`, `cProfile`.

**Knowledge to refresh**

- Recurrent PPO and GAE, V-trace (if going IMPALA), RND and episodic novelty, neural map memory (Parisotto and Salakhutdinov), attention over sets (DeepSets, Set Transformer), auxiliary-task learning (UNREAL), hierarchical RL (options, goal-conditioned policies).
- DOOM mapping basics: sectors, linedefs, things, doors, key-locked doors, exit specials.

## 4. Phase 0: Setup and contract reading (about 2 to 4 hours)

Steps

1. Create the repo from the official template (Use this template on GitHub). Name it clearly. Enable Git LFS for `*.pt`, `*.pth`, `*.onnx`, `*.npz`.
2. Clone, create a virtualenv, install the template requirements.
3. Read, in order: `docs/CONTRACT.md`, `env/doom_env/limits.py`, the wrapper source, `submission/agent.py` skeleton, `validate.py`.
4. Write down in `docs/contract_notes.md` the exact facts you need: action space type and size (discrete or multi-discrete, which buttons), observation dict keys and shapes, ray angular layout and field of view, hit-type encoding, entity list format and maximum length, vitals ordering and scaling, frame resolution, episode termination rules, reward info exposed (if any), how maps and seeds are loaded.
5. Run the provided test map with a random agent. Log: ticks per second, observation statistics, episode length.
6. Run `python validate.py` on the untouched skeleton to confirm your toolchain works.
7. Commit this baseline as the first commit. Commit history matters for the original-work check, so commit often with meaningful messages.

**Exit criteria:** you can step the env, you know every observation field, and the validator passes on the skeleton.

## 5. Phase 1: Training maps and curriculum environments (the highest-leverage phase)

The hidden maps cover four tiers, so build training sets that mirror them.

**Step 1. Define map families**

- A: Combat arenas (single or few rooms, varying enemy types and counts, ammo and health pickups).
- B: Multi-room exploration (rooms and corridors, loops, dead ends, ordinary doors, exit somewhere far from spawn).
- C: Keycard puzzles (key in a side branch, locked door of matching colour blocking the exit path).
- D: Full gauntlet (all of the above, larger, more enemies).

**Step 2. Sources of maps**

1. Provided test map (use for debugging only, never as your only map).
2. Freedoom 2 and Freedoom Phase 1 levels, classic-style community WADs with permissive licences (check licences, keep them out of the repo if unclear; you may reference the download script instead).
3. Procedural generator of your own (strongly recommended, see below).
4. Obsidian (OBLIGE) generated levels for style variety.

**Step 3. Build a procedural generator (`training/mapgen/`)**

- Algorithm: BSP or random rooms-and-corridors on a grid, then add loops, vary room sizes, ceiling and floor heights, light levels and textures.
- Output UDMF TEXTMAP (simple text format) or Doom-format via omgifol. The wrapper loads either.
- Place: player start (thing 1), enemies (zombieman 3004, shotgun guy 9, imp 3001, demon 3002, spawn counts by difficulty), items (medikit 2012, stimpack 2011, clip 2007, shells 2008, shotgun 2001), keycards (blue 5, yellow 6, red 13), ordinary doors (special 1) and key doors (blue 26, yellow 27, red 28 for repeatable doors; verify numbers against the Doom linedef type reference), exit (switch special 11 or walkover special 52).
- Solvability check in the generator (not in the agent): ensure the key is reachable before the locked door, and the exit is reachable after unlocking. Generator-side graph search is allowed because it is data tooling, not the agent.
- Node building: test whether the engine builds nodes on load. If not, run a node builder (ZDBSP or AJBSP) as a post-process.
- Parameters per map: room count, corridor width, enemy density, key colour, door count, tick cap. Sample them randomly (domain randomization).
- Generate at least 500 to 2,000 distinct maps, split into train and validation (hold out about 10 to 15 percent, never train on the validation maps).

**Step 4. Curriculum wrappers**

- Training-only mods are allowed: auto-open doors, infinite ammo, god mode for navigation pretraining, key-door removal for exploration pretraining.
- Stage design:
  1. Stage 0: Single room, one enemy, learn to turn, aim and fire.
  2. Stage 1: Combat arenas with multiple enemies and health management.
  3. Stage 2: Multi-room navigation to the exit, doors auto-open, no enemies.
  4. Stage 3: Add ordinary doors and enemies.
  5. Stage 4: Keycard maps with vanilla rules.
  6. Stage 5: Full gauntlet mix. Keep rehearsing earlier stages (mix 10 to 20 percent) to avoid forgetting.
- Promotion rule: advance when rolling success rate over the last 200 episodes exceeds a threshold (for example 70 percent), using automatic curriculum with a sampling distribution over difficulty (ALP-GMM style or simple bucket sampling).

**Step 5. Env config and speed**

- Match evaluation: frame skip 4 ticks, skill 3, Freedoom 2 IWAD, same observation wrapper.
- Disable rendering extras you do not need, use the lowest RGB resolution allowed.
- Benchmark: aim for at least 800 to 2,000 agent steps per second per core.

**Exit criteria:** a map factory producing varied, verified-solvable maps across all four families, a validation set, and curriculum wrappers that can be toggled by config.

## 6. Phase 2: Observation encoding and preprocessing

Write `submission/features.py` (shared by training and inference so there is zero train/serve skew).

**Inputs and processing**

1. **Rays (64):** per ray, distance normalized to \[0,1\] (log or inverse scaling helps near-field precision) plus a one-hot hit type (wall, door, locked door, enemy, item, none). Treat the rays as a circular 1D signal ordered by angle, so use circular padding in convolutions.
2. **Depth buffer:** downsample (for example average-pool to 1x32 or 8x32), normalize, and treat as an additional ray-like channel set.
3. **Vitals:** health, armor, ammo, keycards, velocity. Scale each to roughly \[0,1\] or standardized values. Keep keycard flags as binary inputs.
4. **Entities:** each visible entity as a row of features: type one-hot, relative x and y (or range and bearing as sin/cos), relative velocity if given, health or state if given. Pad to a fixed maximum with a boolean mask.
5. **RGB (optional, hybrid):** 64x48 or smaller, tiny CNN branch. Start without it. Add only if ablation shows it helps (for example for seeing door colours or switches).
6. **Previous action and previous reward:** feed the last action one-hot into the policy so it can reason about its own motion.
7. **Observation normalization:** running mean and variance statistics frozen at the end of training and saved with the weights.

**Edge cases to handle**

- Missing or empty entity list, NaNs, unexpected dtypes: sanitize with `np.nan_to_num` and clip.
- Observation space introspection in `__init__` so a key rename does not crash you.

**Exit criteria:** unit tests (`tests/test_features.py`) that feed recorded observations and verify shapes, ranges and mask logic.

## 7. Phase 3: Model architecture (the USP)

Name it something distinctive, for example **EgoMap-Landmark Recurrent Policy (EMLRP)**. Keep total parameters between 1 and 5 million so inference fits the latency budget on one CPU thread.

**Module 1: Ray encoder**

- Circular 1D CNN: 3 to 4 conv layers, kernel 5, channels 32, 64, 64, stride 2 on the later layers, GELU, LayerNorm. Output a ray embedding of 128 dimensions plus a per-sector feature map for the egocentric map writer.

**Module 2: Entity set encoder (variable enemies)**

- Per-entity MLP, then 2 layers of multi-head self-attention (4 heads, dim 64) with the mask, then attention pooling with a learned query vector conditioned on the ray embedding. This yields a permutation-invariant threat and item summary independent of count.
- Also output a per-entity importance score as an auxiliary head (for example predicting which entity will damage you next).

**Module 3: Short-term memory (recurrent core)**

- GRU or LSTM (hidden 256), with layer norm. Input: concatenation of ray embedding, entity summary, vitals embedding, previous action embedding, map readout (Module 4).
- Train with truncated BPTT, sequence length 64 to 128, storing and resetting hidden state at episode boundaries.

**Module 4: Egocentric neural spatial map (long-term spatial memory)**

- A learned feature grid around the agent, for example 32x32 cells with 16 channels, each cell covering about 64 map units (tune to the typical room size).
- **Egomotion update:** each step, shift and rotate the grid according to the agent's own motion. The agent knows its commanded turn (discrete action) and receives velocity in vitals; integrate to estimate translation and yaw change. Use differentiable `grid_sample` with an affine transform so the whole thing is trainable. Start with dead-reckoning from the action and velocity inputs and let a small learned correction network refine it.
- **Write:** from the ray and depth embeddings, a learned write head produces cell features at the ray hit points (wall, door, locked door, item, enemy) and free-space evidence along each ray. Blend with decay (a learned gate, like a GRU on the grid) so stale information fades.
- **Read:** a small CNN over the grid, plus attention readout from the GRU state (query) over grid cells, producing a map summary vector. This is what lets the policy remember explored corridors, dead ends and where a locked door was seen.
- Compute cost: 32x32x16 with a few conv layers is cheap on CPU. Verify in profiling.

**Module 5: Landmark slot memory (keycard to locked door)**

- N slots (for example 16) storing key-value entries: type embedding (keycard colour, locked door colour, exit switch, health item, ammo item) and position in the egocentric frame.
- **Write** when an entity or ray hit of a landmark type appears. A learned gate decides whether to create or update a slot (same type and nearby position updates).
- **Update positions** with the same egomotion transform as the map, so slots stay in the agent frame without any global coordinate.
- **Read** via cross-attention from the GRU state. This gives explicit support for: see a locked door, go explore, find the key, return.
- Include a binary memory of keycards held (from vitals) so the policy knows when the locked door is now openable.

**Module 6: Hierarchical control (optional but rewarded)**

- High-level manager (runs every K steps, for example 8) outputs a goal embedding or a discrete mode (explore, fight, retreat or heal, go to landmark, exit). Worker policy conditions on this embedding.
- Train the manager by RL on the sparse and shaped reward, and the worker with dense intrinsic reward for achieving the manager's goal (for example, reduce the distance to the target landmark slot or face the enemy). If time is short, implement the mode as a latent variable learned end to end, not as hand-coded rules.

**Module 7: Heads**

- Policy head (matches the action space from CONTRACT.md; likely factored discrete buttons such as move, turn, shoot, use). Use a factorized categorical (one head per button group) if multi-discrete.
- Value head (separate MLP, optionally two value heads: extrinsic and intrinsic, as in RND).
- Auxiliary heads (training only, discarded at inference): next-ray prediction, relative direction to the nearest seen landmark, inverse dynamics, enemy-kill prediction, damage-in-next-N-steps prediction.

**Diagram of the data flow:** observation → feature builder → ray CNN + entity set encoder + vitals MLP → fuse with map readout and landmark readout → GRU → manager and worker (or flat) → policy, value and auxiliary heads. The egomotion block updates map and landmark memory each step.

**Compliance note:** the map and landmark memories are learned neural state fed to the policy, not classical planners. Never run A\*, hand-written pathfinding or atan2 aiming at inference. Describe the egomotion block in the brief as learned state estimation. When unsure, ask organizers whether dead-reckoned egomotion from the agent's own actions is acceptable (it uses only provided observations plus your own outputs).

## 8. Phase 4: Reward design and exploration

Training may use privileged information for **reward computation only** (for example true position, sector visited, via ViZDoom game variables or the engine API in your own training process). The policy itself never sees it.

**Extrinsic reward (all scaled so episode returns sit around 10 to 30)**

| Event | Reward |
| --- | --- |
| Reach exit | +10 (plus a small bonus for remaining health and time) |
| Keycard pickup | +2 |
| Door opened | +0.5 |
| Locked door unlocked | +2 |
| New area (unique cell or sector) | +0.02 per new cell, with a cap per episode |
| Enemy killed | +0.5 |
| Damage taken | -0.005 per HP |
| Death | -3 |
| Time | -0.001 per step |
| Stuck (no displacement over N steps) | small penalty |

Mirror the evaluation scoring: the scorer rewards exit, progression, exploration and combat. Do not over-reward kills or the agent will camp enemies and forget the exit.

**Potential-based shaping:** add shaping based on training-side distance-to-exit or distance-to-key (computed from the map graph) using potential differences, which preserves the optimal policy. Anneal it toward zero over curriculum so the agent does not depend on it.

**Intrinsic curiosity**

1. **RND (Random Network Distillation):** predictor and fixed target networks over the ray and depth embedding (not the whole observation, so enemy motion does not dominate). Normalize intrinsic reward, use non-episodic intrinsic returns.
2. **Episodic novelty:** reward a state if its embedding is far from embeddings in the current episode's memory buffer (NGU style). This prevents revisiting the same room forever.
3. **Map-coverage bonus:** training-side visited-cell bonus, as above, to ground curiosity in the true scoring signal.
4. **Anti-stuck:** penalize near-zero displacement and wall-hugging jitter; also randomize action repeat or sticky actions slightly to break loops.

**Exit criteria:** reward logging per component, a unit test per reward event, and a short debug run showing each component firing correctly.

## 9. Phase 5: Training pipeline

**Algorithm:** custom recurrent PPO (written by you, not Stable-Baselines3) with GAE, or IMPALA with V-trace if you have many actors. Recurrent PPO is simplest and robust.

**Infrastructure (`training/`)**

- `envs/`: vectorized env pool with asynchronous workers (subprocess per core), each sampling maps from the curriculum distribution at reset.
- `models/`: the architecture modules, reused by `submission/` (import the same code or copy it with a script).
- `algo/ppo_recurrent.py`: rollout storage with hidden-state handling, minibatching by sequences, losses.
- `train.py`: config-driven (YAML), seeds, logging, checkpoint every N updates, resume support.
- `eval.py`: runs a checkpoint on the validation map set with the unmodified rules, reports per-tier scores using your reimplementation of the scoring rubric.

**Starting hyperparameters (tune)**

- Envs: 64 to 256 parallel, rollout length 128, so batch 8k to 32k steps.
- Discount 0.995, GAE lambda 0.95, clip 0.2, 4 epochs, minibatch by sequences of 64.
- Learning rate 3e-4 with linear decay, Adam, gradient clip 0.5.
- Entropy coefficient 0.01 decaying to 0.002 (keep exploration alive early).
- Value loss coefficient 0.5, auxiliary losses 0.05 to 0.2 each.
- Intrinsic reward coefficient 0.1 to 0.5, annealed over time.
- Observation noise and dropout of entity list during training (randomly drop entities or add jitter for robustness).

**Domain randomization**

- Randomize map layout, textures, light, enemy mix, spawn positions, initial yaw, and the number of keys and doors.
- Randomize the seed per episode. Evaluation uses fixed seeds, so also include some training maps run at fixed seeds to avoid overfitting on seed variance.

**Compute plan**

- Short hackathon budget: 30 to 100 million agent steps total. Allocate roughly: 10 percent Stage 0 to 1, 25 percent Stage 2 to 3, 40 percent Stage 4, 25 percent Stage 5 plus fine-tuning.
- Always train in stages from a saved checkpoint so a late failure does not cost everything.
- Run 2 to 3 seeds or architectural variants in parallel early (cheap), then pick one and scale.

**Monitoring**

- Log: episode return per component, success rate per map family, exploration coverage, kills, deaths, entropy, value loss, KL, explained variance, FPS.
- Save sample replay videos or ray traces every N updates for qualitative checks.
- Red flags: entropy collapse, value loss exploding, success rate plateau on keycard maps (usually memory or credit assignment), agent spinning in place (needs anti-stuck and more entropy).

**Checkpoint selection:** choose by validation-map score, not training reward. Keep the top 3 and average (or choose by the lowest variance across seeds).

## 10. Phase 6: Inference agent and latency engineering

Write `submission/agent.py`.

**Class contract**

- `__init__(observation_space, action_space, device)`: load weights from the repo (relative path from the file), build the model, put it in `eval()`, `torch.set_num_threads(1)`, `torch.no_grad()` for all calls, run a warm-up forward pass with dummy observations (JIT and allocator warm-up inside the 60 s init budget), and zero the memory.
- `reset()`: zero GRU hidden state, the neural map, landmark slots, previous action, and any running counters. This is called before every map; stale memory from the last map is a classic bug.
- `act(observation)`: sanitize, featurize, forward, sample or argmax, map to the exact action format from CONTRACT.md, return. Wrap the whole body in try/except and return a safe fallback action (for example the last valid action or a forward move) on any error so a transient bug does not freeze the player.

**Deterministic vs stochastic action choice:** evaluation seeds are fixed but the agent may still benefit from sampling to escape loops. Test both on validation. A common best choice is a sample from the policy with temperature below 1 (for example 0.5 to 0.8), or argmax with a stuck-detector in the network input (the recurrent state can learn this).

**Latency budget (average about 40 ms, hard per-act limit 5 s)**

- Target under 10 ms per act on one CPU thread; leaves 4x headroom for the slower grader CPU.
- If on GPU (2 GB VRAM limit): the model weights and activations will be well under 100 MB, but CPU may be faster for tiny models due to launch overhead. Benchmark both and select a device inside `__init__` when the passed `device` allows.
- Optimizations: fuse operations, use `torch.jit.trace` or `torch.compile` only if it measurably helps (watch compile time against the 10 s first-act limit), avoid Python loops over slots and entities (vectorize), preallocate buffers, avoid `.cpu()` syncs in the middle of the forward, and keep float32 unless quantization is verified.
- Memory: avoid unbounded growth of any list or buffer in `act` (episodes up to about 4,200 ticks, so about 1,050 decisions).

**Packaging**

- Weights under `submission/weights/` through Git LFS. Verify with `git lfs ls-files`.
- Minimal `submission/requirements.txt` only if you truly need extras. Pin versions. Prefer none.
- No network calls, no reading outside the repo, no writing to the read-only repo (write logs only to a temp directory, or not at all).

**Exit criteria:** the agent runs a full validator dry run, under the latency target, with deterministic reproduction of the same scores on a fixed seed.

## 11. Phase 7: Testing, evaluation and ablations

**Local evaluation harness (`training/eval.py`)**

1. Reimplement the published scoring structure: exit, health and time bonus, keycards, doors, locked doors, exploration (unique cells), kills, damage and death penalties. Weights are approximate, so rely on relative ranking.
2. Run on held-out validation maps per tier with fixed seeds and the exact tick caps (about 2,100 to 4,200). Run each map multiple seeds if you generate seeds yourself.
3. Report mean and standard deviation per tier and the weighted total.

**Test layers**

- Unit tests: feature builder, egomotion transform (a known motion must shift a known point correctly), slot write and read, reward events.
- Integration test: agent through the wrapper end to end with the official limits.
- Robustness tests: observation noise, empty entity lists, maximum entities, NaN injection, early death, map with no keycard, map with locked door visible first.
- Memory test: synthetic task where the key is seen early, then the agent must return to the door after wandering, to verify the landmark memory works.
- Generalization test: evaluate on the official test map and on a few classic Freedoom levels that were never in the training set.

**Ablations (these also feed the project brief and the live Q&A)**

- Without the neural map, without landmark slots, without entity attention (use mean pooling), without RND, without the hierarchical manager, rays only vs rays plus RGB.
- Report the score deltas. Judges love seeing evidence that each custom module earns its place.

**Failure analysis loop**

- Record episodes where the agent fails (stuck, died early, missed the key). Replay and classify the failure cause. Feed the findings into new map generator settings (for example more dead ends) or reward tweaks.

## 12. Phase 8: Submission, brief and demo

**Pre-submit checklist**

1. `python validate.py` passes: repo size at most 1 GB, interface, full dry-run episode, latency within limits.
2. Clean clone test: clone the repo into a fresh directory (and a fresh Python 3.13 environment) and run the validator again. Confirms that LFS weights are fetched and no local path dependencies exist.
3. Weights are tracked in LFS and pushed (`git lfs push --all origin`).
4. Repo is public, created from the template, with a clear README (how to train, how to evaluate, how to reproduce results), the training code, map generator and configs. No secrets, no giant checkpoints outside LFS.
5. Seeds fixed and documented, licences for any third-party maps noted.
6. No code path touches the network, the engine internals, or evaluation maps.
7. Tag the final commit (for example `v1.0-submission`) and note the commit hash.

**2-page project brief (PDF)**

- Page 1: problem framing, USP in one sentence, architecture diagram, observation encoding, spatial and landmark memory design.
- Page 2: training and exploration pipeline (curriculum, rewards, RND and episodic novelty), results table (per tier on validation maps), ablation highlights, limitations and future work.
- Keep figures clean and put one strong diagram of the memory system on page 1.

**Live demo and Q&A preparation (Top 10)**

- Prepare a script: load the agent, run a keycard map, show the neural map and landmark slot visualization (a debug view rendering the map memory and slot attention weights).
- Be ready to explain: why these inputs, how variable enemy counts are handled, how memory resets, how you avoid hardcoded rules, how you verified generalization.
- Have fallbacks: recorded videos in case the live run fails.

## 13. Suggested timeline (adapt to your hackathon length)

| Block | Work | Deliverable |
| --- | --- | --- |
| Hours 0 to 4 | Phase 0, contract notes, baseline validator | Repo, notes, random-agent benchmark |
| Hours 4 to 14 | Phase 1 map factory, curriculum wrappers | Map generator, 500+ maps, validation split |
| Hours 10 to 18 | Phase 2 and 3 code, unit tests (overlaps Phase 1) | Features, model modules, tests |
| Hours 16 to 24 | Phase 4 and 5 code, smoke training on Stage 0 to 2 | Working PPO loop with logging |
| Hours 24 to 60 | Main training through Stages 2 to 5, parallel variants, periodic validation | Checkpoints, curves |
| Hours 50 to 66 | Phase 6 agent packaging and latency work (start early with an interim checkpoint) | Validator-passing agent |
| Hours 60 to 72 | Phase 7 ablations, final selection, Phase 8 brief and demo prep | Final tag, PDF, demo script |

**Rule of thumb:** get an end-to-end validator-passing agent submitted as soon as you have any trained checkpoint (even a weak one), then iterate. This protects you from a last-minute packaging failure.

## 14. Risk register

| Risk | Mitigation |
| --- | --- |
| Overfits to a few maps | Large procedural map set, held-out validation, domain randomization |
| Agent never solves keycard puzzles | Dedicated key curriculum stage, landmark memory, key and door shaping rewards, memory test task |
| Stuck in corners or loops | Anti-stuck penalty, RND plus episodic novelty, entropy floor, recurrent state with action history |
| Latency spike or crash | Small model, benchmark on one thread, try/except fallback action, warm-up in init |
| Train/serve skew | Single shared feature module, saved normalization stats, clean-clone validator test |
| Rule violation (scripted behaviour) | No A\*, no atan2 aimbot, all decisions through network, document memory as learned state |
| Large repo or missing LFS | Weights via LFS, keep checkpoints and logs outside the repo, check `git lfs ls-files` |
| Reward hacking (kill farming, camping) | Cap per-event rewards, include time penalty, validate with the official-style score |
| Python version mismatch | Test inference in a 3.13 environment before submitting |
| Sparse reward in Tier 3 | Potential-based shaping annealed over time, curriculum, auxiliary landmark-direction loss |

## 15. Repository layout (suggested)

```
repo/
  submission/
    agent.py              # DoomAgent
    features.py           # shared preprocessing
    model.py              # EMLRP modules
    weights/              # Git LFS
    requirements.txt      # optional, pinned
  training/
    train.py  eval.py  configs/
    algo/ppo_recurrent.py
    envs/ (vector env, curriculum wrappers, reward components)
    mapgen/ (generator, solvability checker, node build script)
    tools/ (visualization, replay, scoring reimplementation)
  tests/
  docs/ (contract_notes.md, brief.pdf, figures)
  README.md
```

## 16. Immediate next actions (first 60 minutes)

1. Create the repo from the template and enable LFS.
2. Read CONTRACT.md and limits.py, fill in `docs/contract_notes.md`.
3. Run the random agent and the validator on the skeleton.
4. Decide the map-generation approach (own UDMF generator first, Obsidian as a supplement) and start it as the first big task.
5. Split work if you have teammates: one on maps and curriculum, one on model and training loop, one on agent packaging, evaluation and the brief.
