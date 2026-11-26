# AGENTS.md — Nordlicht (DevForge DOOM)

Mandatory project rules, contract, and evaluation constraints.

## 1. Mission & Core Rules
- **Goal:** Train a neural `DoomAgent` in `submission/agent.py` that plays unseen DOOM levels **zero-shot** (combat, multi-room navigation, keycards, locked doors, exit).
- **Neural-only decisions:** All actions must come from trained neural parameters.
  - **Forbidden at inference:** Hardcoded if/else rule trees, `A*`/Dijkstra pathfinding, `atan2` aimbots, unmodified off-the-shelf Stable-Baselines3.
- **Engine isolation:** At evaluation, `DoomAgent` never accesses `vizdoom.DoomGame` or global coordinates. It receives only the ego-centric observation dictionary.
- **Privileged state:** `env.privileged_state()` (true position, kills, damage, sector heights, door locks) is for **training reward shaping only**. Never import or call it in `submission/`.
- **No train/serve skew:** Observation preprocessing must be identical between `training/` and `submission/agent.py` (shared in `submission/`).

## 2. Environment & Interface Contract (`1.0.0-draft`)
Source of truth: [docs/CONTRACT.md](docs/CONTRACT.md), [env/doom_env/spaces.py](env/doom_env/spaces.py), [env/doom_env/limits.py](env/doom_env/limits.py).

- **`DoomAgent` (`submission/agent.py`):**
  - `__init__(self, observation_space, action_space, device: str)` — called once; load weights relative to `Path(__file__).resolve().parent`.
  - `reset(self) -> None` — called before every map; clear all recurrent/spatial/episodic memory.
  - `act(self, observation: dict) -> np.ndarray` — called once per `FRAME_SKIP = 4` game ticks.
- **Action space:** `MultiBinary(9)` (`{0, 1}` ints, shape `(9,)`):
  - `0: MOVE_FORWARD`, `1: MOVE_BACKWARD`, `2: MOVE_LEFT`, `3: MOVE_RIGHT`, `4: TURN_LEFT`, `5: TURN_RIGHT`, `6: ATTACK`, `7: USE`, `8: SELECT_NEXT_WEAPON`.
  - `USE` and `SELECT_NEXT_WEAPON` are **edge-triggered** (`0 -> 1`). Holding `USE` does not re-trigger; pressing `USE` on a moving door reverses it.
  - Invalid shape or values terminate scoring for that map (player stands still).
- **Observation space (`Dict`):**
  - `rgb`: `(120, 160, 3) uint8` (HUD hidden).
  - `depth`: `(120, 160) uint8`.
  - `vitals`: `(17,) float32` (`health`, `armor`, `selected_weapon`, `selected_weapon_ammo`, `ammo_bullets`, `ammo_shells`, `ammo_rockets`, `ammo_cells`, 6 binary keycard/skull flags, `vel_forward`, `vel_right`, `vel_up`).
  - `rays`: `distance` `(64,) float32` in `[0, 1]` (`RAY_RANGE = 1024.0`, `0` = straight ahead, counter-clockwise); `hit_type` `MultiDiscrete([6]*64)` (`0: none, 1: wall, 2: door, 3: locked_door, 4: enemy, 5: item`).
  - `visible_entities`: up to `32` entities in `90°` FOV and LOS within `4096.0` units, sorted nearest-first; `features` `(32, 5) float32` (`[dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]`), `entity_type` `MultiDiscrete([12]*32)`, `mask` `MultiBinary(32)`.

## 3. Evaluation & Submission Constraints
- **Scoring:** Tier-weighted sum across unseen maps (Tiers 1–4: Combat, Multi-Room Exploration, Keycard Progression, Full Gauntlet). Partial credit is awarded for progression, exploration, and combat; exit gives the largest reward.
- **Game settings:** `freedoom2.wad`, Skill 3, `2100–4200` ticks per map, vanilla ViZDoom `1.3.3`.
- **Sandbox limits:** `2048 MB` VRAM, `8192 MB` RAM, `1` CPU thread, no network, repo mounted read-only (`/tmp` writable).
- **Timeouts:** `60s` init, `10s` reset, `10s` first `act()`, `5s` subsequent `act()`, `40 ms/step` average budget per map.
- **Repository:** `<= 1 GB` total including Git LFS (`submission/weights/`). No custom `Dockerfile`; optional `submission/requirements.txt` (max 50 lines).
