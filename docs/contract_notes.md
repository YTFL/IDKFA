# Contract & Environment Notes (Local Reference)

> Source of truth: `docs/CONTRACT.md`, `docs/EVALUATION.md`, `env/doom_env/spaces.py`, `env/doom_env/limits.py`, `env/doom_env/wrapper.py`, `env/doom_env/sensors.py`.
> Contract Version: `1.0.0-draft` (`doom_env.CONTRACT_VERSION`)

## 1. Action Space
- Type: `gymnasium.spaces.MultiBinary(9)`
- Expected return from `DoomAgent.act(obs)`: `np.ndarray` of shape `(9,)` with integer/binary values in `{0, 1}`. Invalid shapes or values immediately terminate scoring for the current map (player stands still).
- Button layout (`env/doom_env/spaces.py:BUTTONS`):
  0. `MOVE_FORWARD`
  1. `MOVE_BACKWARD`
  2. `MOVE_LEFT` (strafe left)
  3. `MOVE_RIGHT` (strafe right)
  4. `TURN_LEFT`
  5. `TURN_RIGHT`
  6. `ATTACK`
  7. `USE` (doors, switches) — **Edge-triggered** (fires on 0 -> 1 transition). Holding across steps only triggers once; pressing again on a moving door reverses it (wait 1–2s after pressing).
  8. `SELECT_NEXT_WEAPON` — **Edge-triggered**.
- Frame skip: `FRAME_SKIP = 4` game ticks per `act()` call (buttons held for all 4 tics).

## 2. Observation Space (`gymnasium.spaces.Dict`)

### `rgb` & `depth`
- `rgb`: `Box(0, 255, (120, 160, 3), np.uint8)` (HUD/crosshair/decals/particles hidden; weapon & corpses rendered).
- `depth`: `Box(0, 255, (120, 160), np.uint8)` (aligned with `rgb`).
- *Training tip:* `make_env(..., render_buffers=False)` fills `rgb` and `depth` with zeros for faster training when pixels aren't needed.

### `vitals`: `Box(-inf, inf, (17,), np.float32)`
Ordered per `VITALS_FIELDS` / `VITALS_INDEX`:
- `0`: `health` (0–200)
- `1`: `armor` (0–200)
- `2`: `selected_weapon` (slot 0–7)
- `3`: `selected_weapon_ammo`
- `4–7`: `ammo_bullets`, `ammo_shells`, `ammo_rockets`, `ammo_cells`
- `8–13`: `key_blue_card`, `key_yellow_card`, `key_red_card`, `key_blue_skull`, `key_yellow_skull`, `key_red_skull` (binary `0/1`; stay `1` once picked up)
- `14–16`: `vel_forward`, `vel_right`, `vel_up` (player-frame velocity in map units/tic)

### `rays`: `Dict`
- `NUM_RAYS = 64`, `RAY_RANGE = 1024.0` map units.
- Layout: Spread evenly over 360° counter-clockwise starting straight ahead:
  - Index `0` = straight ahead (`0°`)
  - Index `16` = `90°` left
  - Index `32` = `180°` behind
  - Index `48` = `270°` (`90°` right)
- `distance`: `Box(0.0, 1.0, (64,), np.float32)` — hit distance / `1024.0` (`1.0` = no hit within range).
- `hit_type`: `MultiDiscrete([6] * 64)` — `0`: `none`, `1`: `wall`, `2`: `door`, `3`: `locked_door`, `4`: `enemy`, `5`: `item`.
  - Stops at movement blockers (solid wall, impassable line, step > 24 units, opening < 56 units, alive monster, pickup, closed door). Passes through open doors, corpses, decorations, projectiles.

### `visible_entities`: `Dict`
- Up to `MAX_ENTITIES = 32` entities within 90° horizontal FOV (`HALF_FOV = 45°`) and line-of-sight, up to `4096.0` units away, sorted nearest-first and zero-padded.
- `features`: `Box(-inf, inf, (32, 5), np.float32)`:
  `[distance / 1024.0, sin(rel_angle), cos(rel_angle), rel_vel_forward, rel_vel_right]`
  (`rel_angle = 0` straight ahead, positive to the left; relative velocity is entity vel minus player vel in player's frame, map units/tic).
- `entity_type`: `MultiDiscrete([12] * 32)`:
  `0`: `pad`, `1`: `enemy_melee`, `2`: `enemy_hitscan`, `3`: `enemy_projectile`, `4`: `projectile`, `5`: `health`, `6`: `armor`, `7`: `ammo`, `8`: `weapon`, `9`: `key`, `10`: `barrel`, `11`: `other`.
- `mask`: `MultiBinary(32)` — `1` if real entity, `0` if padding.

## 3. Episode Lifecycle, Maps & Limits
- IWAD: `freedoom2.wad` (bundled with `vizdoom==1.3.3`), Skill 3 ("Hurt me plenty").
- Episode cap: `2100–4200` ticks (`525–1050` decisions at `FRAME_SKIP=4`; 35 ticks = 1s game time).
- Termination vs Truncation:
  - `terminated = True`: player reached exit (`_exit_reached=True`) or died (`is_player_dead()`).
  - `truncated = True`: episode tick cap reached (`is_episode_timeout_reached()`).
- Wrapper default reward (not used by scorer; replace for training): `+1.0` exit, `-1.0` death, `+0.1` per kill.
- Privileged training state: `env.privileged_state()` provides `x, y, z, angle, health, kills, items, damage_taken, keys, dead, timeout, exit_reached, sector_floor, sector_ceiling, door_lock` for reward shaping ONLY (never available at eval).
- Sector limit: ViZDoom reports at most `2048` sectors per map. Keep training maps under 2048 sectors.

## 4. Evaluation Constraints (`env/doom_env/limits.py` & `docs/EVALUATION.md`)
- `DoomAgent` in `submission/agent.py`:
  - `__init__(observation_space, action_space, device)`: called once; `60.0 s` timeout. Load weights relative to `Path(__file__).resolve().parent`.
  - `reset()`: called before every map; `10.0 s` timeout.
  - `act(observation)`: `10.0 s` first call per map (JIT/CUDA warm-up), `5.0 s` subsequent calls, `40.0 ms/step` average budget over the map's tick cap.
- Hardware sandbox: RTX 3090, `2048 MB` VRAM (`MAX_VRAM_MB`), `8192 MB` RAM (`MAX_RAM_MB`, includes `/tmp` and `/dev/shm`), `1` CPU thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), no network, repo mounted read-only, `/tmp` writable.
- Repo size limit: `1 GB` total (`MAX_REPO_BYTES`), weights via Git LFS.
- Note on Windows validation: `validate.py` uses `socket.AF_UNIX` in `env/doom_env/remote.py`, so full validator dry-runs require Docker (`validate.py --docker`) or Linux/WSL.
