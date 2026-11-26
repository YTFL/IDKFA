"""Curriculum environment wrappers and progression manager.

Implements the 6 curriculum stages from PLAN.md Phase 1:
  Stage 0: Single room, one enemy (learn to aim, fire, turn).
  Stage 1: Combat arenas (multiple enemies, health/ammo pickups).
  Stage 2: Multi-room navigation to exit (doors auto-open, no enemies).
  Stage 3: Multi-room with ordinary doors and enemies.
  Stage 4: Keycard puzzle maps with vanilla rules.
  Stage 5: Full gauntlet mix (with 10-20% rehearsal of earlier stages).

Reads stage parameters from training/configs/curriculum.yaml, uses edge-triggered
auto_open_doors and real ViZDoom infinite_ammo/god_mode mods, and delegates dense
reward shaping to ExtrinsicRewardShaper.
"""

from __future__ import annotations

import collections
import enum
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from doom_env import make_env
from doom_env import spaces as S
from doom_env.sensors import RAY_DOOR
from training.mapgen.generator import MapFamily, ProceduralMapGenerator
from training.rewards.shaping import ExtrinsicRewardShaper, RewardConfig

DEFAULT_CURRICULUM_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "curriculum.yaml"
)


class CurriculumStage(int, enum.Enum):
    STAGE_0_AIM_SHOOT = 0
    STAGE_1_COMBAT_ARENA = 1
    STAGE_2_NAV_AUTO_DOORS = 2
    STAGE_3_NAV_DOORS_ENEMIES = 3
    STAGE_4_KEYCARD_PUZZLE = 4
    STAGE_5_FULL_GAUNTLET = 5


@dataclass
class RewardWeights:
    exit_reached: float = 10.0
    key_pickup: float = 2.0
    door_opened: float = 0.5
    locked_door_unlocked: float = 2.0
    enemy_killed: float = 0.5
    damage_taken_per_hp: float = -0.005
    death: float = -3.0
    time_step: float = -0.001
    new_cell_explored: float = 0.02

    def to_reward_config(self) -> RewardConfig:
        return RewardConfig(
            exit_reward=self.exit_reached,
            key_reward=self.key_pickup,
            door_reward=self.door_opened,
            locked_door_reward=self.locked_door_unlocked,
            cell_reward=self.new_cell_explored,
            kill_reward=self.enemy_killed,
            damage_penalty_per_hp=self.damage_taken_per_hp,
            death_penalty=self.death,
            step_penalty=self.time_step,
        )


@dataclass
class CurriculumConfig:
    stage: CurriculumStage = CurriculumStage.STAGE_5_FULL_GAUNTLET
    promotion_threshold: float = 0.70
    window_size: int = 100
    rehearsal_prob: float = 0.15
    god_mode: bool = False
    infinite_ammo: bool = False
    auto_open_doors: bool = False
    render_buffers: bool = True
    cell_size: float = 64.0
    num_rooms: int = 5
    difficulty: float = 0.5


def _parse_yaml_scalar(raw: str) -> Any:
    val = raw.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def load_curriculum_yaml(path: str | Path | None = None) -> dict[str, Any]:
    """Load curriculum.yaml using PyYAML if installed, else a minimal stdlib YAML parser."""
    cfg_path = Path(path) if path is not None else DEFAULT_CURRICULUM_CONFIG_PATH
    if not cfg_path.is_file():
        return {}
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[Any, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key_raw, val_raw = stripped.split(":", 1)
        key_str = key_raw.strip()
        key: Any = int(key_str) if key_str.isdigit() else key_str

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if not val_raw.strip():
            child: dict[Any, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(val_raw)
    return root


class CurriculumEnvWrapper(gym.Wrapper):
    """Wraps DoomEnv with curriculum training mods and dense ExtrinsicRewardShaper."""

    _USE_IDX = S.BUTTONS.index("USE")
    _FRONT_RAY_INDICES = (61, 62, 63, 0, 1, 2, 3)

    def __init__(
        self,
        env: gym.Env,
        stage: CurriculumStage = CurriculumStage.STAGE_5_FULL_GAUNTLET,
        reward_weights: RewardWeights | None = None,
        config: CurriculumConfig | None = None,
    ):
        super().__init__(env)
        self.stage = stage
        self.weights = reward_weights or RewardWeights()
        self.config = config or CurriculumConfig(stage=stage)

        reward_cfg = self.weights.to_reward_config()
        reward_cfg.cell_size = self.config.cell_size
        self.reward_shaper = ExtrinsicRewardShaper(reward_cfg)

        self.step_count: int = 0
        self.total_shaped_reward: float = 0.0
        self._last_obs: dict[str, Any] | None = None
        self._last_use_pressed: bool = False
        self._door_cooldown: int = 0

    def _send_game_cmd(self, cmd: str) -> None:
        game = getattr(self.env.unwrapped, "_game", None)
        if game is not None:
            game.send_game_command(cmd)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_shaper.reset()
        self.step_count = 0
        self.total_shaped_reward = 0.0
        self._last_obs = obs
        self._last_use_pressed = False
        self._door_cooldown = 0

        if self.config.god_mode:
            self._send_game_cmd("god")
        if self.config.infinite_ammo:
            self._send_game_cmd("give Clip")
            self._send_game_cmd("give Shell")

        priv = self.env.unwrapped.privileged_state()
        if priv is not None:
            cx = int(priv["x"] // self.config.cell_size)
            cy = int(priv["y"] // self.config.cell_size)
            self.reward_shaper.visited_cells.add((cx, cy))
            self.reward_shaper.last_health = float(priv.get("health", 100.0))

        info["unique_cells_visited"] = len(self.reward_shaper.visited_cells)
        return obs, info

    def _should_auto_open_door(self) -> bool:
        if self._last_obs is None or self._last_use_pressed or self._door_cooldown > 0:
            return False
        rays = self._last_obs.get("rays")
        if rays is None:
            return False
        dists = rays["distance"]
        hits = rays["hit_type"]
        for idx in self._FRONT_RAY_INDICES:
            if int(hits[idx]) == RAY_DOOR and float(dists[idx]) * S.RAY_RANGE <= 96.0:
                return True
        return False

    def step(self, action: np.ndarray):
        act = np.asarray(action).copy()

        # Edge-triggered auto_open_doors mod for navigation pretraining
        if self.config.auto_open_doors or self.stage == CurriculumStage.STAGE_2_NAV_AUTO_DOORS:
            if self._should_auto_open_door():
                act[self._USE_IDX] = 1
                self._door_cooldown = 10

        self._last_use_pressed = bool(act[self._USE_IDX])
        if self._door_cooldown > 0:
            self._door_cooldown -= 1

        # Replenish engine ammo before it runs dry when infinite_ammo is enabled
        if self.config.infinite_ammo and self._last_obs is not None:
            bullets = float(self._last_obs["vitals"][S.VITALS_INDEX["ammo_bullets"]])
            shells = float(self._last_obs["vitals"][S.VITALS_INDEX["ammo_shells"]])
            if bullets < 20.0:
                self._send_game_cmd("give Clip")
            if shells < 8.0:
                self._send_game_cmd("give Shell")

        obs, base_reward, terminated, truncated, info = self.env.step(act)
        self.step_count += 1
        self._last_obs = obs

        priv = self.env.unwrapped.privileged_state()
        shaped_reward, breakdown = self.reward_shaper.compute(
            obs=obs,
            privileged_state=priv,
            base_reward=base_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        self.total_shaped_reward += shaped_reward
        info["shaped_reward"] = shaped_reward
        info["total_shaped_reward"] = self.total_shaped_reward
        info["reward_breakdown"] = breakdown.to_dict()
        info["unique_cells_visited"] = len(self.reward_shaper.visited_cells)
        return obs, shaped_reward, terminated, truncated, info


class CurriculumManager:
    """Manages progression across stages based on rolling success rates."""

    def __init__(
        self,
        initial_stage: CurriculumStage | None = None,
        promotion_threshold: float | None = None,
        window_size: int | None = None,
        rehearsal_prob: float | None = None,
        map_cache_dir: Path | str | None = None,
        config_path: Path | str | None = None,
    ):
        raw_cfg = load_curriculum_yaml(config_path)
        cur_sec = raw_cfg.get("curriculum", {})
        stages_sec = raw_cfg.get("stages", {})
        rewards_sec = raw_cfg.get("rewards", {})

        self.current_stage = (
            initial_stage
            if initial_stage is not None
            else CurriculumStage(int(cur_sec.get("initial_stage", 0)))
        )
        self.promotion_threshold = (
            promotion_threshold
            if promotion_threshold is not None
            else float(cur_sec.get("promotion_threshold", 0.70))
        )
        self.window_size = (
            window_size if window_size is not None else int(cur_sec.get("window_size", 100))
        )
        self.rehearsal_prob = (
            rehearsal_prob
            if rehearsal_prob is not None
            else float(cur_sec.get("rehearsal_prob", 0.15))
        )

        self.reward_weights = RewardWeights(
            exit_reached=float(rewards_sec.get("exit_reached", 10.0)),
            key_pickup=float(rewards_sec.get("key_pickup", 2.0)),
            door_opened=float(rewards_sec.get("door_opened", 0.5)),
            locked_door_unlocked=float(rewards_sec.get("locked_door_unlocked", 2.0)),
            enemy_killed=float(rewards_sec.get("enemy_killed", 0.5)),
            damage_taken_per_hp=float(rewards_sec.get("damage_taken_per_hp", -0.005)),
            death=float(rewards_sec.get("death", -3.0)),
            time_step=float(rewards_sec.get("time_step", -0.001)),
            new_cell_explored=float(rewards_sec.get("new_cell_explored", 0.02)),
        )

        self.stage_configs: dict[CurriculumStage, CurriculumConfig] = {}
        for st in CurriculumStage:
            st_raw = stages_sec.get(st.value, stages_sec.get(str(st.value), {})) or {}
            self.stage_configs[st] = CurriculumConfig(
                stage=st,
                promotion_threshold=self.promotion_threshold,
                window_size=self.window_size,
                rehearsal_prob=self.rehearsal_prob,
                god_mode=bool(st_raw.get("god_mode", st == CurriculumStage.STAGE_2_NAV_AUTO_DOORS)),
                infinite_ammo=bool(
                    st_raw.get("infinite_ammo", st == CurriculumStage.STAGE_0_AIM_SHOOT)
                ),
                auto_open_doors=bool(
                    st_raw.get("auto_open_doors", st == CurriculumStage.STAGE_2_NAV_AUTO_DOORS)
                ),
                num_rooms=int(st_raw.get("num_rooms", 5)),
                difficulty=float(st_raw.get("difficulty", min(1.0, 0.2 * int(st)))),
            )

        if map_cache_dir is None:
            self.map_cache_dir = Path(tempfile.gettempdir()) / "doom_curriculum_maps"
        else:
            self.map_cache_dir = Path(map_cache_dir)
        self.map_cache_dir.mkdir(parents=True, exist_ok=True)

        self.generator = ProceduralMapGenerator()
        self.history: collections.deque[bool] = collections.deque(maxlen=self.window_size)
        self.total_episodes: int = 0

    def record_episode(self, success: bool) -> None:
        self.history.append(success)
        self.total_episodes += 1

        if len(self.history) >= self.window_size:
            rate = sum(self.history) / len(self.history)
            if (
                rate >= self.promotion_threshold
                and self.current_stage < CurriculumStage.STAGE_5_FULL_GAUNTLET
            ):
                promoted_to = CurriculumStage(self.current_stage + 1)
                print(
                    f"\n[CurriculumManager] Success rate {rate:.1%} exceeds {self.promotion_threshold:.1%}. "
                    f"PROMOTING from {self.current_stage.name} -> {promoted_to.name}!"
                )
                self.current_stage = promoted_to
                self.history.clear()

    @property
    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def sample_stage_for_episode(self) -> CurriculumStage:
        """Samples the active stage, mixing in rehearsals of earlier stages."""
        if self.current_stage == CurriculumStage.STAGE_0_AIM_SHOOT:
            return self.current_stage

        if np.random.rand() < self.rehearsal_prob:
            earlier_stages = list(range(self.current_stage))
            return CurriculumStage(int(np.random.choice(earlier_stages)))

        return self.current_stage

    def get_or_generate_map(self, stage: CurriculumStage, seed: int | None = None) -> Path:
        """Returns a path to a verified-solvable map suited for the given stage."""
        cur_seed = seed if seed is not None else int(np.random.randint(1_000_000))
        st_cfg = self.stage_configs[stage]
        map_path = self.map_cache_dir / f"stage_{stage.value}_{cur_seed}.wad"

        family_by_stage = {
            CurriculumStage.STAGE_0_AIM_SHOOT: MapFamily.FAMILY_A,
            CurriculumStage.STAGE_1_COMBAT_ARENA: MapFamily.FAMILY_A,
            CurriculumStage.STAGE_2_NAV_AUTO_DOORS: MapFamily.FAMILY_B,
            CurriculumStage.STAGE_3_NAV_DOORS_ENEMIES: MapFamily.FAMILY_B,
            CurriculumStage.STAGE_4_KEYCARD_PUZZLE: MapFamily.FAMILY_C,
            CurriculumStage.STAGE_5_FULL_GAUNTLET: MapFamily.FAMILY_D,
        }
        family = family_by_stage.get(stage, MapFamily.FAMILY_D)

        solvable = False
        m = None
        for attempt in range(20):
            gen = ProceduralMapGenerator(cur_seed + attempt)
            m, solvable = gen.generate(
                family,
                num_rooms=st_cfg.num_rooms,
                difficulty=st_cfg.difficulty,
            )
            if solvable:
                break
        if m is None or not solvable:
            raise RuntimeError(f"Could not generate solvable map for {stage.name}")

        m.write_wad(map_path)
        return map_path

    def make_curriculum_env(
        self,
        stage: CurriculumStage | None = None,
        seed: int = 0,
        render_buffers: bool = True,
    ) -> gym.Env:
        """Creates a wrapped curriculum environment configured from curriculum.yaml."""
        active_stage = stage if stage is not None else self.sample_stage_for_episode()
        wad_path = self.get_or_generate_map(active_stage, seed=seed)
        base_env = make_env(str(wad_path), episode_ticks=2100, render_buffers=render_buffers)
        st_cfg = self.stage_configs[active_stage]
        return CurriculumEnvWrapper(
            base_env,
            stage=active_stage,
            reward_weights=self.reward_weights,
            config=st_cfg,
        )
