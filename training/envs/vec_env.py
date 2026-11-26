"""Vectorized DOOM training environments (DummyVecEnv and Windows-safe SubprocVecEnv).

Workers run DoomEnv on CPU, handle curriculum stage sampling, auto-resets,
ExtrinsicRewardShaper, EpisodicNoveltyBuffer, sticky actions, ObservationPreprocessor,
and training-only auxiliary targets (never passed to the policy).
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing as mp
import os
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from doom_env import make_env
from doom_env import spaces as S
from submission.features import ObservationPreprocessor, ProcessedObservation
from training.envs.curriculum import CurriculumConfig, CurriculumEnvWrapper, CurriculumManager, CurriculumStage
from training.rewards.curiosity import EpisodicNoveltyBuffer


@dataclass
class VecEnvStep:
    """Batched output from one vectorized step across `num_envs` workers."""

    obs: list[ProcessedObservation]
    spatial_vec: np.ndarray  # (B, 544) float32 post-step spatial features (before auto-reset)
    reward_ext: np.ndarray  # (B,) float32
    reward_episodic: np.ndarray  # (B,) float32
    done: np.ndarray  # (B,) bool (whether episode ended on THIS transition)
    aux_combat: np.ndarray  # (B, 2) float32 [damage_flag, kill_flag]
    aux_landmark_dir: np.ndarray  # (B, 2) float32 [fwd, left] unit vector
    aux_landmark_mask: np.ndarray  # (B,) float32 (1 if objective exists)
    episode_summaries: list[dict[str, Any]]


@dataclass
class WorkerConfig:
    initial_stage: int = 0
    curriculum_config_path: str | None = None
    preprocess_config: dict[str, Any] | None = None
    episode_ticks: int = 2100
    render_buffers: bool = True
    use_episodic_novelty: bool = True
    sticky_action_prob: float = 0.05
    potential_scale: float = 1.0
    dataset_train_maps: list[str] | None = None
    dataset_mix_prob: float = 0.25
    fixed_wad_path: str | None = None
    fixed_map_name: str = "MAP01"
    map_reuse_episodes: int = 3
    seed: int = 42
    cache_root: str | None = None


def _compute_objective_coords(
    meta: dict[str, Any], priv: dict[str, Any] | None
) -> tuple[float, float] | None:
    key_xy = meta.get("key_xy")
    exit_xy = meta.get("exit_xy")
    if key_xy is not None and priv is not None:
        keys = priv.get("keys")
        has_key = bool(np.sum(keys) > 0) if keys is not None else False
        if not has_key:
            return float(key_xy[0]), float(key_xy[1])
    if exit_xy is not None:
        return float(exit_xy[0]), float(exit_xy[1])
    return None


def _compute_aux_landmark_dir(
    priv: dict[str, Any] | None, target_xy: tuple[float, float] | None
) -> tuple[np.ndarray, float]:
    if priv is None or target_xy is None:
        return np.zeros(2, dtype=np.float32), 0.0
    dx = target_xy[0] - float(priv["x"])
    dy = target_xy[1] - float(priv["y"])
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return np.zeros(2, dtype=np.float32), 0.0
    heading = math.radians(float(priv["angle"]))
    c, s = math.cos(heading), math.sin(heading)
    fwd = (dx * c + dy * s) / dist
    left = (-dx * s + dy * c) / dist
    return np.array([fwd, left], dtype=np.float32), 1.0


class _SingleDoomWorker:
    """Single-environment worker state machine with auto-reset and preprocessing."""

    def __init__(self, worker_id: int, cfg: WorkerConfig) -> None:
        self.worker_id = worker_id
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + worker_id * 1009)

        cache_base = (
            Path(cfg.cache_root)
            if cfg.cache_root is not None
            else Path(tempfile.gettempdir()) / "nordlicht_vec_maps"
        )
        self.worker_cache = cache_base / f"worker_{worker_id}"
        self.worker_cache.mkdir(parents=True, exist_ok=True)

        self.curriculum = CurriculumManager(
            initial_stage=CurriculumStage(cfg.initial_stage),
            map_cache_dir=self.worker_cache,
            config_path=cfg.curriculum_config_path,
        )
        self.preprocessor = ObservationPreprocessor.from_config(cfg.preprocess_config or {})
        self.episodic_novelty = EpisodicNoveltyBuffer() if cfg.use_episodic_novelty else None

        self.env: CurriculumEnvWrapper | None = None
        self.ep_meta: dict[str, Any] = {}
        self.ep_stage: CurriculumStage = self.curriculum.current_stage
        self.ep_count: int = 0
        self.ep_steps: int = 0
        self.ep_ext_reward: float = 0.0
        self.ep_episodic_reward: float = 0.0
        self.last_action: np.ndarray | None = None
        self.prev_kills: int = 0
        self.prev_damage: float = 0.0

    def set_stage(self, stage: int) -> None:
        self.curriculum.current_stage = CurriculumStage(stage)

    def _close_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _open_next_episode(self) -> ProcessedObservation:
        self.ep_count += 1
        ep_seed = int(self.rng.integers(0, 2**30))

        need_new_env = (
            self.env is None
            or self.ep_count == 1
            or (self.cfg.fixed_wad_path is None and (self.ep_count - 1) % max(1, self.cfg.map_reuse_episodes) == 0)
        )
        if need_new_env:
            self._close_env()
            if self.cfg.fixed_wad_path is not None:
                wad_path = Path(self.cfg.fixed_wad_path)
                map_name = self.cfg.fixed_map_name
                st_cfg = self.curriculum.stage_configs[self.curriculum.current_stage]
                self.ep_stage = self.curriculum.current_stage
                self.ep_meta = {}
            else:
                self.ep_stage = self.curriculum.sample_stage_for_episode()
                use_dataset = (
                    bool(self.cfg.dataset_train_maps)
                    and self.ep_stage >= CurriculumStage.STAGE_3_NAV_DOORS_ENEMIES
                    and float(self.rng.random()) < self.cfg.dataset_mix_prob
                )
                if use_dataset and self.cfg.dataset_train_maps:
                    idx = int(self.rng.integers(0, len(self.cfg.dataset_train_maps)))
                    wad_path = Path(self.cfg.dataset_train_maps[idx])
                    map_name = "MAP01"
                    st_cfg = self.curriculum.stage_configs[self.ep_stage]
                    self.ep_meta = {}
                else:
                    slot = f"ep_slot_{self.ep_count % 2}.wad"
                    wad_path, st_cfg, meta = self.curriculum.sample_map_spec(
                        self.ep_stage, seed=ep_seed, filename=slot
                    )
                    map_name = "MAP01"
                    self.ep_meta = meta

            base_env = make_env(
                str(wad_path),
                map_name,
                episode_ticks=self.cfg.episode_ticks,
                render_buffers=self.cfg.render_buffers,
            )
            self.env = CurriculumEnvWrapper(
                base_env,
                stage=self.ep_stage,
                reward_weights=self.curriculum.reward_weights,
                config=st_cfg,
            )
        raw_obs, _ = self.env.reset(seed=ep_seed)

        self.preprocessor.reset()
        if self.episodic_novelty is not None:
            self.episodic_novelty.reset()
        self.last_action = None
        self.ep_steps = 0
        self.ep_ext_reward = 0.0
        self.ep_episodic_reward = 0.0

        priv = self.env.unwrapped.privileged_state()
        self.prev_kills = int(priv.get("kills", 0)) if priv else 0
        self.prev_damage = float(priv.get("damage_taken", 0.0)) if priv else 0.0

        zero_act = np.zeros(S.action_space().shape[0], dtype=np.float32)
        p_obs = self.preprocessor.process(raw_obs, prev_action=zero_act)
        if self.episodic_novelty is not None:
            self.episodic_novelty.compute_novelty_bonus(p_obs)
        return p_obs

    def reset(self) -> ProcessedObservation:
        return self._open_next_episode()

    def step(
        self, action: np.ndarray
    ) -> tuple[ProcessedObservation, float, float, bool, np.ndarray, np.ndarray, float, dict[str, Any] | None]:
        assert self.env is not None
        act = np.asarray(action, dtype=np.int8).copy()
        if (
            self.cfg.sticky_action_prob > 0.0
            and self.last_action is not None
            and float(self.rng.random()) < self.cfg.sticky_action_prob
        ):
            act = self.last_action.copy()
        self.last_action = act.copy()

        # Pass current objective target into ExtrinsicRewardShaper via potential scale
        priv_before = self.env.unwrapped.privileged_state()
        target_xy = _compute_objective_coords(self.ep_meta, priv_before)
        if self.cfg.potential_scale <= 0.0:
            target_xy = None

        raw_obs, r_ext, terminated, truncated, info = self.env.step(act)
        self.ep_steps += 1
        priv_after = self.env.unwrapped.privileged_state()

        # Potential-based shaping bonus if target_xy exists
        if target_xy is not None and priv_after is not None:
            shaper = self.env.reward_shaper
            dist = math.hypot(float(priv_after["x"]) - target_xy[0], float(priv_after["y"]) - target_xy[1])
            pot = -dist / 1000.0
            if shaper.last_potential is not None and shaper.last_target_coords == target_xy:
                pot_delta = (shaper.config.gamma * pot - shaper.last_potential) * self.cfg.potential_scale
                r_ext += pot_delta
            shaper.last_potential = pot
            shaper.last_target_coords = target_xy

        # Stage 0/1 arena clear check (Family A has no exit linedef)
        arena_cleared = False
        num_enemies = int(self.ep_meta.get("num_enemies", 0))
        if (
            not terminated
            and self.ep_meta.get("exit_xy") is None
            and num_enemies > 0
            and priv_after is not None
            and int(priv_after.get("kills", 0)) >= num_enemies
            and float(priv_after.get("health", 0.0)) > 0.0
        ):
            arena_cleared = True
            terminated = True
            clear_bonus = 5.0 + (float(priv_after["health"]) / 100.0)
            r_ext += clear_bonus

        curr_kills = int(priv_after.get("kills", 0)) if priv_after else self.prev_kills
        curr_damage = float(priv_after.get("damage_taken", 0.0)) if priv_after else self.prev_damage
        got_kill = 1.0 if curr_kills > self.prev_kills else 0.0
        took_dmg = 1.0 if curr_damage > self.prev_damage else 0.0
        self.prev_kills = curr_kills
        self.prev_damage = curr_damage
        aux_combat = np.array([took_dmg, got_kill], dtype=np.float32)

        post_target = _compute_objective_coords(self.ep_meta, priv_after)
        aux_dir, aux_mask = _compute_aux_landmark_dir(priv_after, post_target)

        p_obs = self.preprocessor.process(raw_obs, prev_action=act)
        spatial_vec = np.concatenate([p_obs.rays.reshape(-1), p_obs.depth], axis=0).astype(np.float32)
        r_ep_nov = (
            self.episodic_novelty.compute_novelty_bonus(p_obs)
            if self.episodic_novelty is not None
            else 0.0
        )

        self.ep_ext_reward += float(r_ext)
        self.ep_episodic_reward += float(r_ep_nov)

        done = bool(terminated or truncated)
        summary: dict[str, Any] | None = None
        if done:
            exit_reached = bool(info.get("exit_reached", False))
            success = bool(exit_reached or arena_cleared)
            summary = {
                "stage": int(self.ep_stage),
                "success": success,
                "exit_reached": exit_reached,
                "arena_cleared": arena_cleared,
                "steps": self.ep_steps,
                "total_ext_reward": self.ep_ext_reward,
                "total_episodic_reward": self.ep_episodic_reward,
                "unique_cells": int(info.get("unique_cells_visited", 0)),
                "kills": curr_kills,
                "damage_taken": curr_damage,
                "reward_breakdown": dict(info.get("reward_breakdown", {})),
            }
            p_obs = self._open_next_episode()

        return p_obs, spatial_vec, float(r_ext), float(r_ep_nov), done, aux_combat, aux_dir, float(aux_mask), summary

    def close(self) -> None:
        self._close_env()


def _worker_loop(conn: Any, worker_id: int, cfg: WorkerConfig) -> None:
    # Let the parent coordinator handle Ctrl+C / SIGINT and shut workers down cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Silence ViZDoom C++ startup logs at OS fd 1 inside background workers
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    worker = _SingleDoomWorker(worker_id, cfg)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(worker.reset())
            elif cmd == "step":
                conn.send(worker.step(payload))
            elif cmd == "set_stage":
                worker.set_stage(int(payload))
                conn.send(True)
            elif cmd == "close":
                worker.close()
                conn.send(True)
                break
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        worker.close()
    finally:
        worker.close()


class DummyVecEnv:
    """In-process vectorized wrapper (ideal for unit tests and single-env debugging)."""

    def __init__(self, num_envs: int, cfg: WorkerConfig) -> None:
        self.num_envs = num_envs
        self._owned_cache_root: str | None = None
        if cfg.cache_root is None:
            self._owned_cache_root = tempfile.mkdtemp(prefix="nordlicht_vec_")
            cfg = dataclasses.replace(cfg, cache_root=self._owned_cache_root)
        self.workers = [_SingleDoomWorker(i, cfg) for i in range(num_envs)]
        self._pending_slices: dict[tuple[int, int], list[Any]] = {}

    def reset(self) -> list[ProcessedObservation]:
        return [w.reset() for w in self.workers]

    def set_stage(self, stage: int) -> None:
        for w in self.workers:
            w.set_stage(stage)

    def step_async_slice(self, start: int, end: int, actions: np.ndarray) -> None:
        self._pending_slices[(start, end)] = [self.workers[i].step(actions[i - start]) for i in range(start, end)]

    def step_wait_slice(self, start: int, end: int) -> VecEnvStep:
        results = self._pending_slices.pop((start, end))
        return _collate_worker_results(results)

    def step(self, actions: np.ndarray) -> VecEnvStep:
        results = [w.step(actions[i]) for i, w in enumerate(self.workers)]
        return _collate_worker_results(results)

    def close(self) -> None:
        for w in self.workers:
            w.close()
        if self._owned_cache_root is not None:
            shutil.rmtree(self._owned_cache_root, ignore_errors=True)
            self._owned_cache_root = None


class SubprocVecEnv:
    """Multi-process vectorized environment using `spawn` (Windows & Linux safe)."""

    def __init__(self, num_envs: int, cfg: WorkerConfig) -> None:
        self.num_envs = num_envs
        self._owned_cache_root: str | None = None
        if cfg.cache_root is None:
            self._owned_cache_root = tempfile.mkdtemp(prefix="nordlicht_vec_")
            cfg = dataclasses.replace(cfg, cache_root=self._owned_cache_root)
        ctx = mp.get_context("spawn")
        self._remotes: list[Any] = []
        self._work_remotes: list[Any] = []
        self._procs: list[Any] = []
        self._closed = False

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_worker_loop, args=(child_conn, i, cfg), daemon=True)
            proc.start()
            child_conn.close()
            self._remotes.append(parent_conn)
            self._work_remotes.append(child_conn)
            self._procs.append(proc)

    def reset(self) -> list[ProcessedObservation]:
        for remote in self._remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self._remotes]

    def set_stage(self, stage: int) -> None:
        for remote in self._remotes:
            remote.send(("set_stage", int(stage)))
        for remote in self._remotes:
            remote.recv()

    def step_async_slice(self, start: int, end: int, actions: np.ndarray) -> None:
        for i in range(start, end):
            self._remotes[i].send(("step", actions[i - start]))

    def step_wait_slice(self, start: int, end: int) -> VecEnvStep:
        results = [self._remotes[i].recv() for i in range(start, end)]
        return _collate_worker_results(results)

    def step(self, actions: np.ndarray) -> VecEnvStep:
        self.step_async_slice(0, self.num_envs, actions)
        return self.step_wait_slice(0, self.num_envs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for remote in self._remotes:
            try:
                if remote.poll(0.5):
                    remote.recv()
            except Exception:
                pass
            try:
                remote.close()
            except Exception:
                pass
        deadline = time.monotonic() + 2.0
        for proc in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.5)
        if self._owned_cache_root is not None:
            shutil.rmtree(self._owned_cache_root, ignore_errors=True)
            self._owned_cache_root = None


def _collate_worker_results(results: list[tuple[Any, ...]]) -> VecEnvStep:
    obs_list: list[ProcessedObservation] = []
    n = len(results)
    spatial_vecs: list[np.ndarray] = []
    r_ext = np.zeros(n, dtype=np.float32)
    r_ep = np.zeros(n, dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    aux_combat = np.zeros((n, 2), dtype=np.float32)
    aux_dir = np.zeros((n, 2), dtype=np.float32)
    aux_mask = np.zeros(n, dtype=np.float32)
    summaries: list[dict[str, Any]] = []

    for i, (p_obs, svec, re, rnov, d, ac, ad, am, summ) in enumerate(results):
        obs_list.append(p_obs)
        spatial_vecs.append(svec)
        r_ext[i] = re
        r_ep[i] = rnov
        done[i] = d
        aux_combat[i] = ac
        aux_dir[i] = ad
        aux_mask[i] = am
        if summ is not None:
            summaries.append(summ)

    return VecEnvStep(
        obs=obs_list,
        spatial_vec=np.stack(spatial_vecs, axis=0),
        reward_ext=r_ext,
        reward_episodic=r_ep,
        done=done,
        aux_combat=aux_combat,
        aux_landmark_dir=aux_dir,
        aux_landmark_mask=aux_mask,
        episode_summaries=summaries,
    )
