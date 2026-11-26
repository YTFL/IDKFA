"""Evaluation entry point. The scorer instantiates `DoomAgent` from this file.

Contract: docs/CONTRACT.md §6, env/doom_env/limits.py.
All decisions come strictly from trained EMLRPPolicy neural parameters.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_DIR = SUBMISSION_DIR.parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from submission.features import (  # noqa: E402
    ACTION_DIM,
    MAX_ENTITIES,
    NUM_ENTITY_RAW_FEATURES,
    NUM_RAYS,
    VITALS_DIM,
    ObservationPreprocessor,
    ProcessedObservation,
)
from submission.models import EMLRPConfig, EMLRPPolicy, EMLRPState, single_obs_to_tensors  # noqa: E402

_WEIGHT_CANDIDATES = ("agent.pt", "emlrp_best.pt", "emlrp_latest.pt")
_EDGE_BUTTON_INDICES = (7, 8)  # USE, SELECT_NEXT_WEAPON


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size < 1024:
            return path.read_bytes().startswith(b"version https://git-lfs")
    except OSError:
        return False
    return False


def _resolve_weights_path() -> Path:
    env_override = os.environ.get("NORDLICHT_WEIGHTS_PATH")
    if env_override:
        p = Path(env_override)
        if not p.is_absolute():
            p = REPO_DIR / p
        if p.is_file():
            if _is_lfs_pointer(p):
                raise RuntimeError(f"Weights file {p} is an unfetched Git LFS pointer; run `git lfs pull`.")
            return p
        raise FileNotFoundError(f"NORDLICHT_WEIGHTS_PATH not found: {p}")

    for name in _WEIGHT_CANDIDATES:
        candidate = WEIGHTS_DIR / name
        if candidate.is_file():
            if _is_lfs_pointer(candidate):
                raise RuntimeError(f"Weights file {candidate} is an unfetched Git LFS pointer; run `git lfs pull`.")
            return candidate
    raise FileNotFoundError(f"No model weights found in {WEIGHTS_DIR} (checked {_WEIGHT_CANDIDATES}).")


def _load_checkpoint_cpu(weights_path: Path) -> dict[str, Any]:
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint structure in {weights_path}")
    return ckpt


def _space_length(space: Any, default: int) -> int:
    shape = getattr(space, "shape", None)
    if shape and len(shape) >= 1 and int(shape[0]) > 0:
        return int(shape[0])
    return default


def _fit_1d(arr: Any, length: int, dtype: np.dtype, fill: float = 0.0) -> np.ndarray:
    out = np.full((length,), fill, dtype=dtype)
    if arr is None:
        return out
    src = np.asarray(arr)
    if src.ndim == 0:
        out.fill(src.item())
        return out
    flat = src.reshape(-1)
    n = min(length, flat.shape[0])
    if np.issubdtype(dtype, np.floating):
        vals = np.nan_to_num(flat[:n].astype(dtype, copy=False), nan=fill, posinf=fill, neginf=fill)
    else:
        vals = np.nan_to_num(flat[:n], nan=fill).astype(dtype, copy=False)
    out[:n] = vals
    return out


def _fit_2d(arr: Any, rows: int, cols: int, dtype: np.dtype, fill: float = 0.0) -> np.ndarray:
    out = np.full((rows, cols), fill, dtype=dtype)
    if arr is None:
        return out
    src = np.asarray(arr)
    if src.ndim != 2:
        return out
    r = min(rows, src.shape[0])
    c = min(cols, src.shape[1])
    if np.issubdtype(dtype, np.floating):
        vals = np.nan_to_num(src[:r, :c].astype(dtype, copy=False), nan=fill, posinf=fill, neginf=fill)
    else:
        vals = np.nan_to_num(src[:r, :c], nan=fill).astype(dtype, copy=False)
    out[:r, :c] = vals
    return out


class DoomAgent:
    """Zero-shot neural agent backed by EgoMap-Landmark Recurrent Policy (EMLRPPolicy)."""

    def __init__(self, observation_space: Any, action_space: Any, device: str) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.requested_device = str(device)

        torch.set_num_threads(1)

        self._ray_bins = NUM_RAYS
        self._vitals_dim = VITALS_DIM
        self._max_entities = MAX_ENTITIES
        self._entity_raw_dim = NUM_ENTITY_RAW_FEATURES
        self._action_dim = ACTION_DIM
        self._introspect_spaces(observation_space, action_space)

        self.weights_path = _resolve_weights_path()
        ckpt = _load_checkpoint_cpu(self.weights_path)

        model_cfg = EMLRPConfig(**ckpt.get("model_config", {}))
        self.preprocessor = ObservationPreprocessor.from_config(ckpt.get("preprocess_config", {}))
        self.policy = EMLRPPolicy(model_cfg).eval()
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.include_depth = model_cfg.depth_feat_dim > 0 and self.preprocessor.use_depth

        env_temp = os.environ.get("NORDLICHT_TEMPERATURE")
        self.temperature = float(env_temp) if env_temp is not None else float(ckpt.get("inference_temperature", 0.0))

        self.exec_device = self._select_fastest_device(self.requested_device)
        self.policy = self.policy.to(self.exec_device)

        self._prev_action = np.zeros(self._action_dim, dtype=np.int8)
        self._last_safe_action = np.zeros(self._action_dim, dtype=np.int8)
        self._last_safe_action[0] = 1  # MOVE_FORWARD fallback on transient error
        self._logged_errors: set[str] = set()
        self._rng: torch.Generator | None = None
        self.state: EMLRPState = self.policy.init_state(1, device=self.exec_device)

        self._warmup()
        self.reset()

    def _introspect_spaces(self, observation_space: Any, action_space: Any) -> None:
        self._action_dim = _space_length(action_space, ACTION_DIM)
        spaces_map = getattr(observation_space, "spaces", None)
        if not isinstance(spaces_map, Mapping):
            return
        self._vitals_dim = _space_length(spaces_map.get("vitals"), VITALS_DIM)
        rays_space = getattr(spaces_map.get("rays"), "spaces", None)
        if isinstance(rays_space, Mapping):
            self._ray_bins = _space_length(rays_space.get("distance"), NUM_RAYS)
        ent_space = getattr(spaces_map.get("visible_entities"), "spaces", None)
        if isinstance(ent_space, Mapping):
            feat_shape = getattr(ent_space.get("features"), "shape", None)
            if feat_shape and len(feat_shape) == 2:
                self._max_entities = int(feat_shape[0])
                self._entity_raw_dim = int(feat_shape[1])

    def _dummy_processed(self) -> ProcessedObservation:
        return ProcessedObservation(
            rays=np.zeros((self._ray_bins, 8), dtype=np.float32),
            depth=np.zeros((32,), dtype=np.float32),
            vitals=np.zeros((self._vitals_dim,), dtype=np.float32),
            entities=np.zeros((self._max_entities, 17), dtype=np.float32),
            entity_mask=np.zeros((self._max_entities,), dtype=np.float32),
            prev_action=np.zeros((self._action_dim,), dtype=np.float32),
        )

    def _bench_device(self, dev: torch.device, p_obs: ProcessedObservation, iters: int = 15) -> float:
        self.policy.to(dev)
        state = self.policy.init_state(1, device=dev)
        with torch.inference_mode():
            for _ in range(5):
                obs_t, prev_t = single_obs_to_tensors(p_obs, device=dev, include_depth=self.include_depth)
                out = self.policy.step(obs_t, state, prev_t, compute_aux=False)
                state = out.state
                _ = (out.action_logits[0] > 0.0).to(torch.int8).cpu().numpy()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                obs_t, prev_t = single_obs_to_tensors(p_obs, device=dev, include_depth=self.include_depth)
                out = self.policy.step(obs_t, state, prev_t, compute_aux=False)
                state = out.state
                _ = (out.action_logits[0] > 0.0).to(torch.int8).cpu().numpy()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1000.0 / max(1, iters)

    def _select_fastest_device(self, requested: str) -> torch.device:
        try:
            req = torch.device(requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:
            req = torch.device("cpu")

        if req.type != "cuda" or not torch.cuda.is_available():
            return torch.device("cpu")

        dummy = self._dummy_processed()
        cpu_dev = torch.device("cpu")
        cpu_ms = self._bench_device(cpu_dev, dummy)
        try:
            cuda_ms = self._bench_device(req, dummy)
        except Exception:
            self.policy.to(cpu_dev)
            return cpu_dev

        if cpu_ms <= cuda_ms:
            self.policy.to(cpu_dev)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return cpu_dev
        return req

    def _warmup(self) -> None:
        dummy = self._dummy_processed()
        state = self.policy.init_state(1, device=self.exec_device)
        with torch.inference_mode():
            for _ in range(3):
                obs_t, prev_t = single_obs_to_tensors(dummy, device=self.exec_device, include_depth=self.include_depth)
                out = self.policy.step(obs_t, state, prev_t, compute_aux=False)
                state = out.state
                _ = (out.action_logits[0] > 0.0).to(torch.int8).cpu().numpy()

    def _sanitize_observation(self, observation: Any) -> dict[str, Any]:
        obs = observation if isinstance(observation, Mapping) else {}
        rays_raw = obs.get("rays")
        rays = rays_raw if isinstance(rays_raw, Mapping) else {}
        ent_raw = obs.get("visible_entities")
        ent = ent_raw if isinstance(ent_raw, Mapping) else {}

        clean: dict[str, Any] = {
            "rays": {
                "distance": _fit_1d(rays.get("distance"), self._ray_bins, np.dtype(np.float32), fill=1.0),
                "hit_type": _fit_1d(rays.get("hit_type"), self._ray_bins, np.dtype(np.int64), fill=0.0),
            },
            "vitals": _fit_1d(obs.get("vitals"), self._vitals_dim, np.dtype(np.float32), fill=0.0),
            "visible_entities": {
                "features": _fit_2d(
                    ent.get("features"), self._max_entities, self._entity_raw_dim, np.dtype(np.float32), fill=0.0
                ),
                "entity_type": _fit_1d(ent.get("entity_type"), self._max_entities, np.dtype(np.int64), fill=0.0),
                "mask": _fit_1d(ent.get("mask"), self._max_entities, np.dtype(np.int8), fill=0.0),
            },
        }
        depth_raw = obs.get("depth")
        if self.include_depth and depth_raw is not None:
            clean["depth"] = np.nan_to_num(np.asarray(depth_raw), nan=0.0)
        else:
            clean["depth"] = None
        return clean

    def reset(self) -> None:
        """Reset all recurrent, spatial map, landmark slot, and action history state."""
        self.preprocessor.reset()
        self._prev_action.fill(0)
        self._last_safe_action.fill(0)
        self._last_safe_action[0] = 1
        self.state = self.policy.init_state(1, device=self.exec_device)
        if self.temperature > 0.0:
            self._rng = torch.Generator(device=self.exec_device)
            self._rng.manual_seed(0)

    def _decode_action(self, logits: torch.Tensor) -> np.ndarray:
        if self.temperature > 0.0:
            probs = torch.sigmoid(logits / self.temperature)
            act_t = torch.bernoulli(probs, generator=self._rng).to(torch.int8)
        else:
            act_t = (logits > 0.0).to(torch.int8)
        action = act_t.cpu().numpy()
        action = np.asarray(action != 0, dtype=np.int8).reshape(self._action_dim)
        return action

    def act(self, observation: dict) -> np.ndarray:
        """Run one neural forward step and return a valid MultiBinary(9) int8 array."""
        try:
            clean_obs = self._sanitize_observation(observation)
            p_obs = self.preprocessor.process(clean_obs, prev_action=self._prev_action)
            obs_t, prev_act_t = single_obs_to_tensors(
                p_obs, device=self.exec_device, include_depth=self.include_depth
            )
            with torch.inference_mode():
                out = self.policy.step(obs_t, self.state, prev_act_t, compute_aux=False)
                self.state = out.state
                action = self._decode_action(out.action_logits[0])
            self._prev_action[:] = action
            self._last_safe_action[:] = action
            for idx in _EDGE_BUTTON_INDICES:
                if idx < self._action_dim:
                    self._last_safe_action[idx] = 0
            return action
        except Exception as exc:
            err_key = f"{type(exc).__name__}:{exc}"
            if err_key not in self._logged_errors and len(self._logged_errors) < 8:
                self._logged_errors.add(err_key)
                print(f"[DoomAgent] Fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            fallback = self._last_safe_action.copy()
            self._prev_action[:] = fallback
            return fallback
