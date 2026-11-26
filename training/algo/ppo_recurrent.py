"""Custom Recurrent PPO with dual GAE streams, truncated BPTT, and auxiliary tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli

from submission.features import ACTION_DIM, DEPTH_DOWN_WIDTH, MAX_ENTITIES, NUM_RAYS, VITALS_DIM, ProcessedObservation
from submission.models import EMLRPConfig, EMLRPPolicy, EMLRPState, processed_to_tensors


@dataclass
class RecurrentPPOConfig:
    """Hyperparameters for recurrent PPO with extrinsic + intrinsic GAE and aux heads."""

    rollout_len: int = 128
    seq_len: int = 64
    minibatch_seqs: int = 12
    ppo_epochs: int = 4

    gamma: float = 0.995
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    vf_ext_coef: float = 0.5
    vf_int_coef: float = 0.25

    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.002
    intrinsic_coef_start: float = 0.2
    intrinsic_coef_end: float = 0.05

    aux_coef: float = 0.1
    aux_next_ray_weight: float = 1.0
    aux_inverse_dyn_weight: float = 0.5
    aux_combat_weight: float = 0.5
    aux_landmark_dir_weight: float = 0.5
    aux_entity_imp_weight: float = 0.5

    lr: float = 3e-4
    lr_end: float = 3e-5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    entity_dropout_prob: float = 0.05
    obs_noise_std: float = 0.01
    use_bf16: bool = True


def _state_to_cpu_detached(state: EMLRPState) -> EMLRPState:
    return EMLRPState(
        gru_h=state.gru_h.detach().cpu(),
        spatial_map=state.spatial_map.detach().cpu(),
        landmark_keys=state.landmark_keys.detach().cpu(),
        landmark_pos=state.landmark_pos.detach().cpu(),
        landmark_valid=state.landmark_valid.detach().cpu(),
    )


def _slice_and_move_state(
    states: list[EMLRPState],
    chunk_indices: np.ndarray,
    env_indices: np.ndarray,
    device: torch.device,
) -> EMLRPState:
    return EMLRPState(
        gru_h=torch.stack([states[c].gru_h[e] for c, e in zip(chunk_indices, env_indices)]).to(device),
        spatial_map=torch.stack([states[c].spatial_map[e] for c, e in zip(chunk_indices, env_indices)]).to(device),
        landmark_keys=torch.stack([states[c].landmark_keys[e] for c, e in zip(chunk_indices, env_indices)]).to(device),
        landmark_pos=torch.stack([states[c].landmark_pos[e] for c, e in zip(chunk_indices, env_indices)]).to(device),
        landmark_valid=torch.stack([states[c].landmark_valid[e] for c, e in zip(chunk_indices, env_indices)]).to(device),
    )


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    dones: np.ndarray | None,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation over `(T, B)`.

    If `dones` is `None`, computes non-episodic returns (used for intrinsic curiosity).
    """
    T, B = rewards.shape
    advantages = np.zeros((T, B), dtype=np.float32)
    last_gae = np.zeros(B, dtype=np.float32)

    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 if dones is None else (1.0 - dones[t].astype(np.float32))
        delta = rewards[t] + gamma * next_val * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class RecurrentRolloutBuffer:
    """Host-RAM rollout storage supporting chunked truncated BPTT and dual GAE."""

    def __init__(self, rollout_len: int, num_envs: int, seq_len: int, include_depth: bool = True) -> None:
        if rollout_len % seq_len != 0:
            raise ValueError(f"rollout_len ({rollout_len}) must be divisible by seq_len ({seq_len})")
        self.T = rollout_len
        self.B = num_envs
        self.seq_len = seq_len
        self.num_chunks = rollout_len // seq_len
        self.include_depth = include_depth

        T, B = rollout_len, num_envs
        self.ray_distance = np.zeros((T, B, NUM_RAYS), dtype=np.float32)
        self.ray_hit_type = np.zeros((T, B, NUM_RAYS), dtype=np.int64)
        self.entity_features = np.zeros((T, B, MAX_ENTITIES, 5), dtype=np.float32)
        self.entity_types = np.zeros((T, B, MAX_ENTITIES), dtype=np.int64)
        self.entity_mask = np.zeros((T, B, MAX_ENTITIES), dtype=np.float32)
        self.vitals = np.zeros((T, B, VITALS_DIM), dtype=np.float32)
        self.depth_feat = np.zeros((T, B, DEPTH_DOWN_WIDTH), dtype=np.float32)
        self.prev_action = np.zeros((T, B, ACTION_DIM), dtype=np.float32)

        self.reset_before_step = np.zeros((T, B), dtype=bool)
        self.action = np.zeros((T, B, ACTION_DIM), dtype=np.float32)
        self.log_prob = np.zeros((T, B), dtype=np.float32)
        self.value_ext = np.zeros((T, B), dtype=np.float32)
        self.value_int = np.zeros((T, B), dtype=np.float32)
        self.reward_ext = np.zeros((T, B), dtype=np.float32)
        self.reward_int = np.zeros((T, B), dtype=np.float32)
        self.done = np.zeros((T, B), dtype=bool)

        self.next_ray_distance = np.zeros((T, B, NUM_RAYS), dtype=np.float32)
        self.aux_combat = np.zeros((T, B, 2), dtype=np.float32)
        self.aux_landmark_dir = np.zeros((T, B, 2), dtype=np.float32)
        self.aux_landmark_mask = np.zeros((T, B), dtype=np.float32)

        self.adv_ext = np.zeros((T, B), dtype=np.float32)
        self.ret_ext = np.zeros((T, B), dtype=np.float32)
        self.adv_int = np.zeros((T, B), dtype=np.float32)
        self.ret_int = np.zeros((T, B), dtype=np.float32)
        self.advantages = np.zeros((T, B), dtype=np.float32)

        self.chunk_states: list[EMLRPState] = []
        self.ptr = 0

    def reset(self) -> None:
        self.chunk_states.clear()
        self.ptr = 0

    def add(
        self,
        obs: list[ProcessedObservation],
        reset_before_step: np.ndarray,
        state_before_step: EMLRPState,
        action: np.ndarray,
        log_prob: np.ndarray,
        value_ext: np.ndarray,
        value_int: np.ndarray,
        reward_ext: np.ndarray,
        reward_int: np.ndarray,
        done: np.ndarray,
        next_ray_distance: np.ndarray,
        aux_combat: np.ndarray,
        aux_landmark_dir: np.ndarray,
        aux_landmark_mask: np.ndarray,
    ) -> None:
        t = self.ptr
        if t >= self.T:
            raise IndexError("RolloutBuffer is full")

        if t % self.seq_len == 0:
            self.chunk_states.append(_state_to_cpu_detached(state_before_step))

        rays_np = np.stack([x.rays for x in obs], axis=0)
        ents_np = np.stack([x.entities for x in obs], axis=0)

        self.ray_distance[t] = rays_np[..., 0]
        self.ray_hit_type[t] = rays_np[..., 2:8].argmax(axis=-1)
        self.entity_features[t] = ents_np[..., :5]
        self.entity_types[t] = ents_np[..., 5:17].argmax(axis=-1)
        self.entity_mask[t] = np.stack([x.entity_mask for x in obs], axis=0)
        self.vitals[t] = np.stack([x.vitals for x in obs], axis=0)
        if self.include_depth:
            self.depth_feat[t] = np.stack([x.depth for x in obs], axis=0)
        self.prev_action[t] = np.stack([x.prev_action for x in obs], axis=0)

        self.reset_before_step[t] = reset_before_step
        self.action[t] = action
        self.log_prob[t] = log_prob
        self.value_ext[t] = value_ext
        self.value_int[t] = value_int
        self.reward_ext[t] = reward_ext
        self.reward_int[t] = reward_int
        self.done[t] = done

        self.next_ray_distance[t] = next_ray_distance
        self.aux_combat[t] = aux_combat
        self.aux_landmark_dir[t] = aux_landmark_dir
        self.aux_landmark_mask[t] = aux_landmark_mask
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value_ext: np.ndarray,
        last_value_int: np.ndarray,
        gamma: float,
        gamma_int: float,
        gae_lambda: float,
        intrinsic_coef: float,
    ) -> None:
        assert self.ptr == self.T, f"Buffer not full ({self.ptr}/{self.T})"
        self.adv_ext, self.ret_ext = _compute_gae(
            self.reward_ext, self.value_ext, last_value_ext, self.done, gamma, gae_lambda
        )
        self.adv_int, self.ret_int = _compute_gae(
            self.reward_int, self.value_int, last_value_int, None, gamma_int, gae_lambda
        )
        combined = self.adv_ext + intrinsic_coef * self.adv_int
        mean = float(combined.mean())
        std = float(combined.std())
        self.advantages = ((combined - mean) / (std + 1e-8)).astype(np.float32)

    def iter_minibatches(
        self, minibatch_seqs: int, device: torch.device, rng: np.random.Generator
    ) -> Iterator[dict[str, Any]]:
        """Yield `(M, seq_len, ...)` sequence minibatches for truncated BPTT."""
        total_seqs = self.num_chunks * self.B
        perm = rng.permutation(total_seqs)

        for start in range(0, total_seqs, minibatch_seqs):
            idx = perm[start : start + minibatch_seqs]
            c_idx = idx // self.B
            e_idx = idx % self.B

            def _gather_seq(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
                # arr is (T, B, ...); gather slices [c*seq:(c+1)*seq, e] -> (M, seq_len, ...)
                slices = [arr[c * self.seq_len : (c + 1) * self.seq_len, e] for c, e in zip(c_idx, e_idx)]
                return torch.as_tensor(np.stack(slices, axis=0), dtype=dtype, device=device)

            obs_seq: dict[str, Tensor] = {
                "ray_distance": _gather_seq(self.ray_distance),
                "ray_hit_type": _gather_seq(self.ray_hit_type, dtype=torch.long),
                "entity_features": _gather_seq(self.entity_features),
                "entity_types": _gather_seq(self.entity_types, dtype=torch.long),
                "entity_mask": _gather_seq(self.entity_mask),
                "vitals": _gather_seq(self.vitals),
            }
            if self.include_depth:
                obs_seq["depth_feat"] = _gather_seq(self.depth_feat)

            init_state = _slice_and_move_state(self.chunk_states, c_idx, e_idx, device)

            yield {
                "obs_seq": obs_seq,
                "init_state": init_state,
                "prev_action": _gather_seq(self.prev_action),
                "reset_before_step": _gather_seq(self.reset_before_step, dtype=torch.bool),
                "action": _gather_seq(self.action),
                "old_log_prob": _gather_seq(self.log_prob),
                "old_value_ext": _gather_seq(self.value_ext),
                "old_value_int": _gather_seq(self.value_int),
                "ret_ext": _gather_seq(self.ret_ext),
                "ret_int": _gather_seq(self.ret_int),
                "advantages": _gather_seq(self.advantages),
                "done": _gather_seq(self.done, dtype=torch.bool),
                "next_ray_distance": _gather_seq(self.next_ray_distance),
                "aux_combat": _gather_seq(self.aux_combat),
                "aux_landmark_dir": _gather_seq(self.aux_landmark_dir),
                "aux_landmark_mask": _gather_seq(self.aux_landmark_mask),
            }


class RecurrentPPOTrainer:
    """Custom recurrent PPO trainer over EMLRPPolicy with auxiliary supervision."""

    def __init__(
        self,
        policy: EMLRPPolicy,
        config: RecurrentPPOConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.policy = policy
        self.config = config or RecurrentPPOConfig()
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.include_depth = self.policy.config.depth_feat_dim > 0
        self.use_amp = (
            bool(self.config.use_bf16)
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def select_actions(
        self,
        obs_list: list[ProcessedObservation],
        state: EMLRPState,
        reset_mask: np.ndarray | Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, EMLRPState]:
        """Run one batched rollout step and sample MultiBinary(9) actions."""
        was_training = self.policy.training
        self.policy.eval()

        obs_t, prev_act_t = processed_to_tensors(
            obs_list, device=self.device, include_depth=self.include_depth
        )
        done_t = torch.as_tensor(reset_mask, dtype=torch.bool, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.policy.step(obs_t, state, prev_act_t, done=done_t, compute_aux=False)

        logits_f = out.action_logits.float()
        dist = Bernoulli(logits=logits_f)
        if deterministic:
            actions_t = (logits_f > 0.0).float()
        else:
            actions_t = dist.sample()
        log_probs_t = dist.log_prob(actions_t).sum(dim=-1)

        if was_training:
            self.policy.train()

        scalars = torch.stack(
            [log_probs_t, out.value_ext.squeeze(-1).float(), out.value_int.squeeze(-1).float()],
            dim=0,
        ).cpu().numpy()
        return (
            actions_t.cpu().numpy().astype(np.int8),
            scalars[0].astype(np.float32),
            scalars[1].astype(np.float32),
            scalars[2].astype(np.float32),
            out.state,
        )

    @torch.inference_mode()
    def evaluate_values(
        self,
        obs_list: list[ProcessedObservation],
        state: EMLRPState,
        reset_mask: np.ndarray | Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap values for GAE at the end of a rollout."""
        was_training = self.policy.training
        self.policy.eval()

        obs_t, prev_act_t = processed_to_tensors(
            obs_list, device=self.device, include_depth=self.include_depth
        )
        done_t = torch.as_tensor(reset_mask, dtype=torch.bool, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.policy.step(obs_t, state, prev_act_t, done=done_t, compute_aux=False)

        if was_training:
            self.policy.train()

        vals = torch.stack(
            [out.value_ext.squeeze(-1).float(), out.value_int.squeeze(-1).float()], dim=0
        ).cpu().numpy()
        return vals[0].astype(np.float32), vals[1].astype(np.float32)

    def _augment_obs(self, obs_seq: dict[str, Tensor]) -> dict[str, Tensor]:
        cfg = self.config
        if cfg.entity_dropout_prob <= 0.0 and cfg.obs_noise_std <= 0.0:
            return obs_seq
        out = dict(obs_seq)
        if cfg.obs_noise_std > 0.0:
            out["ray_distance"] = (
                out["ray_distance"] + torch.randn_like(out["ray_distance"]) * cfg.obs_noise_std
            ).clamp(0.0, 1.0)
            out["entity_features"] = (
                out["entity_features"] + torch.randn_like(out["entity_features"]) * cfg.obs_noise_std
            )
        if cfg.entity_dropout_prob > 0.0:
            keep = (
                torch.rand(out["entity_mask"].shape, device=out["entity_mask"].device)
                >= cfg.entity_dropout_prob
            ).to(out["entity_mask"].dtype)
            out["entity_mask"] = out["entity_mask"] * keep
        return out

    def update(self, buffer: RecurrentRolloutBuffer, progress_frac: float = 0.0) -> dict[str, float]:
        """Run PPO + auxiliary BPTT epochs over `buffer`."""
        cfg = self.config
        progress = float(np.clip(progress_frac, 0.0, 1.0))
        lr = cfg.lr + progress * (cfg.lr_end - cfg.lr)
        entropy_coef = cfg.entropy_coef_start + progress * (cfg.entropy_coef_end - cfg.entropy_coef_start)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.policy.train()

        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_ext_loss": [],
            "value_int_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_frac": [],
            "aux_loss": [],
            "grad_norm": [],
        }

        early_stop = False
        for _epoch in range(cfg.ppo_epochs):
            if early_stop:
                break
            for mb in buffer.iter_minibatches(cfg.minibatch_seqs, self.device, self.rng):
                aug_obs = self._augment_obs(mb["obs_seq"])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    outputs, _ = self.policy.forward_sequence(
                        aug_obs,
                        mb["init_state"],
                        mb["prev_action"],
                        done_seq=mb["reset_before_step"],
                        compute_aux=True,
                    )

                dist = Bernoulli(logits=outputs["action_logits"].float())
                new_logp = dist.log_prob(mb["action"]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                log_ratio = new_logp - mb["old_log_prob"]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

                adv = mb["advantages"]
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_ext = outputs["value_ext"].squeeze(-1).float()
                v_int = outputs["value_int"].squeeze(-1).float()
                if cfg.value_clip_eps > 0.0:
                    v_ext_clipped = mb["old_value_ext"] + torch.clamp(
                        v_ext - mb["old_value_ext"], -cfg.value_clip_eps, cfg.value_clip_eps
                    )
                    vf_ext_loss = 0.5 * torch.max(
                        (v_ext - mb["ret_ext"]).pow(2), (v_ext_clipped - mb["ret_ext"]).pow(2)
                    ).mean()
                    v_int_clipped = mb["old_value_int"] + torch.clamp(
                        v_int - mb["old_value_int"], -cfg.value_clip_eps, cfg.value_clip_eps
                    )
                    vf_int_loss = 0.5 * torch.max(
                        (v_int - mb["ret_int"]).pow(2), (v_int_clipped - mb["ret_int"]).pow(2)
                    ).mean()
                else:
                    vf_ext_loss = 0.5 * F.mse_loss(v_ext, mb["ret_ext"])
                    vf_int_loss = 0.5 * F.mse_loss(v_int, mb["ret_int"])

                # Auxiliary losses (training only)
                aux_loss = torch.zeros((), device=self.device)
                if cfg.aux_coef > 0.0:
                    nonterm = (~mb["done"]).float()
                    ray_err = (outputs["aux_next_ray"].float() - mb["next_ray_distance"]).pow(2).mean(dim=-1)
                    next_ray_loss = (ray_err * nonterm).sum() / nonterm.sum().clamp(min=1.0)

                    inv_dyn_loss = F.binary_cross_entropy_with_logits(
                        outputs["aux_inverse_dyn"].float(), mb["action"]
                    )
                    combat_loss = F.binary_cross_entropy_with_logits(
                        outputs["aux_combat"].float(), mb["aux_combat"]
                    )

                    dir_mask = mb["aux_landmark_mask"]
                    dir_pred = F.normalize(outputs["aux_landmark_dir"].float(), dim=-1, eps=1e-6)
                    dir_err = (dir_pred - mb["aux_landmark_dir"]).pow(2).sum(dim=-1)
                    landmark_dir_loss = (dir_err * dir_mask).sum() / dir_mask.sum().clamp(min=1.0)

                    # Entity importance: highlight enemies (1,2,3) and keys (9)
                    ent_types = mb["obs_seq"]["entity_types"]
                    imp_target = ((ent_types >= 1) & (ent_types <= 3) | (ent_types == 9)).float()
                    ent_mask = mb["obs_seq"]["entity_mask"]
                    imp_bce = F.binary_cross_entropy_with_logits(
                        outputs["aux_entity_importance"].float(), imp_target, reduction="none"
                    )
                    entity_imp_loss = (imp_bce * ent_mask).sum() / ent_mask.sum().clamp(min=1.0)

                    aux_loss = (
                        cfg.aux_next_ray_weight * next_ray_loss
                        + cfg.aux_inverse_dyn_weight * inv_dyn_loss
                        + cfg.aux_combat_weight * combat_loss
                        + cfg.aux_landmark_dir_weight * landmark_dir_loss
                        + cfg.aux_entity_imp_weight * entity_imp_loss
                    )

                total_loss = (
                    policy_loss
                    + cfg.vf_ext_coef * vf_ext_loss
                    + cfg.vf_int_coef * vf_int_loss
                    - entropy_coef * entropy
                    + cfg.aux_coef * aux_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_ext_loss"].append(float(vf_ext_loss.item()))
                stats["value_int_loss"].append(float(vf_int_loss.item()))
                stats["entropy"].append(float(entropy.item()))
                stats["approx_kl"].append(float(approx_kl.item()))
                stats["clip_frac"].append(float(clip_frac.item()))
                stats["aux_loss"].append(float(aux_loss.item()))
                stats["grad_norm"].append(float(grad_norm.item()))

                if cfg.target_kl > 0.0 and approx_kl.item() > 1.5 * cfg.target_kl:
                    early_stop = True
                    break

        # Explained variance for extrinsic value predictions
        y_true = buffer.ret_ext.reshape(-1)
        y_pred = buffer.value_ext.reshape(-1)
        var_y = float(np.var(y_true))
        explained_var = 0.0 if var_y < 1e-8 else float(1.0 - np.var(y_true - y_pred) / var_y)

        return {
            **{k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()},
            "explained_var_ext": explained_var,
            "lr": float(lr),
            "entropy_coef": float(entropy_coef),
        }
