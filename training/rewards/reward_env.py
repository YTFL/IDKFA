"""Comprehensive Reward Shaping and Exploration Environment Wrapper.

Wraps DoomEnv to combine:
  1. Extrinsic reward shaping (kills, damage, keys, doors, exit, cells)
  2. Potential-based distance shaping
  3. Anti-stuck displacement penalty
  4. RND (Random Network Distillation) intrinsic exploration bonus
  5. Episodic novelty buffer (NGU-style intra-episode curiosity)
  6. Sticky action randomization (to break action-repeat loops)
  7. Per-component reward breakdown logging in info['reward_breakdown']
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from submission.features import ObservationPreprocessor, ProcessedObservation
from training.rewards.curiosity import EpisodicNoveltyBuffer, RNDCuriosity
from training.rewards.shaping import ExtrinsicRewardShaper, RewardBreakdown, RewardConfig


class ShapedRewardEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_config: RewardConfig | None = None,
        use_rnd: bool = True,
        use_episodic_novelty: bool = True,
        sticky_action_prob: float = 0.05,
        rnd: RNDCuriosity | None = None,
        update_rnd: bool = True,
    ):
        super().__init__(env)
        self.reward_shaper = ExtrinsicRewardShaper(reward_config)
        self.preprocessor = ObservationPreprocessor()
        self.use_rnd = use_rnd
        self.use_episodic_novelty = use_episodic_novelty
        self.sticky_action_prob = sticky_action_prob
        self.update_rnd = update_rnd

        self.rnd = rnd if rnd is not None else (RNDCuriosity() if use_rnd else None)
        self.episodic_buffer = EpisodicNoveltyBuffer() if use_episodic_novelty else None

        self.last_action: np.ndarray | None = None
        self.total_episode_reward: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_shaper.reset()
        self.preprocessor.reset()
        if self.episodic_buffer:
            self.episodic_buffer.reset()

        self.last_action = None
        self.total_episode_reward = 0.0

        info["reward_breakdown"] = RewardBreakdown().to_dict()
        return obs, info

    def step(self, action: np.ndarray):
        # Sticky action randomization to prevent getting trapped in repetitive action loops
        act = np.asarray(action).copy()
        if self.last_action is not None and np.random.rand() < self.sticky_action_prob:
            act = self.last_action.copy()
        self.last_action = act.copy()

        obs, base_reward, terminated, truncated, info = self.env.step(act)
        priv = self.env.unwrapped.privileged_state()

        # 1. Extrinsic & potential shaping
        total_extrinsic, breakdown = self.reward_shaper.compute(
            obs=obs,
            privileged_state=priv,
            base_reward=base_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action=act,
        )

        # 2. Preprocess observation for curiosity
        processed_obs = self.preprocessor.process(obs, prev_action=act)

        # 3. Intrinsic Curiosity
        curiosity_bonus = 0.0
        if self.rnd:
            r_rnd = self.rnd.compute_intrinsic_reward(processed_obs, update_predictor=self.update_rnd)
            curiosity_bonus += r_rnd

        if self.episodic_buffer:
            r_nov = self.episodic_buffer.compute_novelty_bonus(processed_obs)
            curiosity_bonus += r_nov

        breakdown.curiosity = curiosity_bonus
        final_reward = float(breakdown.total)

        self.total_episode_reward += final_reward
        info["reward_breakdown"] = breakdown.to_dict()
        info["unique_cells_visited"] = len(self.reward_shaper.visited_cells)
        info["shaped_reward"] = final_reward
        info["total_episode_reward"] = self.total_episode_reward

        return obs, final_reward, terminated, truncated, info
