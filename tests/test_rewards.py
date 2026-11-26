"""Unit tests for Phase 4: Reward Shaping, Exploration, and Curiosity."""

import numpy as np
import pytest

from doom_env import make_env
from fixtures import fixture_wad
from submission.features import ObservationPreprocessor
from training.rewards.curiosity import EpisodicNoveltyBuffer, RNDCuriosity
from training.rewards.reward_env import ShapedRewardEnv
from training.rewards.shaping import ExtrinsicRewardShaper, RewardBreakdown, RewardConfig


def test_extrinsic_reward_components():
    shaper = ExtrinsicRewardShaper()

    # Base step
    r, bd = shaper.compute(
        obs={},
        privileged_state={"x": 100.0, "y": 100.0, "health": 100.0, "killcount": 0},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.time == pytest.approx(-0.001)
    assert bd.exploration == pytest.approx(0.02)  # first cell visited

    # Step in same cell (no exploration bonus)
    r, bd = shaper.compute(
        obs={},
        privileged_state={"x": 105.0, "y": 105.0, "health": 100.0, "killcount": 0},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.exploration == 0.0

    # Enemy killed (+0.5)
    r, bd = shaper.compute(
        obs={},
        privileged_state={"x": 110.0, "y": 110.0, "health": 100.0, "killcount": 1},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.kill == pytest.approx(0.5)

    # Damage taken (-0.005 * 20 = -0.1)
    r, bd = shaper.compute(
        obs={},
        privileged_state={"x": 115.0, "y": 115.0, "health": 80.0, "killcount": 1},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.damage == pytest.approx(-0.1)

    # Keycard pickup (+2.0)
    r, bd = shaper.compute(
        obs={},
        privileged_state={
            "x": 120.0,
            "y": 120.0,
            "health": 80.0,
            "killcount": 1,
            "keys": {"key_red_card": 1},
        },
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.key == pytest.approx(2.0)

    # Door opened (+0.5) & Locked door unlocked (+2.0)
    r, bd = shaper.compute(
        obs={},
        privileged_state={
            "x": 125.0,
            "y": 125.0,
            "health": 80.0,
            "killcount": 1,
            "keys": {"key_red_card": 1},
            "doors": {3: {"is_open": True, "lock": 1}},
        },
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.door == pytest.approx(0.5)
    assert bd.locked_door == pytest.approx(2.0)

    # Exit reached (+10.0 + bonuses)
    r, bd = shaper.compute(
        obs={},
        privileged_state={"x": 130.0, "y": 130.0, "health": 80.0, "killcount": 1},
        base_reward=0.0,
        terminated=True,
        truncated=False,
        info={"exit_reached": True},
    )
    assert bd.exit >= 10.0


def test_anti_stuck_penalty():
    cfg = RewardConfig(stuck_window=4, stuck_min_dist=10.0, stuck_penalty=-0.05)
    shaper = ExtrinsicRewardShaper(cfg)

    # Agent stays in nearly same position for 4 steps
    for _ in range(3):
        r, bd = shaper.compute(
            obs={},
            privileged_state={"x": 100.0, "y": 100.0, "health": 100.0},
            base_reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )
        assert bd.stuck == 0.0

    # 4th step: window filled with no displacement -> stuck penalty fires
    r, bd = shaper.compute(
        obs={},
        privileged_state={"x": 101.0, "y": 101.0, "health": 100.0},
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd.stuck == pytest.approx(-0.05)


def test_rnd_curiosity_bonus():
    rnd = RNDCuriosity(scale=1.0)
    prep = ObservationPreprocessor()

    dummy_obs = {
        "rays": {"distance": np.ones(64, dtype=np.float32), "hit_type": np.zeros(64, dtype=np.int64)},
        "depth": np.zeros((120, 160), dtype=np.uint8),
        "vitals": np.zeros(17, dtype=np.float32),
        "visible_entities": {},
    }
    p_obs = prep.process(dummy_obs)

    r_init = rnd.compute_intrinsic_reward(p_obs, update_predictor=True)
    assert r_init >= 0.0

    # After multiple updates on the exact same observation, error should decrease
    for _ in range(20):
        r_after = rnd.compute_intrinsic_reward(p_obs, update_predictor=True)
    assert r_after < r_init
    assert r_init <= 2.0  # bounded normalization, no early spike


def test_raised_floor_door_not_prematurely_rewarded():
    shaper = ExtrinsicRewardShaper()
    # Closed door on a raised floor (floor=32, ceiling=32) must NOT trigger door_reward
    _, bd_closed = shaper.compute(
        obs={},
        privileged_state={
            "x": 100.0,
            "y": 100.0,
            "health": 100.0,
            "sector_floor": np.array([0.0, 32.0]),
            "sector_ceiling": np.array([128.0, 32.0]),
            "door_lock": np.array([-1, 1]),
        },
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd_closed.door == 0.0
    assert bd_closed.locked_door == 0.0

    # Once opened (ceiling=128 > floor+16), rewards fire
    _, bd_open = shaper.compute(
        obs={},
        privileged_state={
            "x": 100.0,
            "y": 100.0,
            "health": 100.0,
            "sector_floor": np.array([0.0, 32.0]),
            "sector_ceiling": np.array([128.0, 128.0]),
            "door_lock": np.array([-1, 1]),
        },
        base_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert bd_open.door == pytest.approx(0.5)
    assert bd_open.locked_door == pytest.approx(2.0)


def test_episodic_novelty_buffer():
    buf = EpisodicNoveltyBuffer(c_novelty=0.1)
    prep = ObservationPreprocessor()

    dummy_obs1 = {
        "rays": {"distance": np.ones(64, dtype=np.float32), "hit_type": np.zeros(64, dtype=np.int64)},
        "depth": np.zeros((120, 160), dtype=np.uint8),
        "vitals": np.zeros(17, dtype=np.float32),
        "visible_entities": {},
    }
    p_obs1 = prep.process(dummy_obs1)

    # Initial novelty
    n1 = buf.compute_novelty_bonus(p_obs1)
    assert n1 > 0.0

    # Repeat identical observations
    for _ in range(10):
        n_repeat = buf.compute_novelty_bonus(p_obs1)

    # Novelty on identical repeated state decays
    assert n_repeat < n1


def test_shaped_reward_env_full_episode():
    base_env = make_env("maps/two_rooms.wad", episode_ticks=50)
    env = ShapedRewardEnv(base_env, use_rnd=True, use_episodic_novelty=True)

    obs, info = env.reset(seed=42)
    assert "reward_breakdown" in info

    for _ in range(10):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert "reward_breakdown" in info
        assert "total" in info["reward_breakdown"]
        assert "curiosity" in info["reward_breakdown"]
        assert isinstance(reward, float)
        if term or trunc:
            break
    env.close()


def test_shaped_reward_env_real_key_door_exit_progression():
    """Verify key pickup, door open, locked door unlock, exit, and bounded curiosity in real ViZDoom."""
    wad = fixture_wad("############\n#P.r..R...E#\n############")
    env = ShapedRewardEnv(
        make_env(wad, episode_ticks=1000),
        use_rnd=True,
        use_episodic_novelty=True,
        sticky_action_prob=0.0,
    )
    env.reset(seed=0)
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    use = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8)

    totals = {"key": 0.0, "door": 0.0, "locked_door": 0.0, "exit": 0.0, "curiosity": 0.0}
    last_x = None
    for i in range(240):
        st = env.unwrapped.privileged_state()
        stuck = last_x is not None and abs(st["x"] - last_x) < 0.5
        last_x = st["x"]
        act = use if (stuck and i % 6 == 0) else fwd
        _, _, term, trunc, info = env.step(act)
        bd = info["reward_breakdown"]
        for k in totals:
            totals[k] += bd[k]
        if term or trunc:
            break
    env.close()

    assert totals["key"] == pytest.approx(2.0)
    assert totals["door"] == pytest.approx(0.5)
    assert totals["locked_door"] == pytest.approx(2.0)
    assert totals["exit"] >= 10.0
    assert 0.0 < totals["curiosity"] < totals["exit"]
