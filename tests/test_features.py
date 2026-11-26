"""Unit tests for Phase 2: Observation Feature Extraction Pipeline."""

import numpy as np
import pytest

from doom_env import make_env
from submission.features import (
    DEPTH_DOWN_WIDTH,
    ENTITY_FEAT_DIM,
    MAX_ENTITIES,
    NUM_RAYS,
    VITALS_DIM,
    ObservationPreprocessor,
    RunningMeanStd,
)


def test_feature_pipeline_on_real_env():
    env = make_env("maps/two_rooms.wad", episode_ticks=50)
    obs, info = env.reset(seed=42)

    preprocessor = ObservationPreprocessor()
    processed = preprocessor.process(obs)

    # 1. Rays shape & bounds
    assert processed.rays.shape == (NUM_RAYS, 8)
    assert np.all(processed.rays[:, 0] >= 0.0) and np.all(processed.rays[:, 0] <= 1.0)
    assert np.all(processed.rays[:, 1] >= 0.0) and np.all(processed.rays[:, 1] <= 1.0)

    # 2. Depth shape & bounds
    assert processed.depth.shape == (DEPTH_DOWN_WIDTH,)
    assert np.all(processed.depth >= 0.0) and np.all(processed.depth <= 1.0)

    # 3. Vitals shape & scaling
    assert processed.vitals.shape == (VITALS_DIM,)
    assert processed.vitals[0] == pytest.approx(1.0, abs=1e-2)  # Health 100 -> 1.0

    # 4. Entities shape & masking
    assert processed.entities.shape == (MAX_ENTITIES, ENTITY_FEAT_DIM)
    assert processed.entity_mask.shape == (MAX_ENTITIES,)

    # 5. Flat vector
    flat = processed.to_flat_vector()
    expected_dim = (NUM_RAYS * 8) + DEPTH_DOWN_WIDTH + VITALS_DIM + (MAX_ENTITIES * ENTITY_FEAT_DIM) + MAX_ENTITIES + 9
    assert flat.shape == (expected_dim,)
    assert not np.isnan(flat).any()
    env.close()


def test_feature_pipeline_nan_and_inf_resilience():
    """Verify that corrupt observations (NaNs, Infs, extreme values) are sanitized."""
    preprocessor = ObservationPreprocessor()

    corrupt_obs = {
        "rays": {
            "distance": np.full(64, np.nan, dtype=np.float32),
            "hit_type": np.full(64, 999, dtype=np.int64),  # out of bounds
        },
        "depth": np.full((120, 160), np.nan, dtype=np.float32),
        "vitals": np.array([np.nan, np.inf, -np.inf] + [0.0] * 14, dtype=np.float32),
        "visible_entities": {
            "features": np.full((32, 5), np.nan, dtype=np.float32),
            "entity_type": np.full(32, 100, dtype=np.int64),
            "mask": np.zeros(32, dtype=np.int8),
        },
    }

    processed = preprocessor.process(corrupt_obs)

    assert not np.isnan(processed.rays).any()
    assert not np.isnan(processed.depth).any()
    assert not np.isnan(processed.vitals).any()
    assert not np.isnan(processed.entities).any()
    assert not np.isinf(processed.vitals).any()

    flat = processed.to_flat_vector()
    assert not np.isnan(flat).any()
    assert not np.isinf(flat).any()

    # Verify depth=None does not raise IndexError
    no_depth_obs = {
        "rays": corrupt_obs["rays"],
        "depth": None,
        "vitals": np.zeros(17, dtype=np.float32),
        "visible_entities": {},
    }
    p_no_depth = preprocessor.process(no_depth_obs)
    assert p_no_depth.depth.shape == (DEPTH_DOWN_WIDTH,)
    assert np.all(p_no_depth.depth == 0.0)


def test_running_mean_std():
    rms = RunningMeanStd(shape=(5,))
    data = np.random.randn(100, 5).astype(np.float32) * 3.0 + 2.0
    rms.update(data)

    normalized = rms.normalize(data)
    assert np.allclose(np.mean(normalized, axis=0), 0.0, atol=0.2)
    assert np.allclose(np.var(normalized, axis=0), 1.0, atol=0.3)
