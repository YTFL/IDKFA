"""Unit tests for procedural map generator, UDMF compiler, and curriculum wrappers."""

import pytest
import numpy as np
from pathlib import Path

from doom_env import make_env
from training.mapgen.udmf import UDMFMap, THING_PLAYER1, THING_ZOMBIEMAN
from training.mapgen.generator import ProceduralMapGenerator, MapFamily, DoorEdge, Room
from training.envs.curriculum import (
    CurriculumManager,
    CurriculumStage,
    CurriculumEnvWrapper,
    RewardWeights,
)


def test_udmf_compilation_and_loading(tmp_path):
    m = UDMFMap()
    v0 = m.add_vertex(0, 0)
    v1 = m.add_vertex(0, 256)
    v2 = m.add_vertex(256, 256)
    v3 = m.add_vertex(256, 0)
    s = m.add_sector(heightfloor=0, heightceiling=128)
    sd = m.add_sidedef(sector=s, texturemiddle="STARTAN2")
    m.add_linedef(v0, v1, sidefront=sd, blocking=True)
    m.add_linedef(v1, v2, sidefront=sd, blocking=True)
    m.add_linedef(v2, v3, sidefront=sd, blocking=True)
    m.add_linedef(v3, v0, sidefront=sd, blocking=True)
    m.add_thing(128, 128, THING_PLAYER1, angle=0)
    m.add_thing(192, 192, THING_ZOMBIEMAN, angle=180)

    wad_path = tmp_path / "simple_room.wad"
    m.write_wad(wad_path)
    assert wad_path.is_file()

    env = make_env(str(wad_path), episode_ticks=100)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize(
    "family",
    [
        MapFamily.FAMILY_A,
        MapFamily.FAMILY_B,
        MapFamily.FAMILY_C,
        MapFamily.FAMILY_D,
    ],
)
def test_map_families_generation_and_stepping(tmp_path, family):
    gen = ProceduralMapGenerator(seed=123)
    m, solvable = gen.generate(family, num_rooms=5, difficulty=0.5)
    assert solvable, f"Map for {family} should be solvable"

    wad_path = tmp_path / f"family_{family.value}.wad"
    m.write_wad(wad_path)
    assert wad_path.is_file()

    env = make_env(str(wad_path), episode_ticks=60)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)

    # Step multiple times
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    env.close()


def test_solvability_checker_unsolvable_case():
    gen = ProceduralMapGenerator(seed=99)
    rooms = {
        0: Room(id=0, gx=0, gy=0, is_start=True),
        1: Room(id=1, gx=1, gy=0, is_exit=True),
    }
    edges = [(0, 1)]
    # Locked door between 0 and 1 with no key present in room 0
    doors = [DoorEdge(r1=0, r2=1, locked=1)]
    solvable = gen._verify_solvability(rooms, edges, doors, start_id=0, exit_id=1, key_needed=1)
    assert not solvable, "Should detect map as unsolvable when key is missing"


def test_curriculum_wrapper_and_manager(tmp_path):
    mgr = CurriculumManager(map_cache_dir=tmp_path / "curriculum_maps")
    assert mgr.current_stage == CurriculumStage.STAGE_0_AIM_SHOOT

    env = mgr.make_curriculum_env(stage=CurriculumStage.STAGE_0_AIM_SHOOT, seed=7)
    obs, _ = env.reset(seed=42)
    assert "rays" in obs
    assert "vitals" in obs

    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert "shaped_reward" in info
    assert "unique_cells_visited" in info
    assert info["unique_cells_visited"] >= 1
    env.close()

    # Test stage promotion
    for _ in range(100):
        mgr.record_episode(True)
    assert mgr.current_stage == CurriculumStage.STAGE_1_COMBAT_ARENA
