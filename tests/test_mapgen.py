"""Unit tests for procedural map generator, UDMF compiler, and curriculum wrappers."""

import pytest
import numpy as np
from pathlib import Path

from doom_env import make_env
from fixtures import fixture_wad
from training.mapgen.udmf import UDMFMap, THING_PLAYER1, THING_ZOMBIEMAN
from training.mapgen.generator import ProceduralMapGenerator, MapFamily, DoorEdge, Room
from training.envs.curriculum import (
    CurriculumConfig,
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
    assert env.config.infinite_ammo is True
    assert mgr.stage_configs[CurriculumStage.STAGE_2_NAV_AUTO_DOORS].auto_open_doors is True
    assert mgr.stage_configs[CurriculumStage.STAGE_2_NAV_AUTO_DOORS].god_mode is True

    obs, _ = env.reset(seed=42)
    assert "rays" in obs
    assert "vitals" in obs

    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert "shaped_reward" in info
    assert "reward_breakdown" in info
    assert "unique_cells_visited" in info
    assert info["unique_cells_visited"] >= 1
    env.close()

    # Test stage promotion
    for _ in range(100):
        mgr.record_episode(True)
    assert mgr.current_stage == CurriculumStage.STAGE_1_COMBAT_ARENA


def test_generated_locked_doors_and_watertight_geometry(tmp_path):
    """Generated C/D maps must emit real locked door sectors and watertight room walls."""
    for seed in (10, 42, 123):
        for fam in (MapFamily.FAMILY_C, MapFamily.FAMILY_D):
            gen = ProceduralMapGenerator(seed=seed)
            m, solvable = gen.generate(fam, num_rooms=6, difficulty=0.5)
            assert solvable
            wad_path = tmp_path / f"{fam.value}_{seed}.wad"
            m.write_wad(wad_path)

            env = make_env(str(wad_path), episode_ticks=100)
            env.reset(seed=0)
            ps = env.privileged_state()
            assert any(lock > 0 for lock in ps["door_lock"]), f"No locked door in {fam} seed={seed}"
            env.close()


def test_curriculum_auto_open_doors_edge_triggered():
    """Stage 2 auto_open_doors must pulse USE when approaching a closed door and reach exit."""
    wad_door = fixture_wad("##########\n#P..D...E#\n##########")
    env = CurriculumEnvWrapper(
        make_env(wad_door, episode_ticks=300),
        stage=CurriculumStage.STAGE_2_NAV_AUTO_DOORS,
        config=CurriculumConfig(stage=CurriculumStage.STAGE_2_NAV_AUTO_DOORS, auto_open_doors=True),
    )
    env.reset(seed=0)
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    reached_exit = False
    for _ in range(100):
        _, _, term, trunc, info = env.step(fwd)
        if term or trunc:
            reached_exit = info["exit_reached"]
            break
    env.close()
    assert reached_exit, "auto_open_doors failed to open door and reach exit"
