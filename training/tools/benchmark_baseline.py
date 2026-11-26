"""Benchmark test map with random agent.

Logs ticks per second, observation statistics, and episode length.
Follows Phase 0 Step 5 from PLAN.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from doom_env import make_env

MAP_PATH = "maps/two_rooms.wad"


def run_benchmark(num_episodes: int = 3, seed: int = 42):
    print(f"Loading map: {MAP_PATH}")
    env = make_env(MAP_PATH, episode_ticks=2100)
    
    total_steps = 0
    total_ticks = 0
    start_time = time.perf_counter()
    
    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        done = False
        steps = 0
        
        # Track statistics
        ray_dists = []
        ray_hits = []
        entity_counts = []
        vitals_list = []
        
        ep_start = time.perf_counter()
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            ray_dists.append(obs["rays"]["distance"])
            ray_hits.append(obs["rays"]["hit_type"])
            entity_counts.append(obs["visible_entities"]["mask"].sum())
            vitals_list.append(obs["vitals"])
            
            done = terminated or truncated
            
        ep_duration = time.perf_counter() - ep_start
        ticks = steps * 4  # frame_skip is 4
        total_steps += steps
        total_ticks += ticks
        
        print(f"\n--- Episode {ep + 1} Summary ---")
        print(f"Steps: {steps} ({ticks} ticks) in {ep_duration:.3f}s")
        print(f"Ticks/sec: {ticks / ep_duration:.1f} (Steps/sec: {steps / ep_duration:.1f})")
        print(f"Mean ray hit distance: {np.mean(ray_dists):.3f}")
        print(f"Rays hit types distribution: {dict(zip(*np.unique(ray_hits, return_counts=True)))}")
        print(f"Mean visible entities per step: {np.mean(entity_counts):.2f}")
        print(f"Vitals shape: {obs['vitals'].shape}, health final: {obs['vitals'][0]}")

    wall_time = time.perf_counter() - start_time
    env.close()
    
    print("\n================ Benchmark Complete ================")
    print(f"Total Steps: {total_steps} | Total Ticks: {total_ticks} | Wall time: {wall_time:.3f}s")
    print(f"Overall Throughput: {total_ticks / wall_time:.1f} ticks/s ({total_steps / wall_time:.1f} decisions/s)")


if __name__ == "__main__":
    run_benchmark()
