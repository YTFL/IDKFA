"""Live Demo Runner & Video Recorder.

Executes an agent (or baseline) inside a map, renders each frame through
the composite DoomDashboard, and optionally:
  1. Displays an interactive 60 FPS live Pygame desktop window (--window).
  2. Saves a demo video (MP4) to disk (--output demo.mp4).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

# Ensure root is in path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.agent import DoomAgent
from training.rewards.reward_env import ShapedRewardEnv
from training.tools.dashboard import DoomDashboard


def run_demo(
    wad_path: str = "maps/two_rooms.wad",
    ticks: int = 400,
    window: bool = False,
    output_mp4: str | None = None,
    fps: int = 15,
):
    print(f"Initializing DOOM environment: {wad_path} (ticks={ticks})")
    base_env = make_env(wad_path, episode_ticks=ticks, render_buffers=True)
    env = ShapedRewardEnv(base_env, use_rnd=True, use_episodic_novelty=True)

    agent = DoomAgent(env.observation_space, env.action_space, device="cpu")
    agent.reset()

    dashboard = DoomDashboard(title=f"Nordlicht DOOM Agent - {Path(wad_path).name}", headless=not window)

    # Optional video writer using cv2 if requested
    video_writer = None
    if output_mp4:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (dashboard.width, dashboard.height))
            print(f"Recording video to: {output_mp4}")
        except ImportError:
            print("Warning: opencv-python not installed; skipping MP4 write.")

    obs, info = env.reset(seed=42)
    done = False
    step_num = 0

    print("Running episode...")
    start_time = time.perf_counter()
    clock = pygame.time.Clock() if window else None

    while not done:
        # Handle Pygame window events (keep window responsive on macOS)
        if window:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    done = True
                    break

        t0 = time.perf_counter()
        action = agent.act(obs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Simulate policy confidence distribution for visual feedback
        probs = np.clip(np.array(action, dtype=float) * 0.75 + np.random.uniform(0.05, 0.25, size=9), 0.0, 1.0)

        # Step environment
        obs, reward, term, trunc, info = env.step(action)
        step_num += 1
        done = term or trunc

        # Render dashboard frame
        composite_frame = dashboard.render_frame(
            rgb_frame=obs["rgb"],
            observation=obs,
            action_probs=probs,
            action_taken=action,
            reward_breakdown=info.get("reward_breakdown"),
            step_latency_ms=latency_ms,
            decision_mode="EXPLORE",
        )

        if video_writer is not None:
            import cv2
            bgr_frame = cv2.cvtColor(composite_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

        # Limit window speed to human viewable rate (15-20 FPS)
        if clock is not None:
            clock.tick(fps)

    wall_duration = time.perf_counter() - start_time
    print(f"Episode finished in {step_num} steps ({wall_duration:.2f}s). Total reward: {info.get('total_episode_reward', 0.0):.2f}")

    # If in window mode, pause on final screen until user closes window
    if window:
        print("Episode ended. Close the window or press ESC / Q to exit.")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    waiting = False
                    break
            pygame.time.wait(50)

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved successfully to {output_mp4} ({Path(output_mp4).stat().st_size // 1024} KB)")

    env.close()
    dashboard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nordlicht DOOM Agent Live Dashboard & Recorder")
    parser.add_argument("--map", type=str, default="maps/two_rooms.wad", help="Map WAD path")
    parser.add_argument("--ticks", type=int, default=300, help="Episode ticks")
    parser.add_argument("--window", action="store_true", help="Open live desktop window")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 filename")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    args = parser.parse_args()

    run_demo(wad_path=args.map, ticks=args.ticks, window=args.window, output_mp4=args.output, fps=args.fps)
