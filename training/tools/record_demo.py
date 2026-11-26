"""Live Demo Runner & Video/HTML Recorder.

Executes DoomAgent inside a map, renders each frame through the composite DoomDashboard
with authentic EMLRP telemetry (action probabilities, latent mode, spatial map, slots),
and optionally:
  1. Displays a live Pygame desktop window (--window).
  2. Saves an interactive HTML composite replay (--save-html).
  3. Saves an MP4 video if OpenCV is installed (--output demo.mp4; falls back to HTML if absent).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402
from training.rewards.reward_env import ShapedRewardEnv  # noqa: E402
from training.tools.dashboard import DoomDashboard  # noqa: E402
from training.tools.demo_replay import rgb_array_to_png_b64  # noqa: E402


def _write_dashboard_html(frames_b64: list[dict], out_path: Path, map_label: str, steps: int, total_reward: float) -> None:
    import html
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(frames_b64)
    title = html.escape(f"Nordlicht Composite Telemetry Replay — {map_label}")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0e0f14; color: #dce1eb; margin: 20px; }}
  .wrap {{ max-width: 1220px; margin: 0 auto; }}
  img {{ width: 100%; border-radius: 8px; border: 1px solid #282c3a; }}
  .bar {{ display: flex; align-items: center; gap: 12px; margin: 12px 0; }}
  input[type=range] {{ flex: 1; }}
  button {{ background: #4696fa; color: #fff; border: 0; border-radius: 6px; padding: 6px 14px; font-weight: 700; cursor: pointer; }}
</style></head>
<body><div class="wrap">
  <h2>{title}</h2>
  <p>Steps: {steps} | Shaped Reward: {total_reward:.2f} | Frames: {len(frames_b64)}</p>
  <div class="bar"><button id="btn">Play</button><input id="rng" type="range" min="0" max="{max(0, len(frames_b64) - 1)}" value="0" /><span id="lbl">0</span></div>
  <img id="img" />
</div>
<script>
const F = {payload}; let i = 0, t = null;
const rng = document.getElementById('rng');
function show(k) {{ i = k; rng.value = k; if (F[k]) {{ document.getElementById('img').src = 'data:image/png;base64,' + F[k].png; document.getElementById('lbl').textContent = 'step ' + F[k].step; }} }}
rng.addEventListener('input', e => show(parseInt(e.target.value, 10)));
document.getElementById('btn').addEventListener('click', () => {{
  if (t) {{ clearInterval(t); t = null; document.getElementById('btn').textContent = 'Play'; return; }}
  document.getElementById('btn').textContent = 'Pause';
  t = setInterval(() => show((i + 1) % F.length), 120);
}});
show(0);
</script></body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def run_demo(
    wad_path: str = "maps/two_rooms.wad",
    ticks: int = 400,
    window: bool = False,
    output_mp4: str | None = None,
    save_html: str | None = None,
    fps: int = 15,
    seed: int = 42,
    device: str = "cpu",
    stride: int = 2,
) -> dict:
    print(f"Initializing DOOM environment: {wad_path} (ticks={ticks})")
    base_env = make_env(wad_path, episode_ticks=ticks, render_buffers=True)
    env = ShapedRewardEnv(base_env, use_rnd=True, use_episodic_novelty=True, sticky_action_prob=0.0)

    agent = DoomAgent(env.observation_space, env.action_space, device=device)
    agent.reset()

    dashboard = DoomDashboard(title=f"Nordlicht DOOM Agent - {Path(wad_path).name}", headless=not window)

    video_writer = None
    html_target = Path(save_html) if save_html else None
    if output_mp4:
        try:
            import cv2  # type: ignore[import-not-found]

            Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (dashboard.width, dashboard.height))
            print(f"Recording video to: {output_mp4}")
        except ImportError:
            if html_target is None:
                html_target = Path(output_mp4).with_suffix(".html")
            print(f"Info: opencv-python not installed; exporting interactive HTML replay to {html_target} instead.")

    html_frames: list[dict] = []
    step_num = 0
    total_ep_reward = 0.0
    t_start = time.perf_counter()

    try:
        obs, info = env.reset(seed=seed)
        done = False
        clock = pygame.time.Clock() if window else None

        print("Running episode...")
        while not done:
            if window:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        done = True
                        break
                if done:
                    break

            t0 = time.perf_counter()
            action = agent.act(obs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            tel = agent.get_last_telemetry()

            probs = tel["action_probs"]
            mode_probs = tel["mode_probs"]
            top_mode = int(np.argmax(mode_probs)) if mode_probs is not None else 0
            top_p = float(mode_probs[top_mode]) * 100.0 if mode_probs is not None else 0.0
            mode_str = f"MODE_{top_mode} ({top_p:.0f}%)"

            obs, _reward, term, trunc, info = env.step(action)
            step_num += 1
            done = term or trunc
            total_ep_reward = float(info.get("total_episode_reward", 0.0))

            composite_frame = dashboard.render_frame(
                rgb_frame=obs["rgb"],
                observation=obs,
                action_probs=probs,
                action_taken=action,
                reward_breakdown=info.get("reward_breakdown"),
                step_latency_ms=latency_ms,
                decision_mode=mode_str,
                spatial_map=tel["spatial_map"],
                active_slots=tel["active_landmark_slots"],
            )

            if video_writer is not None:
                import cv2

                bgr_frame = cv2.cvtColor(composite_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)

            if html_target is not None and (step_num % max(1, stride) == 0 or done):
                html_frames.append({"step": step_num, "png": rgb_array_to_png_b64(composite_frame)})

            if clock is not None:
                clock.tick(fps)

        wall_duration = time.perf_counter() - t_start
        print(f"Episode finished in {step_num} steps ({wall_duration:.2f}s). Total reward: {total_ep_reward:.2f}")

        if window:
            print("Episode ended. Close the window or press ESC / Q to exit.")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        waiting = False
                        break
                pygame.time.wait(50)

        if html_target is not None and html_frames:
            _write_dashboard_html(html_frames, html_target, Path(wad_path).name, step_num, total_ep_reward)
            print(f"Interactive replay saved to {html_target} ({len(html_frames)} frames)")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved successfully to {output_mp4} ({Path(output_mp4).stat().st_size // 1024} KB)")
        env.close()
        dashboard.close()

    return {"steps": step_num, "total_reward": total_ep_reward, "html_path": str(html_target) if html_target else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nordlicht DOOM Agent Live Dashboard & Recorder")
    parser.add_argument("--map", type=str, default="maps/two_rooms.wad", help="Map WAD path")
    parser.add_argument("--ticks", type=int, default=300, help="Episode ticks")
    parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--window", action="store_true", help="Open live desktop window")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 filename (falls back to HTML if cv2 missing)")
    parser.add_argument("--save-html", type=str, default=None, help="Output standalone HTML composite replay")
    parser.add_argument("--stride", type=int, default=2, help="Capture every Nth step into HTML")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    args = parser.parse_args()

    run_demo(
        wad_path=args.map,
        ticks=args.ticks,
        window=args.window,
        output_mp4=args.output,
        save_html=args.save_html,
        fps=args.fps,
        seed=args.seed,
        device=args.device,
        stride=args.stride,
    )
