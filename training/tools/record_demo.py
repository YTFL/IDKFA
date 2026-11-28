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
import json
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from doom_env import spaces as S  # noqa: E402
from doom_env.wrapper import IWAD as FREEDOOM2_IWAD  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402
from training.rewards.reward_env import ShapedRewardEnv  # noqa: E402
from training.tools.dashboard import DoomDashboard  # noqa: E402
from training.tools.demo_replay import rgb_array_to_png_b64  # noqa: E402

TRACK_LIMITS = {
    "init_max_ms": 60_000.0,
    "reset_max_ms": 10_000.0,
    "first_act_max_ms": 10_000.0,
    "subsequent_act_max_ms": 5_000.0,
    "avg_act_max_ms": 40.0,
    "ram_max_mb": 8192.0,
    "vram_max_mb": 2048.0,
}


def _resolve_wad_path(wad_arg: str) -> str:
    low = wad_arg.strip().lower()
    if low in ("freedoom2", "freedoom2.wad", "official"):
        return FREEDOOM2_IWAD
    return wad_arg


def _get_process_ram_mb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        proc = k32.GetCurrentProcess()
        if k32.K32GetProcessMemoryInfo(proc, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _get_vram_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        alloc = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
        peak = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        return alloc, peak
    except Exception:
        return 0.0, 0.0


def _write_dashboard_html(frames_b64: list[dict], out_path: Path, map_label: str, steps: int, total_reward: float) -> None:
    import html
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(frames_b64)
    title = html.escape(f"IDKFA Composite Telemetry Replay — {map_label}")
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


def _evaluate_track_constraints(
    init_ms: float,
    reset_ms: float,
    first_act_ms: float,
    max_subsequent_act_ms: float,
    mean_latency_ms: float,
    peak_ram_mb: float,
    peak_vram_mb: float,
) -> tuple[dict[str, dict[str, float | bool]], list[str]]:
    checks = {
        "init": {"value_ms": round(init_ms, 2), "limit_ms": TRACK_LIMITS["init_max_ms"], "passed": init_ms <= TRACK_LIMITS["init_max_ms"]},
        "reset": {"value_ms": round(reset_ms, 2), "limit_ms": TRACK_LIMITS["reset_max_ms"], "passed": reset_ms <= TRACK_LIMITS["reset_max_ms"]},
        "first_act": {"value_ms": round(first_act_ms, 2), "limit_ms": TRACK_LIMITS["first_act_max_ms"], "passed": first_act_ms <= TRACK_LIMITS["first_act_max_ms"]},
        "subsequent_act": {"value_ms": round(max_subsequent_act_ms, 2), "limit_ms": TRACK_LIMITS["subsequent_act_max_ms"], "passed": max_subsequent_act_ms <= TRACK_LIMITS["subsequent_act_max_ms"]},
        "avg_budget": {"value_ms": round(mean_latency_ms, 3), "limit_ms": TRACK_LIMITS["avg_act_max_ms"], "passed": mean_latency_ms <= TRACK_LIMITS["avg_act_max_ms"]},
        "ram": {"value_mb": round(peak_ram_mb, 2), "limit_mb": TRACK_LIMITS["ram_max_mb"], "passed": peak_ram_mb <= TRACK_LIMITS["ram_max_mb"]},
        "vram": {"value_mb": round(peak_vram_mb, 2), "limit_mb": TRACK_LIMITS["vram_max_mb"], "passed": peak_vram_mb <= TRACK_LIMITS["vram_max_mb"]},
    }
    violations = [k for k, v in checks.items() if not v["passed"]]
    return checks, violations


def run_demo(
    wad_path: str = "maps/two_rooms.wad",
    map_name: str = "MAP01",
    ticks: int = 4200,
    window: bool = False,
    output_mp4: str | None = None,
    save_html: str | None = None,
    fps: int = 15,
    seed: int = 42,
    device: str = "cpu",
    stride: int = 2,
    log_file: str | None = None,
    strict_constraints: bool = False,
) -> dict:
    resolved_wad = _resolve_wad_path(wad_path)
    print(f"Initializing DOOM environment: {resolved_wad} [{map_name}] (ticks={ticks})")
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    base_env = make_env(resolved_wad, map_name=map_name, episode_ticks=ticks, render_buffers=True)
    env = ShapedRewardEnv(base_env, use_rnd=True, use_episodic_novelty=True, sticky_action_prob=0.0)

    t_init0 = time.perf_counter()
    agent = DoomAgent(env.observation_space, env.action_space, device=device)
    init_ms = (time.perf_counter() - t_init0) * 1000.0
    if init_ms > TRACK_LIMITS["init_max_ms"]:
        print(f"[CONSTRAINT VIOLATION] __init__() took {init_ms:.1f} ms > {TRACK_LIMITS['init_max_ms']:.0f} ms")

    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(log_file)
        if log_file
        else Path("runs/demo_logs") / f"{Path(resolved_wad).stem}_{map_name}_{ts_tag}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = log_path.with_suffix(".summary.json")
    print(f"Logging telemetry -> {log_path} (init={init_ms:.1f} ms)")

    dashboard = DoomDashboard(title=f"IDKFA DOOM Agent - {Path(resolved_wad).name} [{map_name}]", headless=not window)
    if window:
        print(f"Pygame window active (driver={pygame.display.get_driver()})")

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
    summary: dict = {}
    step_num = 0
    total_ep_reward = 0.0

    try:
        while True:
            step_num = 0
            total_ep_reward = 0.0
            latencies: list[float] = []
            actions_hist: list[np.ndarray] = []
            wall_close_steps = 0
            zero_ammo_steps = 0
            peak_ram_mb = _get_process_ram_mb()
            peak_vram_mb = 0.0
            last_priv: dict = {}
            user_quit = False
            hard_violation = False
            t_start = time.perf_counter()

            t_res0 = time.perf_counter()
            agent.reset()
            reset_ms = (time.perf_counter() - t_res0) * 1000.0
            obs, info = env.reset(seed=seed)
            done = False
            clock = pygame.time.Clock() if window else None

            with log_path.open("w", encoding="utf-8") as log_f:
                log_f.write(
                    json.dumps(
                        {
                            "event": "session_start",
                            "wad_path": resolved_wad,
                            "map_name": map_name,
                            "device": str(agent.exec_device),
                            "init_ms": round(init_ms, 2),
                            "reset_ms": round(reset_ms, 2),
                            "ram_mb": round(peak_ram_mb, 2),
                            "limits": TRACK_LIMITS,
                        }
                    )
                    + "\n"
                )

                print(f"Running episode on {Path(resolved_wad).name} [{map_name}]...")
                while not done:
                    if window:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                                user_quit = True
                                done = True
                                break
                        if done:
                            break

                    t0 = time.perf_counter()
                    action = agent.act(obs)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    tel = agent.get_last_telemetry()

                    act_limit_ms = TRACK_LIMITS["first_act_max_ms"] if step_num == 0 else TRACK_LIMITS["subsequent_act_max_ms"]
                    if strict_constraints and latency_ms > act_limit_ms:
                        hard_violation = True
                        done = True

                    probs = tel["action_probs"]
                    act_mask = tel.get("action_mask")
                    mode_probs = tel["mode_probs"]
                    top_mode = int(np.argmax(mode_probs)) if mode_probs is not None else 0
                    top_p = float(mode_probs[top_mode]) * 100.0 if mode_probs is not None else 0.0
                    mode_str = f"MODE_{top_mode} ({top_p:.0f}%)"

                    obs, _reward, term, trunc, info = env.step(action)
                    step_num += 1
                    done = done or term or trunc
                    total_ep_reward = float(info.get("total_episode_reward", 0.0))

                    latencies.append(latency_ms)
                    actions_hist.append(np.asarray(action, dtype=np.int8))
                    ram_mb = _get_process_ram_mb()
                    vram_alloc_mb, vram_peak_cur = _get_vram_mb()
                    peak_ram_mb = max(peak_ram_mb, ram_mb)
                    peak_vram_mb = max(peak_vram_mb, vram_peak_cur)

                    rays_d = obs["rays"]["distance"]
                    min_front_units = float(np.min(rays_d[list(range(0, 6)) + list(range(58, 64))])) * S.RAY_RANGE
                    if min_front_units < 123.0:
                        wall_close_steps += 1
                    vit = obs["vitals"]
                    if float(vit[3]) <= 0.0:
                        zero_ammo_steps += 1
                    last_priv = env.unwrapped.privileged_state() or {}

                    log_f.write(
                        json.dumps(
                            {
                                "step": step_num,
                                "tick": int(last_priv.get("tic", step_num * 4)),
                                "step_latency_ms": round(latency_ms, 3),
                                "ram_mb": round(ram_mb, 2),
                                "vram_alloc_mb": round(vram_alloc_mb, 2),
                                "vram_peak_mb": round(vram_peak_cur, 2),
                                "vitals": {
                                    "health": float(vit[0]),
                                    "armor": float(vit[1]),
                                    "weapon": int(vit[2]),
                                    "ammo": float(vit[3]),
                                    "keys": [int(k) for k in vit[8:14]],
                                },
                                "action_taken": [int(x) for x in action],
                                "action_probs": [round(float(x), 4) for x in probs],
                                "action_mask": [int(x) for x in act_mask] if act_mask is not None else None,
                                "mode_probs": [round(float(x), 4) for x in mode_probs],
                                "active_landmark_slots": int(tel["active_landmark_slots"]),
                                "min_front_ray_units": round(min_front_units, 2),
                                "kills": int(last_priv.get("kills", 0)),
                                "unique_cells": int(info.get("unique_cells_visited", 0)),
                                "reward_breakdown": info.get("reward_breakdown"),
                                "exit_reached": bool(last_priv.get("exit_reached", False)),
                            }
                        )
                        + "\n"
                    )

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
            acts_np = np.stack(actions_hist, axis=0) if actions_hist else np.zeros((1, S.ACTION_DIM), dtype=np.int8)
            lat_np = np.asarray(latencies, dtype=np.float64) if latencies else np.zeros(1, dtype=np.float64)
            first_act_ms = float(latencies[0]) if latencies else 0.0
            max_sub_act_ms = float(np.max(latencies[1:])) if len(latencies) > 1 else 0.0
            mean_lat_ms = float(np.mean(lat_np))

            constraints, violations = _evaluate_track_constraints(
                init_ms=init_ms,
                reset_ms=reset_ms,
                first_act_ms=first_act_ms,
                max_subsequent_act_ms=max_sub_act_ms,
                mean_latency_ms=mean_lat_ms,
                peak_ram_mb=peak_ram_mb,
                peak_vram_mb=peak_vram_mb,
            )
            if hard_violation:
                outcome = "constraint_violation"
            elif user_quit:
                outcome = "user_quit"
            elif last_priv.get("exit_reached", False):
                outcome = "exit_reached"
            elif last_priv.get("dead", False):
                outcome = "dead"
            else:
                outcome = "tick_cap"

            summary = {
                "wad_path": resolved_wad,
                "map_name": map_name,
                "outcome": outcome,
                "steps": step_num,
                "wall_duration_s": round(wall_duration, 2),
                "init_ms": round(init_ms, 2),
                "reset_ms": round(reset_ms, 2),
                "first_act_ms": round(first_act_ms, 3),
                "max_subsequent_act_ms": round(max_sub_act_ms, 3),
                "mean_latency_ms": round(mean_lat_ms, 3),
                "p50_latency_ms": round(float(np.percentile(lat_np, 50)), 3),
                "p99_latency_ms": round(float(np.percentile(lat_np, 99)), 3),
                "max_latency_ms": round(float(np.max(lat_np)), 3),
                "peak_ram_mb": round(peak_ram_mb, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "kills": int(last_priv.get("kills", 0)),
                "unique_cells": int(info.get("unique_cells_visited", 0)) if step_num > 0 else 0,
                "keys_collected": int(np.sum(last_priv.get("keys", 0))) if "keys" in last_priv else 0,
                "wall_close_steps": wall_close_steps,
                "zero_ammo_steps": zero_ammo_steps,
                "total_shaped_reward": round(total_ep_reward, 3),
                "button_rates": {b: round(float(acts_np[:, i].mean()), 4) for i, b in enumerate(S.BUTTONS)},
                "constraints": constraints,
                "constraint_violations": violations,
                "all_constraints_passed": len(violations) == 0,
                "log_path": str(log_path),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            c_status = "PASS" if len(violations) == 0 else f"FAIL ({', '.join(violations)})"
            print(
                f"Episode finished in {step_num} steps ({wall_duration:.2f}s) | outcome={outcome} | "
                f"reward={total_ep_reward:.2f} | kills={summary['kills']} cells={summary['unique_cells']}"
            )
            print(
                f"Track Constraints [{c_status}]: init={init_ms:.1f}ms/60s | reset={reset_ms:.1f}ms/10s | "
                f"1st_act={first_act_ms:.2f}ms/10s | max_act={max_sub_act_ms:.2f}ms/5s | "
                f"avg={mean_lat_ms:.2f}ms/40ms | RAM={peak_ram_mb:.1f}MB | VRAM={peak_vram_mb:.1f}MB"
            )
            print(f"Summary saved -> {summary_path}")

            if html_target is not None and html_frames:
                _write_dashboard_html(html_frames, html_target, Path(resolved_wad).name, step_num, total_ep_reward)
                print(f"Interactive replay saved to {html_target} ({len(html_frames)} frames)")

            if window and not user_quit:
                dashboard.render_end_screen(outcome, summary)
                print("Episode ended. Press R to Restart, or ESC / Q / Close window to exit.")
                restart = False
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                        ):
                            waiting = False
                            break
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            restart = True
                            waiting = False
                            break
                    pygame.time.wait(50)
                if restart:
                    html_frames.clear()
                    continue
            break

        if strict_constraints and violations:
            raise RuntimeError(f"Track constraint violations: {violations}")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved successfully to {output_mp4} ({Path(output_mp4).stat().st_size // 1024} KB)")
        env.close()
        dashboard.close()

    return {
        "steps": step_num,
        "total_reward": total_ep_reward,
        "outcome": summary.get("outcome"),
        "all_constraints_passed": summary.get("all_constraints_passed", True),
        "html_path": str(html_target) if html_target else None,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDKFA DOOM Agent Live Dashboard & Recorder")
    parser.add_argument("--map", type=str, default="freedoom2", help="Map WAD path or 'freedoom2'")
    parser.add_argument("--map-name", type=str, default="MAP01", help="Map lump name inside WAD (e.g. MAP01, MAP02)")
    parser.add_argument("--ticks", type=int, default=4200, help="Episode ticks (35 ticks = 1s, track max = 4200)")
    parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--window", action="store_true", help="Open live desktop window")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 filename (falls back to HTML if cv2 missing)")
    parser.add_argument("--save-html", type=str, default=None, help="Output standalone HTML composite replay")
    parser.add_argument("--stride", type=int, default=2, help="Capture every Nth step into HTML")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    parser.add_argument("--log-file", type=str, default=None, help="Override auto JSONL telemetry log path")
    parser.add_argument("--strict-constraints", action="store_true", help="Abort / fail if any track limit is exceeded")
    args = parser.parse_args()

    run_demo(
        wad_path=args.map,
        map_name=args.map_name,
        ticks=args.ticks,
        window=args.window,
        output_mp4=args.output,
        save_html=args.save_html,
        fps=args.fps,
        seed=args.seed,
        device=args.device,
        stride=args.stride,
        log_file=args.log_file,
        strict_constraints=args.strict_constraints,
    )
