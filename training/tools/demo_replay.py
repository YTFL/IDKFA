"""Live keycard/navigation Q&A demo and self-contained HTML replay exporter.

Renders EMLRPPolicy's egocentric spatial map, active landmark slots, and latent modes
in real time and exports a standalone interactive HTML player (zero extra pip deps).

Usage:
    python training/tools/demo_replay.py --checkpoint submission/weights/agent.pt --save-html runs/diagnostics/demo_replay.html
    python training/tools/demo_replay.py --map maps/two_rooms.wad --ticks 400 --terminal
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from doom_env import spaces as S  # noqa: E402
from submission.models import processed_to_tensors  # noqa: E402
from training.eval import load_policy_from_checkpoint  # noqa: E402
from training.rewards.shaping import ExtrinsicRewardShaper, RewardConfig  # noqa: E402

_ASCII_RAMP = " .:-=+*#%@"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def rgb_array_to_png_b64(rgb: np.ndarray) -> str:
    """Encode an `(H, W, 3) uint8` RGB array into base64 PNG using stdlib zlib only."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    raw_rows = b"".join(b"\x00" + arr[r].tobytes() for r in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png_bytes).decode("ascii")


def spatial_map_to_rgb(spatial_map: np.ndarray, scale: int = 4) -> np.ndarray:
    """Convert `(C, 32, 32)` spatial map features into an upscaled RGB heatmap."""
    energy = np.sqrt(np.mean(np.square(spatial_map), axis=0))
    mx = float(energy.max())
    norm = energy / mx if mx > 1e-6 else np.zeros_like(energy)
    # Dark slate -> cyan -> amber palette
    r = np.clip(norm * 255.0 * 1.1, 16, 255).astype(np.uint8)
    g = np.clip(norm * 210.0, 22, 235).astype(np.uint8)
    b = np.clip((1.0 - norm * 0.5) * 140.0 + norm * 115.0, 36, 255).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    # Mark egocentric center (agent location)
    c = img.shape[0] // 2
    img[c - 1 : c + 1, c - 1 : c + 1] = np.array([255, 80, 80], dtype=np.uint8)
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def spatial_map_to_ascii(spatial_map: np.ndarray, size: int = 16) -> str:
    energy = np.sqrt(np.mean(np.square(spatial_map), axis=0))
    step = max(1, energy.shape[0] // size)
    down = energy[: size * step, : size * step].reshape(size, step, size, step).mean(axis=(1, 3))
    mx = float(down.max())
    norm = down / mx if mx > 1e-6 else np.zeros_like(down)
    lines = []
    center = size // 2
    for r in range(size):
        row_chars = []
        for c in range(size):
            if r == center and c == center:
                row_chars.append("A")
            else:
                idx = min(len(_ASCII_RAMP) - 1, int(norm[r, c] * (len(_ASCII_RAMP) - 1)))
                row_chars.append(_ASCII_RAMP[idx])
        lines.append("".join(row_chars))
    return "\n".join(lines)


def _default_demo_map() -> Path:
    val_dir = ROOT / "maps" / "dataset" / "val"
    if val_dir.is_dir():
        key_maps = sorted(val_dir.glob("keycard_puzzle_*.wad"))
        if key_maps:
            return key_maps[0]
    return ROOT / "maps" / "two_rooms.wad"


def run_demo(
    checkpoint: str | Path,
    wad_path: str | Path | None = None,
    map_name: str = "MAP01",
    episode_ticks: int = 600,
    seed: int = 0,
    device: str = "cpu",
    temperature: float = 0.0,
    stride: int = 3,
    terminal: bool = False,
    record_lmp: str | Path | None = None,
) -> dict[str, Any]:
    wad = Path(wad_path) if wad_path is not None else _default_demo_map()
    if not wad.is_absolute():
        wad = ROOT / wad
    policy, prep, _ = load_policy_from_checkpoint(checkpoint, device=device)
    include_depth = policy.config.depth_feat_dim > 0
    env = make_env(str(wad), map_name, episode_ticks=episode_ticks, render_buffers=True)
    shaper = ExtrinsicRewardShaper(RewardConfig(step_penalty=0.0, stuck_penalty=0.0))

    frames: list[dict[str, Any]] = []
    try:
        rec = str(record_lmp) if record_lmp is not None else None
        raw_obs, _ = env.reset(seed=seed, options={"record": rec} if rec else None)
        prep.reset()
        shaper.reset()
        priv0 = env.unwrapped.privileged_state()
        shaper.last_health = float(priv0.get("health", 100.0))

        state = policy.init_state(1, device=device)
        prev_action = np.zeros(9, dtype=np.int8)
        done_t = torch.zeros(1, dtype=torch.bool, device=device)
        rng = torch.Generator(device=device).manual_seed(int(seed)) if temperature > 0.0 else None

        steps = 0
        total_score = 0.0
        exit_reached = False
        policy.eval()
        with torch.inference_mode():
            while True:
                p_obs = prep.process(raw_obs, prev_action=prev_action)
                obs_t, prev_act_t = processed_to_tensors(p_obs, device=device, include_depth=include_depth)
                out = policy.step(obs_t, state, prev_act_t, done=done_t, compute_aux=False)
                state = out.state
                if temperature > 0.0:
                    action = (
                        torch.bernoulli(torch.sigmoid(out.action_logits[0] / temperature), generator=rng)
                        .to(torch.int8)
                        .cpu()
                        .numpy()
                    )
                else:
                    action = (out.action_logits[0] > 0.0).to(torch.int8).cpu().numpy()

                raw_obs, base_r, term, trunc, info = env.step(action)
                priv = env.unwrapped.privileged_state()
                r_step, _ = shaper.compute(raw_obs, priv, base_r, term, trunc, info)
                total_score += float(r_step)

                smap_np = state.spatial_map[0].detach().cpu().numpy()
                lvalid = state.landmark_valid[0].detach().cpu().numpy()
                lpos = state.landmark_pos[0].detach().cpu().numpy()
                mode_probs = torch.softmax(out.mode_logits[0], dim=-1).detach().cpu().tolist()
                pressed = [S.BUTTONS[i] for i, v in enumerate(action) if int(v) == 1]

                if steps % max(1, stride) == 0 or term or trunc:
                    active_slots = [
                        {
                            "slot": int(i),
                            "valid": round(float(lvalid[i]), 3),
                            "x_fwd": round(float(lpos[i, 0]), 3),
                            "y_left": round(float(lpos[i, 1]), 3),
                        }
                        for i in range(len(lvalid))
                        if lvalid[i] > 0.05
                    ]
                    active_slots.sort(key=lambda d: -d["valid"])
                    frames.append(
                        {
                            "step": steps,
                            "health": round(float(priv.get("health", 0.0)), 1),
                            "kills": int(priv.get("kills", 0)),
                            "keys": len(shaper.keys_held),
                            "cells": len(shaper.visited_cells),
                            "score": round(total_score, 2),
                            "actions": pressed,
                            "mode_probs": [round(float(m), 3) for m in mode_probs],
                            "slots": active_slots[:6],
                            "rgb_b64": rgb_array_to_png_b64(raw_obs["rgb"]),
                            "map_b64": rgb_array_to_png_b64(spatial_map_to_rgb(smap_np)),
                        }
                    )
                    if terminal:
                        print(
                            f"\n[Step {steps:4d}] HP={priv.get('health', 0):.0f} Kills={priv.get('kills', 0)} "
                            f"Keys={len(shaper.keys_held)} Cells={len(shaper.visited_cells)} "
                            f"Score={total_score:.2f} Actions={','.join(pressed) or 'IDLE'}"
                        )
                        print(f"Modes: {[round(float(m), 2) for m in mode_probs]} | Active slots: {len(active_slots)}")
                        print(spatial_map_to_ascii(smap_np, size=12))

                prev_action = action
                steps += 1
                if term or trunc:
                    exit_reached = bool(info.get("exit_reached", False))
                    break
    finally:
        env.close()

    return {
        "map": f"{wad.name}:{map_name}",
        "checkpoint": str(checkpoint),
        "steps": steps,
        "exit_reached": exit_reached,
        "total_score": round(total_score, 3),
        "frames": frames,
    }


def write_replay_html(demo: dict[str, Any], out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(demo)
    title = html.escape(f"IDKFA EMLRP Demo — {demo['map']}")
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 24px; }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  .views {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; margin-top: 12px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }}
  img {{ width: 100%; image-rendering: pixelated; border-radius: 6px; background: #020617; }}
  .controls {{ display: flex; align-items: center; gap: 12px; margin: 14px 0; }}
  input[type=range] {{ flex: 1; }}
  button {{ background: #38bdf8; color: #0f172a; border: 0; border-radius: 6px; padding: 6px 14px; font-weight: 700; cursor: pointer; }}
  code {{ color: #7dd3fc; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th, td {{ text-align: left; padding: 4px 6px; border-bottom: 1px solid #334155; }}
</style>
</head>
<body>
<div class="wrap">
  <h2>{title}</h2>
  <p>Checkpoint: <code>{html.escape(demo['checkpoint'])}</code> | Steps: {demo['steps']} | Score: {demo['total_score']} | Exit: {demo['exit_reached']}</p>
  <div class="controls">
    <button id="playBtn">Play</button>
    <input id="scrubber" type="range" min="0" max="{max(0, len(demo['frames']) - 1)}" value="0" />
    <span id="stepLbl">0</span>
  </div>
  <div class="views">
    <div class="card">
      <strong>Agent RGB Observation</strong>
      <img id="rgbImg" />
      <p id="stats"></p>
    </div>
    <div class="card">
      <strong>32x32 Egocentric Neural Spatial Map</strong>
      <img id="mapImg" />
      <p id="modes"></p>
      <table><thead><tr><th>Slot</th><th>Valid</th><th>X fwd</th><th>Y left</th></tr></thead><tbody id="slotsBody"></tbody></table>
    </div>
  </div>
</div>
<script>
const DEMO = {payload};
let idx = 0, timer = null;
const scrub = document.getElementById('scrubber');
function render(i) {{
  idx = i; scrub.value = i;
  const f = DEMO.frames[i];
  if (!f) return;
  document.getElementById('stepLbl').textContent = `step ${{f.step}}`;
  document.getElementById('rgbImg').src = 'data:image/png;base64,' + f.rgb_b64;
  document.getElementById('mapImg').src = 'data:image/png;base64,' + f.map_b64;
  document.getElementById('stats').textContent = `HP ${{f.health}} | Kills ${{f.kills}} | Keys ${{f.keys}} | Cells ${{f.cells}} | Score ${{f.score}} | Actions: ${{f.actions.join(', ') || 'IDLE'}}`;
  document.getElementById('modes').textContent = 'Latent Mode Probs: ' + f.mode_probs.join(' / ');
  document.getElementById('slotsBody').innerHTML = f.slots.map(s => `<tr><td>#${{s.slot}}</td><td>${{s.valid}}</td><td>${{s.x_fwd}}</td><td>${{s.y_left}}</td></tr>`).join('');
}}
scrub.addEventListener('input', e => render(parseInt(e.target.value, 10)));
document.getElementById('playBtn').addEventListener('click', () => {{
  if (timer) {{ clearInterval(timer); timer = null; document.getElementById('playBtn').textContent = 'Play'; return; }}
  document.getElementById('playBtn').textContent = 'Pause';
  timer = setInterval(() => render((idx + 1) % DEMO.frames.length), 140);
}});
render(0);
</script>
</body>
</html>"""
    target.write_text(doc, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live memory visualization demo & export HTML replay")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/agent.pt")
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--map-name", type=str, default="MAP01")
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--save-html", type=str, default="runs/diagnostics/demo_replay.html")
    parser.add_argument("--record-lmp", type=str, default=None)
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    demo = run_demo(
        checkpoint=args.checkpoint,
        wad_path=args.map,
        map_name=args.map_name,
        episode_ticks=args.ticks,
        seed=args.seed,
        device=dev,
        temperature=args.temperature,
        stride=args.stride,
        terminal=args.terminal,
        record_lmp=args.record_lmp,
    )
    out = write_replay_html(demo, args.save_html)
    print(
        f"Demo complete: map={demo['map']} steps={demo['steps']} score={demo['total_score']} "
        f"exit={demo['exit_reached']} frames={len(demo['frames'])} -> {out}"
    )


if __name__ == "__main__":
    main()
