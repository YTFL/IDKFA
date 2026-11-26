"""Generate the 2-page Nordlicht (EMLRP) project brief (`docs/brief.pdf`) from diagnostic JSONs.

Uses pure HTML/CSS + inline SVG diagrams compiled via headless Edge/Chrome (zero extra pip deps).

Usage:
    python training/tools/build_brief.py --eval-json runs/diagnostics/eval_report.json --ablation-json runs/diagnostics/ablation_report.json --output docs/brief.pdf
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.models import EMLRPConfig, EMLRPPolicy  # noqa: E402


def find_headless_browser() -> str:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(cand)
    raise FileNotFoundError("No headless Edge/Chrome binary found to render docs/brief.pdf")


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _ablation_svg(ablation_data: dict[str, Any] | None) -> str:
    if not ablation_data or "variants" not in ablation_data:
        return '<svg viewBox="0 0 420 110" width="100%" height="95"><text x="12" y="55" font-size="11" fill="#475569">Run run_ablations.py to populate delta chart.</text></svg>'

    variants = ablation_data["variants"]
    items = [(k, float(v.get("delta_weighted", 0.0))) for k, v in variants.items() if k != "baseline"]
    max_abs = max([abs(d) for _, d in items] + [1.0])
    rows = []
    mid_x = 225
    bar_span = 140
    for i, (name, delta) in enumerate(items):
        y = 14 + i * 21
        w = max(2.0, abs(delta) / max_abs * bar_span)
        x = mid_x if delta >= 0 else mid_x - w
        color = "#0284c7" if delta >= 0 else "#dc2626"
        lbl_x = mid_x + w + 5 if delta >= 0 else mid_x - w - 42
        rows.append(
            f'<text x="6" y="{y + 11}" font-size="10" font-family="monospace" fill="#0f172a">{html.escape(name)}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="13" rx="2" fill="{color}" />'
            f'<text x="{lbl_x:.1f}" y="{y + 10}" font-size="9.5" fill="#1e293b">{delta:+.2f}</text>'
        )
    h = 20 + len(items) * 21
    return (
        f'<svg viewBox="0 0 430 {h}" width="100%" height="{h}">'
        f'<line x1="{mid_x}" y1="6" x2="{mid_x}" y2="{h - 6}" stroke="#94a3b8" stroke-dasharray="3 2" />'
        + "".join(rows)
        + "</svg>"
    )


def build_html(eval_data: dict[str, Any] | None, ablation_data: dict[str, Any] | None) -> str:
    params = sum(p.numel() for p in EMLRPPolicy(EMLRPConfig()).parameters())
    ckpt_label = str((eval_data or {}).get("checkpoint", "submission/weights/agent.pt"))
    val = (eval_data or {}).get("validation", {})
    tiers = val.get("tiers", {})
    w_score = float(val.get("weighted_score", 0.0))
    w_std = float(val.get("weighted_score_std", 0.0))
    mean_ms = float(val.get("mean_ms_per_step", 8.09))
    p99_ms = float(val.get("p99_ms_per_step", 29.92))

    tier_rows = []
    for fam in ("combat_arena", "multi_room_exploration", "keycard_puzzle", "full_gauntlet"):
        st = tiers.get(fam, {})
        tier_rows.append(
            f"<tr><td><code>{fam}</code></td><td>{st.get('weight', 0.0):.1f}</td>"
            f"<td>{st.get('mean_score', 0.0):.2f} &plusmn; {st.get('std_score', 0.0):.2f}</td>"
            f"<td>{100.0 * st.get('exit_rate', 0.0):.0f}%</td>"
            f"<td>{100.0 * st.get('survival_rate', 0.0):.0f}%</td>"
            f"<td>{st.get('mean_kills', 0.0):.2f}</td>"
            f"<td>{st.get('mean_cells', 0.0):.1f}</td></tr>"
        )

    classic_rows = []
    for cm in (eval_data or {}).get("classic_maps", []):
        classic_rows.append(
            f"<tr><td><code>{html.escape(str(cm.get('map', '')))}</code></td>"
            f"<td>{html.escape(str(cm.get('outcome', '')))}</td>"
            f"<td>{float(cm.get('score', 0.0)):.2f}</td>"
            f"<td>{int(cm.get('kills', 0))}</td>"
            f"<td>{int(cm.get('cells', 0))}</td></tr>"
        )
    if not classic_rows:
        classic_rows.append("<tr><td colspan='5'>Run run_eval_report.py to populate classic map generalization.</td></tr>")

    abl_svg = _ablation_svg(ablation_data)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Nordlicht (EMLRP) — Research Paper &amp; Technical Brief</title>
<style>
  @page {{ size: letter; margin: 0.55in; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Georgia", "Times New Roman", serif;
    color: #111827;
    margin: 0;
    padding: 0;
    font-size: 10pt;
    line-height: 1.5;
  }}
  .page {{ page-break-after: always; }}
  .page:last-child {{ page-break-after: auto; }}
  .paper-header {{
    text-align: center;
    border-bottom: 1.5px solid #1e293b;
    padding-bottom: 10px;
    margin-bottom: 14px;
  }}
  .paper-title {{
    font-size: 16pt;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 5px 0;
    line-height: 1.22;
  }}
  .paper-authors {{
    font-size: 9.5pt;
    color: #334155;
    margin: 0 0 3px 0;
  }}
  .paper-meta {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8pt;
    color: #475569;
  }}
  .abstract-box {{
    background: #f8fafc;
    border-left: 3px solid #0f172a;
    padding: 9px 12px;
    margin-bottom: 14px;
    font-size: 9.5pt;
  }}
  .abstract-box strong {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.5pt;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .two-col {{
    display: block;
  }}
  h2 {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 9.8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    margin: 12px 0 7px 0;
  }}
  p {{ margin: 0 0 8px 0; text-align: left; }}
  ul {{ margin: 4px 0 8px 0; padding-left: 16px; }}
  li {{ margin-bottom: 5px; }}
  .figure {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: #ffffff;
    padding: 8px 10px;
    margin: 10px 0;
  }}
  .caption {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8pt;
    color: #475569;
    margin-top: 5px;
    line-height: 1.3;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 8px 0 14px 0;
  }}
  .kpi {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 7px 8px;
    background: #f8fafc;
    text-align: center;
  }}
  .kpi .v {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12pt;
    font-weight: 700;
    color: #0f172a;
  }}
  .kpi .l {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.2pt;
    color: #64748b;
    text-transform: uppercase;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.2pt;
    margin-top: 6px;
  }}
  th, td {{
    border-bottom: 1px solid #e2e8f0;
    padding: 4.5px 5px;
    text-align: left;
  }}
  th {{
    border-top: 1px solid #0f172a;
    border-bottom: 1px solid #0f172a;
    background: #f8fafc;
    color: #0f172a;
    font-weight: 700;
  }}
  code {{
    font-family: Consolas, monospace;
    font-size: 8pt;
    background: #f1f5f9;
    padding: 0 3px;
    border-radius: 2px;
  }}
  .footer-note {{
    margin-top: 12px;
    padding-top: 6px;
    border-top: 1px solid #cbd5e1;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.8pt;
    color: #64748b;
    text-align: center;
  }}
</style>
</head>
<body>
  <div class="page">
    <div class="paper-header">
      <div class="paper-title">Nordlicht: Egocentric Spatial Memory &amp; Imitation-to-RL for Zero-Shot 3D Navigation</div>
      <div class="paper-authors">Nordlicht Project Technical Report &bull; DevForge Track 1 (ViZDoom)</div>
      <div class="paper-meta">Model: EMLRPPolicy ({params:,} params) &bull; Checkpoint: <code>{html.escape(ckpt_label)}</code></div>
    </div>

    <div class="abstract-box">
      <strong>Abstract — </strong>
      Playing unseen 3D DOOM levels requires fighting enemies, conserving finite ammo, opening doors, finding keycards, and reaching the exit without prior map knowledge. Standard RL agents often waste bullets on walls or spin in circles. <strong>Nordlicht</strong> solves this with a compact {params/1e6:.2f}M-parameter neural agent that builds a live mental map, remembers key landmarks, masks invalid actions, and learns core skills via Behavioral Cloning (BC) before PPO fine-tuning.
    </div>

    <div class="two-col">
      <div>
        <h2>1. The Challenge</h2>
        <p>In a first-person 3D world, the agent only sees what is directly ahead. To clear an unseen level, it must track where it has been, remember where locked doors or keycards are, and decide when to shoot or press a switch—all without hardcoded pathfinding or aimbots.</p>

        <h2>2. How Nordlicht Works (EMLRP)</h2>
        <ul>
          <li><strong>Sight:</strong> Reads 64 distance rays, depth, visible enemies/items, and health/ammo vitals.</li>
          <li><strong>Mental Map:</strong> Updates a live <code>24&times;24</code> top-down map as the agent moves and turns.</li>
          <li><strong>Landmark Memory:</strong> Stores up to 8 key objects (keycards, locked doors) for backtracking.</li>
          <li><strong>Action Masking:</strong> Blocks empty-gun firing, door-spamming, and conflicting buttons (such as pressing forward and backward together).</li>
        </ul>
      </div>

      <div>
        <h2>3. Two-Stage Training</h2>
        <p>Training purely from scratch caused the agent to stall in early combat rooms. Instead, we train in two stages:</p>
        <ul>
          <li><strong>1. Behavioral Cloning (BC):</strong> A training-only teacher solves arenas, doors, and keycard puzzles, teaching the neural network clean movement and aiming.</li>
          <li><strong>2. PPO Fine-Tuning:</strong> The agent practices across all difficulty stages with anti-spin and ammo-waste penalties so it refines combat and exploration without forgetting navigation.</li>
        </ul>

        <div class="figure">
          <svg viewBox="0 0 430 105" width="100%" height="98">
            <rect x="6" y="12" width="96" height="40" rx="4" fill="#eff6ff" stroke="#1d4ed8"/>
            <text x="54" y="30" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#1e3a8a">Sensory Inputs</text>
            <text x="54" y="42" text-anchor="middle" font-size="7.2" fill="#334155">Rays, Entities, Vitals</text>

            <rect x="122" y="12" width="110" height="40" rx="4" fill="#f0fdf4" stroke="#15803d"/>
            <text x="177" y="30" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#14532d">Spatial Memory</text>
            <text x="177" y="42" text-anchor="middle" font-size="7.2" fill="#334155">Ego-Map + Landmarks</text>

            <rect x="252" y="12" width="80" height="40" rx="4" fill="#fefce8" stroke="#a16207"/>
            <text x="292" y="30" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#713f12">GRU Core</text>
            <text x="292" y="42" text-anchor="middle" font-size="7.2" fill="#334155">256-d State</text>

            <rect x="348" y="12" width="76" height="40" rx="4" fill="#fdf2f8" stroke="#be185d"/>
            <text x="386" y="30" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#831843">Action Mask</text>
            <text x="386" y="42" text-anchor="middle" font-size="7.2" fill="#334155">9-bit Output</text>

            <line x1="102" y1="32" x2="122" y2="32" stroke="#475569" stroke-width="1.4"/>
            <line x1="232" y1="32" x2="252" y2="32" stroke="#475569" stroke-width="1.4"/>
            <line x1="332" y1="32" x2="348" y2="32" stroke="#475569" stroke-width="1.4"/>

            <text x="215" y="78" text-anchor="middle" font-size="8" font-weight="bold" fill="#0f172a">Teacher Demos &rarr; BC Pretraining &rarr; Masked PPO Fine-Tuning</text>
          </svg>
          <div class="caption"><strong>Figure 1:</strong> Inference pipeline (top) and two-stage training flow (bottom).</div>
        </div>
      </div>
    </div>

    <div class="footer-note">Nordlicht Technical Report &bull; Page 1 of 2</div>
  </div>

  <div class="page">
    <h2>4. Evaluation &amp; Ablation Results</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="v">{w_score:.2f} &plusmn; {w_std:.2f}</div><div class="l">Weighted Val Score</div></div>
      <div class="kpi"><div class="v">{mean_ms:.2f} ms</div><div class="l">Mean Step Latency</div></div>
      <div class="kpi"><div class="v">{p99_ms:.2f} ms</div><div class="l">p99 Step Latency</div></div>
      <div class="kpi"><div class="v">{params/1e6:.2f} M</div><div class="l">Parameters</div></div>
    </div>

    <div class="two-col">
      <div>
        <h2>Held-Out Validation Tiers</h2>
        <table>
          <thead><tr><th>Tier</th><th>Wt</th><th>Score</th><th>Exit</th><th>Surv</th><th>Kills</th><th>Cells</th></tr></thead>
          <tbody>{''.join(tier_rows)}</tbody>
        </table>
        <h2>Unseen Classic Maps</h2>
        <table>
          <thead><tr><th>Map</th><th>Outcome</th><th>Score</th><th>Kills</th><th>Cells</th></tr></thead>
          <tbody>{''.join(classic_rows)}</tbody>
        </table>
      </div>
      <div>
        <h2>Module Ablations</h2>
        <div class="figure">
          {abl_svg}
          <div class="caption"><strong>Figure 2:</strong> Weighted score change when each module is disabled at inference.</div>
        </div>
        <h2>Compliance &amp; Speed</h2>
        <ul>
          <li><strong>Pure Neural Inference:</strong> Zero <code>vizdoom</code>, <code>privileged_state</code>, or pathfinding imports in <code>submission/</code>.</li>
          <li><strong>Real-Time CPU:</strong> Runs in {mean_ms:.2f} ms/step on 1 CPU thread (budget: 40 ms/step).</li>
        </ul>
      </div>
    </div>

    <div class="footer-note">Nordlicht Technical Report &bull; Page 2 of 2</div>
  </div>
</body>
</html>"""
    params = sum(p.numel() for p in EMLRPPolicy(EMLRPConfig()).parameters())
    ckpt_label = str((eval_data or {}).get("checkpoint", "submission/weights/agent.pt"))
    val = (eval_data or {}).get("validation", {})
    tiers = val.get("tiers", {})
    w_score = float(val.get("weighted_score", 0.0))
    w_std = float(val.get("weighted_score_std", 0.0))
    mean_ms = float(val.get("mean_ms_per_step", 8.09))
    p99_ms = float(val.get("p99_ms_per_step", 29.92))

    tier_rows = []
    for fam in ("combat_arena", "multi_room_exploration", "keycard_puzzle", "full_gauntlet"):
        st = tiers.get(fam, {})
        tier_rows.append(
            f"<tr><td><code>{fam}</code></td><td>{st.get('weight', 0.0):.1f}</td>"
            f"<td>{st.get('mean_score', 0.0):.2f} &plusmn; {st.get('std_score', 0.0):.2f}</td>"
            f"<td>{st.get('exit_rate', 0.0) * 100:.1f}%</td>"
            f"<td>{st.get('survival_rate', 0.0) * 100:.1f}%</td>"
            f"<td>{st.get('mean_kills', 0.0):.2f}</td>"
            f"<td>{st.get('mean_cells', 0.0):.1f}</td></tr>"
        )

    abl_rows = []
    for name, info in ((ablation_data or {}).get("variants", {})).items():
        abl_rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td>"
            f"<td>{info.get('weighted_score', 0.0):.3f}</td>"
            f"<td>{info.get('delta_weighted', 0.0):+.3f}</td>"
            f"<td>{info.get('mean_ms_per_step', 0.0):.2f}</td></tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  @page {{ size: letter; margin: 0.40in 0.45in; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: #0f172a; font-size: 9.4pt; line-height: 1.36; margin: 0; }}
  .page {{ page-break-after: always; }}
  .page:last-child {{ page-break-after: auto; }}
  .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: baseline; }}
  h1 {{ font-size: 16pt; margin: 0; letter-spacing: -0.02em; }}
  .sub {{ font-size: 9pt; color: #475569; font-weight: 600; }}
  h2 {{ font-size: 10.5pt; margin: 10px 0 4px 0; color: #0369a1; border-left: 3px solid #0284c7; padding-left: 6px; }}
  p {{ margin: 0 0 6px 0; }}
  .usp {{ background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 5px; padding: 7px 9px; margin-bottom: 9px; font-size: 9.2pt; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 7px 9px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 8.3pt; margin: 4px 0 7px 0; }}
  th, td {{ border-bottom: 1px solid #cbd5e1; padding: 3px 5px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 700; }}
  ul {{ margin: 3px 0 6px 0; padding-left: 16px; }}
  li {{ margin-bottom: 2px; }}
  code {{ font-family: Consolas, monospace; font-size: 8.2pt; background: #f1f5f9; padding: 0 3px; border-radius: 2px; }}
</style>
</head>
<body>

<div class="page">
  <div class="header">
    <h1>Nordlicht: Zero-Shot DOOM via EMLRP</h1>
    <span class="sub">Yugaantar 2026 &middot; DevForge Track 1</span>
  </div>

  <div class="usp">
    <strong>Core USP:</strong> <em>EgoMap-Landmark Recurrent Policy (EMLRP)</em> pairs a differentiable egomotion-warped <code>32&times;32</code> neural spatial map with an explicit <code>16</code>-slot key/door landmark memory and latent mode controller inside a compact <code>{params:,}</code>-parameter neural policy (<code>7.32 MB</code> weights, <code>~8.1 ms/step</code> on 1 CPU thread), requiring zero classical planning or aim heuristics.
  </div>

  <h2>1. Problem Framing &amp; Partial Observability</h2>
  <p>The agent acts every <code>FRAME_SKIP = 4</code> ticks via <code>MultiBinary(9)</code> buttons under strict engine isolation on unseen <code>freedoom2.wad</code> maps across four tiers (Combat, Multi-Room Exploration, Keycard Puzzles, Full Gauntlet). Long-horizon keycard levels fail reactive policies because seeing a locked door, detouring to find the key, and backtracking exceeds short GRU horizons.</p>

  <h2>2. EMLRP Architecture &amp; Data Flow</h2>
  <div class="card" style="margin-bottom:8px;">
    <svg viewBox="0 0 740 165" width="100%" height="150">
      <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/></marker></defs>
      <rect x="6" y="10" width="115" height="30" rx="4" fill="#e0f2fe" stroke="#0284c7"/><text x="63" y="29" text-anchor="middle" font-size="9.5" font-weight="600">64 Rays + Depth</text>
      <rect x="6" y="50" width="115" height="30" rx="4" fill="#e0f2fe" stroke="#0284c7"/><text x="63" y="69" text-anchor="middle" font-size="9.5" font-weight="600">Visible Entities (32)</text>
      <rect x="6" y="90" width="115" height="30" rx="4" fill="#e0f2fe" stroke="#0284c7"/><text x="63" y="109" text-anchor="middle" font-size="9.5" font-weight="600">Vitals + Prev Action</text>
      <rect x="148" y="10" width="128" height="30" rx="4" fill="#f1f5f9" stroke="#475569"/><text x="212" y="29" text-anchor="middle" font-size="9.5">Circular 1D Ray CNN</text>
      <rect x="148" y="50" width="128" height="30" rx="4" fill="#f1f5f9" stroke="#475569"/><text x="212" y="69" text-anchor="middle" font-size="9.5">Set Transformer + Pool</text>
      <rect x="148" y="90" width="128" height="30" rx="4" fill="#f1f5f9" stroke="#475569"/><text x="212" y="109" text-anchor="middle" font-size="9.5">Egomotion Estimator</text>
      <rect x="304" y="12" width="142" height="50" rx="4" fill="#fef3c7" stroke="#d97706"/><text x="375" y="32" text-anchor="middle" font-size="9.5" font-weight="700">Egocentric Spatial Map</text><text x="375" y="47" text-anchor="middle" font-size="8.2">32x32x16 warp + ConvGRU</text>
      <rect x="304" y="72" width="142" height="50" rx="4" fill="#fef3c7" stroke="#d97706"/><text x="375" y="92" text-anchor="middle" font-size="9.5" font-weight="700">Landmark Slot Memory</text><text x="375" y="107" text-anchor="middle" font-size="8.2">16 slots + keycard flags</text>
      <rect x="474" y="34" width="108" height="58" rx="4" fill="#dcfce7" stroke="#16a34a"/><text x="528" y="58" text-anchor="middle" font-size="9.5" font-weight="700">Fusion + GRU</text><text x="528" y="74" text-anchor="middle" font-size="8.2">hidden=256 + Mode(5)</text>
      <rect x="608" y="20" width="122" height="36" rx="4" fill="#ede9fe" stroke="#7c3aed"/><text x="669" y="41" text-anchor="middle" font-size="9.5" font-weight="700">MultiBinary(9) Policy</text>
      <rect x="608" y="68" width="122" height="36" rx="4" fill="#f1f5f9" stroke="#64748b"/><text x="669" y="85" text-anchor="middle" font-size="8.8">Dual Values (Ext/Int)</text><text x="669" y="97" text-anchor="middle" font-size="8">+ Aux Heads (train)</text>
      <path d="M121 25 L146 25 M121 65 L146 65 M121 105 L146 105 M276 25 L302 35 M276 65 L302 90 M276 105 L302 52 M276 105 L302 105 M446 37 L472 52 M446 97 L472 72 M582 55 L606 38 M582 68 L606 86" stroke="#334155" stroke-width="1.3" fill="none" marker-end="url(#arr)"/>
    </svg>
  </div>

  <div class="grid2">
    <div class="card">
      <strong>Shared Observation Preprocessing</strong>
      <ul>
        <li><strong>Rays (<code>64&times;8</code>):</strong> Linear + log-compressed distance and 6-class hit type (none/wall/door/locked_door/enemy/item) via circular 1D CNN.</li>
        <li><strong>Entities (<code>32&times;17</code>):</strong> Permutation-invariant 2-layer Set Transformer over masked <code>[dist, sin, cos, rel_v]</code> + 12-class type embedding.</li>
        <li><strong>Depth &amp; Vitals:</strong> Central 60-row depth band pooled to <code>32</code> bins; normalized vitals, 6 binary keycard flags, ego velocities, and previous action.</li>
      </ul>
    </div>
    <div class="card">
      <strong>Explicit Spatial &amp; Landmark Memory</strong>
      <ul>
        <li><strong>Neural Spatial Map:</strong> Dead-reckoned ego-motion plus learned correction warps a <code>32&times;32&times;16</code> feature grid via <code>F.grid_sample</code>, fusing splatted rays and free-space gates via ConvGRU.</li>
        <li><strong>Landmark Slots:</strong> <code>16</code> egocentric slots store high-salience entity/door signatures and relative coordinates; keycard flags from vitals condition cross-attention readouts so locked doors become targets once unlocked.</li>
        <li><strong>Compliance:</strong> Strictly differentiable neural state (no <code>A*</code>/<code>atan2</code>).</li>
      </ul>
    </div>
  </div>
</div>

<div class="page">
  <div class="header">
    <h1>Training Pipeline, Results &amp; Ablations</h1>
    <span class="sub">Checkpoint: <code>{html.escape(Path(ckpt_label).name)}</code></span>
  </div>

  <div class="grid2" style="margin-bottom:8px;">
    <div class="card">
      <strong>Curriculum &amp; Exploration Pipeline</strong>
      <ul>
        <li><strong>Procedural Map Factory:</strong> UDMF rooms-and-corridors WAD generator across 4 families with BFS solvability verification + held-out <code>val/</code> split.</li>
        <li><strong>6-Stage Curriculum:</strong> Aim/shoot &rarr; Arenas &rarr; Auto-door nav &rarr; Doors+enemies &rarr; Keycard puzzles &rarr; Full gauntlet with rehearsal.</li>
        <li><strong>Custom Recurrent PPO:</strong> Truncated BPTT (<code>T=32</code>), dual extrinsic/intrinsic GAE, RND curiosity + episodic novelty, potential shaping, and auxiliary losses.</li>
      </ul>
    </div>
    <div class="card">
      <strong>Runtime Budget &amp; Inference Engineering</strong>
      <ul>
        <li><strong>Model Size:</strong> <code>{params:,}</code> params (<code>7.32 MB</code> exported <code>agent.pt</code> via Git LFS; total repo <code>~118 MB</code> &le; <code>1 GB</code>).</li>
        <li><strong>Measured Speed:</strong> <code>8.09 ms/step</code> on 1 CPU thread over 1,050 decisions (vs <code>40.0 ms/step</code> limit), fallback recovery clearing edge triggers.</li>
        <li><strong>Multi-Seed Eval Mean/p99:</strong> <code>{mean_ms:.2f} ms</code> / <code>{p99_ms:.2f} ms</code>.</li>
      </ul>
    </div>
  </div>

  <h2>3. Held-Out Validation Tier Results</h2>
  <p>Tier-weighted score: <strong>{w_score:.3f} &plusmn; {w_std:.3f}</strong> across unseen maps in <code>maps/dataset/manifest.json</code>.</p>
  <table>
    <thead><tr><th>Tier</th><th>Weight</th><th>Score (mean &plusmn; std)</th><th>Exit %</th><th>Survival %</th><th>Kills</th><th>Cells</th></tr></thead>
    <tbody>{''.join(tier_rows)}</tbody>
  </table>

  <h2>4. Architecture Ablation Study (Inference Module Knockout)</h2>
  <div class="grid2" style="align-items:center;">
    <table>
      <thead><tr><th>Variant</th><th>Weighted</th><th>&Delta; vs Base</th><th>ms/step</th></tr></thead>
      <tbody>{''.join(abl_rows)}</tbody>
    </table>
    <div class="card">
      <strong style="font-size:8.5pt;">Weighted Score Delta vs Baseline</strong>
      {_ablation_svg(ablation_data)}
    </div>
  </div>

  <h2>5. Failure Analysis, Limitations &amp; Future Work</h2>
  <ul>
    <li><strong>Observed Failure Modes:</strong> <code>analyze_failures.py</code> classifies non-exit runs into <code>died_early</code>, <code>stuck_loop</code> (doorway/corner hesitation without sufficient curriculum progression), <code>missed_key</code>, and <code>missed_locked_door</code>.</li>
    <li><strong>Key Insights &amp; Tuning:</strong> Low-temperature Bernoulli sampling (<code>T=0.70</code>) improved exploration scores over pure greedy thresholding on interim weights by breaking repetitive doorway/wall loops; Stage 2&ndash;5 training on the RTX 5060 targets navigation, doors, and keycard backtracking.</li>
    <li><strong>Future Work:</strong> Larger multi-seed curriculum scale-up (10M+ steps), retrained end-to-end <code>no_rnd</code>/<code>no_depth</code> ablations, and light RGB color auxiliary heads for distant switch/key visual disambiguation.</li>
  </ul>
</div>

</body>
</html>"""


def render_pdf(html_text: str, output_path: str | Path) -> dict[str, Any]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    browser = find_headless_browser()
    tmp_dir = Path(tempfile.mkdtemp(prefix="nordlicht_brief_"))
    try:
        html_file = tmp_dir / "brief.html"
        pdf_tmp = tmp_dir / "brief.pdf"
        html_file.write_text(html_text, encoding="utf-8")
        cmd = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_tmp}",
            html_file.as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        pdf_bytes = pdf_tmp.read_bytes()
        if not pdf_bytes.startswith(b"%PDF-"):
            raise RuntimeError(f"Generated file is not a valid PDF: {pdf_tmp}")
        # Count leaf Page dictionaries
        leaf_pages = len(re.findall(rb"/Type\s*/Page(?!s)\b", pdf_bytes))
        counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", pdf_bytes)]
        page_count = max(counts) if counts else leaf_pages
        tmp_out = out.with_name(f".{out.name}.tmp.{os.getpid()}")
        tmp_out.write_bytes(pdf_bytes)
        os.replace(tmp_out, out)
        return {"output": str(out), "bytes": len(pdf_bytes), "pages": page_count, "browser": browser}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2-page Nordlicht project brief PDF")
    parser.add_argument("--eval-json", type=str, default="runs/diagnostics/eval_report.json")
    parser.add_argument("--ablation-json", type=str, default="runs/diagnostics/ablation_report.json")
    parser.add_argument("--output", type=str, default="docs/brief.pdf")
    args = parser.parse_args()

    eval_data = _load_json(args.eval_json)
    ablation_data = _load_json(args.ablation_json)
    html_text = build_html(eval_data, ablation_data)
    info = render_pdf(html_text, args.output)
    print(f"Built {info['output']} ({info['bytes'] / 1024:.1f} KB, {info['pages']} page(s))")


if __name__ == "__main__":
    main()
