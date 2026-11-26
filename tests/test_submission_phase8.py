"""Tests for Phase 8: static compliance, clean-copy submission check, demo replay, and 2-page PDF brief."""

from __future__ import annotations

import re
from pathlib import Path

from training.tools.build_brief import build_html, render_pdf
from training.tools.demo_replay import run_demo, write_replay_html
from training.tools.pre_submit_check import check_submission_compliance, run_checks


def test_submission_static_compliance_and_isolation() -> None:
    violations = check_submission_compliance()
    assert violations == [], f"Compliance violations in submission/: {violations}"


def test_pre_submit_check_passes() -> None:
    res = run_checks()
    assert res["compliance_violations"] == []
    assert res["within_limit"] is True
    assert res["non_lfs_pt"] == []
    assert res["agent_pt_present"] is True
    assert res["requirements_errors"] == []
    assert res["clean_steps"] > 0
    assert res["passed"] is True


def test_demo_replay_generates_html_and_runs(tmp_path: Path) -> None:
    demo = run_demo(
        checkpoint="submission/weights/agent.pt",
        wad_path="maps/two_rooms.wad",
        episode_ticks=80,
        seed=0,
        device="cpu",
        stride=2,
    )
    assert demo["steps"] > 0
    assert len(demo["frames"]) >= 2
    out_html = write_replay_html(demo, tmp_path / "demo.html")
    assert out_html.is_file()
    text = out_html.read_text(encoding="utf-8")
    assert "32x32 Egocentric Neural Spatial Map" in text
    assert "data:image/png;base64," in text


def test_brief_pdf_renders_clean_two_pages(tmp_path: Path) -> None:
    sample_eval = {
        "checkpoint": "submission/weights/agent.pt",
        "validation": {
            "weighted_score": 1.323,
            "weighted_score_std": 7.066,
            "mean_ms_per_step": 13.47,
            "p99_ms_per_step": 29.92,
            "tiers": {
                "combat_arena": {"weight": 1.0, "mean_score": 0.84, "std_score": 0.22, "exit_rate": 0.0, "survival_rate": 1.0, "mean_kills": 0.75, "mean_cells": 32.5},
                "multi_room_exploration": {"weight": 1.5, "mean_score": -0.56, "std_score": 1.21, "exit_rate": 0.0, "survival_rate": 0.75, "mean_kills": 0.0, "mean_cells": 9.8},
                "keycard_puzzle": {"weight": 2.5, "mean_score": 0.13, "std_score": 0.06, "exit_rate": 0.0, "survival_rate": 1.0, "mean_kills": 0.0, "mean_cells": 6.8},
                "full_gauntlet": {"weight": 3.5, "mean_score": 0.28, "std_score": 1.95, "exit_rate": 0.0, "survival_rate": 0.75, "mean_kills": 0.50, "mean_cells": 10.5},
            },
        },
    }
    sample_abl = {
        "variants": {
            "baseline": {"weighted_score": -2.205, "delta_weighted": 0.0, "mean_ms_per_step": 18.0},
            "no_map": {"weighted_score": 4.159, "delta_weighted": 6.364, "mean_ms_per_step": 20.2},
            "no_landmarks": {"weighted_score": -3.880, "delta_weighted": -1.675, "mean_ms_per_step": 19.7},
            "no_entity_attn": {"weighted_score": 4.071, "delta_weighted": 6.276, "mean_ms_per_step": 14.1},
            "no_mode": {"weighted_score": -2.205, "delta_weighted": 0.0, "mean_ms_per_step": 17.1},
            "no_depth": {"weighted_score": -2.607, "delta_weighted": -0.402, "mean_ms_per_step": 16.5},
        }
    }
    html_text = build_html(sample_eval, sample_abl)
    out_pdf = tmp_path / "brief.pdf"
    info = render_pdf(html_text, out_pdf)
    assert out_pdf.is_file()
    raw = out_pdf.read_bytes()
    assert raw.startswith(b"%PDF-")
    assert info["pages"] == 2, f"Expected 2-page PDF brief, got {info['pages']} pages"
    assert len(re.findall(rb"/Type\s*/Page(?!s)\b", raw)) == 2
