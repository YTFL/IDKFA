"""Pre-submission static compliance, LFS, clean-archive, and finalization checker.

Usage:
    python training/tools/pre_submit_check.py
    python training/tools/pre_submit_check.py --finalize-weights checkpoints/emlrp_ppo_v1/best.pt --temperature 0.7
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import limits as L  # noqa: E402
from doom_env import make_env  # noqa: E402

FORBIDDEN_IMPORTS = {"vizdoom", "socket", "urllib", "requests", "http", "httplib2", "aiohttp"}
FORBIDDEN_NAMES = {"privileged_state", "atan2", "arctan2", "astar", "dijkstra"}


def check_submission_compliance(submission_dir: Path | None = None) -> list[str]:
    """Scan all `.py` files in `submission/` for forbidden imports or rule violations."""
    sub = submission_dir or (ROOT / "submission")
    violations: list[str] = []
    for py_file in sorted(sub.rglob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_IMPORTS:
                        violations.append(f"{py_file.relative_to(ROOT)}: forbidden import `{alias.name}`")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in FORBIDDEN_IMPORTS:
                    violations.append(f"{py_file.relative_to(ROOT)}: forbidden import from `{node.module}`")
            elif isinstance(node, ast.Name) and node.id.lower() in FORBIDDEN_NAMES:
                violations.append(f"{py_file.relative_to(ROOT)}: forbidden identifier `{node.id}`")
            elif isinstance(node, ast.Attribute) and node.attr.lower() in FORBIDDEN_NAMES:
                violations.append(f"{py_file.relative_to(ROOT)}: forbidden attribute `{node.attr}`")
    return violations


def check_lfs_and_size(repo: Path = ROOT) -> dict[str, Any]:
    """Verify tracked repository size <= 1 GB and Git LFS tracking on `.pt` weights."""
    files_out = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode("utf-8", "replace")
    rel_files = [f for f in files_out.split("\0") if f]
    total_bytes = sum((repo / f).stat().st_size for f in rel_files if (repo / f).is_file())

    pt_files = [f for f in rel_files if f.endswith(".pt")]
    if pt_files:
        attrs = subprocess.check_output(
            ["git", "-C", str(repo), "check-attr", "filter", "--", *pt_files]
        ).decode("utf-8", "replace")
        non_lfs = [ln.split(": filter: ")[0] for ln in attrs.splitlines() if not ln.endswith(": filter: lfs")]
    else:
        non_lfs = []

    agent_pt = repo / "submission" / "weights" / "agent.pt"
    agent_ok = agent_pt.is_file() and not agent_pt.read_bytes()[:64].startswith(b"version https://git-lfs")

    req_file = repo / "submission" / "requirements.txt"
    req_bad: list[str] = []
    if req_file.is_file():
        lines = [ln.split("#")[0].strip() for ln in req_file.read_text(errors="replace").splitlines()]
        lines = [ln for ln in lines if ln]
        if len(lines) > L.MAX_REQUIREMENTS:
            req_bad.append(f"too many requirements ({len(lines)} > {L.MAX_REQUIREMENTS})")
        for ln in lines:
            if not re.match(L.REQUIREMENT_PATTERN, ln):
                req_bad.append(ln)

    return {
        "total_bytes": total_bytes,
        "within_limit": total_bytes <= L.MAX_REPO_BYTES,
        "non_lfs_pt": non_lfs,
        "agent_pt_present": agent_ok,
        "requirements_errors": req_bad,
    }


def check_clean_copy_rollout(repo: Path = ROOT, episode_ticks: int = 120) -> dict[str, Any]:
    """Copy `submission/` into an isolated temp directory and verify deterministic DoomAgent execution."""
    tmp = Path(tempfile.mkdtemp(prefix="idkfa-clean-"))
    try:
        shutil.copytree(repo / "submission", tmp / "submission", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        agent_path = tmp / "submission" / "agent.py"
        spec = importlib.util.spec_from_file_location("clean_submission_agent", agent_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(tmp / "submission"))
        sys.path.insert(0, str(tmp))
        try:
            spec.loader.exec_module(module)
            DoomAgent = module.DoomAgent
            env = make_env(str(repo / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=episode_ticks)
            try:
                agent = DoomAgent(env.observation_space, env.action_space, "cpu")
                obs0, _ = env.reset(seed=0)
                agent.reset()
                acts = []
                for _ in range(15):
                    a = agent.act(obs0)
                    assert isinstance(a, np.ndarray) and a.shape == (9,) and a.dtype == np.int8
                    assert np.all((a == 0) | (a == 1))
                    acts.append(a.copy())
                    obs0, _, term, trunc, _ = env.step(a)
                    if term or trunc:
                        break
            finally:
                env.close()
        finally:
            for p in (str(tmp), str(tmp / "submission")):
                if p in sys.path:
                    sys.path.remove(p)
        return {"clean_steps": len(acts), "weights_loaded": str((tmp / "submission" / "weights" / "agent.pt").name)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_checks(repo: Path = ROOT) -> dict[str, Any]:
    violations = check_submission_compliance(repo / "submission")
    lfs_info = check_lfs_and_size(repo)
    clean_info = check_clean_copy_rollout(repo)

    passed = (
        not violations
        and lfs_info["within_limit"]
        and not lfs_info["non_lfs_pt"]
        and lfs_info["agent_pt_present"]
        and not lfs_info["requirements_errors"]
        and clean_info["clean_steps"] > 0
    )
    return {
        "passed": passed,
        "compliance_violations": violations,
        **lfs_info,
        **clean_info,
    }


def _run_step(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def finalize_submission(checkpoint: str, temperature: float, quick: bool = False) -> None:
    """Export final weights, regenerate evaluation/ablation/brief artifacts, and run checks."""
    py = sys.executable
    _run_step(
        [
            py,
            "training/tools/export_weights.py",
            "--checkpoint",
            checkpoint,
            "--output",
            "submission/weights/agent.pt",
            "--temperature",
            str(temperature),
        ]
    )
    eval_cmd = [
        py,
        "training/tools/run_eval_report.py",
        "--checkpoint",
        "submission/weights/agent.pt",
        "--temperature",
        str(temperature),
    ]
    if quick:
        eval_cmd.append("--quick")
    _run_step(eval_cmd)

    abl_cmd = [
        py,
        "training/tools/run_ablations.py",
        "--checkpoint",
        "submission/weights/agent.pt",
        "--temperature",
        str(temperature),
        "--max-maps-per-tier",
        "2" if quick else "6",
        "--ticks",
        "400" if quick else "2100",
    ]
    _run_step(abl_cmd)
    _run_step(
        [
            py,
            "training/tools/build_brief.py",
            "--eval-json",
            "runs/diagnostics/eval_report.json",
            "--ablation-json",
            "runs/diagnostics/ablation_report.json",
            "--output",
            "docs/brief.pdf",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 8 submission compliance and clean-copy execution")
    parser.add_argument("--finalize-weights", type=str, default=None, help="Export weights and rebuild all Phase 7/8 reports")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.finalize_weights:
        finalize_submission(args.finalize_weights, temperature=args.temperature, quick=args.quick)

    res = run_checks(ROOT)
    size_mb = res["total_bytes"] / (1024 * 1024)
    print(f"Compliance violations : {res['compliance_violations']}")
    print(f"Repo size             : {size_mb:.1f} MB (within 1 GB: {res['within_limit']})")
    print(f"Non-LFS .pt files     : {res['non_lfs_pt']}")
    print(f"agent.pt resolved     : {res['agent_pt_present']}")
    print(f"requirements.txt errs : {res['requirements_errors']}")
    print(f"Clean-copy steps      : {res['clean_steps']}")
    if not res["passed"]:
        sys.exit(1)
    print("PRE-SUBMIT CHECK PASSED")


if __name__ == "__main__":
    main()
