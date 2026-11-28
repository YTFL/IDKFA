# IDKFA Finalization & Verification Guide (RTX 5060 Machine)

Both `feat/phase-5-training-pipeline` and `feat/phase-6-8-submission` are merged together, including:
- Trained `2,001,920`-step checkpoint (`submission/weights/emlrp_latest.pt`, Update `962`, Stage `1`) and best validation checkpoint (`submission/weights/emlrp_best.pt`, Update `200`, `best_val = 10.5325`), tracked via Git LFS.
- Complete Phase 6–8 pipeline (`submission/agent.py`, `training/eval.py`, `training/tools/*`, `tests/test_agent_phase6.py`, `tests/test_eval_phase7.py`, `tests/test_submission_phase8.py`, `docs/brief.pdf`, `README.md`).

Run the steps below on the RTX 5060 machine in order.

---

## 1. Sync Repo & Git LFS Weights

```bash
git fetch origin
git checkout feat/phase-5-training-pipeline
git pull
git lfs pull
git lfs ls-files
```

Verify `.venv` is active (or run via `.\.venv\Scripts\python.exe` on Windows / `uv run python` on Linux).

---

## 2. Compare Checkpoints & Decoding Temperatures

Benchmark both the `2.00M`-step checkpoint (`emlrp_latest.pt`) and the Update `200` checkpoint (`emlrp_best.pt`) across deterministic (`0.0`) and low-temperature (`0.5`, `0.7`, `0.8`) sampling on held-out `maps/dataset/val/` maps:

```bash
python training/tools/eval_inference_modes.py --checkpoint submission/weights/emlrp_latest.pt --max-maps-per-tier 6 --ticks 2100 --temperatures 0.0 0.5 0.7 0.8
python training/tools/eval_inference_modes.py --checkpoint submission/weights/emlrp_best.pt --max-maps-per-tier 6 --ticks 2100 --temperatures 0.0 0.5 0.7 0.8
```

Pick the `<BEST_CKPT>` (`emlrp_latest.pt` or `emlrp_best.pt`) and `<TEMP>` (e.g. `0.0` or `0.7`) with the highest `weighted` score and exit rate.

---

## 3. Finalize `agent.pt`, Evaluation Reports, Ablations & 2-Page Brief

Run the one-shot finalizer (exports compact `~7.3 MB` `submission/weights/agent.pt`, runs multi-seed held-out evaluation + classic `freedoom1.wad`/`freedoom2.wad` generalization, runs all 6 inference-knockout ablations, rebuilds `docs/brief.pdf`, and executes `pre_submit_check`):

```bash
python training/tools/pre_submit_check.py --finalize-weights submission/weights/<BEST_CKPT> --temperature <TEMP>
```

Or run each step individually with custom seed/map counts:

```bash
python training/tools/export_weights.py --checkpoint submission/weights/<BEST_CKPT> --output submission/weights/agent.pt --temperature <TEMP>
python training/tools/run_eval_report.py --checkpoint submission/weights/agent.pt --temperature <TEMP> --seeds 0 1 2
python training/tools/run_ablations.py --checkpoint submission/weights/agent.pt --temperature <TEMP> --max-maps-per-tier 6 --ticks 2100 --seeds 0 1
python training/tools/analyze_failures.py --checkpoint submission/weights/agent.pt --temperature <TEMP> --max-maps-per-tier 4 --ticks 2100
python training/tools/build_brief.py --eval-json runs/diagnostics/eval_report.json --ablation-json runs/diagnostics/ablation_report.json --output docs/brief.pdf
```

*(Optional training-time ablations if time permits: `python training/tools/run_ablations.py --mode train --variant no_rnd --total-steps 500000` and `--variant no_depth`.)*

---

## 4. Generate Live Q&A Demo & Inspect Outputs

Create the recorded interactive HTML replay fallback (`runs/diagnostics/demo_replay.html`) and inspect terminal map/slot visualizations:

```bash
python training/tools/demo_replay.py --checkpoint submission/weights/agent.pt --temperature <TEMP> --ticks 1200 --terminal --save-html runs/diagnostics/demo_replay.html
```

Inspect:
- `docs/brief.pdf` (confirm 2 clean pages and populated tables/charts)
- `runs/diagnostics/eval_report.md`
- `runs/diagnostics/ablation_report.md`
- `runs/diagnostics/failures/summary.json`
- `runs/diagnostics/demo_replay.html`

---

## 5. Run Full Test Suite & Validator

Run all automated tests (on Windows, ensure a non-sandboxed shell so ViZDoom IPC works):

```bash
python -m pytest tests/test_env.py tests/test_features.py tests/test_mapgen.py tests/test_rewards.py tests/test_model_emlrp.py tests/test_phase2_3_4_integration.py tests/test_ppo_phase5.py tests/test_agent_phase6.py tests/test_eval_phase7.py tests/test_submission_phase8.py -v --tb=short
python training/tools/pre_submit_check.py
```

Run the official validator (use `--docker` on Linux/WSL or a machine with Docker installed, since `validate.py` `dry_run()` uses `socket.AF_UNIX`):

```bash
python validate.py
python validate.py --docker
```

On Linux/Docker, also run the socket/validator tests:

```bash
python -m pytest tests/test_remote.py tests/test_validate.py -v --tb=short
```

---

## 6. Clean Up Weights (Optional), Merge to `main`, Push & Tag

Total repo size with `agent.pt` + `emlrp_best.pt` + `emlrp_latest.pt` is `~118 MB` (well within the `1 GB` limit). If you want only `agent.pt` (`~7.3 MB`) tracked under `submission/weights/` for the final commit, remove the training checkpoints (`git rm submission/weights/emlrp_best.pt submission/weights/emlrp_latest.pt`).

Commit any updated `submission/weights/agent.pt` and `docs/brief.pdf`, merge into `main`, verify LFS, push everything, and tag:

```bash
git add submission/weights/agent.pt docs/brief.pdf
git commit -m "chore(submission): finalize trained 5060 weights and 2-page brief"

git checkout main
git merge --no-ff feat/phase-5-training-pipeline -m "Merge final IDKFA EMLRP submission"
python validate.py --docker   # final check on main
git lfs ls-files
git lfs push --all origin
git push origin main
git tag -a v1.0-submission -m "IDKFA DevForge DOOM final submission"
git push origin v1.0-submission
```
