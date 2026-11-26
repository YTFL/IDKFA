"""Truncated-BPTT Behavioral Cloning (BC) pretrainer for EMLRPPolicy.

Streams sequential chunks `(B, T=seq_len)` across episodes while carrying `EMLRPState`
across chunks until each episode completes (resetting state cleanly at episode starts).
Jointly trains:
  1. Masked factored MultiBinary(9) action logits with positive class weighting on sparse actions
  2. Auxiliary perception/memory heads (next_ray, inverse_dyn, combat, landmark_dir, entity_imp)
  3. Critic value_ext head against Monte-Carlo returns so initial PPO updates start calibrated

Usage:
    python -m training.imitation.pretrain_bc --demo-dir data/bc_demos --out checkpoints/bc_pretrain.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.action_mask import OPPOSING_AXIS_PAIRS
from submission.features import ObservationPreprocessor
from submission.models import EMLRPConfig, EMLRPPolicy, EMLRPState

# Mild positive class weights to keep ATTACK, USE, and strafe bits sharp against imbalances
DEFAULT_POS_WEIGHTS = [1.0, 1.0, 1.5, 1.5, 1.2, 1.2, 2.0, 3.0, 2.0]


def _load_demo_episodes(demo_dir: Path) -> tuple[list[dict[str, np.ndarray]], bool]:
    manifest = json.loads((demo_dir / "manifest.json").read_text(encoding="utf-8"))
    use_depth = bool(manifest.get("use_depth", True))
    episodes: list[dict[str, np.ndarray]] = []
    array_keys = (
        "ray_distance",
        "ray_hit_type",
        "entity_features",
        "entity_types",
        "entity_mask",
        "vitals",
        "depth_feat",
        "prev_action",
        "action",
        "action_mask",
        "next_ray_distance",
        "aux_combat",
        "aux_landmark_dir",
        "aux_landmark_mask",
        "mc_return",
        "done",
    )
    for s in manifest["shards"]:
        data = np.load(demo_dir / s["file"])
        starts = data["ep_starts"]
        lengths = data["ep_lengths"]
        for st, ln in zip(starts, lengths):
            sl = slice(int(st), int(st + ln))
            episodes.append({k: data[k][sl] for k in array_keys})
    return episodes, use_depth


def pretrain_bc(
    demo_dir: str | Path = "data/bc_demos",
    out_path: str | Path = "checkpoints/bc_pretrain.pt",
    epochs: int = 12,
    batch_size: int = 16,
    seq_len: int = 32,
    lr: float = 3e-4,
    aux_coef: float = 0.25,
    vf_coef: float = 0.25,
    device_str: str = "auto",
    seed: int = 42,
    inference_temperature: float = 0.3,
) -> dict[str, Any]:
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    )
    use_bf16 = device.type == "cuda"
    if use_bf16:
        torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    episodes, use_depth = _load_demo_episodes(Path(demo_dir))
    if not episodes:
        raise RuntimeError(f"Empty demonstration dataset at {demo_dir}")

    depth_dim = 32 if use_depth else 0
    prep = ObservationPreprocessor(normalize_vitals=True, use_depth=use_depth)
    model_cfg = EMLRPConfig(depth_feat_dim=depth_dim)
    policy = EMLRPPolicy(model_cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    pos_w = torch.tensor(DEFAULT_POS_WEIGHTS, dtype=torch.float32, device=device)

    total_ep_count = len(episodes)
    avg_len = float(np.mean([len(e["action"]) for e in episodes]))
    print(
        f"[pretrain_bc] Loaded {total_ep_count} episodes (mean_len={avg_len:.1f}) | "
        f"device={device} epochs={epochs} batch_size={batch_size} seq_len={seq_len}",
        flush=True,
    )

    last_stats: dict[str, float] = {}
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        policy.train()
        order = rng.permutation(total_ep_count).tolist()
        ep_cursor = 0

        # Active slots holding (ep_idx, offset)
        slot_ep: list[int | None] = [None] * batch_size
        slot_off = [0] * batch_size
        for b in range(batch_size):
            if ep_cursor < len(order):
                slot_ep[b] = order[ep_cursor]
                slot_off[b] = 0
                ep_cursor += 1

        state = policy.init_state(batch_size, device=device)
        ep_losses: list[float] = []
        ep_bc_losses: list[float] = []
        ep_aux_losses: list[float] = []
        ep_vf_losses: list[float] = []
        ep_accs: list[float] = []

        while any(s is not None for s in slot_ep):
            # Allocate (B, T) chunk arrays
            T = seq_len
            b_ray_d = np.zeros((batch_size, T, 64), dtype=np.float32)
            b_ray_h = np.zeros((batch_size, T, 64), dtype=np.int64)
            b_ent_f = np.zeros((batch_size, T, 32, 5), dtype=np.float32)
            b_ent_t = np.zeros((batch_size, T, 32), dtype=np.int64)
            b_ent_m = np.zeros((batch_size, T, 32), dtype=np.float32)
            b_vit = np.zeros((batch_size, T, 17), dtype=np.float32)
            b_dep = np.zeros((batch_size, T, 32), dtype=np.float32)
            b_prev = np.zeros((batch_size, T, 9), dtype=np.float32)
            b_act = np.zeros((batch_size, T, 9), dtype=np.float32)
            b_mask = np.zeros((batch_size, T, 9), dtype=np.float32)
            b_next_r = np.zeros((batch_size, T, 64), dtype=np.float32)
            b_aux_c = np.zeros((batch_size, T, 2), dtype=np.float32)
            b_aux_d = np.zeros((batch_size, T, 2), dtype=np.float32)
            b_aux_m = np.zeros((batch_size, T), dtype=np.float32)
            b_ret = np.zeros((batch_size, T), dtype=np.float32)
            b_reset = np.zeros((batch_size, T), dtype=bool)
            b_valid = np.zeros((batch_size, T), dtype=bool)
            b_done = np.zeros((batch_size, T), dtype=bool)

            for b in range(batch_size):
                t_pos = 0
                while t_pos < T and slot_ep[b] is not None:
                    ep = episodes[slot_ep[b]]
                    off = slot_off[b]
                    ep_len = len(ep["action"])
                    take = min(T - t_pos, ep_len - off)
                    if off == 0:
                        b_reset[b, t_pos] = True
                    sl_dst = slice(t_pos, t_pos + take)
                    sl_src = slice(off, off + take)
                    b_ray_d[b, sl_dst] = ep["ray_distance"][sl_src]
                    b_ray_h[b, sl_dst] = ep["ray_hit_type"][sl_src]
                    b_ent_f[b, sl_dst] = ep["entity_features"][sl_src]
                    b_ent_t[b, sl_dst] = ep["entity_types"][sl_src]
                    b_ent_m[b, sl_dst] = ep["entity_mask"][sl_src]
                    b_vit[b, sl_dst] = ep["vitals"][sl_src]
                    b_dep[b, sl_dst] = ep["depth_feat"][sl_src]
                    b_prev[b, sl_dst] = ep["prev_action"][sl_src]
                    b_act[b, sl_dst] = ep["action"][sl_src]
                    b_mask[b, sl_dst] = ep["action_mask"][sl_src]
                    b_next_r[b, sl_dst] = ep["next_ray_distance"][sl_src]
                    b_aux_c[b, sl_dst] = ep["aux_combat"][sl_src]
                    b_aux_d[b, sl_dst] = ep["aux_landmark_dir"][sl_src]
                    b_aux_m[b, sl_dst] = ep["aux_landmark_mask"][sl_src]
                    b_ret[b, sl_dst] = ep["mc_return"][sl_src]
                    b_done[b, sl_dst] = ep["done"][sl_src]
                    b_valid[b, sl_dst] = True

                    t_pos += take
                    slot_off[b] += take
                    if slot_off[b] >= ep_len:
                        if ep_cursor < len(order):
                            slot_ep[b] = order[ep_cursor]
                            slot_off[b] = 0
                            ep_cursor += 1
                        else:
                            slot_ep[b] = None
                            slot_off[b] = 0

            obs_seq: dict[str, torch.Tensor] = {
                "ray_distance": torch.as_tensor(b_ray_d, device=device),
                "ray_hit_type": torch.as_tensor(b_ray_h, device=device),
                "entity_features": torch.as_tensor(b_ent_f, device=device),
                "entity_types": torch.as_tensor(b_ent_t, device=device),
                "entity_mask": torch.as_tensor(b_ent_m, device=device),
                "vitals": torch.as_tensor(b_vit, device=device),
            }
            if use_depth:
                obs_seq["depth_feat"] = torch.as_tensor(b_dep, device=device)

            prev_act_t = torch.as_tensor(b_prev, device=device)
            act_t = torch.as_tensor(b_act, device=device)
            mask_t = torch.as_tensor(b_mask, device=device)
            reset_t = torch.as_tensor(b_reset, device=device)
            valid_t = torch.as_tensor(b_valid, device=device)
            done_t = torch.as_tensor(b_done, device=device)
            ret_t = torch.as_tensor(b_ret, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs, next_state = policy.forward_sequence(
                    obs_seq, state, prev_act_t, done_seq=reset_t, compute_aux=True
                )
            state = EMLRPState(
                gru_h=next_state.gru_h.detach(),
                spatial_map=next_state.spatial_map.detach(),
                landmark_keys=next_state.landmark_keys.detach(),
                landmark_pos=next_state.landmark_pos.detach(),
                landmark_valid=next_state.landmark_valid.detach(),
            )

            # Factored masked BCE matching PPO's exact effective mask
            eff_mask = mask_t.clone()
            for pos, neg in OPPOSING_AXIS_PAIRS:
                eff_mask[..., neg] = mask_t[..., neg] * (1.0 - act_t[..., pos])
            eff_mask = eff_mask * valid_t.unsqueeze(-1).float()

            logits = outputs["action_logits"].float()
            bce = F.binary_cross_entropy_with_logits(
                logits, act_t, pos_weight=pos_w, reduction="none"
            )
            bc_loss = (bce * eff_mask).sum() / eff_mask.sum().clamp(min=1.0)

            with torch.no_grad():
                preds = (logits > 0.0).float()
                correct = ((preds == act_t).float() * eff_mask).sum()
                acc = float((correct / eff_mask.sum().clamp(min=1.0)).item())

            # Auxiliary perception & memory losses
            nonterm = (~done_t & valid_t).float()
            next_r_t = torch.as_tensor(b_next_r, device=device)
            ray_err = (outputs["aux_next_ray"].float() - next_r_t).pow(2).mean(dim=-1)
            next_ray_loss = (ray_err * nonterm).sum() / nonterm.sum().clamp(min=1.0)

            v_float = valid_t.float()
            inv_bce = F.binary_cross_entropy_with_logits(
                outputs["aux_inverse_dyn"].float(), act_t, reduction="none"
            ).mean(dim=-1)
            inv_dyn_loss = (inv_bce * v_float).sum() / v_float.sum().clamp(min=1.0)

            aux_c_t = torch.as_tensor(b_aux_c, device=device)
            comb_bce = F.binary_cross_entropy_with_logits(
                outputs["aux_combat"].float(), aux_c_t, reduction="none"
            ).mean(dim=-1)
            combat_loss = (comb_bce * v_float).sum() / v_float.sum().clamp(min=1.0)

            aux_d_t = torch.as_tensor(b_aux_d, device=device)
            aux_m_t = torch.as_tensor(b_aux_m, device=device) * v_float
            dir_pred = F.normalize(outputs["aux_landmark_dir"].float(), dim=-1, eps=1e-6)
            dir_err = (dir_pred - aux_d_t).pow(2).sum(dim=-1)
            landmark_dir_loss = (dir_err * aux_m_t).sum() / aux_m_t.sum().clamp(min=1.0)

            ent_types_t = obs_seq["entity_types"]
            imp_target = ((ent_types_t >= 1) & (ent_types_t <= 3) | (ent_types_t == 9)).float()
            ent_m_seq = obs_seq["entity_mask"] * valid_t.unsqueeze(-1).float()
            imp_bce = F.binary_cross_entropy_with_logits(
                outputs["aux_entity_importance"].float(), imp_target, reduction="none"
            )
            entity_imp_loss = (imp_bce * ent_m_seq).sum() / ent_m_seq.sum().clamp(min=1.0)

            aux_loss = (
                next_ray_loss
                + 0.5 * inv_dyn_loss
                + 0.5 * combat_loss
                + 0.5 * landmark_dir_loss
                + 0.5 * entity_imp_loss
            )

            v_ext = outputs["value_ext"].squeeze(-1).float()
            vf_loss = (F.smooth_l1_loss(v_ext, ret_t, reduction="none") * v_float).sum() / v_float.sum().clamp(min=1.0)

            loss = bc_loss + aux_coef * aux_loss + vf_coef * vf_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            ep_losses.append(float(loss.item()))
            ep_bc_losses.append(float(bc_loss.item()))
            ep_aux_losses.append(float(aux_loss.item()))
            ep_vf_losses.append(float(vf_loss.item()))
            ep_accs.append(acc)

        last_stats = {
            "epoch": float(epoch),
            "loss": round(float(np.mean(ep_losses)), 4),
            "bc_loss": round(float(np.mean(ep_bc_losses)), 4),
            "aux_loss": round(float(np.mean(ep_aux_losses)), 4),
            "vf_loss": round(float(np.mean(ep_vf_losses)), 4),
            "bit_acc": round(float(np.mean(ep_accs)), 4),
            "elapsed_s": round(time.perf_counter() - t_start, 1),
        }
        print(
            f"[pretrain_bc] Epoch {epoch:02d}/{epochs:02d} | "
            f"loss={last_stats['loss']:.4f} bc={last_stats['bc_loss']:.4f} "
            f"aux={last_stats['aux_loss']:.4f} vf={last_stats['vf_loss']:.4f} "
            f"acc={last_stats['bit_acc']:.3f} ({last_stats['elapsed_s']:.1f}s)",
            flush=True,
        )

    save_target = Path(out_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy.state_dict(),
        "model_config": dataclasses.asdict(model_cfg),
        "preprocess_config": prep.to_config(),
        "inference_temperature": float(inference_temperature),
        "bc_stats": last_stats,
        "curriculum_stage": 5,
        "global_step": 0,
        "update_idx": 0,
    }
    torch.save(payload, save_target)
    print(f"[pretrain_bc] Saved BC checkpoint -> {save_target}", flush=True)
    return {"checkpoint": str(save_target), "stats": last_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncated-BPTT Behavioral Cloning for EMLRPPolicy")
    parser.add_argument("--demo-dir", type=str, default="data/bc_demos")
    parser.add_argument("--out", type=str, default="checkpoints/bc_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pretrain_bc(
        demo_dir=args.demo_dir,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
