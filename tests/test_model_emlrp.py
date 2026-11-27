"""Unit tests and CPU latency benchmark for EMLRP model architecture (Phase 3)."""

from __future__ import annotations

import time

import pytest
import torch

from submission.models import EMLRPConfig, EMLRPPolicy


def _make_synthetic_obs(
    batch_size: int,
    config: EMLRPConfig,
    seed: int = 0,
    empty_entities: bool = False,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch_size, config.entity_max, dtype=torch.float32)
    if not empty_entities:
        for b in range(batch_size):
            k = int(torch.randint(1, config.entity_max // 2, (1,), generator=g).item())
            mask[b, :k] = 1.0

    ent_types = torch.randint(
        1, config.entity_type_vocab, (batch_size, config.entity_max), generator=g
    )
    ent_types = ent_types * mask.long()

    return {
        "ray_distance": torch.rand(batch_size, config.ray_bins, generator=g),
        "ray_hit_type": torch.randint(0, config.ray_hit_types, (batch_size, config.ray_bins), generator=g),
        "entity_features": torch.randn(batch_size, config.entity_max, config.entity_feat_dim, generator=g),
        "entity_types": ent_types,
        "entity_mask": mask,
        "vitals": torch.randn(batch_size, config.vitals_dim, generator=g),
        "depth_feat": torch.randn(batch_size, config.depth_feat_dim, generator=g),
    }


def test_step_shapes_and_inference_mode() -> None:
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg).eval()
    B = 3
    obs = _make_synthetic_obs(B, cfg, seed=1)
    state = model.init_state(B)
    prev_act = torch.zeros(B, cfg.action_dim)

    with torch.inference_mode():
        out = model.step(obs, state, prev_act, compute_aux=False)

    assert out.action_logits.shape == (B, cfg.action_dim)
    assert out.value_ext.shape == (B, 1)
    assert out.value_int.shape == (B, 1)
    assert out.mode_logits.shape == (B, cfg.num_latent_modes)
    assert out.aux == {}
    assert out.state.gru_h.shape == (B, cfg.gru_hidden_dim)
    assert out.state.spatial_map.shape == (B, cfg.map_channels, cfg.map_size, cfg.map_size)
    assert out.state.landmark_keys.shape == (B, cfg.landmark_slots, cfg.landmark_dim)
    assert out.state.landmark_pos.shape == (B, cfg.landmark_slots, 2)
    assert out.state.landmark_valid.shape == (B, cfg.landmark_slots)


def test_empty_entity_mask_no_nans() -> None:
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg)
    obs = _make_synthetic_obs(2, cfg, seed=2, empty_entities=True)
    state = model.init_state(2)
    prev_act = torch.zeros(2, cfg.action_dim)

    out = model.step(obs, state, prev_act, compute_aux=True)
    assert torch.isfinite(out.action_logits).all()
    assert torch.isfinite(out.value_ext).all()
    assert torch.isfinite(out.value_int).all()
    loss = out.action_logits.sum() + out.value_ext.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_entity_permutation_invariance() -> None:
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg).eval()
    obs = _make_synthetic_obs(2, cfg, seed=3)

    with torch.inference_mode():
        ray_embed, _ = model.ray_encoder(obs["ray_distance"], obs["ray_hit_type"])
        summary1, _, _ = model.entity_encoder(
            obs["entity_features"], obs["entity_types"], obs["entity_mask"], ray_embed
        )

        perm = torch.randperm(cfg.entity_max)
        summary2, _, _ = model.entity_encoder(
            obs["entity_features"][:, perm],
            obs["entity_types"][:, perm],
            obs["entity_mask"][:, perm],
            ray_embed,
        )

    assert torch.allclose(summary1, summary2, atol=1e-5)


def test_sequence_bptt_reset_and_gradients() -> None:
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg)
    B, T = 4, 10
    steps = [_make_synthetic_obs(B, cfg, seed=10 + t) for t in range(T)]
    obs_seq = {k: torch.stack([s[k] for s in steps], dim=1) for k in steps[0]}
    prev_act_seq = torch.randint(0, 2, (B, T, cfg.action_dim)).float()
    done_seq = torch.zeros(B, T, dtype=torch.bool)
    done_seq[:, 5] = True

    init_state = model.init_state(B)
    outputs, final_state = model.forward_sequence(
        obs_seq, init_state, prev_act_seq, done_seq=done_seq, compute_aux=True
    )

    assert outputs["action_logits"].shape == (B, T, cfg.action_dim)
    assert outputs["aux_next_ray"].shape == (B, T, cfg.ray_bins)
    assert outputs["aux_inverse_dyn"].shape == (B, T, cfg.action_dim)

    # Verify zero leakage across episode boundary at t=5
    tail_obs = {k: v[:, 5:] for k, v in obs_seq.items()}
    tail_outputs, tail_final = model.forward_sequence(
        tail_obs, model.init_state(B), prev_act_seq[:, 5:], done_seq=None, compute_aux=False
    )
    assert torch.allclose(outputs["action_logits"][:, 5:], tail_outputs["action_logits"], atol=1e-5)
    assert torch.allclose(final_state.gru_h, tail_final.gru_h, atol=1e-5)
    assert torch.allclose(final_state.spatial_map, tail_final.spatial_map, atol=1e-5)

    # Verify vectorized forward_sequence matches step-by-step step()
    model.eval()
    with torch.inference_mode():
        seq_eval, seq_final = model.forward_sequence(
            obs_seq, model.init_state(B), prev_act_seq, done_seq=done_seq, compute_aux=True
        )
        st = model.init_state(B)
        step_logits = []
        for t in range(T):
            step_obs = {k: v[:, t] for k, v in obs_seq.items()}
            o = model.step(step_obs, st, prev_act_seq[:, t], done=done_seq[:, t], compute_aux=False)
            st = o.state
            step_logits.append(o.action_logits)
        assert torch.allclose(seq_eval["action_logits"], torch.stack(step_logits, dim=1), atol=1e-5)
        assert torch.allclose(seq_final.gru_h, st.gru_h, atol=1e-5)
        assert torch.allclose(seq_final.spatial_map, st.spatial_map, atol=1e-5)
    model.train()

    loss = (
        outputs["action_logits"].pow(2).mean()
        + outputs["value_ext"].pow(2).mean()
        + outputs["value_int"].pow(2).mean()
        + outputs["aux_next_ray"].pow(2).mean()
        + outputs["aux_landmark_dir"].pow(2).mean()
        + outputs["aux_inverse_dyn"].pow(2).mean()
        + outputs["aux_combat"].pow(2).mean()
        + outputs["aux_entity_importance"].pow(2).mean()
    )
    loss.backward()

    checked_prefixes = (
        "ray_encoder.",
        "entity_encoder.",
        "spatial_map.",
        "landmark_memory.",
        "gru.",
        "policy_head.",
        "value_ext_head.",
        "value_int_head.",
    )
    for prefix in checked_prefixes:
        has_nonzero_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for n, p in model.named_parameters()
            if n.startswith(prefix)
        )
        assert has_nonzero_grad, f"Expected non-zero gradient for {prefix}"


def test_egomotion_landmark_transform() -> None:
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg).eval()
    pos = torch.tensor([[[0.5, 0.0]]])  # 0.5 ahead
    delta_xy = torch.tensor([[0.1, 0.0]])  # moved forward 0.1
    delta_theta = torch.tensor([0.0])
    moved = model.landmark_memory.transform_positions(pos, delta_xy, delta_theta)
    assert torch.allclose(moved, torch.tensor([[[0.4, 0.0]]]), atol=1e-6)

    # Turn 90 deg CCW: landmark ahead (+x) should now be to the right (-y)
    turned = model.landmark_memory.transform_positions(
        pos, torch.zeros(1, 2), torch.tensor([torch.pi / 2])
    )
    assert torch.allclose(turned, torch.tensor([[[0.0, -0.5]]]), atol=1e-5)


def test_parameter_budget_within_1m_to_5m() -> None:
    cfg = EMLRPConfig()
    model = EMLRPPolicy(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[EMLRP] Total parameters: {total_params:,}")
    assert 1_000_000 <= total_params <= 5_000_000, (
        f"Parameter count {total_params:,} outside [1M, 5M] target range"
    )


@pytest.mark.parametrize("threads", [1])
def test_single_thread_cpu_latency(threads: int) -> None:
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        cfg = EMLRPConfig()
        model = EMLRPPolicy(cfg).to("cpu").eval()
        obs = _make_synthetic_obs(1, cfg, seed=99)
        state = model.init_state(1, device="cpu")
        prev_act = torch.zeros(1, cfg.action_dim)

        with torch.inference_mode():
            for _ in range(20):
                out = model.step(obs, state, prev_act, compute_aux=False)
                state = out.state

            t0 = time.perf_counter()
            n_iters = 200
            for _ in range(n_iters):
                out = model.step(obs, state, prev_act, compute_aux=False)
                state = out.state
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

        print(f"\n[EMLRP] 1-thread CPU mean latency: {elapsed_ms:.2f} ms/step")
        assert elapsed_ms < 40.0, f"Latency {elapsed_ms:.2f} ms exceeds 40 ms/step budget"
    finally:
        torch.set_num_threads(prev_threads)
