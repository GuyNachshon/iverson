"""Tests for the discretize / predictor / loss stack."""
from __future__ import annotations

import torch

from models.discretize import (
    expected_value_decode,
    hard_bin_target,
    hl_gauss_target,
)
from models.loss import predictor_loss
from models.terminal_predictor import (
    PredictorConfig,
    TerminalPredictor,
    feature_mask_full,
    feature_mask_invariant,
)


def test_hl_gauss_target_sums_to_one() -> None:
    vals = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    target = hl_gauss_target(vals, n_bins=16, start=0.0, end=1.0)
    assert target.shape == (5, 16)
    assert torch.allclose(target.sum(-1), torch.ones(5), atol=1e-4)


def test_hard_bin_clamps() -> None:
    vals = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
    idx = hard_bin_target(vals, n_bins=10, start=0.0, end=1.0)
    assert idx.min() >= 0 and idx.max() <= 9


def test_expected_value_decode_recovers_mode() -> None:
    # If logits are sharply peaked at bin i, decoded value should equal that bin's center.
    logits = torch.full((1, 10), -10.0)
    logits[0, 3] = 10.0
    decoded = expected_value_decode(logits, start=0.0, end=1.0)
    expected_center = 0.05 + 3 * 0.1  # bin 3 center for 10 bins on [0,1]
    assert abs(decoded.item() - expected_center) < 1e-3


def test_predictor_forward_shapes() -> None:
    cfg = PredictorConfig(embed_dim=64, n_heads=4, n_token_layers=1,
                            n_frame_layers=1, n_terminal_slots=32)
    m = TerminalPredictor(cfg)
    B, K, M, F = 2, 3, 16, 13
    tokens = torch.randn(B, K, M, F)
    mask = torch.ones(B, K, M)
    out = m(tokens, mask, feature_mask=feature_mask_full(torch.device("cpu")))
    assert out["color_id_logits"].shape == (B, 32, 256)
    assert out["cx_logits"].shape == (B, 32, 32)
    assert out["exists_logits"].shape == (B, 32)


def test_invariant_feature_mask_zeros_correct_features() -> None:
    fm = feature_mask_invariant(torch.device("cpu"))
    # 0=color_id, 3-6=bbox should be zero; rest one.
    assert fm[0].item() == 0
    assert fm[3].item() == 0 and fm[4].item() == 0
    assert fm[5].item() == 0 and fm[6].item() == 0
    assert fm[1].item() == 1  # color_rank kept
    assert fm[7].item() == 1 and fm[8].item() == 1  # cx, cy kept
    assert fm[10].item() == 1  # is_singleton kept


def test_predictor_loss_runs_and_is_finite() -> None:
    cfg = PredictorConfig(embed_dim=64, n_heads=4, n_token_layers=1,
                            n_frame_layers=1, n_terminal_slots=32)
    m = TerminalPredictor(cfg)
    B, K, M, F = 2, 3, 16, 13
    tokens = torch.randn(B, K, M, F).clamp(-1, 1)
    mask = torch.ones(B, K, M)
    target = torch.zeros(B, 32, F)
    target[..., 0] = torch.randint(0, 100, (B, 32)).float()  # color_id
    target[..., 7] = torch.rand(B, 32)  # cx
    target[..., 8] = torch.rand(B, 32)  # cy
    target[..., 9] = torch.rand(B, 32) * 5  # aspect
    target[..., 10] = (torch.rand(B, 32) > 0.5).float()  # is_singleton
    target_mask = (torch.rand(B, 32) > 0.7).float()  # ~30% real
    out = m(tokens, mask, feature_mask=feature_mask_full(torch.device("cpu")))
    loss, diag = predictor_loss(out, target, target_mask)
    assert torch.isfinite(loss)
    assert all(isinstance(v, float) for k, v in diag.items() if k != "n_valid_slots")
    assert diag["n_valid_slots"] > 0
