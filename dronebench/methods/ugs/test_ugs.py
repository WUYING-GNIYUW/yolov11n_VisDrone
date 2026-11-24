import math

import pytest
import torch

from dronebench.methods.ugs import (
    decode_logits,
    decode_ltrb,
    encode_ltrb,
    encode_two_hot,
    localization_ce,
    make_support,
    refinement_loss,
    uncertainty_entropy,
    uncertainty_perturbations,
)
from dronebench.methods.ugs.example import run_example


def test_equation_9_and_uniform_limit():
    s = make_support(5, 2, math.log(2), dtype=torch.float64)
    expected = torch.tensor([-2, -2 / 3, 0, 2 / 3, 2], dtype=torch.float64)
    torch.testing.assert_close(s, expected)
    torch.testing.assert_close(make_support(5, 2, 0), torch.linspace(-2, 2, 5))
    even = make_support(16, 15, 2 / 15)
    torch.testing.assert_close(even, -even.flip(0))
    assert not (even == 0).any()
    assert even[8] - even[7] < even[-1] - even[-2]
    with pytest.raises(ValueError, match="distinct"):
        make_support(9, 1, 10000, dtype=torch.float32)


def test_adjacent_physical_interpolation_clips_only_unrepresentable_targets():
    s = make_support(5, 2, math.log(2), dtype=torch.float64)
    t = torch.stack((s[0] - 2, s[0], s[1], s[2], (3 * s[3] + s[4]) / 4, s[-1], s[-1] + 2))
    enc = encode_two_hot(t, s)
    assert torch.equal(enc.right, enc.left + 1)
    assert enc.clipped.tolist() == [True, False, False, False, False, False, True]
    assert enc.right_weight[4].item() == pytest.approx(0.25)
    q = torch.zeros(t.shape + s.shape, dtype=t.dtype)
    q.scatter_(-1, enc.left[:, None], (1 - enc.right_weight)[:, None])
    q.scatter_(-1, enc.right[:, None], enc.right_weight[:, None])
    torch.testing.assert_close(q.sum(-1), torch.ones_like(t))
    torch.testing.assert_close(decode_logits(q.log(), s), t.clamp(-2, 2))
    assert torch.isfinite(localization_ce(q.log(), t, s)).all()


def test_ce_gradient_is_prediction_minus_two_hot_and_half_computes_in_float():
    s = make_support(4, 3, 0.5, dtype=torch.float64)
    t = ((s[1] + 3 * s[2]) / 4).reshape(1)
    logits = torch.tensor([[0.1, 2.0, -0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    loss = localization_ce(logits, t, s)
    q = logits.new_tensor([[0, 0.25, 0.75, 0]])
    torch.testing.assert_close(loss, -(q * logits.log_softmax(-1)).sum(-1))
    loss.sum().backward()
    torch.testing.assert_close(logits.grad, logits.softmax(-1) - q)
    half = torch.tensor([[-1000, -1000, 1000, -1000]], dtype=torch.float16)
    assert uncertainty_entropy(half).dtype == torch.float32
    assert torch.isfinite(localization_ce(half, t.float(), s.float())).all()
    assert decode_logits(half, s.float()).item() == pytest.approx(s[2].item())


def test_um_is_shannon_entropy_of_coordinate_distribution():
    uniform = torch.zeros(3, 5)
    torch.testing.assert_close(uncertainty_entropy(uniform), torch.full((3,), math.log(5)))
    logits = torch.tensor([[2.0, -1.0, 3.0, 0.0, 0.3]], requires_grad=True)
    p = logits.detach().softmax(-1)
    h = -(p * p.log()).sum(-1)
    torch.testing.assert_close(uncertainty_entropy(logits), h)
    assert h.item() < math.log(5)
    torch.testing.assert_close(uncertainty_entropy(logits.flip(-1)), h)
    uncertainty_entropy(logits).sum().backward()
    torch.testing.assert_close(logits.grad, -p * (p.log() + h[:, None]))
    assert uncertainty_entropy(torch.tensor([[0.0, -float("inf")]])).item() == 0


def test_signed_ltrb_roundtrip_across_strides_and_invalid_prediction_policy():
    points = torch.tensor([[0.5, 0.5], [2.5, 1.5]])
    stride = torch.tensor([[8.0], [16.0]])
    boxes_px = torch.tensor([[[8.0, 0.0, 16.0, 8.0], [16.0, 32.0, 32.0, 48.0]]])
    target = encode_ltrb(points, boxes_px / stride)
    expected = torch.tensor([[[-0.5, 0.5, 1.5, 0.5], [1.5, -0.5, -0.5, 1.5]]])
    torch.testing.assert_close(target, expected)
    torch.testing.assert_close(decode_ltrb(points, target) * stride, boxes_px)
    inverted = decode_ltrb(points[0], torch.tensor([-2.0, 0.5, 1.0, 1.0]))
    assert inverted[0] > inverted[2]


def test_refinement_is_per_image_entropy_ascent_on_connected_features():
    f = torch.tensor([[[0.0, 1.0, 3.0]], [[0.5, -1.0, 3.0]]], dtype=torch.float64, requires_grad=True)
    unused = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    features = (f, unused)

    def objective(fs):
        logits = torch.stack((fs[0], -fs[0]), dim=-1)
        return uncertainty_entropy(logits)[..., :2].mean()

    clean = objective(features)
    grad = torch.autograd.grad(clean, f, retain_graph=True)[0]
    rho = 0.02
    deltas = uncertainty_perturbations(features, clean, rho)
    flat = grad.flatten(1)
    torch.testing.assert_close(deltas[0].flatten(1), rho * flat / flat.norm(dim=1, keepdim=True))
    assert deltas[0][0, 0, 0] == 0
    assert (deltas[0][..., 2] == 0).all() and (deltas[1] == 0).all()
    assert all(not d.requires_grad for d in deltas) and f.grad is None
    perturbed = refinement_loss(features, clean, objective, rho)
    assert perturbed.item() > clean.item()
    regularized = refinement_loss(features, clean, objective, rho, feature_penalty=0.125)
    penalty = sum(v.flatten(1).square().sum(1).mean() for v in features)
    torch.testing.assert_close(regularized - perturbed, 0.125 * penalty)
    (clean + perturbed).backward()
    assert f.grad is not None and f.grad.isfinite().all() and f.grad.abs().sum() > 0


def test_empty_positives_and_zero_gradient_have_no_spurious_update():
    f = torch.ones(1, 2, 5, requires_grad=True)
    support = make_support(5, 2, 1)
    logits = f[:, :0]
    ce = localization_ce(logits, torch.empty(1, 0), support)
    assert ce.shape == (1, 0)
    clean = uncertainty_entropy(logits).sum()
    delta = uncertainty_perturbations((f,), clean, 0.5)[0]
    assert torch.isfinite(delta).all() and not delta.any()
    (clean + ce.sum()).backward()
    assert torch.equal(f.grad, torch.zeros_like(f))


def test_minimal_example_trains_and_decodes_without_labels_at_inference():
    result = run_example()
    assert math.isfinite(result["clean_ce"]) and result["clean_ce"] > 0
    assert math.isfinite(result["head_gradient_norm"]) and result["head_gradient_norm"] > 0
    assert result["perturbed_um"] > result["clean_um"]
    assert result["target_ltrb_grid"][0][0][0] < 0
    boxes = torch.tensor(result["predicted_xyxy_px"])
    assert boxes.shape == (1, 2, 4) and boxes.isfinite().all()
