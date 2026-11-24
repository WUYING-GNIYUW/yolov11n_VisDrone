import math

import pytest
import torch
from torch import nn

from . import QueryHead, SparseQueryPlan, cascade_plans, expand_context, map_children, query_focal_loss
from . import query_targets, select_queries


def test_query_children_context_and_strict_threshold():
    logits = torch.full((2, 1, 2, 3), -8.0)
    logits[0, 0, 1, 2] = logits[1, 0, 0, 0] = 8.0
    logits[0, 0, 0, 0] = 0.0
    parents = select_queries(logits, threshold=0.5)
    keys = map_children(torch.cat((parents, parents)), (3, 5))
    assert keys.tolist() == [[0, 2, 4], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    active = expand_context(keys, (3, 5), radius=1)
    assert len(active) == 13
    assert active[active[:, 0] == 0, 1:].tolist() == [[1, 3], [1, 4], [2, 3], [2, 4]]
    logits_at_keys = torch.tensor([0.0, 2.0, -1.0, 0.0, 3.0])
    assert torch.equal(select_queries(logits_at_keys, 0.5, keys), keys[[1, 4]])


def test_query_supervision_uses_nearest_eligible_box_and_pixel_distance():
    logits = torch.randn(2, 1, 5, 6, dtype=torch.float16)
    boxes = [torch.tensor([[0, 4, 2, 6], [3, 1, 11, 9], [0, 0, 0, 0]]), torch.empty(0, 4)]
    variable = query_targets(logits, boxes, stride=(2, 2), scale_range=(2, 9), radius_factor=0.75)
    assert variable.dtype == logits.dtype and not variable.requires_grad
    assert variable[0, 0, 2].tolist() == [1, 0, 1, 1, 1, 1]
    assert not variable[1].any()
    fixed = query_targets(logits, boxes, stride=2, scale_range=(2, 8), radius_px=2)
    assert fixed.nonzero().tolist() == [[0, 0, 2, 0]]


def test_dense_query_focal_training_updates_real_weights():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    head = QueryHead(3, hidden_channels=5, depth=1)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    logits = head(x)
    boxes = [torch.tensor([[2.0, 2.0, 8.0, 12.0]]), torch.empty(0, 4)]
    targets = query_targets(logits, boxes, 4, (0, 32))
    ref = torch.zeros(1, 1, 1, 2)
    zero_loss = query_focal_loss(ref, torch.tensor([[[[0.0, 1.0]]]]))
    assert zero_loss.item() == pytest.approx(math.log(2) / 8)
    before = head[-1].weight.detach().clone()
    query_focal_loss(logits, targets).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert head[-1].weight.grad.isfinite().all()
    opt.step()
    assert not torch.equal(before, head[-1].weight)


def test_submanifold_stack_matches_numerical_reference_at_image_edges():
    torch.manual_seed(8)
    x = torch.randn(2, 4, 7, 9, dtype=torch.double, requires_grad=True)
    net = nn.Sequential(
        nn.Sequential(nn.Conv2d(4, 4, 3, padding=1, groups=2), nn.BatchNorm2d(4), nn.SiLU()),
        nn.Conv2d(4, 8, 3, padding=2, dilation=2, groups=4),
        nn.ReLU(),
        nn.Conv2d(8, 3, 1),
    ).double().eval()
    keys = torch.tensor([[1, 6, 8], [0, 0, 0], [1, 1, 2]])
    active = expand_context(keys, (7, 9), radius=1)
    plan = SparseQueryPlan(keys, torch.cat((active.flip(0), active)), (2, 7, 9))
    actual = plan.run(net, x, chunk_size=3)
    support = x.new_zeros((2, 1, 7, 9))
    bi, yi, xi = active.T
    support[bi, 0, yi, xi] = 1
    expected = x * support
    for op in (*net[0], *list(net)[1:]):
        expected = op(expected) * support
    b, y, k = keys.T
    expected = expected.permute(0, 2, 3, 1)[b, y, k]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0])
    full = SparseQueryPlan(keys, expand_context(keys, (7, 9), radius=3), (2, 7, 9)).run(net, x)
    torch.testing.assert_close(full, net(x).permute(0, 2, 3, 1)[b, y, k])
    assert not torch.allclose(actual, full)


def test_two_stage_cascade_never_promotes_context_or_other_images():
    coarse = torch.full((2, 1, 2, 2), -8.0)
    coarse[0, 0, 0, 0] = 8.0
    middle = torch.full((2, 1, 3, 4), -8.0)
    middle[0, 0, 1, 1] = middle[0, 0, 2, 2] = middle[1, 0, 0, 0] = 8.0
    fine = torch.randn(2, 4, 5, 7)
    detector = nn.Conv2d(4, 6, 3, padding=1)
    plans = cascade_plans((coarse, middle, fine), (nn.Identity(), nn.Identity()), threshold=0.5, radius=1)
    assert plans[0].active.tolist()[-1] == [0, 2, 2]
    assert plans[1].keys.tolist() == [[0, 2, 2], [0, 2, 3], [0, 3, 2], [0, 3, 3]]
    b, y, x = plans[1].keys.T
    torch.testing.assert_close(plans[1].run(detector, fine), detector(fine).permute(0, 2, 3, 1)[b, y, x])
    empty = cascade_plans((coarse, middle, fine), (nn.Identity(), QueryHead(1, depth=0)), threshold=1)
    assert all(p.keys.shape == (0, 3) and p.active.shape == (0, 3) for p in empty)
    assert empty[-1].run(detector, fine).shape == (0, 6)


def test_profile_only_active_kernel_windows_reach_convolution():
    feature = torch.randn(1, 2, 51, 39)
    keys = torch.tensor([[0, 19, 25]])
    active = expand_context(keys, feature.shape[-2:], radius=1)
    plan = SparseQueryPlan(keys, active, (1, 51, 39))
    layer = nn.Conv2d(2, 4, 3, padding=1)
    with torch.autograd.profiler.profile(record_shapes=True) as trace:
        actual = plan.run(layer, feature, chunk_size=4)
    windows = [event.input_shapes[0] for event in trace.function_events if event.name == "aten::conv2d"]
    assert windows == [[4, 2, 3, 3], [4, 2, 3, 3], [1, 2, 3, 3]]
    torch.testing.assert_close(actual[0], layer(feature)[0, :, 19, 25])


def test_unsupported_spatial_operators_cannot_silently_run_dense():
    key = torch.tensor([[0, 0, 0]])
    with pytest.raises(ValueError, match="active context"):
        SparseQueryPlan(key, key[:0], (1, 3, 3))
    with pytest.raises(ValueError, match="outside"):
        SparseQueryPlan(key + 3, key, (1, 3, 3))
    feature = torch.randn(1, 2, 3, 3)
    plan = SparseQueryPlan(key, key, (1, 3, 3))
    with pytest.raises(ValueError, match="stride 1"):
        plan.run(nn.Conv2d(2, 2, 3, stride=2, padding=1), feature)
    with pytest.raises(ValueError, match="running statistics"):
        plan.run(nn.BatchNorm2d(2), feature)
    with pytest.raises(TypeError, match="AdaptiveAvgPool2d"):
        plan.run(nn.AdaptiveAvgPool2d(1), feature)
