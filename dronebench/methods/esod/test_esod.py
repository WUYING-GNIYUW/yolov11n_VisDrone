import math

import pytest
import torch
from torch import nn

from . import ObjSeeker, PatchLayout, foreground_loss, foreground_peaks, foreground_targets, select_patches


def test_gaussian_hybrid_overlap_and_subcell_box():
    prototype = torch.zeros(2, 1, 9, 10)
    boxes = torch.tensor([[1, 1, 8, 8], [8.1, 7.2, 8.3, 7.4], [1, 1, 8, 8], [-3, 0, -1, 3]])
    ids = torch.tensor([0, 1, 0, 0])
    gaussian = foreground_targets(prototype, boxes, ids, 1)
    assert gaussian[0, 0, 4, 4] == 1
    assert gaussian[0, 0, 1, 1].item() == pytest.approx(0.5 ** ((5 / 7) ** 2))
    assert gaussian[0, 0, 0, 0] == 0
    assert gaussian[1, 0, 7, 8] == 1 and gaussian[1].count_nonzero() == 1
    masks = torch.zeros(4, 9, 10)
    masks[0, 4, 4] = masks[2, 4, 4] = 0.75
    masks[1, 0, 0] = 1
    hybrid = foreground_targets(prototype, boxes, ids, 1, masks)
    assert hybrid[0, 0, 4, 4] == 0.75 and hybrid[0].count_nonzero() == 1
    assert torch.equal(hybrid[1], gaussian[1])
    scaled = boxes * torch.tensor([3, 2, 3, 2])
    torch.testing.assert_close(foreground_targets(prototype, scaled, ids, (2, 3)), gaussian)


def test_empty_targets_and_loss_formula():
    z = torch.zeros(2, 1, 2, 2, dtype=torch.float16, requires_grad=True)
    t = foreground_targets(z, torch.empty(0, 4), torch.empty(0, dtype=torch.long), stride=8)
    assert t.count_nonzero() == 0
    loss = foreground_loss(z, t)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(20 * 0.75 * 0.25 * math.log(2) + (1 - 1 / 3))
    loss.backward()
    assert torch.isfinite(z.grad).all() and (z.grad > 0).all()


@pytest.mark.parametrize("algorithm", ["grid", "greedy"])
def test_adaptive_rectangles_real_crops_and_gradient(algorithm):
    p = torch.zeros(2, 1, 7, 11)
    p[0, 0, 1, 6], p[0, 0, 5, 10], p[1, 0, 6, 1] = 0.9, 0.7, 0.5
    layout = select_patches(p, (3, 4), algorithm=algorithm)
    expected = [[0, 6, 1, 10, 4], [0, 7, 4, 11, 7], [1, 1, 4, 5, 7]]
    assert sorted(layout.offsets.tolist()) == expected
    x = torch.arange(2 * 3 * 7 * 11.0).reshape(2, 3, 7, 11).requires_grad_()
    blocks = layout.crop(x)
    assert blocks.shape == (3, 3, 3, 4)
    count = torch.zeros_like(x)
    for block, (i, x1, y1, x2, y2) in zip(blocks, layout.offsets.tolist()):
        torch.testing.assert_close(block, x[i, :, y1:y2, x1:x2])
        count[i, :, y1:y2, x1:x2] += 1
    blocks.sum().backward()
    torch.testing.assert_close(x.grad, count)
    finer = x.detach().repeat_interleave(2, 2).repeat_interleave(2, 3)
    expected_finer = blocks.detach().repeat_interleave(2, 2).repeat_interleave(2, 3)
    torch.testing.assert_close(layout.crop(finer, scale=2), expected_finer)


def test_plateaus_exact_duplicates_and_greedy_size_priority():
    p = torch.zeros(2, 1, 5, 5)
    p[:, 0, 3, 3:] = 0.5
    assert foreground_peaks(p).sum() == 4
    assert select_patches(p, (4, 4)).offsets.tolist() == [[0, 1, 1, 5, 5], [1, 1, 1, 5, 5]]
    p = torch.zeros(1, 1, 14, 14)
    p[0, 0, 1, 1] = 1
    p[0, 0, 5:8, 5:8] = 0.6
    p[0, 0, 6, 6] = 0.8
    assert select_patches(p, (5, 5), algorithm="greedy").offsets.tolist() == [[0, 5, 5, 10, 10], [0, 1, 1, 6, 6]]


@pytest.mark.parametrize("algorithm", ["grid", "greedy"])
def test_padding_empty_pred_and_index_validity(algorithm):
    p = torch.zeros(2, 1, 2, 3)
    empty = select_patches(p, (4, 5), algorithm=algorithm)
    x = torch.ones(2, 2, 2, 3, requires_grad=True)
    empty_blocks = empty.crop(x)
    assert empty.offsets.shape == (0, 5) and empty_blocks.shape == (0, 2, 4, 5)
    empty_blocks.sum().backward()
    assert x.grad.count_nonzero() == 0
    loc, pi, source = empty.project_boxes(torch.ones(2, 4), torch.tensor([0, 1]), 2)
    out, batch = empty.restore_boxes(loc, pi, 2)
    assert out.shape == (0, 4) and source.shape == batch.shape == (0,)
    p[1, 0, 0, 0] = 1
    layout = select_patches(p, (4, 5), algorithm=algorithm)
    blocks = layout.crop(x)
    assert layout.offsets.tolist() == [[1, 0, 0, 3, 2]] and blocks.shape == (1, 2, 4, 5)
    torch.testing.assert_close(blocks[0, :, :2, :3], x[1])
    assert blocks[:, :, 2:, :].count_nonzero() == 0 and blocks[:, :, :, 3:].count_nonzero() == 0
    xy = torch.tensor([[5.99, 3.99], [6, 0], [-1, 0], [2, 4]])
    assert layout.valid_centers(xy, torch.zeros(4, dtype=torch.long), 2).tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        layout.restore_boxes(torch.ones(1, 4), torch.tensor([-1]), 2)


def test_half_open_ownership_overlap_and_full_box_round_trip():
    off = torch.tensor([[0, 0, 1, 4, 4], [0, 2, 1, 6, 4], [1, 0, 0, 4, 3]])
    layout = PatchLayout(off, (5, 6), (3, 4), 2)
    boxes = torch.tensor([[2.0, 0.0, 16.0, 6.0], [11.0, 3.0, 13.0, 5.0], [-2.0, 0.0, 6.0, 4.0]])
    ids = torch.tensor([0, 0, 1])
    local, pi, gi = layout.project_boxes(boxes, ids, (2, 3))
    assert pi.tolist() == [0, 1, 1, 2] and gi.tolist() == [0, 0, 1, 2]
    assert local[0, 1] < 0 and local[1, 0] < 0
    restored, image_ids = layout.restore_boxes(local, pi, (2, 3))
    torch.testing.assert_close(restored, boxes[gi])
    assert torch.equal(image_ids, ids[gi])
    clipped, _, _ = layout.project_boxes(boxes, ids, (2, 3), clip=True)
    torch.testing.assert_close(clipped[0], torch.tensor([2.0, 0.0, 12.0, 4.0]))
    bounded, _ = layout.restore_boxes(local, pi, (2, 3), image_hw=(10, 18))
    assert bounded[-1, 0] == 0


def test_training_and_predicted_only_routing_run_the_real_branch():
    torch.manual_seed(18)
    stem = nn.Conv2d(3, 4, 3, 2, 1)
    seeker, branch = ObjSeeker(4, kernel_size=3), nn.Conv2d(4, 5, 3, padding=1)
    x = stem(torch.rand(2, 3, 20, 24))
    logits = seeker(x)
    t = foreground_targets(logits, torch.tensor([[4.0, 4.0, 6.0, 6.0]]), torch.tensor([0]), stride=2)
    route = select_patches(t, (4, 4))
    block_pred = branch(route.crop(x))
    assert block_pred.shape == (1, 5, 4, 4) and route.offsets[0, 0] == 0
    loss = foreground_loss(logits, t) + block_pred.square().mean()
    loss.backward()
    for param in (stem.weight, seeker[-1].weight, branch.weight):
        assert param.grad.abs().sum() > 0 and torch.isfinite(param.grad).all()
    seeker.eval()
    with torch.no_grad():
        prob = seeker(x).sigmoid()
        predicted = select_patches(prob, (4, 4), threshold=prob.max().item())
        pred = branch(predicted.crop(x))
    assert 0 < len(pred) == len(predicted.offsets) and pred.shape[-2:] == (4, 4)


def test_contract_rejects_misregistered_features_and_logits():
    p = torch.ones(1, 1, 4, 4)
    layout = select_patches(p, (2, 2))
    with pytest.raises(ValueError, match="aligned"):
        layout.crop(torch.ones(1, 1, 7, 8), scale=2)
    with pytest.raises(ValueError, match="probabilities"):
        select_patches(p * 2, (2, 2))
    with pytest.raises(ValueError, match="indices"):
        foreground_targets(p, torch.ones(1, 4), torch.tensor([0.5]), stride=4)
