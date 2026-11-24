import math
import os
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile

import pytest
import torch
from torch import nn

from dronebench.methods.dm_efs import (
    ExpandedPyramid,
    SizeFeatureCodebook,
    SizeRangePredictor,
    expand_features,
    paired_box_ious,
    predicted_size_range,
    run_selected,
    size_range_loss,
)


def test_efs_uses_real_shallow_features_and_skips_unused_neck_steps():
    torch.manual_seed(3)
    maps = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in ((2, 13, 17), (3, 7, 9), (6, 4, 5))]
    features = expand_features(maps[:1], maps[1:])
    assert all(x is y for x, y in zip(features, maps))
    with pytest.raises(ValueError, match="finest to coarsest"):
        expand_features(maps[2:], maps[:2])
    neck = ExpandedPyramid([2, 3, 6], width=5)
    full = neck(features)
    assert [x.shape for x in full] == [(2, 5, 13, 17), (2, 5, 7, 9), (2, 5, 4, 5)]
    seen = []
    handle = neck.down[-1].register_forward_hook(lambda *_: seen.append(True))
    prefix = neck(features, last_level=1)
    handle.remove()
    assert len(prefix) == 2 and not seen
    assert all(torch.allclose(x, y) for x, y in zip(prefix, full))
    full[-1].square().mean().backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in maps)


def test_focal_targets_use_input_pixel_area_floor_and_ignore_empty_images():
    probs = torch.tensor([[[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]] * 2, dtype=torch.double)
    logits = probs.log().requires_grad_()
    boxes = [torch.tensor([[0, 0, 2, 4], [1, 2, 5, 6]]), torch.empty(0, 4)]
    loss = size_range_loss(logits, boxes)
    expected = 0.04 * (-(1 - 0.2) ** 2 * math.log(0.2) - (1 - 0.7) ** 2 * math.log(0.7))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]) == 6 and not logits.grad[1].any()
    zero = size_range_loss(logits, [torch.empty(0, 4)] * 2)
    assert zero.item() == 0 and not torch.autograd.grad(zero, logits)[0].any()
    with pytest.raises(ValueError, match="non-positive"):
        size_range_loss(logits[:1], [torch.tensor([[3, 1, 1, 2]])])


def test_controller_updates_both_branches_and_backbone_then_decodes_ordered_sizes():
    torch.manual_seed(19)
    features = [torch.rand(3, 4, 11, 15, requires_grad=True), torch.rand(3, 8, 6, 8, requires_grad=True)]
    controller = SizeRangePredictor([4, 8], max_size=24, hidden_channels=12)
    logits = controller(features)
    assert logits.shape == (3, 2, 23)
    boxes = [torch.tensor([[0.0, 0, size, size]]) for size in (2, 7, 24)]
    optimizer = torch.optim.SGD(controller.parameters(), lr=0.2)
    before = [branch[-1].weight.detach().clone() for branch in controller.branches]
    size_range_loss(logits, boxes).backward()
    optimizer.step()
    assert all(f.grad.abs().sum() > 0 for f in features)
    assert all(not torch.equal(w, branch[-1].weight) for w, branch in zip(before, controller.branches))
    swapped = torch.tensor([[[0.0, 0, 2], [2.0, 0, 0]]])
    assert predicted_size_range(swapped).tolist() == [[2, 4]]


def test_votes_are_binary_per_image_cumulative_and_checkpointable():
    book = SizeFeatureCodebook(num_levels=3, max_size=12)
    gt = torch.tensor([[0.0, 0, 4, 4], [1.0, 1, 5, 5], [0.0, 0, 12, 3], [0.0, 0, 9, 9]])
    matched = torch.zeros(4, 3, 4)
    matched[0, 0] = torch.tensor([0.0, 0, 2, 4])
    matched[1, 0], matched[2, 1], matched[3, 2] = gt[1], gt[2], gt[3]
    ious = paired_box_ious(gt, matched)
    assert ious[0, 0] == 0.5
    book.observe(gt, ious)
    assert book.votes.sum().item() == 3 and book.votes[0, 4 - 2].item() == 1
    book.observe(gt, ious)
    assert book.votes.sum().item() == 6 and book.votes[2, 9 - 2].item() == 2
    assert not list(book.parameters())
    restored = SizeFeatureCodebook(3, 12)
    restored.load_state_dict(book.state_dict())
    assert torch.equal(restored.votes, book.votes)
    assert restored.level_table().tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    neck, head = restored.gates(torch.tensor([[4, 9], [6, 6], [3, 999]]))
    assert head.tolist() == [[True, False, True], [False, True, False], [True, False, True]]
    assert neck.tolist() == [[True, True, True], [True, True, False], [True, True, True]]


def test_no_votes_open_all_levels_and_nonmonotone_codebook_keeps_literal_neck_gate():
    book = SizeFeatureCodebook(2, 7)
    book.observe(torch.empty(0, 4), torch.empty(0, 2))
    book.observe(torch.tensor([[0, 0, 4, 4]]), torch.full((1, 2), 0.499))
    assert book.level_table().eq(-1).all()
    assert all(g.all() for g in book.gates(torch.tensor([[2, 7]])))
    gt = torch.tensor([[0.0, 0, 1, 1], [0.0, 0, 10, 10]])
    book.observe(gt, torch.tensor([[0.1, 0.9], [0.8, 0.1]]))
    neck, head = book.gates(torch.tensor([[2, 7]]))
    assert neck.tolist() == [[True, False]] and head.tolist() == [[True, True]]
    book.observe(gt[:1], torch.tensor([[0.9, 0.1]]))
    assert book.level_table()[0] == 0


def test_head_dispatch_runs_real_convolutions_only_for_selected_image_level_pairs():
    maps = [torch.randn(3, 2, d, d, requires_grad=True) for d in (8, 4, 2)]
    ops = nn.ModuleList(nn.Conv2d(2, 3, 1) for _ in maps)
    gates = torch.tensor([[False, False, True], [True, False, False], [True, False, True]])
    calls = []
    handles = [
        op.register_forward_hook(lambda _, inputs, output, i=i: calls.append((i, inputs[0].shape[0])))
        for i, op in enumerate(ops)
    ]
    result = run_selected(maps, gates, ops)
    for handle in handles:
        handle.remove()
    assert calls == [(0, 2), (2, 2)]
    assert {i: rows.tolist() for i, (rows, _) in result.items()} == {0: [1, 2], 2: [0, 2]}
    for i, (rows, out) in result.items():
        assert torch.allclose(out, ops[i](maps[i])[rows])
    sum(out.square().sum() for _, out in result.values()).backward()
    assert maps[1].grad is None and not maps[0].grad[0].any() and not maps[2].grad[1].any()


@pytest.mark.skipif(not os.environ.get("VISDRONE_VAL_ZIP"), reason="Set VISDRONE_VAL_ZIP for the real-image check")
@torch.no_grad()
def test_real_image_range_loss_and_label_free_inference():
    from PIL import Image

    with ZipFile(os.environ["VISDRONE_VAL_ZIP"]) as archive:
        files = sorted(PurePosixPath(name) for name in archive.namelist() if name.endswith(".jpg"))
        path = next(p for p in files if p.parent.name == "images" and not p.name.startswith("."))
        labels = path.parent.parent / "annotations" / path.with_suffix(".txt").name
        rows = [line.split(",") for line in archive.read(str(labels)).decode("utf-8-sig").splitlines()]
        with Image.open(BytesIO(archive.read(str(path)))) as image:
            w, h = image.size
            scale = 320 / max(w, h)
            sw, sh = round(w * scale), round(h * scale)
            rgb = image.convert("RGB").resize((sw, sh))
            x = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(sh, sw, 3).permute(2, 0, 1)
    xywh = [[float(v) for v in row[:4]] for row in rows if int(row[4]) > 0 and 1 <= int(row[5]) <= 10]
    boxes = torch.tensor(xywh) * torch.tensor([sw / w, sh / h, sw / w, sh / h])
    boxes[:, 2:] += boxes[:, :2]
    x = x[None].float() / 255
    levels = [nn.functional.avg_pool2d(x, stride) for stride in (4, 8, 16)]
    features = expand_features(levels[:1], levels[1:])
    logits = SizeRangePredictor([3, 3, 3], max_size=320).eval()(features)
    book = SizeFeatureCodebook(3, 320).eval()
    _, mask = book.gates(predicted_size_range(logits))
    out = run_selected(features, mask, [nn.Conv2d(3, 1, 1) for _ in levels])
    assert len(boxes) > 0 and torch.isfinite(size_range_loss(logits, [boxes]))
    assert not book.votes.any() and set(out) == {0, 1, 2}
