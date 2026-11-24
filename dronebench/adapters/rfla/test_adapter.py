import json
from pathlib import Path

import pytest
import torch

from dronebench.adapters.rfla import RFLAAdapter
from dronebench.methods.rfla import hierarchical_assign
from dronebench.model import build_model
from ultralytics.utils.tal import bbox2dist, make_anchors


@pytest.mark.parametrize("metric", ["kl", "wd"])
def test_mask_before_assignment_across_real_anchor_levels(metric):
    strides = [4, 8, 16, 32]
    features = [torch.empty(2, 1, 64 // s, 96 // s) for s in strides]
    anchors, step = make_anchors(features, strides, 0.5)
    order = torch.arange(len(anchors) - 1, -1, -1)
    points, step = (anchors * step)[order], step[order]
    radii = [1, 8, 3, 14]
    radius = torch.cat([torch.full((f[0].numel(),), float(r)) for f, r in zip(features, radii)])[order]
    options = {"radii_px": radii, "metric": metric, "topk": 1, "beta": 0.8}
    adapter = RFLAAdapter(torch.tensor(strides), options)
    boxes = torch.tensor(
        [
            [[0, 0, 0, 0], [17, 21, 19, 23], [0, 0, 0, 0], [34, 2, 62, 30]],
            [[4, 4, 6, 6], [0, 0, 0, 0], [61, 29, 67, 35], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([[99, 1, 99, 0], [2, 99, 0, 99]])[..., None]
    valid = labels < 3
    active = torch.stack((points[:, 1] <= 48, points[:, 0] > 48))
    targets, scores, fg, idx, diag = adapter.assign(points, step, labels, boxes, valid, active, 3)
    assert idx.dtype == torch.long and fg.dtype == torch.bool and scores.dtype == torch.float32
    assert not any(t.requires_grad for t in (targets, scores, fg, idx, *diag.values()))
    assert all(t.dtype == torch.long and t.device == points.device for t in diag.values())
    for i in range(2):
        gi = valid[i, :, 0].nonzero().flatten()
        ai = active[i].nonzero().flatten()
        hla = hierarchical_assign(points[ai], radius[ai], boxes[i, gi], topk=1, beta=0.8, metric=metric)
        pos = hla.gt_index >= 0
        expected = torch.full_like(idx[i], -1)
        expected[ai[pos]] = gi[hla.gt_index[pos]]
        assert torch.equal(idx[i], expected) and torch.equal(fg[i], expected >= 0)
        assert torch.equal(targets[i, fg[i]], boxes[i, expected[fg[i]]])
        assert torch.equal(scores[i, fg[i]].argmax(1), labels[i, expected[fg[i]], 0])
        assert torch.equal(scores[i].sum(1), fg[i].float())
        assert torch.equal(diag["positive_per_gt"][i, gi], hla.counts)
        assert diag["valid_gt_per_image"][i] == len(gi)
        assert diag["unmatched_gt_per_image"][i] == (hla.counts == 0).sum()
        assert diag["positive_per_image"][i] == fg[i].sum()
        assert diag["outside_gt_per_image"][i] == hla.outside_gt.sum()
        assert diag["stage1_per_image"][i] == (hla.stage == 1).sum()
        assert diag["stage2_per_image"][i] == (hla.stage == 2).sum()
    assert torch.equal(step[fg[0]].unique(), torch.tensor([4.0, 32.0]))
    assert not targets[~fg].any() and not diag["positive_per_gt"][~valid[..., 0]].any()


@pytest.mark.parametrize(
    ("n", "g", "has_gt", "enabled"),
    [(3, 0, False, True), (3, 2, False, True), (3, 2, True, False), (0, 2, True, True)],
)
def test_empty_gt_and_empty_candidate_pools(n, g, has_gt, enabled):
    points = torch.ones(n, 2)
    valid = torch.full((2, g, 1), has_gt)
    boxes = torch.tensor([0.0, 0.0, 2.0, 2.0]).expand(2, g, 4)
    result = RFLAAdapter([8], {}).assign(
        points, torch.full((n, 1), 8.0), valid.long(), boxes, valid, torch.full((2, n), enabled), 3
    )
    targets, scores, fg, idx, diag = result
    assert targets.shape == (2, n, 4) and scores.shape == (2, n, 3) and fg.shape == idx.shape == (2, n)
    assert not any(t.any() for t in (targets, scores, fg)) and torch.all(idx == -1)
    expected = valid.sum((1, 2))
    assert torch.equal(diag["valid_gt_per_image"], expected)
    assert torch.equal(diag["unmatched_gt_per_image"], expected)
    assert diag["positive_per_gt"].shape == (2, g)
    assert not any(diag[key].any() for key in diag.keys() - {"valid_gt_per_image", "unmatched_gt_per_image"})


def test_conflict_preserves_outside_geometry_for_stock_dfl():
    adapter = RFLAAdapter([8], {"topk": 1})
    points = torch.tensor([[4.0, 4.0], [12.0, 12.0]])
    step = torch.full((2, 1), 8.0)
    boxes = torch.tensor([[[6.0, 6.0, 18.0, 18.0], [0.0] * 4, [10.0, 10.0, 14.0, 14.0]]])
    labels = torch.tensor([[[0], [99], [1]]])
    targets, scores, fg, idx, diag = adapter.assign(
        points, step, labels, boxes, labels < 2, torch.tensor([[True, False]]), 2
    )
    assert idx.tolist() == [[2, -1]] and scores.tolist() == [[[0.0, 1.0], [0.0, 0.0]]]
    assert diag["positive_per_gt"].tolist() == [[0, 0, 1]]
    assert diag["outside_gt_per_image"].tolist() == diag["unmatched_gt_per_image"].tolist() == [1]
    assert diag["stage1_per_image"].tolist() == [1] and diag["stage2_per_image"].tolist() == [0]
    assert torch.equal(targets[fg], boxes[:, 2])
    signed = bbox2dist(points / step, targets / step)[fg]
    clipped = bbox2dist(points / step, targets / step, reg_max=15)[fg]
    assert torch.equal(signed, torch.tensor([[-0.75, -0.75, 1.25, 1.25]]))
    assert torch.equal(clipped, signed.clamp(0, 14.99))


def test_default_radii_and_invalid_options():
    assert RFLAAdapter([8, 16, 32], {}).radii_px == (16.0, 32.0, 64.0)
    assert RFLAAdapter([4, 8, 16, 32], {}).radii_px == (8.0, 16.0, 32.0, 64.0)
    invalid = ({"top_k": 2}, {"radii_px": [1]}, {"radii_px": [1, 0, 1]}, {"topk": 0}, {"beta": 1.1}, {"metric": "iou"})
    for options in invalid:
        with pytest.raises(ValueError):
            RFLAAdapter([8, 16, 32], options)


def test_yolo11n_public_loss_backward_and_decode():
    model = build_model(json.loads(Path(__file__).with_name("single.json").read_text()), nc=3).train()
    batch = {
        "img": torch.rand(2, 3, 64, 96),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.015, 0.02, 0.01, 0.015], [0.6, 0.5, 0.25, 0.3]]),
    }
    raw = model(batch["img"])
    for k in ("boxes", "scores"):
        raw[k].retain_grad()
    loss, items = model.loss(batch, preds=raw)
    diag = model.last_diagnostics
    assert set(items) == {"box_loss", "cls_loss", "dfl_loss"}
    assert (loss > 0).all() and loss.isfinite().all()
    assert diag["valid_gt_per_image"].tolist() == [1, 1] and not diag["unmatched_gt_per_image"].any()
    assert diag["outside_gt_per_image"][0] == diag["positive_per_image"][0] > 0
    loss.sum().backward()
    for grad in (raw["boxes"].grad, raw["scores"].grad, model.model[0].conv.weight.grad):
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0
    with torch.no_grad():
        masked, _ = model.loss(batch, preds={**raw, "active": torch.zeros_like(raw["active"])})
        assert torch.equal(masked, torch.zeros_like(loss))
        assert model.last_diagnostics["unmatched_gt_per_image"].tolist() == [1, 1]
        decoded, eval_raw = model.eval()(batch["img"])
        assert decoded.shape == (2, 7, 126) and decoded.isfinite().all()
        torch.testing.assert_close(decoded, model.model[-1]._inference(eval_raw))
