import csv
import io
import os
from zipfile import ZipFile

import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from dronebench.methods.rfla import hierarchical_assign, receptive_field_scores


def test_rfd_matches_independent_gaussians_and_squared_wasserstein():
    points = torch.tensor([[1.0, 2.0], [9.0, -3.0], [-2.0, 4.0]], dtype=torch.float64, requires_grad=True)
    radii = torch.tensor([[1.0, 3.0], [2.0, 0.5], [1.25, 7.0]], dtype=torch.float64)
    boxes = torch.tensor([[0.0, -1.0, 2.0, 5.0], [2.0, 3.0, 8.0, 4.0]], dtype=torch.float64)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) / 2
    rf = Independent(Normal(points, radii), 1)
    gt = Independent(Normal(centers[:, None], half[:, None]), 1)
    kl = receptive_field_scores(points, radii, boxes)
    wd = receptive_field_scores(points, radii, boxes, metric="wd")
    torch.testing.assert_close(kl, (1 + kl_divergence(rf, gt)).reciprocal())
    w2 = torch.cdist(torch.cat((centers, half), 1), torch.cat((points, radii), 1)).square()
    torch.testing.assert_close(wd, (1 + w2).reciprocal())
    assert kl[0, 0] == wd[0, 0] == 1
    scale = points.new_tensor([2.0, 0.5])
    shift = points.new_tensor([-20.0, 10.0])
    scaled = receptive_field_scores(points * scale + shift, radii * scale, boxes * scale.repeat(2) + shift.repeat(2))
    torch.testing.assert_close(scaled, kl)
    torch.testing.assert_close(receptive_field_scores(points * 2, radii * 2, boxes * 2, metric="wd"), 1 / (1 + 4 * w2))
    (kl + wd).sum().backward()
    assert torch.isfinite(points.grad).all() and points.grad.abs().sum() > 0


def test_shrink_only_radius_and_keep_outside_candidates():
    points = torch.tensor([[3.0, 5.0], [7.0, 5.0], [13.0, 5.0]], requires_grad=True)
    radii = torch.tensor([1.0, 1 / 0.9, 2.0])
    boxes = torch.tensor([[4.0, 4.0, 6.0, 6.0]])
    original = [value.detach().clone() for value in (points, radii, boxes)]
    result = hierarchical_assign(points, radii, boxes, topk=1)
    assert result.gt_index.tolist() == [0, 0, -1]
    assert result.counts.tolist() == [2]
    assert result.stage.tolist() == [1, 2, 0]
    assert result.outside_gt.tolist() == [True, True, False]
    torch.testing.assert_close(result.score, torch.tensor([1 / 3, 1 / 3, 0.0]))
    assert not result.score.requires_grad
    for before, after in zip(original, (points, radii, boxes)):
        assert torch.equal(before, after)
    unchanged = hierarchical_assign(points, radii[:, None], boxes, topk=1, beta=1)
    assert unchanged.gt_index.tolist() == [0, -1, -1]


def test_second_stage_ranks_all_candidates_but_cannot_steal_or_backfill():
    points = torch.zeros(3, 2)
    radii = torch.tensor([1.0, 2.0, 4.0])
    boxes = torch.tensor([[-1.0, -1.0, 1.0, 1.0], [-2.0, -2.0, 2.0, 2.0]])
    result = hierarchical_assign(points, radii, boxes, topk=1, beta=0.5, gt_chunk_size=1)
    assert result.gt_index.tolist() == [0, 1, 1]
    assert result.stage.tolist() == [1, 1, 2]
    assert result.counts.tolist() == [1, 2]
    assert result.score.tolist() == [1.0] * 3
    assert not result.outside_gt.any()
    assert hierarchical_assign(points, radii, boxes, topk=1, beta=1).gt_index.tolist() == [0, 1, -1]


@pytest.mark.parametrize("chunk", [1, 32])
def test_within_stage_area_or_upstream_overwrite_and_ties(chunk):
    points = torch.zeros(2, 2)
    radii = torch.full((2,), 2.0)
    boxes = torch.tensor([[-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [-2.0, -2.0, 2.0, 2.0]])
    assert receptive_field_scores(points, radii, boxes)[2, 0] > receptive_field_scores(points, radii, boxes)[0, 0]
    paper = hierarchical_assign(points, radii, boxes, topk=1, beta=1, gt_chunk_size=chunk)
    assert paper.gt_index.tolist() == [0, -1]
    assert paper.counts.tolist() == [1, 0, 0]
    upstream = hierarchical_assign(points, radii, boxes, topk=1, beta=1, ranking="upstream", gt_chunk_size=chunk)
    assert upstream.gt_index.tolist() == [2, 2]
    assert upstream.counts.tolist() == [0, 0, 2]


@pytest.mark.parametrize(("n", "g"), [(0, 0), (0, 2), (3, 0)])
def test_empty_inputs(n, g):
    points = torch.zeros(n, 2)
    radii = torch.ones(n, 1)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).expand(g, 4)
    assert receptive_field_scores(points, radii, boxes).shape == (g, n)
    result = hierarchical_assign(points, radii, boxes)
    assert result.gt_index.tolist() == [-1] * n
    assert result.counts.tolist() == [0] * g
    assert result.stage.tolist() == [0] * n
    assert result.score.tolist() == [0.0] * n
    assert not result.outside_gt.any()


def test_half_precision_k_above_pool_size_and_boundary():
    points = torch.tensor([[1.0, 0.0], [640.0, 640.0]], dtype=torch.float16)
    radii = torch.ones(2, dtype=torch.float16)
    boxes = torch.tensor([[-1.0, -1.0, 1.0, 1.0]], dtype=torch.float16)
    result = hierarchical_assign(points, radii, boxes, topk=9)
    assert result.score.dtype == torch.float32 and (result.score > 0).all()
    assert result.counts.tolist() == [2]
    assert result.outside_gt.tolist() == [False, True]
    with pytest.raises(ValueError, match="positive"):
        hierarchical_assign(points, radii * 0, boxes)
    with pytest.raises(ValueError, match="positive"):
        hierarchical_assign(points, radii, boxes * 0)
    with pytest.raises(ValueError, match="finite"):
        hierarchical_assign(points * float("nan"), radii, boxes)
    with pytest.raises(ValueError, match="beta"):
        hierarchical_assign(points, radii, boxes, beta=0)


def test_real_visdrone_validation_annotation():
    archive_path = os.environ.get("VISDRONE_VAL_ZIP")
    if not archive_path:
        pytest.skip("set VISDRONE_VAL_ZIP to the local VisDrone2019-DET-val.zip for this additional geometric check")
    member = "VisDrone2019-DET-val/annotations/0000001_02999_d_0000005.txt"
    with ZipFile(archive_path) as archive, archive.open(member) as raw:
        rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
        xywh = torch.tensor([list(map(float, row[:4])) for row in rows if row[4] != "0" and 1 <= int(row[5]) <= 10])
    boxes = torch.cat((xywh[:, :2], xywh[:, :2] + xywh[:, 2:]), 1) / 3 + torch.tensor([0.0, 140.0, 0.0, 140.0])
    points, radii = [], []
    for stride in (8, 16, 32):
        axis = torch.arange(stride / 2, 640, stride)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        points.append(torch.stack((x.flatten(), y.flatten()), 1))
        radii.append(torch.full((x.numel(),), 2.0 * stride))
    points, radii = torch.cat(points), torch.cat(radii)
    result = hierarchical_assign(points, radii, boxes)
    inside = ((points >= boxes[:, None, :2]) & (points <= boxes[:, None, 2:])).all(2)
    positive = result.gt_index >= 0
    rescued = ~inside.any(1) & (result.counts > 0)
    assert len(boxes) == 127 and len(points) == 8400
    torch.testing.assert_close(result.counts, torch.bincount(result.gt_index[positive], minlength=len(boxes)))
    assert result.outside_gt.any() and rescued.any()
    assert torch.equal(result.outside_gt[positive], ~inside[result.gt_index[positive], positive])
    assert not result.outside_gt[~positive].any()
    print(
        f"{member}: GT={len(boxes)} N={len(points)} positive={positive.sum().item()} "
        f"stage2={(result.stage == 2).sum().item()} outside={result.outside_gt.sum().item()} "
        f"unmatched={(result.counts == 0).sum().item()} no_inside_but_assigned={rescued.sum().item()}"
    )
