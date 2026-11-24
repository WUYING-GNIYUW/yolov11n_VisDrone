import math
from typing import NamedTuple

import torch
from torch import Tensor


class RFLAAssignment(NamedTuple):
    gt_index: Tensor
    counts: Tensor
    stage: Tensor
    outside_gt: Tensor
    score: Tensor


def _geometry(points_xy, rf_radius, gt_xyxy, metric):
    if metric not in ("kl", "wd"):
        raise ValueError("metric must be 'kl' or 'wd'")
    inputs = (points_xy, rf_radius, gt_xyxy)
    if not all(isinstance(value, Tensor) and value.is_floating_point() for value in inputs):
        raise TypeError("points_xy, rf_radius and gt_xyxy must be floating-point tensors")
    if not all(value.device == points_xy.device for value in inputs):
        raise ValueError("all coordinates and radii must be on the same device")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or gt_xyxy.ndim != 2 or gt_xyxy.shape[1] != 4:
        raise ValueError("expected points_xy [N, 2] and gt_xyxy [G, 4]")
    if rf_radius.ndim == 1:
        rf_radius = rf_radius[:, None]
    if rf_radius.ndim != 2 or rf_radius.shape[0] != len(points_xy) or rf_radius.shape[1] not in (1, 2):
        raise ValueError("rf_radius must have shape [N], [N, 1] or [N, 2]")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in inputs) else torch.float32
    points, radius, boxes = (value.to(dtype) for value in (points_xy, rf_radius, gt_xyxy))
    half_size = (boxes[:, 2:] - boxes[:, :2]) * 0.5
    centers = boxes[:, :2] + half_size
    valid = (
        points.isfinite().all()
        & radius.isfinite().all()
        & (radius > 0).all()
        & centers.isfinite().all()
        & half_size.isfinite().all()
        & (half_size > 0).all()
    )
    if not valid:
        raise ValueError("coordinates must be finite; RF radii and both GT side lengths must be positive")
    return points, radius, boxes, centers, half_size


def _scores(points, radius, centers, half_size, metric):
    offset = points - centers[:, None, :]
    gt_sigma = half_size[:, None, :]
    if metric == "kl":
        variance_ratio = (radius / gt_sigma).square()
        log_ratio = gt_sigma.log() - radius.log()
        distance = 0.5 * ((offset / gt_sigma).square() + variance_ratio - 1 + 2 * log_ratio).sum(-1)
    else:
        distance = (offset.square() + (radius - gt_sigma).square()).sum(-1)
    return (1 + distance.clamp_min(0)).reciprocal()


def receptive_field_scores(points_xy: Tensor, rf_radius: Tensor, gt_xyxy: Tensor, *, metric: str = "kl") -> Tensor:
    points, radius, _, centers, half_size = _geometry(points_xy, rf_radius, gt_xyxy, metric)
    return _scores(points, radius, centers, half_size, metric)


def _proposals(scores, k, ranking):
    k = min(k, scores.shape[1])
    cutoff = scores.topk(k, dim=1).values[:, -1:]
    if ranking == "upstream":
        return scores >= cutoff
    above, tied = scores > cutoff, scores == cutoff
    remaining = k - above.sum(1, keepdim=True, dtype=torch.int32)
    return above | (tied & (tied.cumsum(1, dtype=torch.int32) <= remaining))


@torch.no_grad()
def hierarchical_assign(
    points_xy: Tensor,
    rf_radius: Tensor,
    gt_xyxy: Tensor,
    *,
    topk: int = 3,
    beta: float = 0.9,
    metric: str = "kl",
    ranking: str = "paper",
    gt_chunk_size: int = 32,
) -> RFLAAssignment:
    if type(topk) is not int or topk < 1 or type(gt_chunk_size) is not int or gt_chunk_size < 1:
        raise ValueError("topk and gt_chunk_size must be positive integers")
    if not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if ranking not in ("paper", "upstream"):
        raise ValueError("ranking must be 'paper' or 'upstream'")
    points, radius, boxes, centers, half_size = _geometry(points_xy, rf_radius, gt_xyxy, metric)
    n, g = len(points), len(boxes)
    gt_index = torch.full((n,), -1, dtype=torch.long, device=points.device)
    stage = torch.zeros_like(gt_index)
    score = points.new_zeros(n)
    outside = torch.zeros_like(gt_index, dtype=torch.bool)
    if n and g:
        if ranking == "paper":
            order = half_size.prod(1).argsort(stable=True)
        else:
            order = torch.arange(g - 1, -1, -1, device=points.device)
        for number, scale, k in ((1, 1.0, topk), (2, beta, 1)):
            scaled = radius * scale
            for ids in order.split(gt_chunk_size):
                scores = _scores(points, scaled, centers[ids], half_size[ids], metric)
                claimed, row = _proposals(scores, k, ranking).max(0)
                take = claimed & (gt_index < 0)
                gt_index = torch.where(take, ids[row], gt_index)
                score = torch.where(take, scores.gather(0, row[None]).squeeze(0), score)
                stage.masked_fill_(take, number)
        matched = boxes[gt_index.clamp_min(0)]
        outside = (gt_index >= 0) & ((points < matched[:, :2]) | (points > matched[:, 2:])).any(1)
    counts = gt_index.new_zeros(g + 1).scatter_add_(0, gt_index + 1, torch.ones_like(gt_index))[1:]
    return RFLAAssignment(gt_index, counts, stage, outside, score)
