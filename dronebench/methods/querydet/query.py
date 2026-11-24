from __future__ import annotations

import math
from operator import index
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QueryHead(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: int | None = None, depth: int = 4):
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        hidden = in_channels if hidden_channels is None else hidden_channels
        layers = []
        channels = in_channels
        for _ in range(depth):
            layers.extend((nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU()))
            channels = hidden
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        super().__init__(*layers)
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


def _query_map(logits):
    if logits.ndim != 4 or logits.shape[1] != 1 or not logits.is_floating_point():
        raise ValueError("query logits must be a floating B x 1 x H x W tensor")


@torch.no_grad()
def query_targets(
    logits: Tensor,
    boxes_xyxy: Sequence[Tensor],
    stride: float | tuple[float, float],
    scale_range: tuple[float, float],
    *,
    radius_px: float | None = None,
    radius_factor: float = 1.0,
) -> Tensor:
    _query_map(logits)
    b, _, h, w = logits.shape
    sy, sx = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    lo, hi = scale_range
    if len(boxes_xyxy) != b or not (0 <= lo < hi):
        raise ValueError("expected one box tensor per image and 0 <= scale_min < scale_max")
    if not all(math.isfinite(v) and v > 0 for v in (sy, sx, radius_factor if radius_px is None else radius_px)):
        raise ValueError("stride and query radius must be finite and positive")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    y = (torch.arange(h, device=logits.device, dtype=dtype) + 0.5) * sy
    x = (torch.arange(w, device=logits.device, dtype=dtype) + 0.5) * sx
    grid = torch.stack((x.repeat(h), y.repeat_interleave(w)), dim=1)
    target = logits.new_zeros(logits.shape)
    for bi, boxes in enumerate(boxes_xyxy):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("each image needs an N x 4 tensor of pixel-space xyxy boxes")
        boxes = boxes.to(device=logits.device, dtype=dtype)
        wh = boxes[:, 2:] - boxes[:, :2]
        scale = wh.prod(dim=1).clamp_min(0).sqrt()
        keep = torch.isfinite(boxes).all(dim=1) & (wh > 0).all(dim=1) & (scale >= lo) & (scale < hi)
        centers = (boxes[keep, :2] + boxes[keep, 2:]) * 0.5
        if centers.shape[0] == 0:
            continue
        radii = scale[keep] * radius_factor
        step = max(1, 262144 // centers.shape[0])
        values = target[bi, 0].view(-1)
        for start in range(0, len(grid), step):
            segment = grid[start : start + step]
            distances = torch.cdist(segment, centers, compute_mode="donot_use_mm_for_euclid_dist")
            nearest, owner = distances.min(dim=1)
            radius = radii[owner] if radius_px is None else radius_px
            values[start : start + step] = nearest < radius
    return target


def query_focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    _query_map(logits)
    if logits.shape != targets.shape or not (0 <= alpha <= 1 and gamma >= 0):
        raise ValueError("focal loss needs matching shapes, alpha in [0, 1] and nonnegative gamma")
    x = logits if logits.dtype == torch.float64 else logits.float()
    t = targets.to(x)
    wrong_logit = (1.0 - 2.0 * t) * x
    ce = F.softplus(wrong_logit)
    weight = alpha * t + (1.0 - alpha) * (1.0 - t)
    if gamma:
        weight = weight * (gamma * F.logsigmoid(wrong_logit)).exp()
    return (weight * ce).mean()


def _coords(coords):
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != torch.long:
        raise ValueError("coordinates must be an int64 N x 3 tensor of (batch, y, x)")


def _hw(shape):
    h, w = map(index, shape)
    if min(h, w) < 1:
        raise ValueError("height and width must be positive")
    return h, w


def _flat(coords, h, w):
    return (coords[:, 0] * h + coords[:, 1]) * w + coords[:, 2]


def _unflat(values, h, w):
    row = values // w
    return torch.stack((row // h, row % h, values % w), dim=1)


def _offsets(start_y, stop_y, start_x, stop_x, device):
    ys = torch.arange(start_y, stop_y, device=device)
    xs = torch.arange(start_x, stop_x, device=device)
    return ys.repeat_interleave(xs.numel()), xs.repeat(ys.numel())


def _collect(b, y, x, h, w):
    valid = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    linear = ((b * h + y) * w + x)[valid]
    return _unflat(linear.unique(sorted=True), h, w)


@torch.no_grad()
def select_queries(logits: Tensor, threshold: float, coords: Tensor | None = None) -> Tensor:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be a probability in [0, 1]")
    if coords is None:
        _query_map(logits)
        return (logits[:, 0].sigmoid() > threshold).nonzero(as_tuple=False)
    _coords(coords)
    if (
        logits.shape not in ((len(coords),), (len(coords), 1))
        or logits.device != coords.device
        or not logits.is_floating_point()
    ):
        raise ValueError("sparse logits must be floating K or K x 1 values aligned with the coordinates")
    return coords[logits.reshape(-1).sigmoid() > threshold]


def map_children(parents: Tensor, fine_hw: tuple[int, int], scale: int = 2) -> Tensor:
    _coords(parents)
    h, w = _hw(fine_hw)
    scale = index(scale)
    if scale < 1 or (parents < 0).any():
        raise ValueError("scale must be positive and parent coordinates nonnegative")
    dy, dx = _offsets(0, scale, 0, scale, parents.device)
    b, y, x = parents.T.unsqueeze(-1)
    return _collect(b, scale * y + dy, scale * x + dx, h, w)


def expand_context(keys: Tensor, fine_hw: tuple[int, int], radius: int = 2) -> Tensor:
    _coords(keys)
    h, w = _hw(fine_hw)
    radius = index(radius)
    if radius < 0 or (keys < 0).any() or (keys[:, 1] >= h).any() or (keys[:, 2] >= w).any():
        raise ValueError("context needs a nonnegative radius and in-bounds keys")
    dy, dx = _offsets(-radius, radius + 1, -radius, radius + 1, keys.device)
    b, y, x = keys.T.unsqueeze(-1)
    return _collect(b, y + dy, x + dx, h, w)
