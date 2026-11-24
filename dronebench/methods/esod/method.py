from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _map_shape(t: Tensor):
    if t.ndim != 4 or t.shape[1] != 1 or min(t.shape) < 1:
        raise ValueError("expected a nonempty [B, 1, H, W] map")
    return t.shape[0], t.shape[2], t.shape[3]


def _scale_xyxy(t: Tensor, stride) -> Tensor:
    sy, sx = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    sy, sx = float(sy), float(sx)
    if not (math.isfinite(sy) and math.isfinite(sx) and min(sy, sx) > 0):
        raise ValueError("stride must be positive and finite")
    return t.new_tensor((sx, sy, sx, sy))


def _boxes_float(boxes: Tensor) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("expected [N, 4] xyxy boxes")
    return boxes if boxes.dtype == torch.float64 else boxes.float()


def _ids(indices: Tensor, n: int, limit: int, device) -> Tensor:
    if indices.shape != (n,):
        raise ValueError("expected one index per row")
    raw = indices.to(device)
    ids = raw.long()
    if ((raw != ids) | (ids < 0) | (ids >= limit)).any():
        raise ValueError("image or patch indices must be integers in range")
    return ids


class ObjSeeker(nn.Sequential):
    def __init__(self, in_channels: int, kernel_size: int = 13):
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, 1),
        )


@torch.no_grad()
def foreground_targets(
    logits: Tensor,
    boxes_xyxy: Tensor,
    batch_indices: Tensor,
    stride,
    instance_masks: Tensor | None = None,
    tau: float = 0.5,
) -> Tensor:
    b, h, w = _map_shape(logits)
    if not 0 < tau < 1:
        raise ValueError("Gaussian corner value tau must be between zero and one")
    boxes = _boxes_float(boxes_xyxy).to(logits.device)
    ids = _ids(batch_indices, len(boxes), b, logits.device)
    if instance_masks is not None:
        if instance_masks.shape != (len(boxes), h, w):
            raise ValueError("instance_masks must be [N, H, W] on the stem grid")
        instance_masks = instance_masks.to(logits.device)
        if not ((instance_masks >= 0) & (instance_masks <= 1)).all():
            raise ValueError("instance_masks must contain finite probabilities in [0, 1]")
    grid_boxes = boxes / _scale_xyxy(boxes, stride)
    out = boxes.new_zeros((b, 1, h, w))
    xs = torch.arange(w, dtype=out.dtype, device=out.device) + 0.5
    ys = torch.arange(h, dtype=out.dtype, device=out.device) + 0.5
    rows = torch.cat((ids[:, None], grid_boxes), dim=1).tolist()
    for i, (bi, x1, y1, x2, y2) in enumerate(rows):
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            raise ValueError("box coordinates must be finite")
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        x = slice(math.floor(x1), math.ceil(x2))
        y = slice(math.floor(y1), math.ceil(y2))
        dx = ((xs[x] - (x1 + x2) / 2).abs() - 0.5).clamp_min(0) / ((x2 - x1) / 2)
        dy = ((ys[y] - (y1 + y2) / 2).abs() - 0.5).clamp_min(0) / ((y2 - y1) / 2)
        patch = ((dy.square()[:, None] + dx.square()) * (0.5 * math.log(tau))).exp()
        if instance_masks is not None:
            mask = instance_masks[i, y, x].to(patch.dtype)
            patch = torch.where(mask.any(), patch * mask, patch)
        region = out[int(bi), 0, y, x]
        region.copy_(torch.maximum(region, patch))
    return out


def foreground_loss(
    logits: Tensor,
    targets: Tensor,
    focal_weight: float = 20.0,
    dice_weight: float = 1.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    _map_shape(logits)
    if logits.shape != targets.shape:
        raise ValueError("foreground targets must match logits")
    if not (0 <= alpha <= 1 and gamma >= 0 and min(focal_weight, dice_weight) >= 0):
        raise ValueError("invalid focal or Dice loss parameters")
    z = logits if logits.dtype == torch.float64 else logits.float()
    t = targets.to(z)
    p = z.sigmoid()
    pt = p * t + (1 - p) * (1 - t)
    at = alpha * t + (1 - alpha) * (1 - t)
    focal = (at * (1 - pt).pow(gamma) * F.binary_cross_entropy_with_logits(z, t, reduction="none")).mean()
    axes = (1, 2, 3)
    dice = 1 - (2 * (p * t).sum(axes) + 1) / (p.sum(axes) + t.sum(axes) + 1)
    return focal_weight * focal + dice_weight * dice.mean()


@dataclass(frozen=True)
class PatchLayout:
    offsets: Tensor
    feature_hw: tuple[int, int]
    patch_hw: tuple[int, int]
    batch_size: int

    def crop(self, features: Tensor, scale: int = 1) -> Tensor:
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("crop scale must be a positive integer")
        h, w = (v * scale for v in self.feature_hw)
        if features.ndim != 4 or features.shape[0] != self.batch_size or features.shape[2:] != (h, w):
            raise ValueError("features do not match this layout's batch and aligned spatial grid")
        ph, pw = (v * scale for v in self.patch_hw)
        if ph > h or pw > w:
            features = F.pad(features, (0, max(0, pw - w), 0, max(0, ph - h)))
        b, x, y = self.offsets[:, :3].to(features.device).unbind(1)
        xi = x[:, None, None] * scale + torch.arange(pw, device=features.device)[None, None, :]
        yi = y[:, None, None] * scale + torch.arange(ph, device=features.device)[None, :, None]
        blocks = features.permute(0, 2, 3, 1)[b[:, None, None], yi, xi]
        return blocks.permute(0, 3, 1, 2).contiguous()

    def project_boxes(self, boxes_xyxy: Tensor, batch_indices: Tensor, stride, clip: bool = False):
        boxes = _boxes_float(boxes_xyxy)
        ids = _ids(batch_indices, len(boxes), self.batch_size, boxes.device)
        offsets = self.offsets.to(boxes.device)
        rects = offsets[:, 1:].to(boxes.dtype) * _scale_xyxy(boxes, stride)
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        matches = (center >= rects[:, None, :2]) & (center < rects[:, None, 2:])
        valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
        pi, gi = torch.where(matches.all(2) & (offsets[:, :1] == ids) & valid)
        origin = rects[pi, :2]
        local = boxes[gi] - origin.repeat(1, 2)
        if clip:
            local = torch.minimum(local.clamp_min(0), (rects[pi, 2:] - origin).repeat(1, 2))
        return local, pi, gi

    def valid_centers(self, local_xy: Tensor, patch_indices: Tensor, stride) -> Tensor:
        if local_xy.ndim != 2 or local_xy.shape[1] != 2:
            raise ValueError("expected [N, 2] local xy coordinates in input pixels")
        xy = local_xy if local_xy.dtype == torch.float64 else local_xy.float()
        ids = _ids(patch_indices, len(xy), len(self.offsets), xy.device)
        off = self.offsets.to(xy.device)[ids]
        extent = (off[:, 3:5] - off[:, 1:3]) * _scale_xyxy(xy, stride)[:2]
        return ((xy >= 0) & (xy < extent)).all(1)

    def restore_boxes(self, local_xyxy: Tensor, patch_indices: Tensor, stride, image_hw=None):
        boxes = _boxes_float(local_xyxy)
        ids = _ids(patch_indices, len(boxes), len(self.offsets), boxes.device)
        off = self.offsets.to(boxes.device)[ids]
        global_boxes = boxes + off[:, 1:3].repeat(1, 2) * _scale_xyxy(boxes, stride)
        if image_hw is not None:
            h, w = image_hw
            global_boxes = torch.minimum(global_boxes.clamp_min(0), boxes.new_tensor((w, h, w, h)))
        return global_boxes, off[:, 0]


def _activation(probabilities: Tensor, threshold: float):
    _map_shape(probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    p = probabilities if probabilities.dtype == torch.float64 else probabilities.float()
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("expected finite probabilities in [0, 1], not logits")
    active = p >= threshold
    return active, active & (F.max_pool2d(p, 3, 1, 1) == p)


@torch.no_grad()
def foreground_peaks(probabilities: Tensor, threshold: float = 0.5) -> Tensor:
    return _activation(probabilities, threshold)[1]


def _grid_offsets(active: Tensor, peaks: Tensor, ph: int, pw: int) -> Tensor:
    b, _, h, w = active.shape
    ny, nx = (h + ph - 1) // ph, (w + pw - 1) // pw

    def tiles(mask):
        padded = F.pad(mask[:, 0], (0, nx * pw - w, 0, ny * ph - h))
        return padded.reshape(b, ny, ph, nx, pw).permute(0, 1, 3, 2, 4)

    bi, row, col = torch.where(tiles(peaks).any(-1).any(-1))
    occupied = tiles(active)[bi, row, col]
    dx = occupied.any(1).long().argmax(1)
    dy = occupied.any(2).long().argmax(1)
    x = (col * pw + dx).clamp_max(w - pw)
    y = (row * ph + dy).clamp_max(h - ph)
    return torch.stack((bi, x, y, x + pw, y + ph), 1).unique(dim=0)


def _greedy_offsets(active: Tensor, peaks: Tensor, ph: int, pw: int) -> Tensor:
    marks = peaks[:, 0].cpu()
    a = active[:, 0].cpu()
    b, h, w = a.shape
    sizes = F.avg_pool2d(a[:, None].float(), 9, 1, 4)[:, 0]
    rows = []
    for bi in range(b):
        coords = marks[bi].nonzero()
        scores = sizes[bi, coords[:, 0], coords[:, 1]].tolist()
        covered = torch.zeros_like(a[bi])
        for _, (cy, cx) in sorted(zip(scores, coords.tolist()), key=lambda item: -item[0]):
            if covered[cy, cx]:
                continue
            left = max(0, min(cx - pw // 2, w - pw))
            top = max(0, min(cy - ph // 2, h - ph))
            window = a[bi, top : top + ph, left : left + pw]
            x = min(left + int(window.any(0).long().argmax()), w - pw)
            y = min(top + int(window.any(1).long().argmax()), h - ph)
            rows.append((bi, x, y, x + pw, y + ph))
            covered[y : y + ph, x : x + pw] = True
    return torch.tensor(rows, dtype=torch.long, device=active.device).reshape(-1, 5)


@torch.no_grad()
def select_patches(
    probabilities: Tensor, patch_hw: tuple[int, int], threshold: float = 0.5, algorithm: str = "grid"
) -> PatchLayout:
    active, peaks = _activation(probabilities, threshold)
    b, _, h, w = active.shape
    ph, pw = patch_hw
    if not (isinstance(ph, int) and isinstance(pw, int) and min(ph, pw) > 0):
        raise ValueError("patch_hw must be a pair of positive integers")
    if algorithm == "grid":
        select = _grid_offsets
    elif algorithm == "greedy":
        select = _greedy_offsets
    else:
        raise ValueError("algorithm must be 'grid' (Algorithm 2) or 'greedy' (Algorithm 1)")
    offsets = select(active, peaks, min(ph, h), min(pw, w))
    return PatchLayout(offsets, (h, w), (ph, pw), b)
