from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def expand_features(shallow: Sequence[Tensor], base: Sequence[Tensor]) -> tuple[Tensor, ...]:
    """Expose actual backbone maps to an extended neck, in fine-to-coarse order, without resampling or copying."""
    if not shallow or not base:
        raise ValueError("Provide at least one real shallow map and one base map")
    levels = (*shallow, *base)
    if any(x.ndim != 4 or x.shape[0] != levels[0].shape[0] for x in levels):
        raise ValueError("All levels must be NCHW tensors with the same batch size")
    if any(a.shape[-2] < b.shape[-2] or a.shape[-1] < b.shape[-1] for a, b in zip(levels, levels[1:])):
        raise ValueError("Supply levels from finest to coarsest on both axes")
    return levels


class ExpandedPyramid(nn.Module):
    """Small runnable two-pass neck for EFS; an existing task-specific neck can be used instead."""

    def __init__(self, in_channels: Sequence[int], width: int):
        super().__init__()
        if not in_channels or min(*in_channels, width) < 1:
            raise ValueError("Provide at least one level, with positive channel counts and width")
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_channels)
        self.down = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(len(in_channels) - 1))

    def forward(self, features: Sequence[Tensor], last_level: int | None = None) -> tuple[Tensor, ...]:
        if len(features) != len(self.lateral):
            raise ValueError("Supply one fine-to-coarse tensor per declared level")
        last = len(features) - 1 if last_level is None else last_level
        if not 0 <= last < len(features):
            raise ValueError("last_level must be a fine-to-coarse level index")
        first = [layer(x) for layer, x in zip(self.lateral, features)]
        for i in range(len(first) - 2, -1, -1):
            first[i] = first[i] + F.interpolate(first[i + 1], size=first[i].shape[-2:], mode="nearest")
        out = [first[0]]
        for i in range(1, last + 1):
            pooled = F.adaptive_avg_pool2d(out[-1], output_size=first[i].shape[-2:])
            out.append(first[i] + self.down[i - 1](pooled))
        return tuple(out)


class SizeRangePredictor(nn.Module):
    """Predict two single-label distributions over integer pixel sizes 2 through max_size from backbone features."""

    def __init__(
        self, in_channels: Sequence[int], max_size: int, projection_channels: int = 16, hidden_channels: int = 64
    ):
        super().__init__()
        self.in_channels = tuple(in_channels)
        if not self.in_channels or min(*self.in_channels, projection_channels, hidden_channels) < 1 or max_size < 2:
            raise ValueError("Provide positive channel counts and max_size >= 2")
        self.projections = nn.ModuleList(nn.Conv2d(c, projection_channels, 1) for c in self.in_channels)
        self.shared = nn.Sequential(nn.Linear(len(self.in_channels) * projection_channels, hidden_channels), nn.ReLU())
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, max_size - 1)
            )
            for _ in range(2)
        )

    def forward(self, backbone_features: Sequence[Tensor]) -> Tensor:
        if len(backbone_features) != len(self.in_channels) or any(
            x.ndim != 4 or x.shape[1] != c for x, c in zip(backbone_features, self.in_channels)
        ):
            raise ValueError("Provide one NCHW backbone tensor per declared channel count")
        pooled = torch.cat([proj(x).mean(dim=(-2, -1)) for proj, x in zip(self.projections, backbone_features)], dim=1)
        shared = self.shared(pooled)
        return torch.stack([branch(shared) for branch in self.branches], dim=1)


def _box_sizes(boxes: Tensor, max_size: int) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Ground truth must have shape [N,4] in input-pixel xyxy coordinates")
    xyxy = boxes.float()
    wh = xyxy[:, 2:] - xyxy[:, :2]
    if not (torch.isfinite(wh) & (wh > 0)).all():
        raise ValueError("Filter non-finite or non-positive-area ground truth before measuring sizes")
    return wh.prod(dim=1).sqrt().floor().clamp(min=2, max=max_size).long()


def size_range_loss(
    logits: Tensor, boxes_per_image: Sequence[Tensor], weight: float = 0.04, gamma: float = 2.0
) -> Tensor:
    """Sum two softmax focal losses over min/max sizes; average over images that have at least one valid box."""
    if logits.ndim != 3 or logits.shape[1] != 2 or logits.shape[2] < 1 or len(boxes_per_image) != logits.shape[0]:
        raise ValueError("Provide logits [B,2,max_size-1] and one [N,4] pixel box tensor per image")
    if weight < 0 or gamma < 0:
        raise ValueError("Loss weight and focal gamma must be nonnegative")
    rows, labels = [], []
    for row, boxes in enumerate(boxes_per_image):
        sizes = _box_sizes(boxes, logits.shape[-1] + 1)
        if sizes.numel():
            rows.append(row)
            labels.append(torch.stack((sizes.min(), sizes.max())))
    if not rows:
        return logits.reshape(-1)[:0].sum()
    scores = logits[rows].flatten(0, 1)
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    target = torch.stack(labels).to(scores.device).flatten() - 2
    ce = F.cross_entropy(scores, target, reduction="none")
    miss_probability = -torch.expm1(-ce)
    return weight * (miss_probability.pow(gamma) * ce).view(-1, 2).sum(dim=1).mean()


def predicted_size_range(logits: Tensor) -> Tensor:
    """Decode [B,2,max_size-1] logits to ordered integer min/max sizes [B,2] in input pixels."""
    if logits.ndim != 3 or logits.shape[1] != 2 or logits.shape[2] < 1:
        raise ValueError("Expected logits [B,2,max_size-1]")
    return (logits.argmax(dim=-1) + 2).sort(dim=1).values


def paired_box_ious(gt_xyxy: Tensor, matched_xyxy: Tensor) -> Tensor:
    """Compute [N,L] IoUs for supplied per-target, per-level predictions [N,L,4] in the same coordinate frame."""
    if gt_xyxy.ndim != 2 or gt_xyxy.shape[1] != 4:
        raise ValueError("Expected ground truth [N,4]")
    if matched_xyxy.ndim != 3 or matched_xyxy.shape[0] != len(gt_xyxy) or matched_xyxy.shape[-1] != 4:
        raise ValueError("Expected one matched prediction per ground truth and level [N,L,4]")
    gt, pred = gt_xyxy[:, None, :].float(), matched_xyxy.float()
    overlap = torch.minimum(gt[..., 2:], pred[..., 2:]) - torch.maximum(gt[..., :2], pred[..., :2])
    inter = overlap.clamp_min(0).prod(-1)
    gt_area = (gt[..., 2:] - gt[..., :2]).clamp_min(0).prod(-1)
    pred_area = (pred[..., 2:] - pred[..., :2]).clamp_min(0).prod(-1)
    union = gt_area + pred_area - inter
    return torch.where(union > 0, inter / union.clamp_min(torch.finfo(inter.dtype).tiny), 0.0)


class SizeFeatureCodebook(nn.Module):
    """Accumulate per-image binary size/level votes (Eqs. 2–4); there are no gradient-trained parameters here."""

    def __init__(self, num_levels: int, max_size: int, iou_threshold: float = 0.5):
        super().__init__()
        if num_levels < 1 or max_size < 2 or not 0 <= iou_threshold <= 1:
            raise ValueError("Require num_levels >= 1, max_size >= 2 and an IoU threshold in [0,1]")
        self.iou_threshold = iou_threshold
        self.register_buffer("votes", torch.zeros(num_levels, max_size - 1, dtype=torch.long))

    @torch.no_grad()
    def observe(self, gt_xyxy: Tensor, matched_iou: Tensor) -> None:
        """Add one image's observations [N,L]; repeated successes in a (size,level) cell cast a single image vote."""
        levels, bins = self.votes.shape
        sizes = _box_sizes(gt_xyxy, bins + 1).to(self.votes.device) - 2
        if matched_iou.shape != (sizes.numel(), levels):
            raise ValueError("Expected one IoU per ground truth and level in a single image, shape [N,L]")
        successes = (matched_iou.to(self.votes.device) >= self.iou_threshold).T.long()
        image_votes = torch.zeros_like(self.votes)
        image_votes.scatter_add_(1, sizes.expand(levels, -1), successes)
        self.votes.add_(image_votes.clamp_(max=1))

    def level_table(self) -> Tensor:
        """Return winning level indices by size, filling holes by nearest supported size; all -1 before any success."""
        bins = self.votes.shape[1]
        indices = torch.arange(bins, device=self.votes.device)
        observed = self.votes.any(dim=0)
        left = torch.where(observed, indices, -2 * bins).cummax(0).values
        right = torch.where(observed, indices, 2 * bins).flip((0,)).cummin(0).values.flip((0,))
        nearest = torch.where(indices - left <= right - indices, left, right).clamp(0, bins - 1)
        winners = self.votes.argmax(dim=0)[nearest]
        return torch.where(observed.any(), winners, -1)

    def gates(self, sizes: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-image fine-to-coarse (neck, head) boolean gates from endpoint sizes under Eq. 6."""
        if sizes.ndim != 2 or sizes.shape[1] != 2:
            raise ValueError("Expected minimum and maximum input-pixel sizes [B,2]")
        levels, bins = self.votes.shape
        sizes = sizes.to(device=self.votes.device, dtype=torch.long).sort(dim=1).values.clamp(2, bins + 1) - 2
        small, large = self.level_table()[sizes].unbind(dim=1)
        idx = torch.arange(levels, device=self.votes.device)
        cold = small[:, None] < 0
        neck = (idx <= large[:, None]) | cold
        head = (idx == small[:, None]) | (idx == large[:, None]) | cold
        return neck, head


def run_selected(
    features: Sequence[Tensor], gates: Tensor, operators: Sequence[Callable[[Tensor], Tensor]]
) -> dict[int, tuple[Tensor, Tensor]]:
    """Run independent level operators only on selected image rows; map level -> (original batch indices, outputs)."""
    if gates.ndim != 2 or gates.dtype != torch.bool:
        raise ValueError("Supply boolean level gates [B,L]")
    if len(features) != gates.shape[1] or len(operators) != len(features):
        raise ValueError("Supply exactly one feature and operator per gate column")
    result = {}
    for level, (feature, op) in enumerate(zip(features, operators)):
        if feature.shape[0] != gates.shape[0] or feature.device != gates.device:
            raise ValueError("Features must share the gate's batch size and device")
        rows = gates[:, level].nonzero(as_tuple=True)[0]
        if rows.numel():
            x = feature if rows.numel() == feature.shape[0] else feature.index_select(0, rows)
            result[level] = (rows, op(x))
    return result
