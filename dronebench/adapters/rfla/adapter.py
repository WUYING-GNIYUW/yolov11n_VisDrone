from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from dronebench.methods.rfla import hierarchical_assign


class RFLAAdapter:
    """Map YOLO pixel anchors and padded batch targets to the single-image RFLA assigner."""

    def __init__(self, strides, options: dict):
        if not isinstance(options, dict):
            raise TypeError("options.rfla must be a dict")
        unknown = options.keys() - {"radii_px", "topk", "beta", "metric"}
        if unknown:
            raise ValueError(f"Unknown options.rfla keys: {sorted(unknown)}")
        self.strides = tuple(map(float, strides.tolist() if isinstance(strides, Tensor) else strides))
        if not self.strides or len(set(self.strides)) != len(self.strides):
            raise ValueError("strides must be nonempty and distinct")
        radii = options.get("radii_px", tuple(2 * s for s in self.strides))
        if not isinstance(radii, (list, tuple)) or len(radii) != len(self.strides):
            raise ValueError("radii_px must contain one pixel radius for each detection stride")
        self.radii_px = tuple(map(float, radii))
        if not all(math.isfinite(v) and v > 0 for v in self.strides + self.radii_px):
            raise ValueError("strides and radii_px must be finite and positive")
        self.topk = options.get("topk", 3)
        self.beta = options.get("beta", 0.9)
        self.metric = options.get("metric", "kl")
        if type(self.topk) is not int or self.topk < 1:
            raise ValueError("topk must be a positive integer")
        if isinstance(self.beta, bool) or not 0 < self.beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        if self.metric not in ("kl", "wd"):
            raise ValueError("metric must be 'kl' or 'wd'")

    @torch.no_grad()
    def assign(
        self,
        points_px: Tensor,
        stride_px: Tensor,
        gt_labels: Tensor,
        gt_boxes_px: Tensor,
        valid_gt: Tensor,
        active: Tensor,
        nc: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Assign active pixel anchors only; return original xyxy boxes, unit class targets and padded GT indices."""
        b, g = gt_boxes_px.shape[:2]
        n = len(points_px)
        if points_px.shape != (n, 2) or stride_px.shape != (n, 1) or active.shape != (b, n):
            raise ValueError("expected points_px [N,2], stride_px [N,1] and active [B,N]")
        if gt_boxes_px.shape != (b, g, 4) or gt_labels.shape != (b, g, 1) or valid_gt.shape != gt_labels.shape:
            raise ValueError("expected gt_boxes_px [B,G,4], gt_labels and valid_gt [B,G,1]")
        if active.dtype != torch.bool or valid_gt.dtype != torch.bool:
            raise TypeError("active and valid_gt must be boolean masks")
        if type(nc) is not int or nc < 1:
            raise ValueError("nc must be a positive integer")

        radii = stride_px.new_zeros(n, dtype=torch.float32)
        for stride, radius in zip(self.strides, self.radii_px):
            radii.masked_fill_(stride_px[:, 0] == stride, radius)
        boxes = gt_boxes_px.new_zeros((b, n, 4))
        scores = points_px.new_zeros((b, n, nc), dtype=torch.float32)
        gt_idx = points_px.new_full((b, n), -1, dtype=torch.long)
        counts = gt_idx.new_zeros((b, g))
        stages_outside = gt_idx.new_zeros((b, 3))
        valid = valid_gt.squeeze(-1)
        for i in range(b):
            anchors = active[i].nonzero(as_tuple=True)[0]
            gt = valid[i].nonzero(as_tuple=True)[0]
            if not anchors.numel() or not gt.numel():
                continue
            result = hierarchical_assign(
                points_px[anchors],
                radii[anchors],
                gt_boxes_px[i, gt],
                topk=self.topk,
                beta=self.beta,
                metric=self.metric,
            )
            positive = (result.gt_index >= 0).nonzero(as_tuple=True)[0]
            chosen = anchors[positive]
            source = gt[result.gt_index[positive]]
            gt_idx[i, chosen] = source
            boxes[i, chosen] = gt_boxes_px[i, source]
            scores[i, chosen] = F.one_hot(gt_labels[i, source, 0].long(), num_classes=nc).to(scores.dtype)
            counts[i, gt] = result.counts
            stages_outside[i] = torch.stack(
                ((result.stage == 1).sum(), (result.stage == 2).sum(), result.outside_gt.sum())
            )
        diagnostics = {
            "positive_per_gt": counts,
            "valid_gt_per_image": valid.sum(1),
            "positive_per_image": counts.sum(1),
            "unmatched_gt_per_image": (valid & (counts == 0)).sum(1),
            "stage1_per_image": stages_outside[:, 0],
            "stage2_per_image": stages_outside[:, 1],
            "outside_gt_per_image": stages_outside[:, 2],
        }
        return boxes, scores, gt_idx >= 0, gt_idx, diagnostics
