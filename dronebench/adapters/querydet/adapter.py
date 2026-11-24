from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from dronebench.methods.querydet import QueryHead, SparseQueryPlan, cascade_plans, query_focal_loss, query_targets
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import xywhn2xyxy


@dataclass(frozen=True)
class _Options:
    threshold: float = 0.15
    context_radius: int = 2
    hidden_channels: int | None = 32
    depth: int = 4
    scale_ranges: list | None = None
    radius_px: float | None = None
    radius_factor: float = 1.0
    alpha: float = 0.25
    gamma: float = 2.0
    loss_weight: float = 1.0

    def __post_init__(self):
        if not 0 <= self.threshold <= 1 or not 0 <= self.alpha <= 1:
            raise ValueError("threshold and alpha must be probabilities in [0, 1]")
        if not all(type(n) is int and n >= 0 for n in (self.depth, self.context_radius)):
            raise ValueError("depth and context_radius must be nonnegative integers")
        if self.hidden_channels is not None and (type(self.hidden_channels) is not int or self.hidden_channels < 1):
            raise ValueError("hidden_channels must be a positive integer or null")
        if not all(math.isfinite(n) and n >= 0 for n in (self.gamma, self.loss_weight)):
            raise ValueError("gamma and loss_weight must be finite and nonnegative")
        if not math.isfinite(self.radius_factor) or self.radius_factor <= 0:
            raise ValueError("radius_factor must be finite and positive")
        if self.radius_px is not None and not (
            math.isfinite(self.radius_px) and self.radius_px > 0 and self.radius_factor == 1.0
        ):
            raise ValueError("radius_px must be finite and positive and cannot override a non-default radius_factor")


def _operators(module):
    if isinstance(module, nn.Sequential):
        for child in module:
            yield from _operators(child)
    elif type(module) in (Conv, DWConv):
        yield module.conv
        if hasattr(module, "bn"):
            yield module.bn
        yield module.act
    else:
        yield module


def _scatter(rows, keys, shape, fill):
    b, h, w = shape
    grid = rows.new_full((b, h, w, rows.shape[1]), fill)
    grid[tuple(keys.T)] = rows
    return grid.reshape(b, h * w, -1).transpose(1, 2)


class QueryDetAdapter(nn.Module):
    """Train dense query heads and evaluate stock one-to-many Detect heads on cascaded local grids."""

    def __init__(self, channels, strides, options: dict):
        super().__init__()
        self.strides = tuple(float(s) for s in strides)
        if len(channels) != len(self.strides) or len(channels) not in (3, 4):
            raise ValueError("provide channels and strides for three or four fine-to-coarse levels")
        if not all(math.isfinite(s) and s > 0 for s in self.strides):
            raise ValueError("strides must be finite and positive")
        ratios = [coarse / fine for fine, coarse in zip(self.strides, self.strides[1:])]
        if not all(r.is_integer() and r >= 2 for r in ratios):
            raise ValueError("each stride must be an integer multiple of the preceding finer stride")
        self.scales = tuple(int(r) for r in reversed(ratios))
        self.options = _Options(**options)
        ranges = self.options.scale_ranges
        self.scale_ranges = tuple((0, 4 * s) for s in self.strides[1:]) if ranges is None else tuple(map(tuple, ranges))
        if len(self.scale_ranges) != len(channels) - 1 or not all(0 <= lo < hi for lo, hi in self.scale_ranges):
            raise ValueError("scale_ranges needs [lo, hi) for each non-finest level in fine-to-coarse order")
        self.query_heads = nn.ModuleList(
            QueryHead(c, self.options.hidden_channels, self.options.depth) for c in channels[1:]
        )

    def forward(self, head: Detect, feats: list[torch.Tensor], level_mask: torch.Tensor) -> dict:
        """Return Detect-layout logits and eligibility in local, fine-to-coarse anchor order."""
        if len(feats) != head.nl or len(feats) != len(self.strides):
            raise ValueError("Detect, features and configured strides must have the same level count")
        if level_mask.shape != (feats[0].shape[0], head.nl) or level_mask.dtype != torch.bool:
            raise ValueError("level_mask must be a B x L bool tensor")
        if level_mask.device != feats[0].device:
            raise ValueError("level_mask and features must share a device")
        if getattr(head, "one2one_cv2", None) is not None or head.cv2 is None:
            raise ValueError("this adapter targets the YOLO11 one-to-many Detect head")
        if self.training:
            raw = head.forward_head(feats, **head.one2many)
            raw["active"] = torch.cat(
                [level_mask[:, i : i + 1].expand(-1, f.shape[2] * f.shape[3]) for i, f in enumerate(feats)], dim=1
            )
            logits = tuple(q(f) for q, f in zip(self.query_heads, feats[1:]))
            raw["query_state"] = {"mode": "dense", "logits": logits}
            return raw
        if head.training:
            raise ValueError("call eval() on Detect before sparse inference")
        return self._sparse(head, feats, level_mask)

    @torch.no_grad()
    def _sparse(self, head, feats, level_mask):
        plans = cascade_plans(
            feats[::-1],
            tuple(reversed(self.query_heads)),
            threshold=self.options.threshold,
            radius=self.options.context_radius,
            scales=self.scales,
        )
        boxes, scores, active, levels = [], [], [], []
        for i, (f, plan) in enumerate(zip(feats, [*reversed(plans), None])):
            b, _, h, w = f.shape
            shape = (b, h, w)
            enabled = level_mask[:, i]
            if plan is None:
                keys = context = torch.ones(shape, dtype=torch.bool, device=f.device).nonzero()
            else:
                keys, context = plan.keys, plan.active
            det_keys = keys[enabled[keys[:, 0]]]
            filtered = len(det_keys) != len(keys)
            det_context = det_keys if plan is None else context
            if plan is not None and filtered:
                det_context = context[enabled[context[:, 0]]]
            if not len(det_keys):
                box_rows = f.new_empty((0, 4 * head.reg_max))
                cls_rows = f.new_empty((0, head.nc))
            elif plan is None:
                x = f[enabled]
                box_rows = head.cv2[i](x).permute(0, 2, 3, 1).reshape(-1, 4 * head.reg_max)
                cls_rows = head.cv3[i](x).permute(0, 2, 3, 1).reshape(-1, head.nc)
            else:
                if filtered:
                    plan = SparseQueryPlan(det_keys, det_context, shape)
                box_rows = plan.run(nn.Sequential(*_operators(head.cv2[i])), f)
                cls_rows = plan.run(nn.Sequential(*_operators(head.cv3[i])), f)
            boxes.append(_scatter(box_rows, det_keys, shape, 0.0))
            scores.append(_scatter(cls_rows, det_keys, shape, -1000.0))
            mask = torch.zeros(shape, dtype=torch.bool, device=f.device)
            mask[tuple(det_keys.T)] = True
            active.append(mask.flatten(1))
            levels.append({"keys": keys, "context": context, "det_keys": det_keys, "det_context": det_context})
        return {
            "feats": feats,
            "boxes": torch.cat(boxes, dim=2),
            "scores": torch.cat(scores, dim=2),
            "active": torch.cat(active, dim=1),
            "query_state": {"mode": "sparse", "levels": levels},
        }

    def aux_loss(self, raw: dict, batch: dict, hw: tuple[int, int]) -> torch.Tensor:
        """Compute dense focal supervision in local pixel coordinates, or zero for sparse evaluation."""
        state = raw["query_state"]
        if state["mode"] == "sparse":
            return raw["boxes"].new_zeros(())
        logits = state["logits"]
        h, w = hw
        boxes = xywhn2xyxy(batch["bboxes"].to(device=logits[0].device, dtype=torch.float32), w=w, h=h)
        image_ids = batch["batch_idx"].reshape(-1).to(boxes.device)
        boxes_by_image = [boxes[image_ids == i] for i in range(logits[0].shape[0])]
        opt = self.options
        losses = []
        for q, stride, limits in zip(logits, self.strides[1:], self.scale_ranges):
            target = query_targets(
                q, boxes_by_image, stride, limits, radius_px=opt.radius_px, radius_factor=opt.radius_factor
            )
            losses.append(query_focal_loss(q, target, alpha=opt.alpha, gamma=opt.gamma))
        return torch.stack(losses).mean() * opt.loss_weight
