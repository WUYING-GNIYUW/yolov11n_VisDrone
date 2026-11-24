import math

import torch
from torch import nn

from dronebench.methods.ugs import (
    decode_logits,
    decode_ltrb,
    encode_ltrb,
    localization_ce,
    make_support,
    refinement_loss,
    uncertainty_entropy,
)
from ultralytics.utils.metrics import bbox_iou


def _refined_logits(towers, features):
    tracking = [
        m
        for m in towers.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training and m.track_running_stats
    ]
    try:
        for bn in tracking:
            bn.track_running_stats = False
        return torch.cat([tower(f).flatten(2) for tower, f in zip(towers, features)], dim=2)
    finally:
        for bn in tracking:
            bn.track_running_stats = True


class UGSAdapter(nn.Module):
    def __init__(self, reg_max, options):
        super().__init__()
        if type(reg_max) is not int or reg_max < 2:
            raise ValueError("UGS needs at least two support points per coordinate")
        if not isinstance(options, dict):
            raise TypeError("options.ugs must be a dictionary")
        unknown = options.keys() - {"nonuniform", "um", "ur", "alpha", "beta", "lambda_um", "gamma_ur", "rho"}
        if unknown:
            raise ValueError(f"Unknown options.ugs keys: {sorted(unknown)}")
        for name in ("nonuniform", "um", "ur"):
            if type(options.get(name, True)) is not bool:
                raise ValueError(f"options.ugs.{name} must be boolean")

        def number(name, default):
            value = options.get(name, default)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"options.ugs.{name} must be a finite nonnegative number")
            return float(value)

        alpha = number("alpha", reg_max - 1)
        if alpha == 0:
            raise ValueError("options.ugs.alpha must be strictly positive")
        beta = number("beta", 2 / alpha)
        self.reg_max = reg_max
        self.register_buffer("support", make_support(reg_max, alpha, beta if options.get("nonuniform", True) else 0))
        self.lambda_um = number("lambda_um", 0.5) * options.get("um", True)
        self.gamma_ur = number("gamma_ur", 0.5) * options.get("ur", True)
        self.rho = number("rho", 0.5)

    def _apply(self, fn):
        grid = self.support
        super()._apply(fn)
        if self.support.dtype != torch.float32:
            self.support = grid.to(device=self.support.device, dtype=torch.float32)
        return self

    @torch.no_grad()
    def initialize_head(self, head):
        if head.reg_max != self.reg_max:
            raise ValueError("UGS and Detect must use the same number of bins")
        bias = -(8 * (self.support / self.support[-1]) - 1).abs()
        for tower in head.cv2:
            tower[-1].bias.copy_(bias.repeat(4))

    def decode(self, pred_distri, anchors):
        if (
            pred_distri.ndim != 3
            or pred_distri.shape[-1] != 4 * self.reg_max
            or anchors.shape != (pred_distri.shape[1], 2)
        ):
            raise ValueError("expected pred_distri [B,N,4R] and anchors [N,2]")
        logits = pred_distri.reshape(*pred_distri.shape[:2], 4, self.reg_max)
        distances = decode_logits(logits, self.support)
        lo, hi = decode_ltrb(anchors, distances).chunk(2, dim=-1)
        extent = hi - lo
        collapsed = extent <= (4 * torch.finfo(distances.dtype).eps) * self.support[-1]
        mid = lo + 0.5 * extent
        return torch.cat((torch.where(collapsed, mid, lo), torch.where(collapsed, mid, hi)), dim=-1)

    def losses(self, head, raw, pred_boxes, anchors, target_boxes, target_scores, normalizer, fg, *, refine):
        pos = fg.nonzero(as_tuple=True)
        target = target_boxes[pos].detach().to(pred_boxes.dtype)
        distances = encode_ltrb(anchors[pos[1]].to(target.dtype), target)
        weights = target_scores[pos].detach().sum(-1, dtype=target.dtype) / normalizer.detach()

        def select(box_logits):
            return box_logits.transpose(1, 2)[pos].reshape(-1, 4, self.reg_max)

        def entropy(logits):
            return (uncertainty_entropy(logits).mean(-1) * weights).sum()

        logits = select(raw["boxes"])
        zero = logits[..., :0].sum(dtype=target.dtype)
        iou = bbox_iou(pred_boxes[pos], target, xywh=False, CIoU=True).squeeze(-1)
        box = ((1 - iou) * weights).sum()
        loc = (localization_ce(logits, distances, self.support).mean(-1) * weights).sum()
        train_grad = refine and self.training and head.training and torch.is_grad_enabled()
        run_ur = bool(self.gamma_ur and train_grad and len(logits))
        clean = entropy(logits) if self.lambda_um or run_ur else zero
        um = self.lambda_um * clean if self.lambda_um else zero
        ur = zero
        if run_ur:
            def perturbed(features):
                return entropy(select(_refined_logits(head.cv2, features)))

            ur = self.gamma_ur * refinement_loss(raw["feats"], clean, perturbed, rho=self.rho)

        degenerate = (pred_boxes[..., 2:] <= pred_boxes[..., :2]).any(-1)
        negative = distances < 0
        diagnostics = {
            "ugs_target_coords": distances.numel(),
            "ugs_outside_fg": negative.any(-1).sum(),
            "ugs_negative_coords": negative.sum(),
            "ugs_clipped_low_coords": (distances < self.support[0]).sum(),
            "ugs_clipped_high_coords": (distances > self.support[-1]).sum(),
            "ugs_degenerate_fg": degenerate[pos].sum(),
            "ugs_degenerate_active": (degenerate & raw["active"]).sum(),
            "ugs_refined": run_ur,
        }
        if self.lambda_um or run_ur:
            diagnostics["ugs_weighted_clean_entropy"] = clean.detach()
        return box, loc, um, ur, diagnostics
