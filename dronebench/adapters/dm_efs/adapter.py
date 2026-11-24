import math

import torch
from torch import nn

from dronebench.methods.dm_efs import (
    SizeFeatureCodebook,
    SizeRangePredictor,
    expand_features,
    paired_box_ious,
    predicted_size_range,
    size_range_loss,
)
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.torch_utils import initialize_weights


class DMEFSAdapter(nn.Module):
    """Expose P2 to Detect and route image- or patch-local detection levels from backbone size estimates."""

    def __init__(self, shallow_channels, channels, strides, options, *, backbone_channels=None):
        super().__init__()
        if not isinstance(options, dict):
            raise TypeError("options.dm_efs must be a dictionary")
        unknown = options.keys() - {
            "efs", "dfm", "max_size", "iou_threshold", "loss_weight", "gamma", "projection_channels", "hidden_channels"
        }
        if unknown:
            raise ValueError(f"Unknown options.dm_efs keys: {sorted(unknown)}")
        self.efs, self.dfm = options.get("efs", True), options.get("dfm", True)
        if type(self.efs) is not bool or type(self.dfm) is not bool:
            raise ValueError("efs and dfm must be booleans")
        max_size = options.get("max_size", 640)
        projection, hidden = options.get("projection_channels", 16), options.get("hidden_channels", 64)
        if type(max_size) is not int or max_size < 2:
            raise ValueError("max_size must be an integer >= 2")
        if any(type(c) is not int or c < 1 for c in (projection, hidden)):
            raise ValueError("projection_channels and hidden_channels must be positive integers")
        threshold = options.get("iou_threshold", 0.5)
        self.loss_weight, self.gamma = options.get("loss_weight", 0.04), options.get("gamma", 2.0)
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (self.loss_weight, self.gamma)):
            raise ValueError("loss_weight and gamma must be finite and nonnegative")
        self.channels = ([shallow_channels] if self.efs else []) + list(channels)
        steps = ([4.0] if self.efs else []) + [float(s) for s in strides]
        if (
            not len(channels)
            or len(steps) != len(self.channels)
            or not all(s.is_integer() and s > 0 for s in steps)
            or any(a >= b for a, b in zip(steps, steps[1:]))
        ):
            raise ValueError("provide one positive, increasing integer stride per fine-to-coarse channel count")
        self.strides = [int(s) for s in steps]
        backbone_channels = (shallow_channels,) if backbone_channels is None else backbone_channels
        self.predictor = SizeRangePredictor(backbone_channels, max_size, projection, hidden) if self.dfm else None
        self.codebook = SizeFeatureCodebook(len(self.strides), max_size, threshold) if self.dfm else None

    def extend_head(self, head):
        """Prepend only a freshly initialized stride-4 branch; never initialize the imported head again."""
        if not self.efs:
            return
        if head.nl != len(self.strides) - 1 or getattr(head, "one2one_cv2", None) is not None:
            raise ValueError("EFS expects the unextended stock YOLO11 one-to-many Detect head")
        p2 = Detect(nc=head.nc, reg_max=head.reg_max, ch=(self.channels[0],))
        p2.stride.fill_(self.strides[0])
        p2.bias_init()
        initialize_weights(p2)
        p2.to(head.cv2[0][-1].weight).train(head.training)
        head.cv2.insert(0, p2.cv2[0])
        head.cv3.insert(0, p2.cv3[0])
        head.nl += 1
        head.stride = torch.cat((head.stride.new_tensor(self.strides[:1]), head.stride))
        head.shape = None

    def route(self, shallow, native_feats, *, backbone_feats=None):
        """Return fine-to-coarse Detect inputs, enabled rows per level, and the requested versus actual neck route."""
        feats = expand_features((shallow,), native_feats) if self.efs else tuple(native_feats)
        full = torch.ones((shallow.shape[0], len(feats)), dtype=torch.bool, device=shallow.device)
        logits = sizes = None
        neck = head = full
        if self.dfm:
            logits = self.predictor((shallow,) if backbone_feats is None else backbone_feats)
            sizes = predicted_size_range(logits)
            neck, head = self.codebook.gates(sizes)
        enabled = full if self.training else head
        return feats, enabled, {
            "logits": logits,
            "size_range": sizes,
            "neck_requested": neck,
            "neck_executed": full,
            "head_requested": head,
            "head_enabled": enabled,
        }

    @torch.no_grad()
    def _vote(self, gt_by_image, boxes_px, feats):
        grids = [f.shape[-2:] for f in feats]
        counts = [h * w for h, w in grids]
        if len(grids) != len(self.strides) or boxes_px.shape != (len(gt_by_image), sum(counts), 4):
            raise ValueError("expected decoder xyxy pixels [B,sum(H*W),4] in fine-to-coarse row-major order")
        levels = boxes_px.split(counts, dim=1)
        for row, gt in enumerate(gt_by_image):
            if not len(gt):
                continue
            centre = (gt[:, :2] + gt[:, 2:]) * 0.5
            matched = []
            for pred, (h, w), stride in zip(levels, grids, self.strides):
                x, y = (centre / stride).floor().long().unbind(1)
                index = y.clamp(0, h - 1) * w + x.clamp(0, w - 1)
                matched.append(pred[row].index_select(0, index))
            self.codebook.observe(gt, paired_box_ious(gt, torch.stack(matched, dim=1)))

    def aux_loss(self, state, batch, hw, boxes_px, feats, update_votes):
        """Train the two size endpoints and optionally accumulate this training batch's per-image, per-level votes."""
        if not self.dfm:
            return feats[0].new_zeros(())
        logits = state["logits"]
        gt = xywhn2xyxy(batch["bboxes"].to(logits.device, torch.float32), w=hw[1], h=hw[0])
        image_ids = batch["batch_idx"].reshape(-1).to(logits.device)
        valid = (gt[:, 2:] > gt[:, :2]).all(1) & torch.isfinite(gt).all(1)
        gt_by_image = [gt[valid & (image_ids == i)] for i in range(len(logits))]
        if self.training and update_votes:
            self._vote(gt_by_image, boxes_px, feats)
        return size_range_loss(logits, gt_by_image, weight=self.loss_weight, gamma=self.gamma)
