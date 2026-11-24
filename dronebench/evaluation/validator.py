from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import RANK, ops
from ultralytics.utils.metrics import DetMetrics, bbox_ioa

BIN_NAMES = ("lt8", "8to16", "ge16")
REFERENCE = 640
RECALL_CONF = 0.25
RECALL_CAP = 300


def _original_boxes(boxes, batch):
    coordinates = boxes.detach().cpu().numpy().astype(np.float64)
    return ops.scale_boxes(batch["imgsz"], coordinates, batch["ori_shape"], batch["ratio_pad"])


def image_counts(pred: dict, batch: dict, integer_gt: bool = True) -> np.ndarray:
    """Return per-bin [GT, TP] counts for one prepared validation image, without mutating either input."""
    gt = _original_boxes(batch["bboxes"], batch)
    if integer_gt:
        pixels = np.rint(gt)
        tolerance = np.tile(np.asarray(batch["ori_shape"])[::-1], 2) * 2e-6
        on_grid = (np.abs(gt - pixels) <= tolerance).all(axis=1)
        gt[on_grid] = pixels[on_grid]
    long_side = max(batch["ori_shape"])
    short_side = (gt[:, 2:] - gt[:, :2]).min(axis=1)
    bins = np.searchsorted([8 * long_side, 16 * long_side], REFERENCE * short_side, side="right")
    counts = np.stack([np.bincount(bins, minlength=3), np.zeros(3, dtype=np.int64)])
    scores = pred["conf"].detach().cpu().numpy()
    kept = np.flatnonzero(scores >= RECALL_CONF)
    if not len(gt) or not len(kept):
        return counts
    order = kept[np.argsort(-scores[kept], kind="stable")[:RECALL_CAP]]
    boxes = _original_boxes(pred["bboxes"], batch)[order]
    ious = bbox_ioa(boxes, gt, iou=True, eps=np.finfo(np.float64).tiny)
    gt_cls = batch["cls"].detach().cpu().numpy().reshape(-1)
    det_cls = pred["cls"].detach().cpu().numpy().reshape(-1)[order]
    ious[det_cls[:, None] != gt_cls] = 0
    for overlaps in ious:
        j = overlaps.argmax()
        if overlaps[j] >= 0.5:
            counts[1, bins[j]] += 1
            ious[:, j] = 0
    return counts


class _FixedBinMetrics(DetMetrics):
    def __init__(self):
        super().__init__()
        self.counts = np.zeros((2, 3), dtype=np.int64)
        self.fixed_bin_protocol = {}

    @property
    def fixed_bins(self):
        return {
            name: {"gt": int(gt), "tp": int(tp), "recall": float(tp / gt) if gt else None}
            for name, (gt, tp) in zip(BIN_NAMES, self.counts.T)
        }

    @property
    def results_dict(self):
        result = super().results_dict
        for name, values in self.fixed_bins.items():
            result[f"metrics/R50_640_{name}(B)"] = values["recall"] if values["gt"] else float("nan")
        return result


class FixedBinValidator(DetectionValidator):
    """Augment native YOLO box metrics with GT-size recall at IoU 0.5 on the fixed 640 reference frame."""

    integer_gt = True

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.metrics = _FixedBinMetrics()

    def init_metrics(self, model):
        super().init_metrics(model)
        self.metrics.counts.fill(0)
        self.metrics.fixed_bin_protocol = {
            "reference_long_side": REFERENCE,
            "iou": 0.5,
            "confidence_gte": RECALL_CONF,
            "top_per_image": RECALL_CAP,
            "integer_gt_pixels": self.integer_gt,
            "matching": "confidence descending, maximum IoU among unused GT of same class",
            "nms_conf": self.args.conf,
            "nms_iou": self.args.iou,
            "nms_max_det": self.args.max_det,
            "agnostic_nms": self.args.agnostic_nms,
            "single_cls": self.args.single_cls,
            "end2end_head": self.end2end,
        }

    def _process_batch(self, preds, batch):
        self.metrics.counts += image_counts(preds, batch, self.integer_gt)
        return super()._process_batch(preds, batch)

    def get_stats(self):
        stats = super().get_stats()
        if self.training and not self.args.val:
            for name in BIN_NAMES:
                stats.pop(f"metrics/R50_640_{name}(B)")
        return stats

    def gather_stats(self):
        super().gather_stats()
        if RANK != -1:
            counts = torch.as_tensor(self.metrics.counts, device=self.device)
            dist.reduce(counts, dst=0, op=dist.ReduceOp.SUM)
            if RANK == 0:
                self.metrics.counts[:] = counts.cpu().numpy()
