from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from dronebench.methods.esod import ObjSeeker, foreground_loss, foreground_targets, select_patches
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn


class ESODAdapter(nn.Module):
    stride = 4

    def __init__(self, shallow_channels, options):
        super().__init__()
        known = {"patch_divisor", "threshold", "warmup_epochs", "algorithm", "kernel_size", "loss_weight"}
        unknown = options.keys() - known
        if unknown:
            raise ValueError(f"Unknown options.esod keys: {sorted(unknown)}")
        self.patch_divisor = options.get("patch_divisor", 4)
        self.threshold = options.get("threshold", 0.3)
        self.warmup_epochs = options.get("warmup_epochs", 3)
        self.algorithm = options.get("algorithm", "grid")
        kernel = options.get("kernel_size", 13)
        self.loss_weight = options.get("loss_weight", 1.0)
        if type(self.patch_divisor) is not int or self.patch_divisor < 1:
            raise ValueError("patch_divisor must be a positive integer")
        if type(self.threshold) not in (int, float) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be a probability in [0, 1]")
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        if self.algorithm not in ("grid", "greedy"):
            raise ValueError("algorithm must be 'grid' or 'greedy'")
        if type(kernel) is not int or kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel_size must be an odd positive integer")
        if type(self.loss_weight) not in (int, float) or not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError("loss_weight must be finite and nonnegative")
        self.seeker = ObjSeeker(shallow_channels, kernel)
        self.epoch = 0

    @staticmethod
    def _boxes(batch, image_hw, device):
        gt = batch["bboxes"].to(device=device, dtype=torch.float32)
        valid = torch.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1) & ((gt[:, :2] >= 0) & (gt[:, :2] < 1)).all(1)
        h, w = image_hw
        boxes = xywhn2xyxy(gt[valid], w=w, h=h)
        return boxes, batch["batch_idx"].to(device).reshape(-1)[valid], valid

    def _target(self, batch, image_hw, logits):
        boxes, ids, _ = self._boxes(batch, image_hw, logits.device)
        return foreground_targets(logits, boxes, ids, self.stride)

    def split(self, shallow, hw, batch=None):
        h, w = hw
        if min(h, w) < 64 or h % 32 or w % 32 or shallow.shape[-2:] != (h // self.stride, w // self.stride):
            raise ValueError("ESOD requires aligned stride-4 P2 and input sides >=64 divisible by 32")
        patch_hw = tuple(max(64, 32 * math.ceil(d / (32 * self.patch_divisor))) for d in (h, w))
        logits = self.seeker(shallow)
        route = logits.detach().float().sigmoid()
        target = None
        warming = self.training and 0 <= self.epoch < self.warmup_epochs and batch is not None
        if warming and {"batch_idx", "bboxes"} <= batch.keys():
            target = self._target(batch, (h, w), logits)
            has_gt = target.amax(dim=(2, 3), keepdim=True) > 0
            route = torch.where(has_gt, target, route)
        layout = select_patches(route, tuple(d // self.stride for d in patch_hw), self.threshold, self.algorithm)
        state = {"layout": layout, "hw": patch_hw, "image_hw": (h, w), "logits": logits, "target": target}
        return layout.crop(shallow), state

    def local_batch(self, batch, state):
        layout = state["layout"]
        device = layout.offsets.device
        boxes, ids, valid = self._boxes(batch, state["image_hw"], device)
        local, patch_ids, source = layout.project_boxes(boxes, ids, self.stride, clip=False)
        h, w = state["hw"]
        return {
            "batch_idx": patch_ids,
            "cls": batch["cls"].to(device)[valid][source],
            "bboxes": xyxy2xywhn(local, w=w, h=h),
        }

    def aux_loss(self, batch, state):
        target, logits = state["target"], state["logits"]
        if target is None:
            target = self._target(batch, state["image_hw"], logits)
        return foreground_loss(logits, target) * self.loss_weight

    def restore(self, decoded, state):
        layout = state["layout"]
        k, c, a = decoded.shape
        if k != len(layout.offsets) or c < 5:
            raise ValueError("decoded must be [K, 4+nc, A] for the selected layout")
        if not k or not a:
            return decoded.new_zeros((layout.batch_size, c, 0))
        off = layout.offsets.to(decoded.device)
        centres = decoded[:, :2] + off[:, 1:3, None].to(decoded.dtype) * self.stride
        coords = torch.cat((centres, decoded[:, 2:4]), 1)
        valid = torch.isfinite(coords).all(1) & (coords[:, 2:] > 0).all(1)
        anchor_valid = state.get("anchor_valid")
        if anchor_valid is not None:
            if anchor_valid.dtype != torch.bool or anchor_valid.shape != (k, a):
                raise ValueError("anchor_valid must be a [K, A] boolean mask in decoded order")
            valid &= anchor_valid.to(decoded.device)
        elif not torch.equal(off[:, 3:5] - off[:, 1:3], off.new_tensor(layout.patch_hw[::-1]).expand(k, 2)):
            raise ValueError("padded layouts require anchor_valid from pre-regression anchor centres")
        scores = decoded[:, 4:].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        rows = torch.cat((coords, scores), 1).masked_fill(~valid[:, None], 0).transpose(1, 2)
        image_ids = off[:, 0]
        images = [rows[image_ids == i].reshape(-1, c) for i in range(layout.batch_size)]
        return pad_sequence(images, batch_first=True).transpose(1, 2)
