from __future__ import annotations

from operator import index
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .query import _coords, _flat, _hw, _offsets, _unflat, expand_context, map_children, select_queries


class SparseQueryPlan:
    def __init__(self, keys: Tensor, active: Tensor, shape: tuple[int, int, int]):
        b, h, w = map(index, shape)
        _hw((h, w))
        _coords(keys)
        _coords(active)
        if b < 1 or keys.device != active.device:
            raise ValueError("expected a positive batch and coordinates on the same device")
        bounds = keys.new_tensor((b, h, w))
        for coords in (keys, active):
            if ((coords < 0) | (coords >= bounds)).any():
                raise ValueError("coordinates outside the (batch, height, width) grid")
        self.shape = (b, h, w)
        self.keys = keys
        self._ids = _flat(active, h, w).unique(sorted=True)
        self.active = _unflat(self._ids, h, w)
        key_ids = _flat(keys, h, w)
        self._readout = torch.searchsorted(self._ids, key_ids)
        if (self._readout == len(self._ids)).any() or not torch.equal(self._ids[self._readout], key_ids):
            raise ValueError("every output key must be in the active context")
        self._neighbors = {}

    def _lookup(self, kernel, dilation):
        key = (kernel, dilation)
        if key not in self._neighbors:
            _, h, w = self.shape
            ry, rx = kernel[0] // 2, kernel[1] // 2
            dy, dx = _offsets(-ry, ry + 1, -rx, rx + 1, self.active.device)
            b, y, x = self.active.T.unsqueeze(-1)
            y = y + dy * dilation[0]
            x = x + dx * dilation[1]
            ids = (b * h + y) * w + x
            slots = torch.searchsorted(self._ids, ids)
            n = len(self._ids)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            hit = inside & (slots < n) & (self._ids[slots.clamp(max=n - 1)] == ids)
            self._neighbors[key] = torch.where(hit, slots, n)
        return self._neighbors[key]

    def _convolve(self, layer, values, chunk_size):
        kernel, dilation = layer.kernel_size, layer.dilation
        pad = tuple(k // 2 * d for k, d in zip(kernel, dilation))
        if (
            layer.stride != (1, 1)
            or any(k % 2 != 1 for k in kernel)
            or layer.padding not in (pad, "same")
            or layer.padding_mode != "zeros"
        ):
            raise ValueError("sparse convolution requires stride 1, odd kernels and same-size zero padding")
        if values.shape[1] != layer.in_channels:
            raise ValueError("feature channels do not match the convolution input")
        if not len(values):
            return values.new_empty((0, layer.out_channels))
        if kernel == (1, 1) and layer.groups == 1:
            return F.linear(values, layer.weight[:, :, 0, 0], layer.bias)
        neighbors = self._lookup(kernel, dilation)
        padded = torch.cat((values, values.new_zeros((1, values.shape[1]))))
        outputs = []
        for sites in neighbors.split(chunk_size):
            patches = padded[sites].reshape(len(sites), *kernel, layer.in_channels).permute(0, 3, 1, 2)
            out = F.conv2d(patches, layer.weight, layer.bias, groups=layer.groups)
            outputs.append(out[:, :, 0, 0])
        return torch.cat(outputs)

    def _apply(self, head, values, chunk_size):
        if isinstance(head, nn.Sequential):
            for layer in head:
                values = self._apply(layer, values, chunk_size)
            return values
        if isinstance(head, nn.Conv2d):
            return self._convolve(head, values, chunk_size)
        if isinstance(head, nn.BatchNorm2d):
            if head.training or not head.track_running_stats:
                raise ValueError("BatchNorm2d needs fixed running statistics (call eval() before sparse inference)")
            return head(values[:, :, None, None])[:, :, 0, 0]
        if isinstance(head, (nn.ReLU, nn.SiLU, nn.LeakyReLU, nn.Identity)):
            return head(values)
        raise TypeError(f"unsupported sparse head operator: {type(head).__name__}")

    def run(self, head: nn.Module, feature: Tensor, *, chunk_size: int = 256) -> Tensor:
        if feature.ndim != 4 or (feature.shape[0], *feature.shape[2:]) != self.shape:
            raise ValueError("expected a B x C x H x W feature matching the plan grid")
        if feature.device != self.active.device or index(chunk_size) < 1:
            raise ValueError("the plan and feature must share a device, and chunk_size must be positive")
        b, y, x = self.active.T
        values = feature.permute(0, 2, 3, 1)[b, y, x]
        return self._apply(head, values, chunk_size)[self._readout]


@torch.no_grad()
def cascade_plans(
    features: Sequence[Tensor],
    query_heads: Sequence[nn.Module],
    *,
    threshold: float = 0.15,
    radius: int = 2,
    scales: Sequence[int] | None = None,
) -> list[SparseQueryPlan]:
    if len(features) < 2 or len(query_heads) != len(features) - 1:
        raise ValueError("provide features from coarse to fine and one query head per transition")
    scales = [2] * len(query_heads) if scales is None else scales
    if len(scales) != len(query_heads):
        raise ValueError("provide one integer stride ratio per transition")
    plans = []
    for coarse, fine, head, scale in zip(features, features[1:], query_heads, scales):
        scale = index(scale)
        if coarse.ndim != 4 or fine.ndim != 4 or coarse.shape[0] != fine.shape[0] or coarse.device != fine.device:
            raise ValueError("all pyramid levels must be NCHW tensors in the same batch on the same device")
        hw = _hw(fine.shape[-2:])
        if scale < 1 or tuple((side + scale - 1) // scale for side in hw) != tuple(coarse.shape[-2:]):
            raise ValueError("the coarse grid must cover the fine grid using the supplied stride ratio")
        if plans:
            previous = plans[-1]
            parents = select_queries(previous.run(head, coarse), threshold, previous.keys)
        else:
            logits = head(coarse)
            if (logits.shape[0], *logits.shape[2:]) != (coarse.shape[0], *coarse.shape[2:]):
                raise ValueError("the dense query head must preserve the batch and spatial grid")
            parents = select_queries(logits, threshold)
        keys = map_children(parents, hw, scale)
        context = expand_context(keys, hw, radius)
        plans.append(SparseQueryPlan(keys, context, (fine.shape[0], *hw)))
    return plans
