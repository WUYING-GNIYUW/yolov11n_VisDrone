import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


class TwoHot(NamedTuple):
    left: Tensor
    right: Tensor
    right_weight: Tensor
    clipped: Tensor


def make_support(
    num_bins: int,
    alpha: float,
    beta: float,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer >= 2")
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(beta) or beta < 0:
        raise ValueError("alpha must be finite and positive; beta must be finite and nonnegative")
    strength = alpha * beta
    if not math.isfinite(strength) or not dtype.is_floating_point:
        raise ValueError("alpha * beta must be finite and the support dtype must be floating point")
    n = num_bins - 1
    u = (2 * torch.arange(num_bins, dtype=torch.float64, device="cpu") - n) / n
    if strength:
        a = u.abs()
        ratio = torch.exp(strength * (a - 1)) * (torch.expm1(-strength * a) / math.expm1(-strength))
        u = u.sign() * ratio
    grid = (alpha * u).to(dtype)
    if not torch.isfinite(grid).all() or not (grid[1:] > grid[:-1]).all():
        raise ValueError("support points are not finite and distinct at this precision; adjust alpha, beta or dtype")
    return grid.to(device=device)


def _check_support(support: Tensor, values: Tensor) -> None:
    if support.ndim != 1 or support.numel() < 2 or not support.is_floating_point():
        raise ValueError("support must be a one-dimensional floating-point grid of at least two points")
    if support.device != values.device:
        raise ValueError("support must be on the same device as the inputs")


def _work_logits(logits: Tensor) -> Tensor:
    if logits.ndim < 1 or logits.shape[-1] < 2 or not logits.is_floating_point():
        raise ValueError("logits must be floating point, with at least two bins on the last axis")
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def encode_two_hot(target: Tensor, support: Tensor) -> TwoHot:
    _check_support(support, target)
    dtype = torch.promote_types(torch.promote_types(target.dtype, support.dtype), torch.float32)
    t, s = target.to(dtype), support.to(dtype)
    clipped = (t < s[0]) | (t > s[-1])
    t = t.clamp(min=s[0], max=s[-1])
    right = torch.searchsorted(s, t.contiguous()).clamp(1, s.numel() - 1)
    left = right - 1
    right_weight = (t - s[left]) / (s[right] - s[left])
    return TwoHot(left, right, right_weight, clipped)


def localization_ce(logits: Tensor, target: Tensor, support: Tensor) -> Tensor:
    if logits.shape[:-1] != target.shape or logits.shape[-1:] != support.shape:
        raise ValueError("expected logits.shape == (*target.shape, support.numel())")
    left, right, w, _ = encode_two_hot(target, support)
    logp = _work_logits(logits).log_softmax(-1)
    selected = logp.gather(-1, torch.stack((left, right), dim=-1))
    weights = torch.stack((1 - w, w), dim=-1)
    return -(weights * selected.masked_fill(weights == 0, 0)).sum(-1)


def decode_logits(logits: Tensor, support: Tensor) -> Tensor:
    _check_support(support, logits)
    if logits.shape[-1:] != support.shape:
        raise ValueError("the last logits dimension must equal the number of support points")
    return (_work_logits(logits).softmax(-1) * support).sum(-1)


def uncertainty_entropy(logits: Tensor) -> Tensor:
    logp = _work_logits(logits).log_softmax(-1)
    p = logp.exp()
    return -(p * logp.masked_fill(p == 0, 0)).sum(-1)


def encode_ltrb(points_xy: Tensor, boxes_xyxy: Tensor) -> Tensor:
    if points_xy.shape[-1:] != (2,) or boxes_xyxy.shape[-1:] != (4,):
        raise ValueError("points must be (..., 2) xy; boxes must be (..., 4) xyxy")
    return torch.cat((points_xy - boxes_xyxy[..., :2], boxes_xyxy[..., 2:] - points_xy), dim=-1)


def decode_ltrb(points_xy: Tensor, distances_ltrb: Tensor) -> Tensor:
    if points_xy.shape[-1:] != (2,) or distances_ltrb.shape[-1:] != (4,):
        raise ValueError("points must be (..., 2) xy; distances must be (..., 4) ltrb")
    return torch.cat((points_xy - distances_ltrb[..., :2], points_xy + distances_ltrb[..., 2:]), dim=-1)


def uncertainty_perturbations(features: Sequence[Tensor], um_loss: Tensor, rho: float = 0.5) -> Tuple[Tensor, ...]:
    features = tuple(features)
    if not features or any(f.ndim < 2 or not f.is_floating_point() for f in features):
        raise ValueError("features must be a nonempty sequence of floating-point tensors with a batch axis")
    if um_loss.ndim != 0 or not math.isfinite(rho) or rho < 0:
        raise ValueError("um_loss must be a scalar and rho must be finite and nonnegative")
    if rho == 0:
        return tuple(torch.zeros_like(f) for f in features)
    if not um_loss.requires_grad or not all(f.requires_grad for f in features):
        raise ValueError("UR needs a differentiable clean entropy and the features used to compute it")
    gradients = torch.autograd.grad(um_loss, features, retain_graph=True, create_graph=False, allow_unused=True)
    if all(grad is None for grad in gradients):
        raise ValueError("clean entropy is disconnected from every supplied feature")
    deltas = []
    for f, g in zip(features, gradients):
        if g is None:
            deltas.append(torch.zeros_like(f))
            continue
        flat = g.detach().to(torch.promote_types(g.dtype, torch.float32)).flatten(1)
        norm = flat.norm(p=2, dim=1, keepdim=True)
        unit = flat / norm.masked_fill(norm == 0, 1)
        deltas.append((rho * unit).reshape_as(f).to(f.dtype))
    return tuple(deltas)


def refinement_loss(
    features: Sequence[Tensor],
    clean_um: Tensor,
    perturbed_um: Callable[[Tuple[Tensor, ...]], Tensor],
    rho: float = 0.5,
    feature_penalty: float = 0.0,
) -> Tensor:
    if not math.isfinite(feature_penalty) or feature_penalty < 0:
        raise ValueError("feature_penalty must be finite and nonnegative")
    features = tuple(features)
    deltas = uncertainty_perturbations(features, clean_um, rho)
    loss = perturbed_um(tuple(f + d for f, d in zip(features, deltas)))
    if loss.ndim != 0:
        raise ValueError("perturbed_um must return a scalar")
    if feature_penalty:
        loss = loss + feature_penalty * sum(
            f.to(torch.promote_types(f.dtype, torch.float32)).flatten(1).square().sum(1).mean() for f in features
        )
    return loss
