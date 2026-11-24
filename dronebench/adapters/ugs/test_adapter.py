from pathlib import Path

import pytest
import torch
from torch import nn

from dronebench.adapters.ugs import UGSAdapter
from dronebench.config import read_spec
from dronebench.methods.ugs import decode_ltrb, encode_two_hot, localization_ce, make_support, uncertainty_entropy
from dronebench.model import build_model
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import make_anchors


def _case(options, training=True, empty=False):
    torch.manual_seed(13)
    head = Detect(nc=2, reg_max=8, ch=(4, 6)).train(training)
    adapter = UGSAdapter(8, options).train(training)
    adapter.initialize_head(head)
    head.cv2[0][0].bn.eval()
    head.cv2[0][1].bn.track_running_stats = False
    feats = [torch.randn(2, 4, 4, 6, requires_grad=True), torch.randn(2, 6, 2, 3, requires_grad=True)]
    raw = head.forward_head(feats, **head.one2many)
    anchors, _ = make_anchors(feats, [8, 16])
    boxes = adapter.decode(raw["boxes"].transpose(1, 2), anchors)
    soft = torch.zeros(2, len(anchors), 2)
    if not empty:
        soft[0, 3, 0], soft[0, -1, 1], soft[1, 4, 0] = 0.25, 1, 0.5
    fg = soft.any(-1)
    raw["active"] = torch.ones_like(fg)
    target = torch.cat((anchors - 0.75, anchors + 0.5), dim=-1).expand_as(boxes)
    return adapter, head, raw, (boxes, anchors, target, soft, soft.sum().clamp_min(1), fg)


def test_prior_on_every_level_and_persistent_signed_support():
    head = Detect(nc=2, reg_max=16, ch=(4, 8, 16, 32)).eval()
    head.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
    head.bias_init()
    adapter = UGSAdapter(16, {})
    support = make_support(16, 15, 2 / 15)
    cls_bias = [t[-1].bias.clone() for t in head.cv3]
    assert torch.equal(adapter.state_dict()["support"], support)
    assert torch.equal(adapter.half().float().support, support)
    uniform = UGSAdapter(16, {"nonuniform": False}).support
    torch.testing.assert_close(uniform, torch.linspace(-15, 15, 16))
    assert not torch.allclose(uniform, support)
    adapter.initialize_head(head)
    feats = [torch.zeros(1, c, 16 // 2**i, 16 // 2**i) for i, c in enumerate((4, 8, 16, 32))]
    with torch.no_grad():
        raw = head.forward_head(feats, **head.one2many)
        anchors, _ = make_anchors(feats, head.stride)
        boxes = adapter.decode(raw["boxes"].transpose(1, 2), anchors)
    assert (boxes[..., :2] < anchors).all() and (boxes[..., 2:] > anchors).all()
    assert all(torch.equal(t[-1].bias, before) for t, before in zip(head.cv3, cls_bias))


@pytest.mark.parametrize("options", [{"unknown": 1}, {"um": 1}, {"alpha": 0}, {"rho": -1}, {"beta": float("nan")}])
def test_reject_bad_options(options):
    with pytest.raises(ValueError):
        UGSAdapter(16, options)


def test_signed_geometry_endpoint_clip_and_real_diagnostics():
    adapter = UGSAdapter(5, {"alpha": 2, "beta": 0.5, "um": False, "ur": False})
    head = Detect(nc=1, reg_max=5, ch=(4,))
    anchor = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
    distances = torch.tensor([[[-0.5, 0.5, 1.5, 0.5], [-3, 0.5, 6, 0.5], [-2, 0.5, -2, 0.5]]])
    target = decode_ltrb(anchor, distances)
    s = adapter.support
    left, right, w, _ = encode_two_hot(distances, s)
    q = torch.zeros(*distances.shape, len(s))
    q.scatter_(-1, left[..., None], (1 - w)[..., None]).scatter_(-1, right[..., None], w[..., None])
    logits = q.clamp_min(1e-30).log().requires_grad_()
    flat = logits.flatten(2)
    boxes = adapter.decode(flat, anchor)
    stride = torch.tensor([[8], [16], [32]])
    torch.testing.assert_close((boxes * stride)[0, 0], torch.tensor([8.0, 0.0, 16.0, 8.0]))
    torch.testing.assert_close((boxes * stride)[0, 1], torch.tensor([56.0, 0.0, 56.0, 16.0]))
    torch.testing.assert_close(boxes[0, 2], torch.tensor([2.5, 0.0, 2.5, 1.0]))
    fg = torch.tensor([[True, True, False]])
    raw = {"boxes": flat.transpose(1, 2), "active": fg}
    soft = torch.tensor([[[0.25], [0.75], [0.0]]])
    box, loc, um, ur, diag = adapter.losses(head, raw, boxes, anchor, target, soft, soft.sum(), fg, refine=True)
    exp_ce = localization_ce(logits[fg], distances[fg], s).mean(-1)
    torch.testing.assert_close(loc, (exp_ce * soft[fg].flatten()).sum())
    exp_ciou = bbox_iou(boxes[fg], target[fg], xywh=False, CIoU=True)
    torch.testing.assert_close(box, ((1 - exp_ciou) * soft[fg]).sum())
    assert diag["ugs_target_coords"] == 8 and diag["ugs_outside_fg"] == diag["ugs_negative_coords"] == 2
    assert diag["ugs_clipped_low_coords"] == diag["ugs_clipped_high_coords"] == 1
    assert diag["ugs_degenerate_active"] == diag["ugs_degenerate_fg"] == 1 and not diag["ugs_refined"]
    assert um.item() == ur.item() == 0
    (box + loc).backward()
    assert logits.grad.isfinite().all() and not logits.grad[~fg].any()


@pytest.mark.parametrize("rho", [0.0, 0.2])
def test_refinement_reuses_clean_positives_and_batch_moments(rho):
    adapter, head, raw, args = _case({"rho": rho, "lambda_um": 0.25, "gamma_ur": 0.75})
    bns = [m for m in head.cv2.modules() if isinstance(m, nn.BatchNorm2d)]
    flags = [(bn.training, bn.track_running_stats) for bn in bns]
    stats = [(bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()) for bn in bns]
    calls = []
    hooks = [
        t.register_forward_hook(lambda m, x, y: calls.append((m, x[0].detach(), y.detach())))
        for t in (*head.cv2, *head.cv3)
    ]
    try:
        _, _, um, ur, diag = adapter.losses(head, raw, *args, refine=True)
    finally:
        for h in hooks:
            h.remove()
    assert diag["ugs_refined"] and [m for m, _, _ in calls] == list(head.cv2)
    for bn, flags_before, (mean, var, count) in zip(bns, flags, stats):
        assert (bn.training, bn.track_running_stats) == flags_before
        assert torch.equal(bn.running_mean, mean) and torch.equal(bn.running_var, var)
        assert torch.equal(bn.num_batches_tracked, count)
    soft, norm, fg = args[-3:]
    weights = soft[fg].sum(-1) / norm

    def expected(packed):
        logits = packed.transpose(1, 2)[fg].reshape(-1, 4, 8)
        return (uncertainty_entropy(logits).mean(-1) * weights).sum()

    torch.testing.assert_close(um, 0.25 * expected(raw["boxes"]))
    torch.testing.assert_close(ur, 0.75 * expected(torch.cat([y.flatten(2) for _, _, y in calls], dim=2)))
    if rho == 0:
        torch.testing.assert_close(ur / 0.75, um / 0.25)
    else:
        for clean, (_, perturbed, _) in zip(raw["feats"], calls):
            sizes = (perturbed - clean).flatten(1).norm(dim=1)
            torch.testing.assert_close(sizes, torch.full_like(sizes, rho))
    grad = torch.autograd.grad(um, raw["boxes"], retain_graph=True)[0].transpose(1, 2)
    assert not grad[~fg].any() and grad[fg].abs().sum() > 0
    ur.backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in raw["feats"])
    assert all(t[-1].weight.grad.norm() > 0 for t in head.cv2)
    assert all(t[-1].weight.grad is None for t in head.cv3)


@pytest.mark.parametrize(
    "training,grad,um,ur,refine",
    [
        (True, True, False, True, True),
        (True, True, True, False, True),
        (False, True, True, True, True),
        (True, False, True, True, True),
        (True, True, True, True, False),
    ],
)
def test_independent_entropy_and_refinement_gates(training, grad, um, ur, refine):
    with torch.set_grad_enabled(grad):
        adapter, head, raw, args = _case({"um": um, "ur": ur}, training)
        calls = []
        handle = head.cv2[0].register_forward_hook(lambda *args: calls.append(1))
        try:
            _, _, clean_loss, perturbed_loss, diag = adapter.losses(head, raw, *args, refine=refine)
        finally:
            handle.remove()
    ran = training and grad and ur and refine
    assert len(calls) == int(ran) and diag["ugs_refined"] is ran
    assert (clean_loss.item() > 0) == um and (perturbed_loss.item() > 0) == ran


def test_empty_foreground_backpropagates_zero_through_the_real_head():
    adapter, head, raw, args = _case({}, empty=True)
    *losses, diag = adapter.losses(head, raw, *args, refine=True)
    assert all(loss.requires_grad and loss.item() == 0 for loss in losses)
    assert not diag["ugs_refined"] and diag["ugs_target_coords"] == 0
    sum(losses).backward()
    assert all(f.grad is not None and not f.grad.any() for f in raw["feats"])
    assert all(t[-1].weight.grad is not None and not t[-1].weight.grad.any() for t in head.cv2)


def test_single_yolo11n_trains_and_validates_with_shared_geometry():
    torch.manual_seed(7)
    spec = read_spec(config=Path(__file__).with_name("single.json"))
    model = build_model(spec, nc=2).train()
    imgs = torch.rand(2, 3, 64, 96)
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[1.0], [0.0]]),
        "bboxes": torch.tensor([[0.42, 0.42, 0.14, 0.18], [0.27, 0.68, 0.12, 0.13]]),
    }
    total, items = model(batch)
    assert list(items) == ["box_loss", "cls_loss", "ugs_loc_loss", "ugs_um_loss", "ugs_ur_loss"]
    assert total.isfinite().all() and (total > 0).all() and model.last_diagnostics["ugs_refined"]
    total.sum().backward()
    assert model.model[-1].cv2[0][-1].weight.grad.norm() > 0
    model.eval()
    with torch.no_grad():
        proj = model.model[-1].cv2[0][-1]
        proj.weight.zero_()
        bias = proj.bias.view(4, model.ugs.reg_max)
        bias.fill_(-30)
        bias[:, 0] = 30
        decoded, raw = model(imgs)
        _, stride, boxes = model.geometry(raw)
        torch.testing.assert_close(decoded[:, :4].transpose(1, 2), xyxy2xywh(boxes * stride))
        first_level = raw["feats"][0].shape[-2] * raw["feats"][0].shape[-1]
        assert not decoded[:, 4:, :first_level].any() and raw["scores"][:, :, :first_level].isfinite().all()
        _, val = model.loss(batch, (decoded, raw))
    assert val["ugs_ur_loss"].item() == 0 and not model.last_diagnostics["ugs_refined"]
    assert model.last_diagnostics["ugs_degenerate_active"] >= imgs.shape[0] * first_level
