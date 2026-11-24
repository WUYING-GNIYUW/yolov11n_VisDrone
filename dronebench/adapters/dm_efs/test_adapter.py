import io
import json
from pathlib import Path

import pytest
import torch

from dronebench.adapters.dm_efs import DMEFSAdapter
from dronebench.model import build_model, gated_detect
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import dist2bbox


@torch.no_grad()
def test_extension_keeps_original_objects_logits_and_loaded_biases():
    torch.manual_seed(21)
    base = build_model({}, nc=3).eval()
    head = base.model[-1]
    head.cv2[0][-1].bias.fill_(0.375)
    head.cv3[1][-1].bias.fill_(-7.25)
    old_box, old_cls = tuple(head.cv2), tuple(head.cv3)
    seen = []
    handle = base.model[2].register_forward_hook(lambda _, __, out: seen.append(out))
    _, stock = base(torch.rand(2, 3, 64, 128))
    handle.remove()
    shallow, native = seen[0], stock["feats"]
    args = (shallow.shape[1], [f.shape[1] for f in native], head.stride)
    for dfm in (False, True):
        dm = DMEFSAdapter(*args, {"efs": False, "dfm": dfm}).eval()
        dm.extend_head(head)
        feats, gate, _ = dm.route(shallow, native)
        assert tuple(feats) == tuple(native) and gate.all() and dm.strides == [8, 16, 32]
        assert tuple(head.cv2) == old_box and tuple(head.cv3) == old_cls
        assert (dm.predictor is not None) == (dm.codebook is not None) == dfm
    dm = DMEFSAdapter(*args, {"dfm": False}).eval()
    dm.extend_head(head)
    feats, gate, state = dm.route(shallow, native)
    assert feats[0] is shallow and tuple(feats[1:]) == tuple(native)
    assert gate.all() and head.nl == 4 and head.stride.tolist() == dm.strides == [4, 8, 16, 32]
    assert tuple(head.cv2)[1:] == old_box and tuple(head.cv3)[1:] == old_cls
    assert not list(dm.parameters()) and dm.codebook is None
    assert dm.aux_loss(state, {}, (64, 128), None, feats, True).item() == 0
    assert not head.cv2[0].training and head.cv2[0][-1].bias.eq(2.0).all()
    assert head.cv2[1][-1].bias.eq(0.375).all() and head.cv3[2][-1].bias.eq(-7.25).all()
    full = head.forward_head(feats, **head.one2many)
    n2 = shallow.shape[-2] * shallow.shape[-1]
    for k in ("scores", "boxes"):
        torch.testing.assert_close(full[k][:, :, n2:], stock[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unknown options.dm_efs"):
        DMEFSAdapter(*args, {"iou_threshol": 0.75})
    with pytest.raises(ValueError, match="booleans"):
        DMEFSAdapter(*args, {"efs": "false"})


def test_endpoint_loss_and_per_image_cell_votes_from_supplied_decoder_boxes():
    torch.manual_seed(22)
    args = (2, [3, 4, 5], [8, 16, 32], {"max_size": 32})
    dm = DMEFSAdapter(*args)
    maps = [torch.randn(3, c, 64 // s, 128 // s, requires_grad=True) for c, s in zip(dm.channels, dm.strides)]
    feats, mask, state = dm.route(maps[0], maps[1:])
    assert mask.all()
    a, b, c, d = torch.tensor([[16, 16, 24, 24], [32, 16, 40, 24], [64, 16, 88, 40], [96, 40, 104, 48]]).float()
    gt = torch.stack((d, a, c, b, torch.zeros(4)))
    batch = {"batch_idx": torch.tensor([1, 0, 0, 0, 2]), "bboxes": xyxy2xywh(gt) / torch.tensor([128, 64] * 2)}
    pred = torch.zeros(3, 680, 4)
    pred[0, 165], pred[0, 169], pred[0, 652] = a, b, c
    pred[1, 377], pred[1, 679] = d, torch.tensor([96, 40, 100, 48])
    pred[0, 512] = a
    pred.requires_grad_()
    loss = dm.aux_loss(state, batch, (64, 128), pred, feats, True)
    indices = torch.tensor([[8, 24], [8, 8]]) - 2
    logp = state["logits"][:2].log_softmax(-1).gather(-1, indices[..., None]).squeeze(-1)
    torch.testing.assert_close(loss, -0.04 * ((1 - logp.exp()).square() * logp).sum(1).mean())
    loss.backward()
    assert pred.grad is None and maps[0].grad.abs().sum() > 0
    expected = torch.zeros(4, 31, dtype=torch.long)
    expected[0, 8 - 2], expected[2, 24 - 2], expected[3, 8 - 2] = 2, 1, 1
    assert torch.equal(dm.codebook.votes, expected)
    with torch.no_grad():
        dm.aux_loss(state, batch, (64, 128), pred, feats, False)
        assert torch.equal(dm.codebook.votes, expected)
        dm.aux_loss(state, batch, (64, 128), pred, feats, True)
        dm.eval().aux_loss(state, batch, (64, 128), pred, feats, True)
        assert torch.equal(dm.codebook.votes, 2 * expected)
    buf = io.BytesIO()
    torch.save(dm.state_dict(), buf)
    buf.seek(0)
    restored = DMEFSAdapter(*args).half()
    restored.load_state_dict(torch.load(buf, weights_only=True))
    assert restored.codebook.votes.dtype == torch.long and torch.equal(restored.codebook.votes, 2 * expected)


@torch.no_grad()
def test_real_per_row_head_dispatch_and_requested_vs_executed_neck():
    torch.manual_seed(23)
    dm = DMEFSAdapter(
        1, [4, 8, 16], [8, 16, 32], {"max_size": 8, "projection_channels": 1, "hidden_channels": 1},
        backbone_channels=[1, 2],
    )
    maps = [torch.randn(2, c, 64 // s, 128 // s) for c, s in zip(dm.channels, dm.strides)]
    deep = torch.zeros(2, 2, 8, 16)
    deep[1] = 1
    for param in dm.predictor.parameters():
        param.zero_()
    dm.predictor.projections[1].weight[0, 0] = 1
    dm.predictor.shared[0].weight[0, 1] = 1
    for branch in dm.predictor.branches:
        branch[0].weight.fill_(1)
        branch[-1].bias[4 - 2] = 1
        branch[-1].weight[8 - 2] = 2
    dm.codebook.votes[0, 4 - 2] = dm.codebook.votes[2, 8 - 2] = 1
    backbone = (maps[0], deep)
    _, train_mask, train_state = dm.route(maps[0], maps[1:], backbone_feats=backbone)
    assert train_mask.all() and not train_state["head_requested"].all()
    feats, mask, state = dm.eval().route(maps[0], maps[1:], backbone_feats=backbone)
    assert state["size_range"].tolist() == [[4, 4], [8, 8]]
    assert mask.tolist() == [[True, False, False, False], [False, False, True, False]]
    assert state["neck_requested"].tolist() == [[True, False, False, False], [True, True, True, False]]
    assert state["neck_executed"].all() and torch.equal(state["head_enabled"], state["head_requested"])
    head = Detect(nc=3, reg_max=4, ch=(4, 8, 16)).eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    dm.extend_head(head)
    dense = head.forward_head(feats, **head.one2many)
    calls = []
    hooks = [
        op.register_forward_hook(lambda _, args, __, name=name, i=i: calls.append((name, i, args[0].shape[0])))
        for name, branches in head.one2many.items() for i, op in enumerate(branches)
    ]
    sparse = gated_detect(head, feats, mask)
    for hook in hooks:
        hook.remove()
    assert sorted(calls) == [(name, i, 1) for name in ("box_head", "cls_head") for i in (0, 2)]
    active = sparse["active"]
    for k in ("boxes", "scores"):
        torch.testing.assert_close(sparse[k].transpose(1, 2)[active], dense[k].transpose(1, 2)[active])


def test_public_single_experiment_train_eval_and_pan_execution():
    torch.manual_seed(16)
    spec = json.loads(Path(__file__).with_name("single.json").read_text())
    model = build_model(spec, nc=3)
    img = torch.rand(2, 3, 96, 128)
    batch = {
        "img": img, "cls": torch.tensor([[1], [2]]), "batch_idx": torch.arange(2),
        "bboxes": torch.tensor([[0.5, 0.5, 60 / 128, 60 / 96]] * 2),
    }
    backbone, controller_inputs = [], []
    hooks = [model.model[i].register_forward_hook(lambda _, __, out: backbone.append(out)) for i in (2, 4, 6, 10)]
    hooks.append(model.dm.predictor.register_forward_pre_hook(lambda _, args: controller_inputs.extend(args[0])))
    raw = model(img)
    for hook in hooks:
        hook.remove()
    assert len(controller_inputs) == 4 and all(a is b for a, b in zip(controller_inputs, backbone))
    assert [f.shape[-2:] for f in raw["feats"]] == [(24, 32), (12, 16), (6, 8), (3, 4)]
    assert raw["boxes"].shape == (2, 64, 1020) and raw["active"].all()
    losses, items = model.loss(batch, raw)
    assert set(items) == {"size_loss", "box_loss", "cls_loss", "dfl_loss"} and items["size_loss"] > 0
    losses.sum().backward()
    params = [model.model[2].cv2.conv.weight, model.model[-1].cv2[0][-1].weight]
    params.extend(branch[-1].weight for branch in model.model[-1].cv3)
    params.extend(branch[-1].weight for branch in model.dm.predictor.branches)
    params.extend(p.weight for p in model.dm.predictor.projections)
    assert all(p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in params)
    assert model.criterion.assigner.stride_val == 8
    votes = model.dm.codebook.votes.clone()
    assert votes[0, 60 - 2] == 2 and votes[1:].sum() == 0
    calls = []
    hooks = [
        m.register_forward_hook(lambda _, __, out, i=m.i: calls.append((i, out.shape[0]))) for m in model.model[11:23]
    ]
    with torch.no_grad():
        pred, val_raw = model.eval()(img)
        val_losses, _ = model.loss(batch, (pred, val_raw))
    for hook in hooks:
        hook.remove()
    assert calls == [(i, 2) for i in range(11, 23)]
    assert val_raw["level_mask"].tolist() == [[True, False, False, False]] * 2
    assert torch.equal(val_raw["dm_state"]["head_executed"], val_raw["level_mask"])
    assert pred.shape == (2, 7, 1020) and not pred[:, 4:, 24 * 32:].any()
    assert val_losses.isfinite().all() and torch.equal(model.dm.codebook.votes, votes)


@torch.no_grad()
def test_public_vote_uses_ugs_decoded_box_not_stock_dfl_on_the_same_logits():
    torch.manual_seed(24)
    model = build_model({"dm_efs": True, "ugs": True}, nc=3)
    head = model.model[-1]
    for tower in head.cv2:
        tower[-1].weight.zero_()
        bias = tower[-1].bias.view(4, -1)
        bias.fill_(-12)
        bias[:, 10] = 8
    img = torch.rand(1, 3, 64, 128)
    gt = torch.tensor([[25.0, 25, 43, 43]])
    batch = {"img": img, "batch_idx": torch.tensor([0]), "cls": torch.tensor([[1]]),
             "bboxes": xyxy2xywh(gt) / torch.tensor([128, 64] * 2)}
    raw = model(img)
    anchors, step, ugs_boxes = model.geometry(raw)
    stock_boxes = dist2bbox(head.dfl(raw["boxes"]).transpose(1, 2), anchors, xywh=False)
    cell = 8 * 32 + 8
    assert box_iou(gt, (ugs_boxes * step)[0, cell : cell + 1]).item() >= 0.5
    assert box_iou(gt, (stock_boxes * step)[0, cell : cell + 1]).item() < 0.5
    _, items = model.loss(batch, raw)
    assert items["size_loss"].isfinite()
    assert model.dm.codebook.votes.nonzero().tolist() == [[0, 18 - 2]]
