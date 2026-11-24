import json
from pathlib import Path

import pytest
import torch
from torch import nn

from dronebench.adapters.querydet import QueryDetAdapter
from dronebench.methods.querydet import QueryHead, query_focal_loss, query_targets
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel


def _adapt(model, feats, options):
    return QueryDetAdapter([f.shape[1] for f in feats], model.stride, options)


def test_dense_yolo_training_and_local_multilevel_focal():
    torch.manual_seed(11)
    model = DetectionModel("yolo11n.yaml", nc=3, verbose=False)
    model.args = get_cfg()
    head = model.model[-1]
    native = model(torch.randn(2, 3, 96, 128))
    feats = native["feats"]
    adapter = _adapt(model, feats, {"threshold": 1.0})
    raw = adapter(head, feats, torch.ones(2, 3, dtype=torch.bool))
    assert raw["feats"] is feats and raw["active"].all()
    for name in ("scores", "boxes"):
        torch.testing.assert_close(raw[name], native[name])
    batch = {
        "batch_idx": torch.tensor([1, 0]),
        "cls": torch.tensor([[0], [2]]),
        "bboxes": torch.tensor([[0.75, 0.25, 0.25, 0.25], [0.25, 0.75, 0.125, 0.25]]),
    }
    boxes_px = [torch.tensor([[24, 60, 40, 84]]), torch.tensor([[80, 12, 112, 36]])]
    logits = raw["query_state"]["logits"]
    assert all(type(q) is QueryHead for q in adapter.query_heads)
    assert [q.shape[-2:] for q in logits] == [(6, 8), (3, 4)]
    expected = [query_focal_loss(q, query_targets(q, boxes_px, s, (0, 4 * s))) for q, s in zip(logits, (16, 32))]
    aux = adapter.aux_loss(raw, batch, hw=(96, 128))
    torch.testing.assert_close(aux, torch.stack(expected).mean())
    no_boxes = {name: tensor[:0] for name, tensor in batch.items()}
    assert adapter.aux_loss(raw, no_boxes, (96, 128)).isfinite()
    order = batch["batch_idx"].argsort()
    loss, items = model.loss({k: v[order] for k, v in batch.items()}, preds=raw)
    assert set(items) == {"box_loss", "cls_loss", "dfl_loss"}
    (loss.sum() + 2 * aux).backward()
    for weight in [model.model[0].conv.weight, head.cv2[0][-1].weight, head.cv3[0][-1].weight]:
        assert weight.grad.isfinite().all() and weight.grad.abs().sum() > 0
    for q in adapter.query_heads:
        assert q[-1].weight.grad.isfinite().all() and q[-1].weight.grad.abs().sum() > 0


@torch.no_grad()
def test_checked_positions_context_and_disabled_query_carriers():
    torch.manual_seed(12)
    channels, strides = (8, 12, 16, 24), (4, 8, 16, 32)
    head = Detect(nc=3, reg_max=4, ch=channels).eval()
    head.stride = torch.tensor(strides)
    feats = [torch.randn(2, c, 96 // s, 128 // s) for c, s in zip(channels, strides)]
    adapter = QueryDetAdapter(channels, strides, {"depth": 0, "threshold": 0.5}).eval()
    for f, q in zip(feats[1:], adapter.query_heads):
        f[:, 0].fill_(-8)
        q[0].weight.zero_()
        q[0].weight[0, 0, 1, 1] = 1
        q[0].bias.zero_()
    feats[3][0, 0, 1, 1] = feats[3][1, 0, 0, 2] = 8
    feats[2][0, 0, 2, 3] = feats[2][0, 0, 2, 5] = 8
    feats[1][0, 0, 5, 7] = feats[1][0, 0, 4, 9] = 8
    mask = torch.tensor([[True, False, False, False], [False, False, True, True]])
    dense = head.forward_head(feats, **head.one2many)
    dense_conv_inputs = []
    hooks = [
        module.register_forward_pre_hook(lambda m, args: dense_conv_inputs.append(tuple(args[0].shape)))
        for branches in (head.cv2, head.cv3)
        for module in branches.modules()
        if isinstance(module, nn.Conv2d)
    ]
    try:
        raw = adapter(head, feats, mask)
    finally:
        for hook in hooks:
            hook.remove()
    assert dense_conv_inputs and all((b, h, w) == (1, 3, 4) for b, _, h, w in dense_conv_inputs)
    levels = raw["query_state"]["levels"]
    assert [len(level["keys"]) for level in levels] == [4, 4, 8, 24]
    assert [len(level["det_keys"]) for level in levels] == [4, 0, 4, 12]
    assert levels[0]["keys"].tolist() == [[0, 10, 14], [0, 10, 15], [0, 11, 14], [0, 11, 15]]
    assert len(levels[0]["det_context"]) == 36
    assert len(levels[1]["context"]) > len(levels[1]["keys"]) and not len(levels[1]["det_context"])
    assert (levels[2]["det_context"][:, 0] == 1).all()
    active = raw["active"]
    assert active.sum(1).tolist() == [4, 16]
    assert active[0].nonzero().flatten().tolist() == [334, 335, 366, 367]
    for name in ("boxes", "scores"):
        torch.testing.assert_close(raw[name].transpose(1, 2)[active], dense[name].transpose(1, 2)[active])
    decoded = head._inference(raw).transpose(1, 2)
    torch.testing.assert_close(decoded[active], head._inference(dense).transpose(1, 2)[active])
    assert decoded[~active][:, 4:].count_nonzero() == 0


@torch.no_grad()
def test_empty_queries_and_dense_training_level_mask():
    channels, strides = (8, 12, 16), (8, 16, 32)
    head = Detect(nc=2, reg_max=4, ch=channels)
    head.stride = torch.tensor(strides)
    feats = [torch.randn(2, c, 96 // s, 96 // s) for c, s in zip(channels, strides)]
    adapter = QueryDetAdapter(channels, strides, {"threshold": 1})
    mask = torch.tensor([[True, False, False], [False, False, True]])
    dense = adapter(head, feats, mask)
    assert dense["active"].sum(1).tolist() == [144, 9]
    assert all(q.shape[0] == 2 for q in dense["query_state"]["logits"])
    raw = adapter.eval()(head.eval(), feats, mask)
    levels = raw["query_state"]["levels"]
    assert all(level["keys"].shape == level["context"].shape == (0, 3) for level in levels[:-1])
    assert raw["active"].sum(1).tolist() == [0, 9]
    decoded = head._inference(raw)
    assert decoded.shape == (2, 6, 189) and decoded.isfinite().all() and decoded[0, 4:].count_nonzero() == 0
    assert adapter.aux_loss(raw, {}, (96, 96)).item() == 0


@pytest.mark.parametrize("size,fuse", [(96, False), (96, True), (640, False)])
@torch.no_grad()
def test_real_yolo_decoder_and_cpu_defaults(size, fuse):
    torch.manual_seed(13)
    model = DetectionModel("yolo11n.yaml", nc=10, verbose=False).eval()
    if fuse:
        model.fuse(verbose=False)
        assert not hasattr(model.model[-1].cv3[0][0][0], "bn")
    decoded, native = model(torch.randn(1, 3, size, size))
    adapter = _adapt(model, native["feats"], {} if size == 640 else {"threshold": 0.0}).eval()
    raw = adapter(model.model[-1], native["feats"], torch.ones(1, 3, dtype=torch.bool))
    actual = model.model[-1]._inference(raw)
    assert actual.shape == decoded.shape == (1, 14, sum((size // s) ** 2 for s in (8, 16, 32)))
    assert raw["active"][0, -(size // 32) ** 2 :].all() and actual.isfinite().all()
    if size == 96:
        assert raw["active"].all()
        torch.testing.assert_close(actual, decoded)


def test_single_config_and_unknown_option():
    config = json.loads(Path(__file__).with_name("single.json").read_text())
    assert config["querydet"] is True
    options = config["options"]["querydet"]
    QueryDetAdapter([8, 16, 32], [8, 16, 32], options)
    with pytest.raises(TypeError, match="threshod"):
        QueryDetAdapter([8, 16, 32], [8, 16, 32], {**options, "threshod": 0.1})
