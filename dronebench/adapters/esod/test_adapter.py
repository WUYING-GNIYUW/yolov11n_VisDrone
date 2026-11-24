from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dronebench.adapters.esod import ESODAdapter
from dronebench.config import from_id
from dronebench.methods.esod import PatchLayout, select_patches
from dronebench.model import build_model
from ultralytics.utils.nms import non_max_suppression


def _picture(hw):
    path = Path(__file__).resolve().parents[3] / "ultralytics/assets/bus.jpg"
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB").resize(hw[::-1], Image.Resampling.BILINEAR))
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div_(255)


@pytest.mark.parametrize(
    "hw,expected", [((96, 96), (64, 64)), ((128, 128), (64, 64)), ((640, 640), (160, 160)), ((384, 640), (96, 160))]
)
def test_real_yolo_after_p2_on_selected_blocks_only(hw, expected):
    torch.manual_seed(17)
    model = build_model(from_id("01000")).train()
    x = _picture(hw)
    batch = {
        "img": x, "bboxes": torch.tensor([[0.18, 0.60, 0.24, 0.46]]),
        "batch_idx": torch.tensor([0]), "cls": torch.zeros(1, 1),
    }
    stem, deep = [], []
    model.model[2].register_forward_hook(lambda m, args, out: stem.append(out.detach()))
    model.model[3].register_forward_pre_hook(lambda m, args: deep.append(args[0].detach()))
    raw = model._raw(x, batch=batch)
    layout = raw["esod_state"]["layout"]
    k = len(layout.offsets)
    assert k and len(stem) == len(deep) == 1 and stem[0].shape[-2:] == (hw[0] // 4, hw[1] // 4)
    torch.testing.assert_close(deep[0], layout.crop(stem[0]), rtol=0, atol=0)
    assert raw["hw"] == expected and deep[0].shape == (k, stem[0].shape[1], expected[0] // 4, expected[1] // 4)
    assert ((layout.offsets[:, 3:] - layout.offsets[:, 1:3]) == torch.tensor(layout.patch_hw[::-1])).all()
    assert [tuple(f.shape[-2:]) for f in raw["feats"]] == [(expected[0] // s, expected[1] // s) for s in (8, 16, 32)]
    assert all(f.shape[0] == k for f in raw["feats"])
    total, items = model.loss(batch, preds=raw)
    assert model.last_diagnostics["positive"] > 0 and all(torch.isfinite(v) for v in items.values())
    total.sum().backward()
    for weight in (model.model[0].conv.weight, model.model[3].conv.weight, model.esod.seeker[-1].weight):
        assert weight.grad is not None and weight.grad.abs().sum() > 0 and torch.isfinite(weight.grad).all()
    box_grads = [level[-1].weight.grad for level in model.model[-1].cv2]
    assert all(torch.isfinite(g).all() for g in box_grads) and sum(g.abs().sum() for g in box_grads) > 0
    torch.optim.SGD(model.parameters(), lr=0.0001).step()
    with torch.inference_mode():
        pred, raw = model.eval()(x)
    assert raw["esod_state"]["target"] is None and len(deep) == len(stem) == 2
    assert pred.shape[:2] == (1, 14) and pred.shape[-1] > 0 and torch.isfinite(pred).all()
    assert len(non_max_suppression(pred, nc=10)) == 1


def test_teacher_requires_labelled_training_and_empty_skips_every_deep_layer():
    torch.manual_seed(8)
    model = build_model({"esod": True, "options": {"esod": {"threshold": 1.0, "warmup_epochs": 1}}})
    x = _picture((64, 96))
    batch = {
        "img": x, "bboxes": torch.tensor([[0.6, 0.6, 0.2, 0.4]]),
        "batch_idx": torch.tensor([0]), "cls": torch.zeros(1, 1),
    }
    calls = []
    for m in model.model[3:-1]:
        m.register_forward_pre_hook(lambda m, args: calls.append(m.i))
    model.train()
    yes = model._raw(x, batch)
    assert yes["esod_state"]["target"] is not None and calls == list(range(3, 23))
    calls.clear()
    assert not model._raw(x)["feats"] and not model._raw(x, {"img": x})["feats"]
    model.eval()
    with torch.inference_mode():
        pred, raw = model(x)
        assert model._raw(x, batch)["esod_state"]["target"] is None
        actual, items = model.loss(batch, preds=(pred, raw))
        assert torch.isfinite(actual).all() and items["foreground_loss"] > 0
    assert pred.shape == (1, 14, 0) and not raw["feats"] and not calls
    assert non_max_suppression(pred)[0].shape == (0, 6)
    model.train()
    model.esod.epoch = 1
    total, items = model.loss(batch)
    assert not calls and items["box_loss"] == items["dfl_loss"] == items["cls_loss"] == 0
    total.sum().backward()
    assert model.esod.seeker[-1].weight.grad.abs().sum() > 0 and model.model[3].conv.weight.grad is None


def test_centre_owned_full_box_overlap_and_per_image_per_class_nms():
    a = ESODAdapter(4, {})
    off = torch.tensor([[0, 0, 0, 16, 16], [0, 8, 0, 24, 16], [2, 0, 0, 16, 16]])
    s = {"layout": PatchLayout(off, (16, 24), (16, 16), 3), "hw": (64, 64), "image_hw": (64, 96)}
    gt = torch.tensor([[48, 32, 80, 20], [64, 32, 12, 12], [48, 32, -1, 8], [2, 2, 4, 4], [2, 2, 4, 4]]).float()
    batch = {
        "batch_idx": torch.tensor([0, 0, 0, 1, 2]), "cls": torch.arange(5)[:, None],
        "bboxes": gt / torch.tensor([96, 64, 96, 64]),
    }
    local = a.local_batch(batch, s)
    assert set(local) == {"batch_idx", "cls", "bboxes"} and local["batch_idx"].tolist() == [0, 1, 1, 2]
    assert local["cls"].flatten().tolist() == [0, 0, 1, 4]
    torch.testing.assert_close(local["bboxes"][:, 0] * 64, torch.tensor([48., 16., 32., 2.]))
    torch.testing.assert_close(local["bboxes"][:2, 2], torch.full((2,), 1.25))
    raw = torch.zeros(3, 4, 6)
    raw[0, 0] = torch.tensor([48, 32, 50, 20, 0.9, 0])
    raw[0, 1] = torch.tensor([64, 32, 0, 10, 1., 1.])
    raw[1, 0] = torch.tensor([16, 32, 50, 20, 0.8, 0])
    raw[1, 1] = torch.tensor([16, 32, 50, 20, 0, 0.7])
    raw[1, 2] = torch.tensor([16, 32, -3, 10, 1., 1.])
    raw[2, 0] = torch.tensor([0, 4, 20, 12, 0.95, 0])
    packed = a.restore(raw.transpose(1, 2), s)
    assert packed.shape == (3, 6, 8) and packed[1].count_nonzero() == 0 and packed[2, :, 4:].count_nonzero() == 0
    assert packed[0, :, 1].count_nonzero() == packed[0, :, 6].count_nonzero() == 0
    rows = non_max_suppression(packed, nc=2)
    assert [len(r) for r in rows] == [2, 0, 1]
    torch.testing.assert_close(rows[0], torch.tensor([[23, 22, 73, 42, 0.9, 0], [23, 22, 73, 42, 0.7, 1]]))
    torch.testing.assert_close(rows[2][0, :4], torch.tensor([-10, -2, 10, 10]).float())


def test_single_full_block_matches_stock_decode_and_nms_without_clipping():
    x = _picture((96, 128))
    torch.manual_seed(31)
    stock = build_model(from_id("00000")).eval()
    torch.manual_seed(31)
    model = build_model({"esod": True, "options": {"esod": {"threshold": 0, "patch_divisor": 1}}}).eval()
    with torch.inference_mode():
        expected, _ = stock(x)
        actual, raw = model(x)
    state = raw["esod_state"]
    assert state["layout"].offsets.tolist() == [[0, 0, 0, 32, 24]]
    torch.testing.assert_close(actual, expected)
    opts = {"conf_thres": .00001, "iou_thres": .65, "nc": 10}
    stock_rows = non_max_suppression(expected, **opts)[0]
    assert len(stock_rows) > 0
    torch.testing.assert_close(non_max_suppression(actual, **opts)[0], stock_rows)
    local = torch.tensor([[2, 15, 16, 10, .99], [5, 15, 10, 10, .90], [-3, 25, 14, 10, .85]]).T[None]
    same = model.esod.restore(local, state)
    torch.testing.assert_close(same, local, rtol=0, atol=0)
    after = non_max_suppression(same, iou_thres=.65)[0]
    torch.testing.assert_close(after, non_max_suppression(local, iou_thres=.65)[0], rtol=0, atol=0)
    assert len(after) == 3 and after[0, :4].tolist() == [-6, 10, 10, 20] and after[2, 0] == -10


def test_padded_layout_requires_true_anchor_mask_and_filters_bad_geometry():
    adapter = ESODAdapter(5, {})
    p = torch.ones(1, 1, 16, 24)
    layout = select_patches(p, (24, 32))
    s = {"layout": layout, "hw": (96, 128), "image_hw": (64, 96)}
    boxes = torch.tensor(
        [[99, 70, 4, 4, .75], [10, 10, 4, 4, 1], [10, 5, 4, 4, 1], [1, 1, 8, -1, 1],
         [0, 0, 4, 0, 1], [8, 8, 2, float("inf"), 1], [float("nan"), 8, 2, 4, 1]]
    )
    with pytest.raises(ValueError, match="pre-regression"):
        adapter.restore(boxes.T[None], s)
    anchors = torch.tensor([[92, 60], [100, 4], [4, 68], [4, 4], [4, 4], [12, 12], [12, 12]])
    s["anchor_valid"] = layout.valid_centers(anchors, torch.zeros(7, dtype=torch.long), 4)[None]
    restored = adapter.restore(boxes.T[None], s)
    torch.testing.assert_close(restored[0, :, 0], boxes[0])
    assert restored[:, :, 1:].count_nonzero() == 0 and torch.isfinite(restored).all()
    s["layout"] = select_patches(p * 0, (24, 32))
    assert adapter.restore(boxes.new_empty((0, 5, 7)), s).shape == (1, 5, 0)
    empty = {"batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)}
    assert adapter.local_batch(empty, s)["bboxes"].shape == (0, 4)


@pytest.mark.parametrize("options", [{"thresh": 0.5}, {"warmup_epochs": -1}, {"threshold": 1.1}, {"patch_divisor": 0}])
def test_reject_invalid_options(options):
    with pytest.raises(ValueError):
        ESODAdapter(4, options)
