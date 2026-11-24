import math

import numpy as np
import pytest
import torch

from dronebench.evaluation.validator import FixedBinValidator, image_counts
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops


def _pred(boxes, scores, classes):
    return {
        "bboxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "conf": torch.tensor(scores, dtype=torch.float32),
        "cls": torch.tensor(classes, dtype=torch.float32),
    }


def _batch(boxes, classes, shape=(100, 640), gains=(1, 1), pads=(0, 0), imgsz=(100, 640)):
    return {
        "bboxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "cls": torch.tensor(classes, dtype=torch.float32),
        "ori_shape": shape,
        "imgsz": imgsz,
        "ratio_pad": (gains, pads),
    }


@pytest.mark.parametrize(
    "imgsz,gy,gx,px,py",
    [((512, 640), 360 / 765, 640 / 1360, 0, 76), ((640, 992), 540 / 765, 960 / 1360, 16, 50)],
)
def test_reference_bins_letterbox_rounded_labels_and_clipping(imgsz, gy, gx, px, py):
    gt = np.array([[100, 0, 130, 16], [100, 48, 137, 65], [180, 120, 242, 153], [0, 200, 34, 250]], dtype=float)
    frame = np.array([1360, 765, 1360, 765])
    normalized = np.round(ops.xyxy2xywh(gt) / frame, 6)
    stored = ops.xywh2xyxy(normalized * frame)
    gains, pads = np.array([gx, gy, gx, gy]), np.array([px, py, px, py])
    batch = _batch(stored * gains + pads, [0, 1, 2, 0], (765, 1360), (gy, gx), (px, py), imgsz)
    boxes = gt[[0, 1, 1, 2, 2, 3]].copy()
    boxes[0, 1] = -40
    pred = _pred(boxes * gains + pads, [0.25, 0.99, 0.8, 0.249, 0.95, 0.7], [0, 1, 1, 2, 1, 0])
    untouched = [batch["bboxes"].clone(), pred["bboxes"].clone()]
    assert image_counts(pred, batch).tolist() == [[1, 2, 1], [1, 1, 1]]
    assert torch.equal(batch["bboxes"], untouched[0]) and torch.equal(pred["bboxes"], untouched[1])


def test_portrait_uses_long_axis_and_real_rounded_resize_gains():
    gains = (64 / 2048, 11 / 321)
    original = np.array([[10, 700, 35, 780]], dtype=float)
    input_boxes = original * np.array(gains[::-1] * 2) + [26, 0, 26, 0]
    batch = _batch(input_boxes, [0], (2048, 321), gains, (26, 0), (64, 64))
    assert image_counts(_pred(input_boxes, [0.7], [0]), batch).tolist() == [[1, 0, 0], [1, 0, 0]]


def test_subpixel_extents_are_not_rounded_across_thresholds():
    boxes = [[1, 2, 8.9, 20], [20, 2, 45, 17.99], [60, 2, 76.1, 20]]
    batch = _batch(boxes, [0, 0, 0])
    assert image_counts(_pred(boxes, [0.9] * 3, [0] * 3), batch).tolist() == [[1, 1, 1], [1, 1, 1]]


def test_equal_iou_uses_first_ground_truth():
    batch = _batch([[0, 0, 4, 10], [0, 0, 9, 10]], [1, 1])
    pred = _pred([[0, 0, 6, 10]], [0.75], [1])
    assert image_counts(pred, batch).tolist() == [[1, 1, 0], [1, 0, 0]]


def test_global_score_order_and_cross_bin_assignment():
    batch = _batch([[0, 0, 7, 10], [0, 0, 9, 10]], [0, 0])
    pred = _pred([[3, 0, 9, 10], [0, 0, 8, 10]], [0.8, 0.9], [0, 0])
    assert image_counts(pred, batch).tolist() == [[1, 1, 0], [0, 1, 0]]
    pred["conf"] = pred["conf"].flip(0)
    assert image_counts(pred, batch).tolist() == [[1, 1, 0], [1, 1, 0]]
    pred["conf"][:] = 0.8
    assert image_counts(pred, batch).tolist() == [[1, 1, 0], [1, 1, 0]]


def test_score_threshold_iou_equality_image_cap_and_empty_sides():
    box = [30, 40, 46, 60]
    batch = _batch([box], [1])
    half_box = [30, 40, 38, 60]
    pred = _pred([half_box] + [box] * 300, [0.25] + [0.99] * 300, [1] + [0] * 300)
    assert image_counts(pred, batch).tolist() == [[0, 0, 1], [0, 0, 0]]
    pred["conf"][-1] = 0.249
    assert image_counts(pred, batch).tolist() == [[0, 0, 1], [0, 0, 1]]
    assert image_counts(_pred([], [], []), batch).tolist() == [[0, 0, 1], [0, 0, 0]]
    assert not image_counts(pred, _batch([], [])).any()


def test_baseline_map_is_identical_for_nontrivial_detections(tmp_path):
    model = YOLO("yolo11n.yaml").model
    shape = (64, 128)
    gt = torch.tensor([[0, 0, 2, 3], [10, 20, 14, 24]], dtype=torch.float32)
    batch = {
        "img": torch.zeros(1, 3, *shape), "batch_idx": torch.tensor([0, 0]), "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": ops.xyxy2xywh(gt) / torch.tensor([128, 64, 128, 64]), "ori_shape": [shape],
        "ratio_pad": [((1, 1), (0, 0))], "im_file": ["controlled.jpg"],
    }
    native, extra = [cls(save_dir=tmp_path, args={"plots": False}) for cls in (DetectionValidator, FixedBinValidator)]
    stats = []
    for validator in (native, extra):
        validator.device = torch.device("cpu")
        validator.data = {"val": "controlled"}
        validator.init_metrics(model)
        predictions = _pred([[0, 0, 2, 3], [10, 20, 13, 24], [5, 5, 7, 7]], [0.9, 0.24, 0.7], [0, 0, 0])
        validator.update_metrics([predictions], batch)
        validator.gather_stats()
        stats.append(validator.get_stats())
    assert 0 < stats[0]["metrics/mAP50-95(B)"] < 1
    assert all(v == stats[1][k] for k, v in stats[0].items())
    assert extra.metrics.fixed_bins == {
        "lt8": {"gt": 0, "tp": 0, "recall": None}, "8to16": {"gt": 1, "tp": 1, "recall": 1.0},
        "ge16": {"gt": 1, "tp": 0, "recall": 0.0},
    }
    assert math.isnan(stats[1]["metrics/R50_640_lt8(B)"])
    assert extra.metrics.results_dict["metrics/R50_640_8to16(B)"] == 1
    extra.init_metrics(model)
    assert not extra.metrics.counts.any()
    extra.args.val = False
    extra.update_metrics([predictions], batch)
    assert extra.get_stats().keys() == stats[0].keys()
    assert extra.metrics.fixed_bins["8to16"]["tp"] == 1
