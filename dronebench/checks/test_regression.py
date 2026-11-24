import argparse
import hashlib

import torch

from dronebench.config import from_id
from dronebench.model import build_model
from dronebench.smoke import LR, detections, exercise, labelled_batch, runtime, save_report, tensor_bytes
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def _flatten(value, name=""):
    if isinstance(value, (tuple, list, dict)):
        for key, item in (value.items() if isinstance(value, dict) else enumerate(value)):
            yield from _flatten(item, f"{name}/{key}")
    else:
        yield name, value


def _exact(actual, reference):
    a, b = dict(_flatten(actual)), dict(_flatten(reference))
    assert list(a) == list(b), "tensor names or order differ"
    digest, elements, tensors = hashlib.sha256(), 0, 0
    for name, t in a.items():
        expected = b[name]
        digest.update(name.encode() + b"\0")
        if t is None:
            assert expected is None, name
            digest.update(b"None\0")
            continue
        assert t.shape == expected.shape and t.dtype == expected.dtype and t.device == expected.device, name
        assert torch.isfinite(t).all(), name
        observed = tensor_bytes(t)
        assert observed == tensor_bytes(expected), f"first non-identical tensor: {name}"
        digest.update(f"{t.dtype}:{list(t.shape)}:".encode() + observed)
        tensors += 1
        elements += t.numel()
    return {"equal_bytes": True, "tensor_count": tensors, "none_count": len(a) - tensors,
            "elements": elements, "common_sha256": digest.hexdigest()}


def baseline_regression(imgsz=96, seed=17):
    torch.manual_seed(seed)
    reference = DetectionModel("yolo11n.yaml", nc=10, verbose=False)
    reference.args = get_cfg()
    torch.manual_seed(seed)
    actual = build_model(from_id("00000"), nc=10)
    rb, ab = labelled_batch(imgsz, seed), labelled_batch(imgsz, seed)
    assert type(actual) is type(reference) is DetectionModel and vars(actual.args) == vars(reference.args)
    rp, ap = dict(reference.named_parameters()), dict(actual.named_parameters())
    assert {k: p.requires_grad for k, p in rp.items()} == {k: p.requires_grad for k, p in ap.items()}
    assert all(ap[k].data_ptr() != p.data_ptr() for k, p in rp.items())
    checks = {"initial_state": _exact(actual.state_dict(), reference.state_dict()), "batch": _exact(ab, rb)}
    raw_ref, raw = reference.train()(rb["img"]), actual.train()(ab["img"])
    checks["train_raw"] = _exact(raw, raw_ref)
    rl, ri = reference.loss(rb, preds=raw_ref)
    al, ai = actual.loss(ab, preds=raw)
    checks["train_loss_and_items"] = _exact((al, ai), (rl, ri))
    al.sum().backward()
    rl.sum().backward()
    checks["parameter_grads"] = _exact({k: p.grad for k, p in ap.items()}, {k: p.grad for k, p in rp.items()})
    before = {k: p.detach().clone() for k, p in ap.items() if p.requires_grad}
    for model in (actual, reference):
        torch.optim.SGD(model.parameters(), lr=LR).step()
    updated = sum(not torch.equal(t, ap[k]) for k, t in before.items())
    assert updated > 0
    checks["post_sgd_state"] = _exact(actual.state_dict(), reference.state_dict())
    with torch.no_grad():
        ref_eval, actual_eval = reference.eval()(rb["img"]), actual.eval()(ab["img"])
        decoded, eval_raw = actual_eval
        checks["eval_raw"] = _exact(eval_raw, ref_eval[1])
        checks["eval_decoded"] = _exact(decoded, ref_eval[0])
        el, ei = actual.loss(ab, preds=actual_eval)
        checks["eval_loss_and_items"] = _exact((el, ei), reference.loss(rb, preds=ref_eval))
        rows = detections(decoded, len(ab["img"]), 10)
        checks["eval_nms"] = _exact(rows, detections(ref_eval[0], len(rb["img"]), 10))
        assert all(len(x) for x in rows), "expected a nonempty NMS exercise for each baseline image"
    return {
        "kind": "synthetic all-off regression against independently seeded pinned DetectionModel", "status": "passed",
        **runtime(ab, seed), "command": "python -m dronebench.checks.test_regression --imgsz "
        f"{imgsz} --seed {seed} --output experiment_records/baseline-regression.json",
        "reference": "DetectionModel('yolo11n.yaml', nc=10, verbose=False); model.args = get_cfg()",
        "strides_px": actual.stride.tolist(), "tal_min_px": actual.criterion.assigner.stride_val,
        "trainable_parameter_tensors": len(before), "updated_parameter_tensors": updated,
        "train_loss_items": {k: t.item() for k, t in ai.items()}, "eval_loss_items": {k: t.item() for k, t in ei.items()},
        "eval_decoded_shape": list(decoded.shape), "nms_counts": [len(x) for x in rows],
        "first_nms_xyxy_conf_class": [x[0].tolist() for x in rows], "exact_comparisons": checks,
    }


def test_all_off_matches_pinned_detection_model():
    with torch.random.fork_rng(devices=[]):
        baseline_regression()


def test_all_five_empty_tile_batch_vs_empty_spatial_selection():
    options = ({"esod": {"threshold": 1.0}}, {"dm_efs": {"iou_threshold": 0.0}, "querydet": {"threshold": 1.0}})
    with torch.random.fork_rng(devices=[]):
        empty_tiles, empty_keys = [exercise({**from_id("11111"), "options": o}, 96, 17)["eval"] for o in options]
    assert empty_tiles["decoded_shape"] == [2, 14, 0] and empty_tiles["route"]["esod_tiles_per_image"] == [0, 0]
    assert empty_keys["decoded_shape"][-1] > 0 and min(empty_keys["route"]["esod_tiles_per_image"]) > 0
    assert not any(any(row) for row in empty_keys["route"]["active_positions"])
    for result in (empty_tiles, empty_keys):
        assert result["nms_counts"] == [0, 0] and result["loss_items"]["foreground_loss"] > 0
        assert result["loss_items"]["box_loss"] == result["loss_items"]["ugs_loc_loss"] == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exact all-off CPU regression; no training metrics are computed.")
    p.add_argument("--imgsz", type=int, default=96)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output")
    args = p.parse_args()
    save_report(baseline_regression(args.imgsz, args.seed), args.output)
