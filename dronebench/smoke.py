import hashlib
import json
import os
import platform
import shlex
import sys
from pathlib import Path

import torch
import torchvision

import ultralytics
from ultralytics.utils.nms import non_max_suppression

from .config import BASE, bits_of, combinations, from_id, normalize, read_spec
from .model import build_model
from .records import source_hash

LR = 0.0001
NMS = {"conf_thres": 0.00001, "iou_thres": 0.7, "max_det": 32, "max_time_img": 60.0}


def labelled_batch(imgsz, seed):
    if not __debug__:
        raise RuntimeError("QA assertions require Python without -O")
    if imgsz < 64 or imgsz % 32:
        raise ValueError("imgsz must be at least 64 and divisible by 32")
    return {
        "img": torch.rand(2, 3, imgsz, imgsz, generator=torch.Generator().manual_seed(seed + 1)),
        "batch_idx": torch.tensor([0, 0, 0, 1, 1]),
        "cls": torch.tensor([[1.0], [4.0], [7.0], [2.0], [6.0]]),
        "bboxes": torch.tensor(
            [[0.18, 0.21, 0.04, 0.08], [0.71, 0.68, 0.36, 0.32], [0.48, 0.52, 0.12, 0.21],
             [0.77, 0.19, 0.06, 0.04], [0.36, 0.58, 0.43, 0.46]]
        ),
    }


def tensor_bytes(t):
    return t.detach().cpu().contiguous().numpy().tobytes()


def runtime(batch, seed):
    labels = torch.cat((batch["batch_idx"][:, None], batch["cls"], batch["bboxes"]), 1)
    return {
        "base_commit": BASE, "experiment_source_sha256": source_hash(), "device": "cpu", "dtype": "float32",
        "python": platform.python_version(), "python_executable": sys.executable, "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "ultralytics": ultralytics.__version__,
        "torch_threads": torch.get_num_threads(), "seed": seed, "optimizer": {"name": "SGD", "lr": LR, "steps": 1},
        "batch": {
            "source": "synthetic uniform [0,1); CPU image generator seeded with seed + 1",
            "shape": list(batch["img"].shape), "image_sha256": hashlib.sha256(tensor_bytes(batch["img"])).hexdigest(),
            "labels_batch_class_xywhn": labels.tolist(),
        },
        "nms": NMS, "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "YOLO_OFFLINE")},
    }


def detections(decoded, batch_size, nc):
    assert decoded.ndim == 3 and decoded.shape[:2] == (batch_size, 4 + nc), decoded.shape
    assert torch.isfinite(decoded).all() and ((decoded[:, 4:] >= 0) & (decoded[:, 4:] <= 1)).all()
    rows = non_max_suppression(decoded, nc=nc, **NMS)
    assert len(rows) == batch_size
    for row in rows:
        assert row.ndim == 2 and row.shape[1] == 6 and torch.isfinite(row).all()
        assert (row[:, 2:4] > row[:, :2]).all() and (row[:, 4] > NMS["conf_thres"]).all()
        assert ((row[:, 5] >= 0) & (row[:, 5] < nc) & (row[:, 5] == row[:, 5].long())).all()
    return rows


def save_report(report, output):
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(text)


def _value(x):
    if isinstance(x, dict):
        return {k: _value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_value(v) for v in x]
    if isinstance(x, torch.Tensor):
        assert torch.isfinite(x).all(), "nonfinite diagnostic"
        return x.detach().tolist()
    return x


def _routing(model, raw, batch):
    img, head = batch["img"], model.model[-1]
    b = len(img)
    assert raw.get("image_count", b) == b
    h, w = raw.get("hw", img.shape[-2:])
    es = raw.get("esod_state")
    route = {"local_hw": [h, w], "esod_tiles_per_image": None}
    rows = b
    if es is not None:
        layout = es["layout"]
        offsets = layout.offsets
        assert layout.batch_size == b and ((offsets[:, 0] >= 0) & (offsets[:, 0] < b)).all()
        assert model.training or es["target"] is None, "eval tile selection used a target"
        assert torch.isfinite(es["logits"]).all()
        rows = len(offsets)
        route.update(esod_tiles_per_image=offsets[:, 0].bincount(minlength=b).tolist(),
                     offsets_image_xyxy_p2=offsets.tolist(), esod_target_in_route=es["target"] is not None)
    feats = raw["feats"]
    assert bool(feats) == bool(rows), "deep path and selected tile count disagree"
    route["local_rows"] = rows
    if not feats:
        return route
    assert len(feats) == head.nl
    grids = [f.shape[-2:] for f in feats]
    steps = head.stride.tolist()
    assert all(f.shape[0] == rows and fh * s == h and fw * s == w for f, (fh, fw), s in zip(feats, grids, steps))
    lengths = [fh * fw for fh, fw in grids]
    n = sum(lengths)
    assert raw["boxes"].shape == (rows, 4 * head.reg_max, n) and raw["scores"].shape == (rows, head.nc, n)
    assert all(torch.isfinite(v).all() for v in (*feats, raw["scores"], raw["boxes"]))
    gates = raw["level_mask"] if "level_mask" in raw else torch.ones(rows, head.nl, dtype=torch.bool)
    active = raw["active"] if "active" in raw else torch.ones(rows, n, dtype=torch.bool)
    assert gates.shape == (rows, head.nl) and active.shape == (rows, n) and gates.dtype == active.dtype == torch.bool
    by_level = active.split(lengths, dim=1)
    counts = torch.stack([part.sum(1) for part in by_level], dim=1)
    assert not counts[~gates].any()
    if model.training:
        assert gates.all() and active.all(), "expected dense detection supervision in training"
    route.update(grids_hw=[list(hw) for hw in grids], head_levels=gates.tolist(), active_positions=counts.tolist())
    dm = raw.get("dm_state")
    if dm is not None:
        keys = ("size_range", "neck_requested", "neck_executed", "head_requested")
        route["dm"] = {key: _value(dm[key]) for key in keys}
        if dm["logits"] is not None:
            assert torch.isfinite(dm["logits"]).all()
    q = raw.get("query_state")
    if q is not None:
        route["query_mode"] = q["mode"]
        if model.training:
            assert q["mode"] == "dense" and all(torch.isfinite(v).all() for v in q["logits"])
        else:
            assert q["mode"] == "sparse" and len(q["levels"]) == head.nl
            route["query_positions_by_level"] = [{k: len(v) for k, v in level.items()} for level in q["levels"]]
            for level, part, (fh, fw) in zip(q["levels"], by_level, grids):
                mask = torch.zeros((rows, fh, fw), dtype=torch.bool)
                mask[tuple(level["det_keys"].T)] = True
                assert torch.equal(mask.flatten(1), part), "selected detection keys disagree with active mask"
    return route


def _loss_record(model, raw, batch, loss, items):
    assert list(items) == list(model.criterion.loss_names) and torch.isfinite(loss).all()
    torch.testing.assert_close(loss.detach(), torch.stack(list(items.values())) * len(batch["img"]), atol=0, rtol=0)
    diag = getattr(model, "last_diagnostics", {})
    if getattr(model, "rfla", None) is not None and raw["feats"]:
        assert "positive_per_gt" in diag and torch.equal(diag["positive_per_gt"].sum(1), diag["positive_per_image"])
        if "ugs_outside_fg" in diag:
            assert diag["ugs_outside_fg"] == diag["outside_gt_per_image"].sum(), "signed UGS and RFLA disagree"
    if not model.training:
        assert not loss.requires_grad and not diag.get("ugs_refined", False)
    routed = {"active", "level_mask", "patch_offsets", "dm"}
    return {"loss_items": _value(items), "route": _routing(model, raw, batch),
            "diagnostics": {k: _value(v) for k, v in diag.items() if k not in routed}}


def _sgd_step(model, loss):
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.SGD(params.values(), lr=LR)
    loss.sum().backward()
    before = {k: p.detach().clone() for k, p in params.items()}
    optimizer.step()
    groups = {}
    for k, p in params.items():
        group = "head" if k.startswith(f"model.{len(model.model) - 1}.") else k.split(".", 1)[0]
        stats = groups.setdefault(group, {"trainable": 0, "with_grad": 0, "nonzero_grad": 0, "updated": 0})
        stats["trainable"] += 1
        assert torch.isfinite(p).all(), f"nonfinite parameter: {k}"
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"nonfinite gradient: {k}"
            stats["with_grad"] += 1
            stats["nonzero_grad"] += int(p.grad.any())
        stats["updated"] += int(not torch.equal(before[k], p))
    assert any(g["nonzero_grad"] and g["updated"] for g in groups.values()), "SGD updated no trainable tensors"
    return groups


def _assigner(model, spec):
    if spec["rfla"]:
        a = model.rfla
        return {"type": type(a).__name__, "radii_px": list(a.radii_px), "metric": a.metric,
                "topk": a.topk, "beta": a.beta}
    a = model.criterion.assigner
    floor = a.stride_val
    assert floor == (model.stride[1].item() if spec["tal_min"] == "auto" else spec["tal_min"])
    return {"type": type(a).__name__, "min_px": floor, "policy": spec["tal_min"], "topk": a.topk}


def exercise(spec, imgsz, seed):
    torch.manual_seed(seed)
    model = build_model(spec, nc=10).train()
    batch = labelled_batch(imgsz, seed)
    img = batch["img"]
    raw = model._raw(img, batch) if hasattr(model, "experiment") else model(img)
    loss, items = model.loss(batch, preds=raw)
    train = _loss_record(model, raw, batch, loss, items)
    update = _sgd_step(model, loss)
    assert not raw["feats"] or update["head"]["updated"], "nonempty detection route made no head update"
    del raw, loss, items
    buffers = {k: t.clone() for k, t in model.named_buffers()}
    with torch.no_grad():
        decoded, raw = model.eval()(img)
        loss, items = model.loss(batch, preds=(decoded, raw))
        ev = _loss_record(model, raw, batch, loss, items)
        rows = detections(decoded, len(img), model.model[-1].nc)
        es = raw.get("esod_state")
        if es is not None:
            for i, count in enumerate(ev["route"]["esod_tiles_per_image"]):
                assert count or not decoded[i, 4:].any(), "detections for an image with zero tiles"
        elif "active" in raw:
            assert not decoded[:, 4:].transpose(1, 2)[~raw["active"]].any(), "scores on excluded positions"
        for name, value in model.named_buffers():
            assert torch.equal(value, buffers[name]), f"eval mutated buffer: {name}"
        ev.update(decoded_shape=list(decoded.shape), decoded_sha256=hashlib.sha256(tensor_bytes(decoded)).hexdigest(),
                  nms_counts=[len(x) for x in rows],
                  first_nms_xyxy_conf_class=[x[0].tolist() if len(x) else None for x in rows])
    return {"id": bits_of(spec), "config": spec, "status": "passed", "strides_px": model.model[-1].stride.tolist(),
            "assigner": _assigner(model, spec), "train": train, "sgd_parameter_tensors": update, "eval": ev}


def controls():
    yield "P2 with TAL floor fixed at 16 input pixels", normalize({"dm_efs": True, "tal_min": 16})
    yield "Warm codebook and selective spatial queries on tiles", normalize({
        **from_id("11111"), "options": {"dm_efs": {"iou_threshold": 0.0}, "querydet": {"threshold": 0.5}}
    })


def smoke(a):
    specs = combinations() if a.all else [read_spec(a.id, a.config)]
    batch = labelled_batch(a.imgsz, a.seed)
    report = {"kind": "synthetic tensor integration QA, not training or validation metrics", **runtime(batch, a.seed),
              "command": shlex.join([sys.executable, "-m", "dronebench", *sys.argv[1:]]), "cases": [], "controls": []}
    for spec in specs:
        print(f"Tensor smoke: {bits_of(spec)}", file=sys.stderr, flush=True)
        report["cases"].append(exercise(spec, a.imgsz, a.seed))
    if a.all:
        for name, spec in controls():
            print(f"Tensor control: {name}", file=sys.stderr, flush=True)
            report["controls"].append({"name": name, **exercise(spec, a.imgsz, a.seed)})
    report["status"] = "passed"
    save_report(report, a.output)
    return report
