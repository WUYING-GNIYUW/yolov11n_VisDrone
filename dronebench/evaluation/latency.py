from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO, __version__
from ultralytics.utils.torch_utils import get_cpu_info

from .digests import file_sha256

IMAGE_TYPES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(
    weights: str | Path,
    images: str | Path,
    *,
    device: str = "cpu",
    imgsz: int = 640,
    warmup: int = 10,
    runs: int = 100,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
) -> dict:
    """Measure single-image wall time from opening a local file to decoded, post-NMS boxes in host memory."""
    if min(warmup, runs, imgsz, max_det) < 1 or not (0 <= conf <= 1 and 0 <= iou <= 1):
        raise ValueError("warmup, runs, imgsz and max_det must be positive; conf and iou must be in [0,1]")
    device = str(device)
    if not re.fullmatch(r"cpu|mps|cuda|(?:cuda:)?\d+", device):
        raise ValueError("Specify one device: cpu, mps, cuda, or a CUDA index such as 0 or cuda:1")
    weights = Path(weights).expanduser().resolve(strict=True)
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise ValueError("Provide a trusted local detection checkpoint (.pt), not a model name or export")
    images = Path(images).expanduser().resolve(strict=True)
    candidates = images.iterdir() if images.is_dir() else [images]
    paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not paths:
        raise ValueError(f"No supported local images in {images}")
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError(f"Expected detection weights, got task {model.task!r}")
    options = dict(
        device=device, imgsz=imgsz, batch=1, rect=False, quantize=32, conf=conf, iou=iou, max_det=max_det,
        nms=None, agnostic_nms=False, classes=None, augment=False, save=False, save_txt=False, verbose=False,
    )

    def predict(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        result, = model.predict(source=image, **options)
        return len(result.boxes.data.detach().cpu().numpy())

    for i in range(warmup):
        predict(paths[i % len(paths)])
    actual = model.predictor.device
    expected_type = device if device in {"cpu", "mps"} else "cuda"
    index = device.rsplit(":", 1)[-1]
    if actual.type != expected_type or (index.isdigit() and actual.index != int(index)):
        raise ValueError(f"Requested {device} but predictor is on {actual}; do not compare mixed-device runs")
    samples = []
    for i in range(runs):
        path = paths[i % len(paths)]
        _sync(actual)
        start = time.perf_counter_ns()
        detections = predict(path)
        _sync(actual)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        samples.append({"file": str(path), "ms": elapsed, "detections": detections})
    p50, p95 = np.percentile([s["ms"] for s in samples], [50, 95])
    end2end = bool(model.predictor.model.end2end)
    image_manifest = [{"file": p.name, "sha256": file_sha256(p)} for p in paths[:runs]]
    encoded = json.dumps(image_manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    return {
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "images_sha256": hashlib.sha256(encoded).hexdigest(),
        "image_manifest": image_manifest,
        "device": str(actual),
        "requested_device": device,
        "hardware": torch.cuda.get_device_name(actual) if actual.type == "cuda" else get_cpu_info(),
        "torch": str(torch.__version__),
        "ultralytics": __version__,
        "opencv": cv2.__version__,
        "cpu_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "batch": 1,
        "precision": "fp32",
        "imgsz": imgsz,
        "input_hw": list(model.predictor.imgsz),
        "rect": False,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "end2end_head": end2end,
        "postprocess": "end-to-end head selection" if end2end else "class-aware YOLO detection NMS",
        "boundary": "local file read/decode -> preprocessing, transfer, predict/postprocess -> original boxes on CPU",
        "warmup": warmup,
        "runs": runs,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch=1 local-image-to-host-box latency for a detection checkpoint.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path, help="One image or directory of images (nonrecursive)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, help="Also write the report as JSON")
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = json.dumps(benchmark(**args), indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
