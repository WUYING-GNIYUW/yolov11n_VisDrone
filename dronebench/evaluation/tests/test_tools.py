import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from dronebench.evaluation.smoke_data import build_smoke


def test_balanced_disjoint_rebuild_and_baseline_filter(tmp_path):
    archive = tmp_path / "fixture.zip"
    image = io.BytesIO()
    Image.new("RGB", (64, 32)).save(image, format="JPEG")
    with zipfile.ZipFile(archive, "w") as z:
        for sequence in range(4):
            for frame in range(2):
                stem = f"{sequence:07}_{frame:05}_d_0000001"
                z.writestr(f"VisDrone2019-DET-val/images/{stem}.jpg", image.getvalue())
                z.writestr(f"VisDrone2019-DET-val/annotations/{stem}.txt", "0,0,8,4,0,0,0,0\n8,4,16,8,1,2,2,1\n")
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    one, two = (build_smoke(archive, tmp_path / n, 2, 2, 17) for n in ("a", "b"))
    assert yaml.safe_load(one.read_text())["names"][9] == "motor"
    manifest = json.loads((one.parent / "manifest.json").read_text())
    assert manifest == json.loads((two.parent / "manifest.json").read_text())
    assert manifest["archive_sha256"] == before == hashlib.sha256(archive.read_bytes()).hexdigest()
    sequences = []
    for split in ("train", "val"):
        assert len(manifest["splits"][split]) == 2
        for record in manifest["splits"][split]:
            sequences.append(record["file"].split("_")[0])
            assert record["gt"] == 1
            label = Path(record["file"]).with_suffix(".txt")
            assert (one.parent / "labels" / split / label).read_text() == "1 0.250000 0.250000 0.250000 0.250000\n"
    assert len(set(sequences)) == 4
    with pytest.raises(FileExistsError):
        build_smoke(archive, one.parent, 2, 2)


def test_public_val_and_disk_to_detections_latency(tmp_path):
    from dronebench.evaluation.latency import benchmark
    from dronebench.evaluation.validator import FixedBinValidator
    from ultralytics import YOLO

    images, labels = tmp_path / "images" / "val", tmp_path / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name in ("b", "a"):
        Image.new("RGB", (320, 180), (10, 90, 140)).save(images / f"{name}.jpg")
        (labels / f"{name}.txt").write_text("0 0.500000 0.500000 0.012500 0.044444\n")
    data = tmp_path / "local.yaml"
    config = {"path": str(tmp_path), "train": "images/val", "val": "images/val", "names": ["box"]}
    data.write_text(yaml.safe_dump(config))
    weights = tmp_path / "untrained-engineering-fixture.pt"
    YOLO("yolo11n.yaml").save(weights)
    model = YOLO(weights)
    m = model.val(
        validator=FixedBinValidator, data=str(data), batch=1, imgsz=96, device="cpu", workers=0, max_det=500,
        plots=False, project=str(tmp_path), name="val",
    )
    assert [v["gt"] for v in m.fixed_bins.values()] == [0, 2, 0]
    assert m.fixed_bin_protocol["nms_max_det"] == 500 and m.fixed_bin_protocol["top_per_image"] == 300
    assert m.box.map == m.results_dict["metrics/mAP50-95(B)"]
    report = benchmark(weights, images, warmup=1, runs=3, imgsz=96, device="cpu")
    assert report["device"] == "cpu" and report["batch"] == 1 and report["input_hw"] == [96, 96]
    assert report["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = [
        {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(images.iterdir())
    ]
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    assert report["images_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert [Path(s["file"]).name for s in report["samples"]] == ["a.jpg", "b.jpg", "a.jpg"]
    times = [s["ms"] for s in report["samples"]]
    assert min(times) > 0
    assert [report["p50_ms"], report["p95_ms"]] == pytest.approx(np.percentile(times, [50, 95]))
    with pytest.raises(FileNotFoundError):
        benchmark(tmp_path / "absent.pt", images)
