from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image

from .digests import file_sha256

BASE_CONFIG = Path(__file__).resolve().parents[2] / "ultralytics/cfg/datasets/VisDrone.yaml"
SOURCE = PurePosixPath("VisDrone2019-DET-val")


def convert_labels(text, size):
    dw, dh = (1 / length for length in size)
    lines = []
    for line in text.strip().splitlines():
        row = line.split(",")
        if row[4] != "0":
            x, y, w, h = map(int, row[:4])
            lines.append(f"{int(row[5]) - 1} {(x + w / 2) * dw:.6f} {(y + h / 2) * dh:.6f} {w * dw:.6f} {h * dh:.6f}\n")
    return "".join(lines)


def build_smoke(archive: str | Path, out: str | Path, train_count: int = 4, val_count: int = 2, seed: int = 17) -> Path:
    archive = Path(archive).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    if min(train_count, val_count) < 1:
        raise ValueError("Both smoke splits need at least one image")
    with zipfile.ZipFile(archive, "r") as source:
        names = [PurePosixPath(n) for n in source.namelist() if n.lower().endswith(".jpg")]
        names = [n for n in names if n.parent == SOURCE / "images"]
        names.sort(key=lambda n: hashlib.sha256(f"{seed}:{n.name}".encode()).digest())
        selected = []
        sequences = set()
        for name in names:
            sequence = name.stem.split("_", 1)[0]
            if sequence not in sequences:
                sequences.add(sequence)
                selected.append(name)
            if len(selected) == train_count + val_count:
                break
        if len(selected) != train_count + val_count:
            raise ValueError(f"Not enough distinct image sequence prefixes in {archive}")
        out.mkdir(parents=True, exist_ok=False)
        splits = {}
        for split, members in (("train", selected[:train_count]), ("val", selected[train_count:])):
            images, labels = out / "images" / split, out / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            splits[split] = []
            for member in members:
                data = source.read(str(member))
                with Image.open(io.BytesIO(data)) as image:
                    size = image.size
                txt = member.with_suffix(".txt").name
                raw = source.read(str(SOURCE / "annotations" / txt)).decode("utf-8")
                converted = convert_labels(raw, size)
                (images / member.name).write_bytes(data)
                (labels / txt).write_text(converted, encoding="utf-8")
                splits[split].append({"file": member.name, "size_wh": size, "gt": len(converted.splitlines())})
    manifest = {
        "purpose": "engineering smoke only; both splits come from official val",
        "archive_sha256": file_sha256(archive),
        "seed": seed,
        "splits": splits,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    names = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))["names"]
    config = {"path": str(out), "train": "images/train", "val": "images/val", "names": names}
    path = out / "smoke.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Build two engineering-only splits from a local VisDrone val ZIP.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New directory; existing paths are never overwritten")
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(build_smoke(args.zip, args.out, args.train_count, args.val_count, args.seed))


if __name__ == "__main__":
    main()
