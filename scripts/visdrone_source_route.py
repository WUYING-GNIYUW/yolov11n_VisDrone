# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reproduce the isolated P3 C3k2 experiment without importing another Ultralytics installation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C2f
from ultralytics.utils import YAML

OUT = ROOT / "artifacts/source_route"
WEIGHTS = Path("/root/home/2112406148/ultralytics/weights/yolo11n.pt")
DATA = ROOT / "ultralytics/cfg/datasets/VisDrone.yaml"
MODELS = {
    "baseline": "baseline",
    "separate": "source-separate",
    "detail": "source-route",
}


def build(variant):
    """Use identical random initialization seeds, then load the same original pretrained checkpoint."""
    torch.manual_seed(0)
    model = YOLO(str(ROOT / f"ultralytics/cfg/models/11/yolo11n-visdrone-{MODELS[variant]}.yaml"))
    model.load(str(WEIGHTS))
    return model


def real_batch(device):
    """Prepare two real training images with isotropic 640 letterboxing for loss/gradient checks only."""
    root = Path(YAML.load(DATA)["path"])
    images, labels, indices = [], [], []
    for index, path in enumerate(sorted((root / "images/train").glob("*.jpg"))[:2]):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        ratio = min(640 / w, 640 / h)
        nw, nh = round(w * ratio), round(h * ratio)
        left, top = round((640 - nw) / 2 - 0.1), round((640 - nh) / 2 - 0.1)
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh))
        images.append(torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))))
        rows = np.loadtxt(root / "labels/train" / f"{path.stem}.txt", ndmin=2)
        rows[:, 1] = (rows[:, 1] * w * ratio + left) / 640
        rows[:, 2] = (rows[:, 2] * h * ratio + top) / 640
        rows[:, 3] *= w * ratio / 640
        rows[:, 4] *= h * ratio / 640
        labels.append(torch.tensor(rows, dtype=torch.float32))
        indices.append(torch.full((len(rows),), index))
    targets = torch.cat(labels)
    return {
        "img": torch.stack(images).to(device).float() / 255,
        "batch_idx": torch.cat(indices).to(device),
        "cls": targets[:, :1].to(device),
        "bboxes": targets[:, 1:].to(device),
    }


def backward_check(model, device):
    """Check real losses and an optimizer update without altering benchmark or training initialization."""
    net = copy.deepcopy(model).to(device).train()
    net.args = get_cfg()
    batch = real_batch(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    loss, items = net(batch)
    loss.sum().backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    grouped_grad = net.model[16].cv1.conv.weight.grad
    assert grouped_grad[:32].abs().sum() > 0 and grouped_grad[32:].abs().sum() > 0
    before = net.model[16].cv1.conv.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, net.model[16].cv1.conv.weight)
    result = {
        "loss_items": {key: value.detach().cpu().tolist() for key, value in items.items()},
        "all_trainable_gradients_finite": True,
    }
    if device.type == "cuda":
        result["training_batch2_peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def benchmark(model, device):
    """Measure FP32 batch-one forward/decode only; report timing samples, not a deployment speed claim."""
    net = model.to(device).eval()
    image = torch.zeros(1, 3, 640, 640, device=device)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    samples = []
    with torch.inference_mode():
        for _ in range(20):
            net(image)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
        for _ in range(3):
            synchronize()
            start = time.perf_counter()
            for _ in range(50):
                net(image)
            synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 50)
        result = {
            "fp32_b1_ms": samples, "median_ms": statistics.median(samples),
            "cpu_threads_during_benchmark": torch.get_num_threads(),
        }
        if device.type == "cuda":
            result["forward_peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["forward_extra_peak_MiB"] = (
                torch.cuda.max_memory_allocated(device) - allocated_before
            ) / 2**20
    net.cpu()
    return result


def paired_benchmark(args):
    """Interleave fused models to reduce ordering bias under the shared GPU's changing workload."""
    device = torch.device(args.device)
    models = {name: build(name).model.to(device).eval().fuse(verbose=False) for name in MODELS}
    image = torch.zeros(1, 3, 640, 640, device=device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    samples = {name: [] for name in models}
    names = list(models)
    with torch.inference_mode():
        for net in models.values():
            for _ in range(20):
                net(image)
        for repeat in range(6):
            for name in names[repeat % 3 :] + names[: repeat % 3]:
                sync()
                start = time.perf_counter()
                for _ in range(50):
                    models[name](image)
                sync()
                samples[name].append((time.perf_counter() - start) * 1000 / 50)
    result = {
        "scope": "Conv-BN fused FP32 B1 640 forward+decode; no preprocess/NMS; 20 warmup; 6x50 interleaved",
        "shared_gpu": device.type == "cuda",
        "device": str(device),
        "samples_ms": samples,
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
    }
    (OUT / f"paired_fused_{device.type}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def verify(args):
    """Exercise actual models, pretrained transfer, source invariance, shapes, losses, fusion, and native export."""
    device = torch.device(args.device)
    models = {name: build(name).model.eval() for name in MODELS}
    image = torch.zeros(1, 3, 640, 640)
    stats = {}
    for name, model in models.items():
        # CPU profiler counts Conv/MatMul and supported elementwise ops; native get_flops silently returns 0
        # with this environment's THOP because its profile() does not support stride/min_cells arguments.
        with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
        ) as prof:
            prediction = model(image)[0]
        stats[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "profiler_GFLOPs_640": sum(x.flops for x in prof.key_averages()) / 1e9,
            "output": list(prediction.shape),
        }
        for h, w in [(384, 640), (672, 672)]:
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, h, w))[0]
            assert output.shape == (1, 14, h * w // 64 + h * w // 256 + h * w // 1024)
        block = model.model[16]
        x = torch.randn(1, 256, 16, 16)
        with torch.inference_mode():
            torch.testing.assert_close(block(x), block.forward_split(x), rtol=0, atol=0)
            if name == "baseline":
                torch.testing.assert_close(block(x), C2f.forward(block, x), rtol=0, atol=0)
            else:
                inputs = []
                handle = block.cv2.register_forward_pre_hook(lambda m, a: inputs.append(a[0].clone()))
                block(x)
                x[:, :128] += 1
                block(x)
                handle.remove()
                torch.testing.assert_close(inputs[0][:, :32], inputs[1][:, :32], rtol=0, atol=0)
                assert not torch.equal(inputs[0][:, 32:64], inputs[1][:, 32:64])
        stats[name]["benchmark"] = benchmark(model, device)
    assert stats["baseline"]["parameters"] - stats["detail"]["parameters"] == 8192
    source = YOLO(str(WEIGHTS)).model.state_dict()["model.16.cv1.conv.weight"]
    expected = torch.cat((source[:32, :128], source[32:, 128:]), 0).float()
    torch.testing.assert_close(models["detail"].model[16].cv1.conv.weight, expected, rtol=0, atol=0)
    clone = build("detail").model
    clone.load(models["detail"], verbose=False)
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in models["detail"].state_dict().items())
    net = models["detail"]
    with torch.inference_mode():
        reference = net(image)[0]
        fused = copy.deepcopy(net).fuse(verbose=False)
        torch.testing.assert_close(reference, fused(image)[0], rtol=1e-4, atol=1e-3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stats["detail"]["backward"] = backward_check(net, device)
    stats["checks"] = {
        "default_C2f_equivalence": True,
        "protected_detail_bypass_invariant_to_context": True,
        "dense_to_grouped_diagonal_transfer": True,
        "variant_to_variant_reload": True,
        "forward_split": True,
        "conv_bn_fuse": True,
        "rectangular_and_larger_inputs": True,
    }
    if args.export:
        # Use the native exporter and write only under this worktree's artifact directory.
        wrapper = build("detail")
        wrapper.model.pt_path = str(OUT / "source-route.pt")
        exported = wrapper.export(format="torchscript", imgsz=640, device="cpu", batch=1, nms=False)
        restored = torch.jit.load(str(exported)).eval()
        with torch.inference_mode():
            expected = copy.deepcopy(wrapper.model).eval().fuse(verbose=False)(image)[0]
            torch.testing.assert_close(restored(image), expected, rtol=1e-4, atol=1e-3)
        stats["checks"]["native_torchscript_export_reload"] = str(exported)
    stats["environment"] = {
        "ultralytics": ultralytics.__version__, "import": ultralytics.__file__,
        "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device), "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "timing_scope": "FP32 B1 640; 20 warmup; 3x50; forward+decode; excludes preprocess/NMS; shared GPU",
    }
    path = OUT / f"verification_{device.type}.json"
    path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def train(args):
    """Launch one controlled experiment; the caller chooses the seed and model, never an automatic sweep."""
    recipe = YAML.load(ROOT / "scripts/visdrone_source_route_train.yaml")
    recipe.update(
        data=str(DATA), device=args.device, seed=args.seed, batch=args.batch, epochs=args.epochs,
        project=str(ROOT / "runs/visdrone_source_route"),
        name=f"{args.variant}_b{args.batch}_{args.epochs}e_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    audit = json.loads((OUT / "data_audit.json").read_text())
    data_root = Path(YAML.load(DATA)["path"])
    for split in ("train", "val", "test"):
        digest = hashlib.sha256()
        images = sorted((data_root / "images" / split).glob("*.jpg"))
        for image in images:
            label = data_root / "labels" / split / f"{image.stem}.txt"
            digest.update(label.name.encode() + b"\0" + label.read_bytes())
        if len(images) != audit[split]["images"] or digest.hexdigest() != audit[split]["sha256"]:
            raise RuntimeError(f"{split} data changed since this experiment's data_audit.json")
    # Match each experiment's random initialization to its requested seed, including unmatched class logits.
    torch.manual_seed(args.seed)
    model = YOLO(str(ROOT / f"ultralytics/cfg/models/11/yolo11n-visdrone-{MODELS[args.variant]}.yaml"))
    # Let the real trainer load the original 80-class checkpoint after setting dataset names, so class-name remapping
    # is identical for all three variants. Loading an intermediate 10-class model here would change that procedure.
    manifest = {
        "recipe": recipe, "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
        "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "model_yaml": model.model.yaml, "ultralytics_import": ultralytics.__file__,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "ultralytics/nn/modules/block.py", ROOT / "ultralytics/nn/tasks.py", DATA,
                Path(__file__), ROOT / "scripts/visdrone_source_route_train.yaml",
            )
        },
        "data_audit": audit,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    manifest_dir = ROOT / "logs/visdrone_source_route"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{recipe['name']}.json").write_text(json.dumps(manifest, indent=2))
    model.train(**recipe)


def validate(args):
    """Request native APsmall only if the existing optional evaluator is installed; never silently install it."""
    if importlib.util.find_spec("faster_coco_eval") is None:
        raise SystemExit("APsmall 未评估：环境缺少已有验证器依赖 faster-coco-eval；不会将占位 0 当作结果。")
    YOLO(args.checkpoint).val(
        data=str(DATA), split="val", imgsz=640, batch=20, device=args.device,
        conf=0.001, iou=0.7, max_det=1000, augment=False, save_json=True,
        project=str(ROOT / "runs/visdrone_source_route_eval"),
    )


def ddp_check(args):
    """Run one real-image forward/backward/update on each GPU and verify the grouped gradients synchronize."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    model = build("detail").model.to(device).train()
    model.args = get_cfg()
    wrapped = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    batch = real_batch(device)
    if rank % 2:
        batch["img"] = batch["img"].flip(-1)
        batch["bboxes"][:, 0] = 1 - batch["bboxes"][:, 0]
    loss, items = wrapped(batch)
    loss.sum().backward()
    gradient = model.model[16].cv1.conv.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    gathered = [torch.zeros_like(gradient) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, gradient)
    assert all(torch.equal(gradient, other) for other in gathered)
    optimizer.step()
    if rank == 0:
        result = {
            "world_size": dist.get_world_size(), "batch_per_rank": 2,
            "different_augmentation_per_rank": True,
            "finite_detection_loss": bool(torch.isfinite(loss).all()),
            "grouped_gradient_equal_across_ranks": True,
            "optimizer_step": True,
            "loss_items_rank0": {key: value.detach().cpu().tolist() for key, value in items.items()},
        }
        (OUT / "ddp_smoke.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "benchmark", "train", "val", "ddp-check"])
    parser.add_argument("--variant", choices=MODELS, default="detail")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.action == "val" and not args.checkpoint:
        parser.error("val requires --checkpoint")
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    {"verify": verify, "benchmark": paired_benchmark, "train": train, "val": validate, "ddp-check": ddp_check}[
        args.action
    ](args)
