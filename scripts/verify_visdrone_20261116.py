# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Check the numerical contracts and real detection training path of one VisDrone experiment."""
# ruff: noqa: E402 -- Import the selected worktree before any installed ultralytics package.

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import ultralytics
from ultralytics.nn.modules import C2PSA, Concat
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss


def finite(tensor):
    """Assert that a real output or gradient contains no NaN/Inf."""
    assert tensor is not None and torch.isfinite(tensor).all()


def check_operator(variant):
    """Exercise active operators, including nonzero gains and degenerate inputs."""
    if variant == "capacity":
        logits = torch.randn(2, 2, 17, 17, requires_grad=True)
        with torch.no_grad():
            logits[..., 0] += 5
        weights = logits.softmax(-1)
        capacity = torch.linspace(0.6, 1.4, 17).reshape(1, 1, 1, 17).requires_grad_()
        bounded, retained = Attention._limit_attention(weights, capacity)
        assert torch.all(bounded.sum(-2, keepdim=True) <= capacity + 2e-6)
        torch.testing.assert_close(bounded.sum(-1) + retained, torch.ones_like(retained))
        assert retained.min() >= 0
        values = torch.randn(2, 2, 5, 17)
        context = values @ bounded.transpose(-2, -1) + values * retained.unsqueeze(-2)
        context.square().mean().backward()
        finite(logits.grad)
        finite(capacity.grad)
        assert capacity.grad.abs().sum() > 0
        assert logits.grad.abs().sum() > 0
        uniform = torch.full((1, 1, 17, 17), 1 / 17)
        same, residual = Attention._limit_attention(uniform, torch.ones(1, 1, 1, 17))
        torch.testing.assert_close(same, uniform)
        assert residual.abs().max() < 1e-6
        one, residual_one = Attention._limit_attention(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
        assert one.item() == 1 and residual_one.item() == 0
        module = C2PSA(128, 128, n=2, capacity_attention=True).eval()
        for shape in ((2, 128, 7, 9), (1, 128, 1, 1)):
            x = torch.randn(shape, requires_grad=True)
            output = module(x)
            assert output.shape == x.shape
            output.square().mean().backward()
            finite(x.grad)
        # Reconstruct the original C2PSA path using the same tensors and compare the disabled option.
        module.capacity_attention = False
        x = torch.randn(1, 128, 7, 9)
        a, b = module.cv1(x).chunk(2, 1)
        expected = module.cv2(torch.cat((a, module.m(b)), 1))
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
        return {"capacity_column_bound": True, "row_mass": True, "retained_mass_max": retained.max().item()}
    module = Concat(1, conservative=True)
    semantic = torch.randn(2, 5, 10, 14, requires_grad=True)
    lateral = torch.randn(2, 7, 10, 14, requires_grad=True)
    original = torch.cat((semantic, lateral), 1)
    initial = module([semantic, lateral])
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    (initial[:, :5] * torch.randn_like(semantic)).sum().backward()
    finite(module.transport_gain.grad)
    gain_grad = module.transport_gain.grad.item()
    assert abs(gain_grad) > 1e-7
    module.zero_grad(set_to_none=True)
    semantic.grad = lateral.grad = None
    with torch.no_grad():
        module.transport_gain.fill_(0.8)
    output = module([semantic, lateral])
    error = (F.avg_pool2d(output[:, :5], 2) - F.avg_pool2d(semantic, 2)).abs().max().item()
    assert error < 2e-6
    assert (output[:, :5] - semantic).abs().max() > 1e-5
    torch.testing.assert_close(output[:, 5:], lateral, rtol=0, atol=0)
    output[:, :5].square().mean().backward()
    finite(semantic.grad)
    finite(lateral.grad)
    assert lateral.grad.abs().sum() > 0
    constant = torch.ones(1, 5, 10, 14)
    torch.testing.assert_close(module([constant, lateral[:1]])[:, :5], constant, atol=1e-6, rtol=0)
    finite(module([torch.ones(1, 5, 2, 2), torch.zeros(1, 7, 2, 2)]))
    try:
        module([torch.randn(1, 5, 7, 9), torch.randn(1, 7, 7, 9)])
    except ValueError:
        pass
    else:
        raise AssertionError("Odd spatial dimensions should be rejected by the opt-in Concat")
    return {"nonzero_gain_mean_error": error, "initial_gain_gradient": gain_grad, "constant_preservation": True}


def main():
    """Validate both the low-level invariant and the unchanged model interfaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("capacity", "conservative"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    report = {
        "variant": args.variant,
        "torch": torch.__version__,
        "package": ultralytics.__file__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "operator": check_operator(args.variant),
    }
    cfg = ROOT / "ultralytics/cfg/models/11"
    torch.manual_seed(17)
    baseline = DetectionModel(cfg / "yolo11n-visdrone-baseline.yaml", verbose=False).eval()
    model = DetectionModel(cfg / f"yolo11n-visdrone-{args.variant}.yaml", verbose=False).eval()
    loaded = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert loaded.missing_keys == ([] if args.variant == "capacity" else ["model.15.transport_gain"])
    report["baseline_parameters"] = sum(p.numel() for p in baseline.parameters())
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["baseline_state_tensors_transferred"] = len(baseline.state_dict())
    report["new_state_keys"] = loaded.missing_keys

    # Exercise the real 80->10 class transfer owner with synthetic source weights; do not claim trained-weight accuracy.
    source_cfg = copy.deepcopy(baseline.yaml)
    source_cfg["nc"] = 80
    source = DetectionModel(source_cfg, verbose=False)
    transferred = copy.deepcopy(model)
    transferred.load(source, verbose=False)
    transferred_state = transferred.state_dict()
    source_backbone = {k: v for k, v in source.state_dict().items() if not k.startswith("model.23.")}
    assert all(torch.equal(v, transferred_state[k]) for k, v in source_backbone.items())
    report["synthetic_80_to_10_backbone_transfer"] = len(source_backbone)
    report["parameters_after_80_class_transfer"] = sum(p.numel() for p in transferred.parameters())
    del source, transferred

    shapes = {}
    handles = [
        model.model[i].register_forward_hook(
            lambda module, inputs, output: shapes.update({str(module.i): list(output.shape)})
        )
        for i in (10, 15, 16, 19, 22)
    ]
    image = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        prediction = model(image)[0]
        finite(prediction)
        assert prediction.shape == (1, 14, 8400)
        if args.variant == "conservative":
            torch.testing.assert_close(prediction, baseline(image)[0], rtol=0, atol=0)
        for handle in handles:
            handle.remove()
        assert model(torch.randn(1, 3, 384, 640))[0].shape == (1, 14, 5040)
        # Make the Concat path active for the remaining tests.
        if args.variant == "conservative":
            model.model[15].transport_gain.fill_(0.25)
        benchmark = {}
        for name, candidate in (("baseline", baseline), ("variant", model)):
            for _ in range(2):
                candidate(image)
            samples = []
            for _ in range(9):
                started = time.perf_counter()
                candidate(image)
                samples.append(1000 * (time.perf_counter() - started))
            benchmark[name] = statistics.median(samples)
        report["cpu_640_forward_median_ms"] = benchmark
        # CPU profiler counts supported arithmetic operations, not every memory/pooling/softmax operation.
        flop_counts = {}
        for name, candidate in (("baseline", baseline), ("variant", model)):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                candidate(image)
            flop_counts[name] = sum(event.flops for event in prof.key_averages())
        report["profiler_supported_flops_640"] = flop_counts
    report["shapes_640"] = shapes

    model.train()
    model.args = copy.deepcopy(DEFAULT_CFG)
    criterion = v8DetectionLoss(model)
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 0, 1, 1]),
        "cls": torch.tensor([[0.0], [3.0], [1.0], [9.0]]),
        "bboxes": torch.tensor(
            [[0.3, 0.4, 0.1, 0.12], [0.7, 0.6, 0.2, 0.18], [0.4, 0.5, 0.15, 0.1], [0.65, 0.7, 0.2, 0.2]]
        ),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    losses, _ = criterion(model(batch["img"]), batch)
    losses.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    for gradient in gradients:
        finite(gradient)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    if args.variant == "conservative":
        report["detection_loss_gain_gradient"] = model.model[15].transport_gain.grad.item()
        assert abs(report["detection_loss_gain_gradient"]) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    report["synthetic_detection_loss"] = losses.detach().tolist()
    report["finite_parameter_gradients"] = len(gradients)

    model.eval()
    sample = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        expected = model(sample)[0]
        fused = copy.deepcopy(model).fuse(verbose=False)
        fused_output = fused(sample)[0]
        torch.testing.assert_close(fused_output, expected, rtol=1e-4, atol=1e-4)
        report["fused_max_abs_error"] = (fused_output - expected).abs().max().item()
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        torch.testing.assert_close(restored(sample)[0], expected, rtol=0, atol=0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            finite(model(sample)[0])
        for parameter in model.parameters():
            finite(parameter)
    report["checkpoint_roundtrip"] = True
    report["cpu_bfloat16_forward"] = True
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
