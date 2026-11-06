# YOLO11n-VisDrone module variants

## Audit and design boundary

The verified base is `origin/main` (`55989f76c`, remote HEAD `main`). The primary checkout had pre-existing tracked edits and untracked experiment artifacts; they were not modified, reset, stashed, or copied into these worktrees. The remote audit fetched 261 remote refs. Existing local VisDrone worktrees covered SPPF residual gating, P3 source routing, anti-aliasing, and C2PSA local centering; these variants use separate owners and do not reuse those implementations.

Relevant literature reviewed in the audit includes QueryDet (CVPR 2022, DOI `10.1109/CVPR52688.2022.01330`), RFAG-YOLO (Sensors 2025, DOI `10.3390/s25072193`), YOLO-MARS (Sensors 2025, DOI `10.3390/s25082534`), SCConv (CVPR 2023), *Feature disentanglement in one-stage object detection* (Pattern Recognition 2024, DOI `10.1016/j.patcog.2023.109878`), *Exploring Orthogonality in Open World Object Detection* (CVPR 2024, DOI `10.1109/CVPR52733.2024.01638`), and *Weighted parallel decoupled feature pyramid network for object detection* (Neurocomputing 2024, DOI `10.1016/j.neucom.2024.127809`). The implementations below are engineering hypotheses. They are not claimed to be novel or to improve accuracy without controlled VisDrone training.

## Variants

| Branch | Change | YAML |
|---|---|---|
| `codex/visdrone-c3k2-crp` | `C3k2CRP` estimates channel-wise projection of each successive bottleneck output on the first split and applies a learnable suppression gain. | `yolo11n-visdrone-c3k2-crp.yaml` |
| `codex/visdrone-c2psa-local-global` | `C2PSALocalGlobal` retains the original PSA path, computes a depthwise local path, and fuses both with a per-channel pooled gate. | `yolo11n-visdrone-c2psa-local-global.yaml` |
| `codex/visdrone-neck-diff-fusion` | `DiffGuidedConcat` computes an absolute cross-scale difference map, generates a spatial gate, and compensates both inputs before concatenation. | `yolo11n-visdrone-neck-diff-fusion.yaml` |
| `codex/visdrone-combined` | All three changes, with a 10-class YOLO11n baseline YAML for controlled comparison. | `yolo11n-visdrone-combined.yaml`, `yolo11n-visdrone-baseline.yaml` |

For the 640 input, the modified layers preserve the original P3/P4/P5 output shapes `(1, 64, 8400)`, `(1, 128, 40, 40)`, and `(1, 256, 20, 20)` (with 10 classes). `C3k2CRP` adds one scalar gain at the YOLO11n depth, local-global C2PSA adds 34,304 parameters, and difference-guided fusion adds 129 parameters. THOP measurements at 640×640 were:

| Model | Parameters | MACs | GFLOPs (2×MAC convention) |
|---|---:|---:|---:|
| 10-class baseline | 2,591,790 | 3,225,139,200 | 6.4502784 |
| CRP | 2,591,791 | 3,225,139,200 | 6.4502784 |
| Local-global PSA | 2,626,094 | 3,225,940,224 | 6.451880448 |
| Difference-guided neck | 2,591,919 | 3,225,958,400 | 6.4519168 |
| Combined | 2,626,224 | 3,226,759,424 | 6.453518848 |

THOP does not count every elementwise operation, so these numbers are a repeatable reference rather than a complete hardware cost model.

## Validation

Each individual branch and the combined branch passed Python compilation, `git diff --check`, model construction, 640×640 batch-1 forward, finite outputs, finite gradients, and direct odd/even spatial-shape checks. The combined branch also passed a one-epoch CPU smoke training run with `imgsz=64`, `batch=2`, `workers=0`, and `fraction=0.001` against the converted local VisDrone data. The three individual branches passed the same smoke run. The smoke run is a pipeline check, not an accuracy result; validation on the fraction produced zero meaningful training progress and must not be interpreted as model performance.

The host has PyTorch `2.5.1+cu121`, but `torch.cuda.is_available()`/NVML and `nvidia-smi` report no usable driver. CUDA latency, GPU memory, FPS, pretrained-weight coverage, and full VisDrone accuracy were therefore not claimed or fabricated. The required controlled experiment remains: same pretrained checkpoint, split, image size, optimizer, schedule, augmentation, seed set, and evaluation settings for baseline, each single variant, and the combined model.
