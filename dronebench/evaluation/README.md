# YOLO11n 评测：固定 GT 分桶与单图端到端延迟

这是基于 [基线 main 快照](https://github.com/WUYING-GNIYUW/yolov11n_VisDrone/tree/55989f76c1e85c192ba12cb3e533e3de842a2301) 的实验协议，**不是 VisDrone 官方评测**。正式实验请对各配置使用同一数据版本、固定随机种子与预算，并保存实际权重、输入分辨率、验证设置和设备。下方的真实小样本只检验工程链路，不能用来比较方法。

## 验证器

从仓库根目录运行，共享解释器是 `/home/user/work/yolo11n-env/bin/python`。[FixedBinValidator](validator.py) 沿用 `DetectionValidator` 的构造方法、验证循环、NMS、原生 mAP50–95 和 fitness，仅在父类的逐图统计入口另累计 GT 与 TP。可传给 `YOLO.val`，也可作为普通 `DetectionValidator` 实例使用。适配器若使用局部特征块，必须在此之前将各块的预测放回**整张网络输入图**的坐标，并为整图做后处理；本验证器不把一块当一图。

```python
from ultralytics import YOLO
from dronebench.evaluation.validator import FixedBinValidator

metrics = YOLO("/absolute/experiment/weights/best.pt").val(
    validator=FixedBinValidator,
    data="/absolute/converted/VisDrone.yaml",
    split="val", imgsz=640, batch=1, device="cpu", workers=0, rect=True,
    conf=0.001, iou=0.7, max_det=500, single_cls=False, agnostic_nms=False,
    plots=False, save_json=False,
)
print(metrics.box.map)
print(metrics.fixed_bins)
print(metrics.fixed_bin_protocol)
```

- 正式约定显式请求 `conf=0.001, iou=0.7, max_det=500`，类别不折叠，验证为普通 YOLO11 检测头。mAP 仍用相同设置的原生验证器结果；下面的 `0.25`/`300` **不筛掉任何 mAP 输入**。基线可能按数据集最大 GT 数调高默认 `max_det=300`；故正式配置传非默认的 500，并在 `fixed_bin_protocol` 记录实际值。更严的上游置信阈值、更小的检测上限或不同 NMS 都可能改变后续召回，不能混合比较。
- 对每图，使用基线准备好的输入像素系 `xyxy`。用该图 `ratio_pad=((gain_h,gain_w),(left,top))` 对 GT 和预测分别减 padding，并分别以 `gain_w/gain_h` 还原横/纵坐标，裁剪至原图边界。原 VisDrone 标注边缘为整数，而基线 YOLO 标签仅保留 6 位归一化小数：反变换后四条边均在相应轴尺寸 `2e-6` 的误差内接近整数时，仅将该 GT 恢复到原像素格，否则保留亚像素坐标。预测始终不取整。若数据有精确亚像素标注且不应作该恢复，设置验证器子类的 `integer_gt=False`，不可混用两种协议。
- 令原图高宽为 `H,W`，GT 短边为 `min(x2-x1,y2-y1)`。参考长度 `s = 短边 × 640 / max(H,W)`；`lt8` 为 `[0,8)`，`8to16` 为 `[8,16)`，`ge16` 为 `[16,∞)`。实现使用乘积比较，8/16 正好落入右桶。改变网络输入到 960、改变矩形 padding 或 batch 大小，不重定义桶。
- 在**全图、所有桶和类别合计**的后 NMS 预测中，保留置信度 `>=0.25`，按分数降序取最多 300 个，同分保留已有预测顺序。依次匹配同类、尚未匹配的所有 GT 中 IoU 最大者，要求原图 `IoU >= 0.5`，相同 IoU 取准备后 GT 列表的首项。预测和 GT 各至多使用一次，不能分别在三个桶里重新匹配。以匹配 GT 的桶计 TP；汇总整套图像的 `TP / GT`，不是按图或类别平均。
- `metrics.fixed_bins` 每桶返回 `{"gt": int, "tp": int, "recall": float或None}`。空桶无分母，不视作 0；有 GT 无命中是 0。`metrics.results_dict` 保留 `metrics/mAP50-95(B)` 等原有键，并新增 `metrics/R50_640_lt8(B)`、`metrics/R50_640_8to16(B)`、`metrics/R50_640_ge16(B)`；训练/日志用的扁平数值字典用 `NaN` 表示空桶。标准 JSON 建议保存 `metrics.fixed_bins`（空值为 `null`）、`metrics.fixed_bin_protocol` 和 `float(metrics.box.map)`。不把原始 counts 或 `None` 塞进训练器的数值日志。若训练显式 `val=False`、仅最后一轮内部验证，保持此前训练 CSV 的原列数，不在末行插列；额外指标仍可从验证器的 `metrics` 读取。

[官方说明](https://aiskyeye.com/evaluate/object-detection-2022/)要求考虑忽略对象、截断规则，按每图跨类别最高 500 个检测计算 AP，并报告 AR@1/10/100/500；[官方工具的 ignore-region 处理](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/dropObjectsInIgr.m)同时处理 GT 和检测。此处只使用[基线转换](../../ultralytics/cfg/datasets/VisDrone.yaml)生成的标签，不根据 ignore 区域另行移除检测，固定桶 R50 也不是官方 AR。

## 延迟：从本地压缩图像文件到主机检测框

```bash
PY=/home/user/work/yolo11n-env/bin/python
"$PY" -m dronebench.evaluation.latency \
  --weights /absolute/experiment/weights/best.pt --images /absolute/converted/images/val \
  --device 0 --imgsz 640 --conf 0.25 --iou 0.7 --max-det 300 \
  --warmup 10 --runs 100 --output /absolute/experiment/latency.json
```

[latency.py](latency.py) 接受可信的本地检测 `.pt` 和一张图片或一个非递归图片目录；不以内置模型名代替不存在的权重。每个请求 batch=1，FP32，`rect=False` 固定方形输入，记录检查步长后的实际 `input_hw`。图像按路径排序，循环取图；warmup 和正式计数分别从头开始。如果需要覆盖每一图，将 `--runs` 至少设为目录中图片数。

**计时内：** OpenCV 打开、读取并解码本地文件 → 基线 `YOLO.predict` 的 letterbox、BGR→RGB、归一化与必要的 H2D → 网络前向及检测头框解码 → 后处理（普通 YOLO11 的 class-aware NMS）和回映射到原图 → 检测数组可在 CPU 读取（含必要的 D2H）。不计网络/相机输入、枚举目录、权重加载、初始化、warmup，或绘图/写盘；也不是 `Results.speed` 的三个阶段之和。每次真的打开文件，但操作系统可能使用热页缓存，**不是冷盘测试**。

**同步与输出：** 在相同设备、图片、分辨率、阈值、线程设置和运行负载下测试。使用同一个已热身的预测器；每张计时前、计时结束处同步实际 CUDA 或 MPS 设备，CPU 为同步执行。`perf_counter_ns` 计墙钟时间，按毫秒样本用线性插值分位数得到 `p50_ms`/`p95_ms`。JSON 包含实际 `device`/`hardware`、软件版本、线程数、输入形状、`warmup`/`runs`、`postprocess`、`boundary` 及逐次 `samples`（文件、耗时、检测数）。`weights_sha256` 是实际权重文件摘要；`image_manifest` 按正式计时首次使用顺序记录每个唯一文件的相对文件名和内容 SHA256，不包含只用于 warmup 的图。`images_sha256` 对该列表的 `json.dumps(..., sort_keys=True, separators=(",", ":"))` ASCII 字节取 SHA256，循环次数和请求顺序保存在 `runs`/`samples`。摘要在计时结束后读取；运行中不能替换输入文件。不支持把一个请求分摊在多设备；未测过的硬件不推算成绩。

## 只用于工程冒烟的两份 split

```bash
PY=/home/user/work/yolo11n-env/bin/python
"$PY" -m dronebench.evaluation.smoke_data \
  --zip /tmp/visdrone-2019-val.AwxYDK66.zip \
  --out /home/user/work/yolo11n-smoke-data --seed 17 --train-count 4 --val-count 2
```

[smoke_data.py](smoke_data.py) 只读 ZIP，不联网；如临时缓存已清理，可另存[相同来源的原包](https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip)，对照清单中的 SHA256。目标必须是**尚不存在**的目录，复建时请换新路径，不删除已用数据/缓存。按 `sha256("种子:图片文件名")` 排序，每个序列前缀只选一图，前 4 张为 smoke-train，后 2 张为 smoke-val，故两个 split 连序列前缀也不同。保存原 JPEG 字节；标注完全按基线规则：仅丢弃第 5 列文本为 `0` 的行，类别号减 1，整像素 `xywh` 转归一化中心/宽高并保留 6 位小数，不额外按截断/遮挡过滤。数据 YAML、归档 SHA256、图名、尺寸和框数清单都在输出目录，不放入仓库。这两个 split **都来自正式 val**，不要把该权重拿去正式比较、训练或调参。

本机已产出 [smoke.yaml](</home/user/work/yolo11n-smoke-data/smoke.yaml>) 和 [manifest.json](</home/user/work/yolo11n-smoke-data/manifest.json>)：train 4 图 / 295 框；val 2 图 / 124 框。完整原 ZIP（548 图）的 38,759 个有效 GT 中，参考三桶为 17,003 / 13,641 / 8,115；使用基线格式 6 位小数标签、模拟 640 / 960 letterbox 往返后逐图核对，两次桶计数都与原标注一致。这是**标注分布**，没有在全部图片上跑检测模型。

## 测试与工程记录（非正式成绩）

```bash
YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false OMP_NUM_THREADS=1 \
  /home/user/work/yolo11n-env/bin/python -m pytest -q -o addopts='' -p no:cacheprovider dronebench/evaluation/tests
```

[test_validator.py](tests/test_validator.py) 检查 640/960 参考不变、非方形及逐轴缩放、padding 裁剪、跨桶竞争、稳定的同分匹配、阈值/上限、空桶、非零受控 mAP 与基线逐项精确一致；[test_tools.py](tests/test_tools.py) 走真实 `YOLO.val`/本地 `.pt`/JPEG、检查分位数和数据可重建性，不依赖原始 VisDrone 下载。

本机另用 4 张真实图随机初始化 YOLO11n 训练 1 epoch（128、batch=2、seed=17、SGD、CPU，共 2 个训练 batch），仅为得到可加载的[烟测权重](</home/user/work/yolo11n-smoke-data/engineering/eval/smoke-train/weights/last.pt>)。[640 验证](</home/user/work/yolo11n-smoke-data/engineering/eval/smoke-validation-640.json>)、[960 验证](</home/user/work/yolo11n-smoke-data/engineering/eval/smoke-validation-960.json>) 都是 39 / 52 / 33 个 GT、三个桶均 0 TP；640 原生和扩展 mAP 均为 0。[延迟烟测 JSON](</home/user/work/yolo11n-smoke-data/engineering/eval/smoke-latency-640-hashed.json>)使用同一权重、两张真实图，CPU i5-8250U、FP32、640×640、3 次 warmup / 10 次计时，p50=124.40 ms，p95=186.72 ms，返回检测数为 0。[工程记录摘要](experiment_records/metric-smoke.json)可随代码分发，不含图片和权重。这些数只说明工具有实际输入输出；正式权重、完整 val、专用设备稳定负载下的精度/延迟仍待测。
