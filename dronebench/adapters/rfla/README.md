# RFLA → YOLO11n：只替换训练目标分配

固定基线 `55989f76c1e85c192ba12cb3e533e3de842a2301`。默认 `rfla=false`，不构建此适配器。RFLA 单项（`00010`）保留 YOLO11n 的 P3–P5、Detect 分类与回归头、BCE、CIoU、DFL 及推理解码；训练和计算验证 loss 时，用几何分布的 [HLA](../../methods/rfla/core.py) 取代 TAL。推理不用 GT，不计算 RFLA。

当不少小框里没有可用网格中心时，可以检验：按感受野相似度排名能否降低缺配率，同时不过度增加框外回归目标。缺配率降低不保证召回提高；两者需分别实测。[方法卡](../../methods/rfla/README.md) 列出了论文与作者实现的差异；这里不是在 YOLO 上复现论文中 Faster R-CNN/FCOS 的分数。

```bash
python -m dronebench train --id 00010 --data VisDrone.yaml
```

[单项 JSON](single.json) 显式给出相同设置，可用 `--config dronebench/adapters/rfla/single.json` 代替 `--id 00010`；其他主开关不写即为关。`options.rfla` 只接受以下四个键：

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `radii_px` | 对构造时实际检测步长 `s` 取 `2*s` | 每层一个有限正半径，作为 x/y 两轴共同的标准差，单位为**输入像素**，不是直径或网格单位。顺序与 `strides` 相同；每个位置凭自己的 `stride_px` 查找，不猜测层在展平数组中的分界。 |
| `topk` | `3` | 第一轮每个有效 GT 在全部 active 层和位置中提名至多 k 个点。 |
| `beta` | `0.9` | 第二轮把 RF 半径乘以此因子，重新排名提名一个点；已占用的点不改派，也不保证每个 GT 都补到一个。 |
| `metric` | `"kl"` | `"kl"` 为 RF 到 GT 的 KL，`"wd"` 为平方 Wasserstein 距离，转成相似度排名。 |

`[8,16,32] → [16,32,64] px` 是**本实验对 YOLO11n 的假设**，不是论文给 YOLO11n 的数值或实测有效感受野。加入 P2 后默认 `[4,8,16,32] → [8,16,32,64] px`；显式半径列表也必须补到四层。不同分辨率下这些半径仍是输入像素。对比时记录步长和半径，不把扩大/缩小 RF 的变化归到层路由本身。RFLA 不检查点是否在 GT 内，也不用 TAL 的候选边长下限。

## 公共接口与坐标

`from dronebench.adapters.rfla import RFLAAdapter`。`RFLAAdapter(strides, options)` 的 `options` 是 `options.rfla` 子字典。

```python
adapter = RFLAAdapter(strides, options)
boxes, scores, fg, gt_idx, diagnostics = adapter.assign(
    anchor_points * stride_px, stride_px, gt_labels, gt_boxes_px, valid_gt, active, nc
)
```

- `anchor_points` 是 `make_anchors(..., 0.5)` 的网格中心；乘 `[N,1]` 步长一次得到 `[N,2]` 像素 xy。`gt_boxes_px` 是同一坐标系的浮点 `[B,G,4]` xyxy，`gt_labels` 是 `[B,G,1]` 的整数类别（可浮点存储，取值 `[0,nc)`）。`valid_gt [B,G,1]` 和 `active [B,N]` 为 bool，所有张量同设备。
- 整图或 ESOD 块均可。块内点与 GT 都应已换到同一局部原点、同一像素尺度；分块、过滤/裁剪、反投影与跨块去重不在本适配器内。每图先提取所有 `active` 点和有效 GT，再做 HLA；未激活点不能占据提名。冲突由通用 HLA 处理，默认同轮优先小面积 GT。有效 GT 可能没有匹配，尤其候选数少或冲突时。
- `boxes [B,N,4]` 是前景对应的原 GT 像素框，保持 GT dtype；`scores [B,N,nc]` 为 FP32，前景正确类别恰好 `1`，其余为 `0`。**RFD 只排名，不当作分类置信度或回归权重**。`fg [B,N]` 为 bool，`gt_idx [B,N]` 为 long，前景索引到传入图的 **padded GT 行**（可有空洞）；背景/未激活处框和分类全零，索引 `-1`，只对 `fg` gather。
- 公共 loss 先将未激活处的 BCE 屏蔽（零目标本身不等于忽略），全 batch 用 `scores.sum().clamp_min(1)` 归一化 BCE 与框损失。因此分母是最终匹配的点数、不是 GT 数或 RFD 总和；每个前景的框权重为 1。这和 TAL 的软目标不同，是对 YOLO 分类/框权重的明确选择，而非论文规定的 YOLO 损失。

## 框外目标与回归

保留真实 xyxy，不移动 GT 或中心。若点 `p` 在分配框外，网格单位原始 `ltrb = (p.x-x1, p.y-y1, x2-p.x, y2-p.y) / stride` 至少一项为负。stock [BboxLoss](../../../ultralytics/utils/loss.py#L140-L149) 的 CIoU 对真实框计算；DFL 调用 [bbox2dist](../../../ultralytics/utils/tal.py#L590-L605)，上游把 signed ltrb 裁到 `[0, reg_max-1-0.01]` 再做双邻分箱。YOLO11n 默认支持点是 `0..15`，不是 signed 支持；框外匹配不能被这些非负距离精确表达。适配器不提前裁剪、不丢弃此正样本，也不更改 DFL 解码。

与 UGS 组合时共享同一组 `(points_px, stride_px, boxes, fg)` 与同一原始 signed ltrb。支持点、负值/超界如何编码和解码由 [UGS 适配层](../ugs/) 定义，不要先调用 stock `bbox2dist(..., reg_max-1)` 再喂给 UGS。比较两条回归路径时同时记录框外比例。

## 诊断与代价

`diagnostics` 值均为同设备 int64 张量，不做逐步 `.item()`、`.cpu()` 或打印。日志层可在记录边界拷至 CPU，或先在设备上累加总数再求比率。

| 键 | 形状 | 口径 |
| --- | --- | --- |
| `positive_per_gt` | `[B,G]` | 最终计数，对齐 padded GT。无匹配和 padding 都是 0；想逐 GT 分辨时保留调用时的 `valid_gt`。 |
| `valid_gt_per_image`、`unmatched_gt_per_image` | 各 `[B]` | 有效 GT 总数、其中最终计数为 0 的 GT 数（不含 padding）。 |
| `positive_per_image`、`outside_gt_per_image` | 各 `[B]` | 最终前景点数、其中严格在所分配原框之外的点数；恰在边界不算框外。 |
| `stage1_per_image`、`stage2_per_image` | 各 `[B]` | 分别由两轮贡献且最终保留的前景点数，两者之和等于 `positive_per_image`。 |

例如，比较整图与路由后的 `sum(unmatched_gt_per_image)/sum(valid_gt_per_image)`、`sum(outside_gt_per_image)/sum(positive_per_image)`，分母为零时标为无观测。ESOD 中每行是一块而非原图；恢复原图统计需结合块元数据，不能把重复出现的块内 GT 当成独立整图 GT。缺配率变化是分配结果，不是召回率或 mAP。

两轮都需要对每图 `G_valid × N_active` 的 GT–位置对评分。实现按图处理，不分配 `[B,G,N]`；通用 HLA 默认再按最多 32 个 GT 分块，评分中间张量峰值量级为 `O(min(32,G_valid) × N_active)`，总评分工作量仍为 `O(G_valid × N_active)`。640 输入下 P3–P5 有 8,400 个位置，加 P2 后有 34,000 个位置；密集候选时不能把分块等同于稀疏计算。筛选变长子集和通用方法的几何校验仍可能同步；返回设备张量只避免了诊断标量化带来的额外同步。

## 本地验证

```bash
OMP_NUM_THREADS=1 /home/user/work/yolo11n-env/bin/python -m pytest -q dronebench/adapters/rfla/test_adapter.py
```

测试包含跨层真实 `make_anchors` 及重排后的 stride 映射、多图/带洞 padding、空目标和全屏蔽、框外冲突、stock DFL 截断，以及单项公共模型的训练前向、真实 BCE/CIoU/DFL 反向与推理解码。单测无需下载权重或数据集。

另实测了公共 `train` 和重新加载 `best.pt` 的独立 `val`。可用本地官方 val ZIP 重现；两个不重叠小 split **都取自 val，仅供工程冒烟**：

```bash
PY=/home/user/work/yolo11n-env/bin/python
work=$(mktemp -d)
export OMP_NUM_THREADS=1
"$PY" -m dronebench.evaluation.smoke_data --zip "$VISDRONE_VAL_ZIP" --out "$work/data" --train-count 4 --val-count 2 --seed 17
"$PY" -m dronebench train --id 00010 --data "$work/data/smoke.yaml" --kind smoke --epochs 1 --batch 2 --imgsz 96 --seed 0 --workers 0 --device cpu --project "$work/runs"
"$PY" -m dronebench val --weights "$work"/runs/*/weights/best.pt --data "$work/data/smoke.yaml" --imgsz 96 --batch 2 --device cpu
```

CPU、PyTorch 2.9.1 下，1 个 epoch 正常结束；重新加载后验证了 2 图、124 个 GT。训练目录 `routing.jsonl` 中的 **epoch 末训练批** 两图各有 21、11 个有效 GT，37、15 个最终正样本，4、3 个未匹配 GT，32、14 个框外正样本。两图第二轮新增均为 0，逐 GT 数组也已写入。这些数值验证记录链路，不外推为整轮统计、640 分辨率下的结果或方法收益。
