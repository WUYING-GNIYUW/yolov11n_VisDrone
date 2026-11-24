# ESOD 接入 YOLO11n：在 P2 后切块

单项为 **`01000`**。在仓库根目录激活实验环境后运行：

```bash
source /home/user/work/yolo11n-env/bin/activate
python -m dronebench train --id 01000 --data VisDrone.yaml
```

[单项配置](./single.json) 显式列出相同默认值，可用 `--config dronebench/adapters/esod/single.json` 替换 `--id 01000`。独立验证用训练命令输出的权重路径：

```bash
python -m dronebench val --weights /实际运行目录/weights/best.pt --data VisDrone.yaml --imgsz 640
```

[适配代码](./adapter.py) 复用[通用方法](../../methods/esod/method.py)；[方法卡](../../methods/esod/README.md) 记录论文、作者实现、许可证、两种择块算法及适用条件。其他开关全关时不引入择层、逐点稀疏头、新分配器或定位表示。全关 `00000` 不构建本适配器。

## 数据怎样流动

1. [公共模型](../../model.py) 在整图上只执行 YOLO 的 0、1、2 号层。`split(shallow, hw, batch=None)` 接收第 2 号层输出 `[B,C,H/4,W/4]` 和网络输入像素 `(H,W)`。ObjSeeker 在该图上计算单通道前景 logits。`select_patches` 输出 `[K,5]` 的 `[原图号,x1,y1,x2,y2]`，用 P2 格和右/下不含的区间表示；`layout.crop` 真正取出 K 个等形特征块，不切 RGB、不放大。
2. **只有这些块经过第 3 号层及剩余骨干、PAN 与检测头。** 归属图可以有零块。若全批 `K=0`，公共模型不执行任何深层；训练仍计算整图前景损失，预测返回 `[B,4+nc,0]`。联合训练时裁剪可反传至浅层；离散择块不反传。
3. `local_batch` 只返回 `batch_idx`（块号）、`cls` 和归一化局部 `xywh`，不复制图。对有限、正面积、中心在整图有效范围的 GT，按**原 GT 中心**是否落在各块半开有效格内归属。中心恰在一块右边或下边时不归该块；多个块含同一中心就各训练一次；仅与块相交、中心不在块内则不分给它。局部中心减去 `4 × (x1,y1)`，**保留整个框的宽高**。因此局部 `xywh` 的宽高可大于 1；不把跨块大目标改成一截小目标。原 TAL/CIoU/DFL 算法不变，单项 TAL 候选边长下限仍为 16 输入像素；候选位置和正例数会随切块改变。检测分配按 K 个块分组，公共损失最终仍按原批图数 B 缩放，不把新增块数当成更多独立原图。
4. 公共层先按 **局部** P3/8、P4/16、P5/32（组合时按实际检测层）解码为 `[K,4+nc,A]` 的像素 `xywh` 与类别概率。`restore` 只给中心加上块原点，**不在 NMS 前把框截到块边或图边，也不因回归后的中心越界删框**。有效网格点和其回归后预测中心是两个量。按原图打包到 `[B,4+nc,max_i(K_i)×A]`；非有限坐标、非正宽高、补齐的槽位均置零，非有限类别分数置零。公共验证/预测在这批整图坐标上做原生按图/类 NMS，重叠区预测可以互相抑制。这里的「原图」指网络输入画布：从 letterbox 坐标还原到源照片仍由原评测/显示流程处理。

**没有块 padding 的默认路径。** 每条输入边 `s` 要求 ≥64 且为 32 的倍数。相应块边（输入像素）为 `max(64, 32×ceil(s/(32×patch_divisor)))`；高宽分别计算。96/128 方图对应 64×64，640 方图对应 160×160，384×640 对应 96×160。边缘窗口平移回图内；不因候选网格不能整除就在深层执行纯补零块。最小 64 给单块训练留出大于 1 个 P5 空间位置。若外部手工传入带 padding 的通用 `PatchLayout`，须在 state 提供 `[K,A]` 的 bool `anchor_valid`；可用 `layout.valid_centers` 对**真实回归前 anchor 中心的局部输入像素**计算，并按本次解码的层序和格序排列。缺此掩膜会显式报错，不会把回归后中心误作 anchor。该扩展是解码接口，不改变公共训练网格。

## 前景监督与可控参数

GT 只在 `self.training`、`0 ≤ epoch < warmup_epochs`、训练批显式给出标注、且**该图有有效目标**时参与择块。无有效 GT 的图使用预测。其他时期、单纯调用 `model(img)`、正式验证和推理均按预测择块；即使误给 `.eval()` 的 `split` 传了标签，也不会读取。`aux_loss` 在给定标注时以框高斯热图监督全图 logits（没有 SAM），目标仅在同一训练前向内复用。[公共损失](../../loss.py) 可以在**推理结束后**用验证标注计算损失；不借此修改块表或输出。

| `options.esod` | 默认 | 作用 |
|---|---:|---|
| `patch_divisor` | 4 | 每条边的目标分块比例，正整数；1 为整图大小的块，并非强制选中。 |
| `threshold` | 0.3 | `[0,1]`，在前景概率图上保留不低于此值的局部极大值。低阈值减少冷启动时全部漏选的风险，也可能不节省算量。 |
| `warmup_epochs` | 3 | 前三轮有标签训练允许 GT 择块，从第 4 轮起只用预测；0 为全程预测。与优化器学习率预热无关。 |
| `algorithm` | `grid` | 并行网格择块；可切换 `greedy` 的 CPU 顺序参考算法。 |
| `kernel_size` | 13 | ObjSeeker 逐通道卷积的正奇数核大小。 |
| `loss_weight` | 1.0 | 非负；前景项为该值乘 `20 × soft focal + Dice`。设为 `0` 则不提供前景损失梯度，不是关闭 ESOD。 |

未知子选项报错。块原点只要求落在 P2 格上，不保证与全图 P3–P5 格重合；块边上下文与整图不同。ESOD 不自行选择特征层，也不实现块内逐点稀疏卷积；这两项分别由 DM-EFS、QueryDet 控制。

## 实测：先看覆盖和正例，再谈性能

[阶段二测试](./test_adapter.py) 不需网络或额外权重。在仓库自带的真实照片上，以工程框驱动 YOLO11n 四种形状的前向、反向和一次 SGD；捕获真实第 2/3 号层边界，逐元素核对第 3 层收到的确是所选 P2 切片。另测空深层、跨块映射、无泄漏、padding anchor、退化框和重叠区原图 NMS。单整块路由与相同初始化的 stock 模型对比解码和 NMS；越界框 `[-6,10,10,20]` 以及回归中心在图外的有效候选不会被 ESOD 额外截断或删除。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -c /dev/null -p no:cacheprovider \
  dronebench/adapters/esod/test_adapter.py dronebench/methods/esod/test_esod.py
```

实测 **22 passed**。另将一张有 17 个原始 GT 的 VisDrone 图依次直接缩放至 96×96、128×128、640×640、384×640，各做一次单图 SGD：四次训练分别选中 2、4、6、6 个块，TAL 正例分别为 62、79、142、137。这个形状/梯度检查不采用评测预处理、不报告 AP。

使用[工程子集生成器](../../evaluation/smoke_data.py)从 VisDrone 官方 val 抽取不重序列的 4 训/2 验，做真正的训练器、EMA、保存、重载及独立验证。**样本来自 val，仅验证工程路径，不能当作正式 VisDrone 成绩。** 命令如下；两项对照只改一个参数（`threshold=0.5` 或 `warmup_epochs=0`），其余相同。

```bash
python -m dronebench train --id 01000 --data /home/user/work/yolo11n-smoke-data/smoke.yaml \
  --epochs 4 --imgsz 128 --batch 2 --workers 0 --device cpu --seed 17 --kind smoke --project /tmp/dronebench-esod-tested
```

Python 3.13.5、PyTorch 2.9.1+cpu、torchvision 0.24.1+cpu。此工程批次很小：默认运行收据为 4 轮、每轮 2 批，梯度累积后实际只有 **1 次 optimizer step**。下表中训练数字均为**每轮最后一批**，不是全轮总数；验证覆盖使用各自 best 检查点、`eval()` 和同样的两张 `96×160` 矩形网络输入，先在**不传 GT**的情况下固定路由，再用 124 个标注统计。事后调用 `local, _, source = layout.project_boxes(gt_xyxy_px, image_ids, 4)`；去重 `source` 得中心命中数，在 `local` 的四条边均在对应块内时去重 `source` 得全框命中数。

| 阈值 / 预热轮数 | 训练 K（1→4 轮） | TAL 正例数（1→4 轮） | 验证每图 K | 至少一块含中心 / 含全框 |
|---|---|---|---|---|
| **0.3 / 3（默认）** | 8, 8, 8, 8 | 82, 49, 100, **80** | 6, 6 | 124/124；116/124 |
| 0.5 / 3 | 8, 8, 8, 6 | 82, 49, 100, 90 | 0, 0 | 0/124；0/124 |
| 0.3 / 0 | 8, 8, 8, 8 | 57, 43, 85, 81 | 6, 6 | 124/124；116/124 |

这个短训说明暖启动后检测器仍得到训练信号，但不说明默认值对所有数据最优。阈值 0.5 在这两张图上过早拒绝了所有块；0.3 未拒绝目标中心，但因为块重叠，实际处理的特征面积比 `Σ K_i·ph·pw / (B·H·W)` 为 **1.6**，这不是加速。默认模型在这两张图的独立验证 mAP50–95 为 **0**；覆盖不等于检测成功。正式全量训练、固定尺度分桶召回和同设备端到端延迟仍需单独运行。在开发集上联合检查覆盖、块面积比、正例数和延迟，不应把暖启动的 GT 路由召回当作推理召回。
