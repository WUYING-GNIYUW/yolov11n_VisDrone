# RFLA 方法卡：给小目标分到训练位置

**问题。** 一个标注框（GT）可能比采样网格还小，里面没有任何位置。只认“中心落在框内”会让它拿不到训练样本；只按框的交并比，也可能让很多相邻候选一起得零分。RFLA 改的是**训练时谁负责哪个目标**：一个位置即使在框外，它所看到的区域也可能和目标相符。它不产生图像特征、不预测框，也不增加推理分支。

## 从输入到输出

1. **描述一个位置能看到哪里。** 输入中心 `(x, y)` 和有效感受野半径 `r`。把中心视为二维高斯的均值，把 `r` 当成标准差，不是直径或方差。矩形 GT 也用一个高斯表示：均值是框中心，两轴标准差是 `(宽/2, 高/2)`。这是假设中心附近的信息更有用，不是对当前网络感受野的测量。
2. **评分（RFD），所有候选都参加。** 两个分布的中心和宽窄越接近，分数越高。默认算 `D = KL(感受野 || GT)`，不是反向 KL；也可以取平方 Wasserstein 距离 `D = W₂²`。两者都返回 `1/(1+D)`，相同分布是 1，无须先有交叠。分数用于排序，不是类别置信度或 IoU。
3. **第一轮分配。** 每个 GT 在本图的全部有效位置里提名得分最高的 `k` 个，不分层设额、不用 IoU 门限。默认同一位置被多个 GT 提名时，面积较小的 GT 获得它，而不是分数最大的 GT。位置得分并列时按输入位置下标，GT 面积相同时按 GT 下标。这两个同分约定是本实现为可复现而补充的。
4. **第二轮分配。** 中心和 GT 不动，只把候选标准差乘 `β`（方差随之乘 `β²`），从头评分，每个 GT 再提名全池的第一名。**先排名，再合并**：只在第一轮仍空闲的位置写入第二轮结果。如果第一名已被占用，不改归属，也不改找第二名。两轮归属冲突时第一轮优先。因此没有每个框必得 `k+1` 个、甚至至少一个的保证，须读实际计数。

## 可调用接口

[公开导出](./__init__.py) · [独立 PyTorch 实现](./core.py) · [最小样例](./example.py) · [测试](./test_rfla.py)

```python
from dronebench.methods.rfla import receptive_field_scores, hierarchical_assign

scores = receptive_field_scores(points_xy, rf_radius, gt_xyxy, metric="kl")
a = hierarchical_assign(points_xy, rf_radius, gt_xyxy, topk=3, beta=0.9, metric="kl")
foreground = a.gt_index >= 0
outside_locations = points_xy[a.outside_gt]
```

| 参数或结果 | 约定 |
| --- | --- |
| `points_xy`, `rf_radius` | 一张图已启用的 `N` 个位置，形状 `[N,2]`；半径 `[N]` 或 `[N,1]` 为两轴相同，`[N,2]` 按 x、y 分别指定半径。 |
| `gt_xyxy` | `[G,4]`，连续坐标 `(x1,y1,x2,y2)`、正宽高，不加 1。位置、半径和框必须是同设备的有限浮点数，使用同一坐标系及单位，通常是预处理后的图像像素。GT 不含 batch 填充行和忽略标注。允许 `G=0` 或 `N=0`。 |
| `scores` | `[G,N]`，行是 GT，列是候选。`metric="kl"` 或 `"wd"`；半精度输入至少用 float32 算，相同输入没有随机操作。独立评分可微，分配不记梯度。 |
| `a.gt_index`, `a.stage`, `a.score` | 均为 `[N]`。GT 下标从 0 开始，背景为 `-1`。轮次为 `0/1/2`，0 表示背景。分数取**归属 GT、获胜轮次**的 RFD，背景为 0，不是所有 GT 的最大值。 |
| `a.counts`, `a.outside_gt` | `[G]` 整数是冲突和两轮合并后的实际正样本数，含 0；`[N]` 布尔量只标记中心严格在**自己归属框外**的正样本（边界不算外）。不裁剪坐标、不把框外正样本改成背景。 |
| `topk`, `beta`, `ranking`, `gt_chunk_size` | 第一轮默认 `k=3`，候选不足时取 `min(k,N)`；第二轮固定取第一名，默认 `β=0.9`，允许 `0<β≤1`。`ranking="paper"` 或 `"upstream"` 的差别见下。按 `gt_chunk_size` 个 GT 分块（默认 32），不改变归属。 |

没有数据集名、类别数、网络层或固定步长依赖。单独返回完整评分矩阵需 `G×N` 个数；分配对两轮各做 `G×N` 次配对评分，临时成对张量按 `min(G,gt_chunk_size)×N` 展开。`a.stage == 1` 可单独检查第一轮结果，`(a.counts == 0).sum()` 可数出没有正样本的 GT。不继承分类权重、不做采样器的二次抽样。

## 论文、作者程序与本地选择的边界

- **半径的来源。** [论文 §3.1，式 (2)–(4)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf)对普通串行卷积以 `T₀=1, J₀=1` 起算，`Tᵢ=Tᵢ₋₁+(kᵢ−1)Jᵢ₋₁`、`Jᵢ=Jᵢ₋₁sᵢ`。`T` 是理论感受野**边长**；文中用理论半径的一半作有效半径，正方形时即 `r=T/4`。[作者的 RF 生成器](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/mmdet/core/anchor/rf_generator.py)按 ResNet-50 层路径算 `T`，`fraction=0.5` 乘 `T` 得的是有效**边长**，还有宽高比变体；[作者的 FCOS 头](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/mmdet/models/dense_heads/rfla_fcos_head.py)给每个点用正方形。不能把该层表当作别的网络的测量值。换结构时由调用者传每点半径：可用已推导的理论范围、输入梯度实测，或显式记录“半径 = 步长 × 待验证系数”。后者是实验假设，不是论文的通用默认值。
- **评分口径。** 本实现用论文 §3.2 式 (3)–(6)、(8) 的均值和协方差独立编写。令 `s = (GT 宽/2, GT 高/2)`，`r` 为候选标准差，`d` 为两中心之差；KL 为 `½ Σ轴[(d/s)²+(r/s)²−1−2 ln(r/s)]`，平方 Wasserstein 为 `Σ轴[d²+(r−s)²]`。以标准差为输入时采用式 (6) 的矩阵定义，不直接套用印刷式 (7) 中不同的尺度系数。把候选直径设成 `2r` 即与[作者的 BboxDistanceMetric（GT 在前、候选在后）](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/mmdet/core/bbox/iou_calculators/metric_calculator.py)方向和尺度一致；原程序给边长等项加 `1e-6`，这里要求输入非退化，不加该项。共同平移不会改分；中心、半径、GT 同比例缩放，KL 不变，平方 Wasserstein 距离乘比例的平方。
- **冲突与并列。** 论文 §3.3、式 (9) 说发生争用时小 GT 优先、第一轮正样本冻结。[作者 HieAssigner](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/mmdet/core/bbox/assigners/hierarchical_assigner.py)逐 GT 写入，没在此处按面积排序；同轮争用实际是**输入顺序靠后者覆盖**。它按分值相等选中与前 k 名并列的所有候选，因此同一 GT 在单轮可多于 k 个。`ranking="upstream"` 可对照这两条**正样本**规则，默认仍是论文版本。两种模式都冻结第一轮。上游框架区分非正样本是背景还是忽略（例如未入选且该轮最大分 ≥0.8 时可保留忽略值）；这里合并为 `-1`，不复刻忽略框规则、抽样器、上游负样本标签或浮点 epsilon，不保证逐位重现整个检测器。
- **超参数依据。** 论文 §4.3 推荐 `k=3`，表 3 选 `β=0.9`。[作者 AI-TOD Faster R-CNN 配置](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/configs/rfla/aitod_faster_r50_rfla_kld_1x.py)用了 `topk=[3,1], ratio=0.9, assign_metric='kl'`；不能误读成分配器构造函数本身的初值 `[2,1], ratio=1`。默认不启用其可选中心在框内过滤。
- **框外位置交给谁处理？** §3.4 在 FCOS 去掉框内限制，式 (10) 调整 centerness，给截断后的分子加 `c=0.01`；[作者 FCOS 头的 centerness_target](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/mmdet/models/dense_heads/rfla_fcos_head.py)则直接把两个轴的最小侧距夹到至少 0.01，仅改权重计算。这不是一个有符号的预测解码器。本模块根本不计算 centerness 或回归损失。从框外点到框边的 `(l,t,r,b)` 至少一项为负；如果定位头只支持非负分箱，直接截为零再当成原框是错的。须由适配层明确编码范围、哪些损失能用这些点，以及解码时如何还原；保留原归属和框外计数以便对照。

## 什么条件下值得试

| 可观测条件 | 需要的接口、可检验的假设 |
| --- | --- |
| 小框的正样本数明显偏少，或不少 GT 内没有任何有效候选中心。 | 能取得各候选的图像坐标、可解释的感受野尺度和标注区域。固定同一批 GT、同一参考尺度分桶，对比分配器的零正例比例、每框计数分布、框外正例比例和最终召回，而不是只看总正例数。 |
| 两个近邻对象常争同一个位置，或同分候选很多。 | 分别检查两轮和冲突后的数量；按密度分组，交换 GT 顺序测试顺序敏感性。小框优先可能挤掉邻居；半径太大、路由后候选太少，都可能让第二轮无新样本。 |
| 同一层对不同位置覆盖的输入范围近似可用一个宽度表示。 | 用梯度响应范围检查假定的半径；中心偏离主体、细长倾斜或多峰时，轴对齐高斯可能不合适。在有有效层/块/查询位置筛选时，只传实际可执行的候选；把分块偏移换回图像坐标，不要用 GT 内外掩码先删点。 |

**跨任务尝试而非已证结论。** 在细胞实例分割的种子提议阶段，若许多实例内部没有粗网格中心，可用每个实例的范围给“哪个种子应负责该实例”排序；不能把框外种子直接当成前景像素。在关键点估计里，若有标注误差范围或可估计的部位宽度，可把它转成有面积的监督窗口，再比较不同位置负责它的效果；只有孤立点、没有尺度时，应先验证窗口假设。两种尝试都要求后端能从分到的位置预测偏移。

## 实验出处（不是本仓库成绩）

Chang Xu 等，*RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection*，ECCV 2022。[论文 §4.1–4.6、表 1–8](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf)。论文在 AI-TOD、TinyPerson、VisDrone2019、DOTA-v2.0 上实验。表 7 单列的 **VisDrone2019 train→val** 结果为 FCOS `14.1→15.1` AP、Faster R-CNN `22.3→23.4`、DetectoRS `25.7→27.4`。这不是 YOLO11n 的分数或增益，也不是本模块测试结果。那些模型、步长、类别数、训练预算均不构成接口要求。

**代码出处与许可。** [作者仓库固定提交 `198977b5e198005fec783e81f10f9363946d2ef8`](https://github.com/Chasel-Tsui/mmdet-rfla/tree/198977b5e198005fec783e81f10f9363946d2ef8)；上述实现及配置链接均固定该提交。[上游 LICENSE](https://github.com/Chasel-Tsui/mmdet-rfla/blob/198977b5e198005fec783e81f10f9363946d2ef8/LICENSE) 为 MIT，Copyright (c) 2022 Chang Xu。本目录是对论文公式与分配规则的独立 PyTorch 实现，未复制上游文件，不依赖 MMDetection、MMCV、YOLO 或训练权重。本工作副本另见[根目录许可](../../../LICENSE)。

## 运行验证

在工作副本根目录运行（无需网络）。第一条样例中，一个不含任何候选中心的框在两轮各得到一个框外正例。

```bash
PY=/home/user/work/yolo11n-env/bin/python
"$PY" -m dronebench.methods.rfla.example
OMP_NUM_THREADS=1 "$PY" -m pytest -c /dev/null -p no:cacheprovider -q dronebench/methods/rfla/test_rfla.py
```

可选的真实标注用例不下载或解压数据到磁盘。给 `VISDRONE_VAL_ZIP` 传[本地 VisDrone2019-DET 验证集压缩包](https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip)的路径；不传时仅跳过该用例。它读 `0000001_02999_d_0000005` 的原始框，排除忽略区，只保留 1–10 类，将 1920×1080 等比缩至 640×360，再上下各填充 140。候选取 8/16/32 步长的网格中心，测试用半径为两倍步长，**仅是固定测试输入**。检查逐框计数和有无中心在框内的情况；这不是训练、召回或 AP 评测。

```bash
VISDRONE_VAL_ZIP=/path/to/VisDrone2019-DET-val.zip OMP_NUM_THREADS=1 "$PY" -m pytest -c /dev/null -p no:cacheprovider -q -s dronebench/methods/rfla/test_rfla.py
```
