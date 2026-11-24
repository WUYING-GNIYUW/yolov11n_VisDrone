# ESOD：先在浅层找前景，只把选中的特征块交给后续网络

一张大图上可能只有几处需要仔细看。检测器的浅层已经看过整图；让它顺便估计「这里可能有东西吗」，不必再跑一套网络挑图。把可能有前景的位置围进若干同样大小的块，深层网络一次处理这一批块。**切的是特征，不是原图；也不是先算完整图再把背景预测置零。** 本提取包括前景监督、两种择块算法、真正的裁剪和框的来回映射。块内部用什么检测头，由调用者决定。

## 什么时候值得尝试

| 可观察的条件 | 怎么检验假设 | 不一定适用的情况 |
| --- | --- | --- |
| 前景集中在少数区域，后半段网络占主要耗时 | 在同一组图上记录每图块数、`K × ph × pw / (B × H × W)` 和整条路径延迟；比同分辨率的整图运行，而不是只比卷积次数 | 满屏物体、散得很开的目标、重叠块多时，实际处理面积甚至可能大于整图 |
| 浅层还能看见小结构的线索 | 分开统计「至少一个块含框中心」和「至少一个块含全框」的比例，再按**固定参考尺度**的尺寸桶测最终召回；比较预测掩膜与仅作诊断的真值掩膜 | 挑块阶段漏掉的区域，后面再好的检测头也看不到；中心命中不等于看到了整个目标 |
| 后半段能在局部上下文中完成判断 | 改块大小，看边缘目标、跨块大目标、需要远处背景的类别是否退化 | 依赖全图关系的任务，缩小块可能省算量却损失判断依据 |

例如，高分辨率表面检查中，如果缺陷只出现在少数部位，可把局部检测器作为后半段；显微图上若细胞稀疏，可尝试只在块内做实例分割，再用块原点拼回掩膜；地理影像上也可试局部地标关键点，把点坐标平移回整图。这些是**待验证用途**，不是论文已验证的任务。分割重叠区的融合、关键点的去重需由相应任务定义。

## 调用与坐标约定

独立实现见 [method.py](./method.py)，公共符号从 [包入口](./__init__.py) 导出。运行时只需 PyTorch 和标准库。下列 `stride` 是浅层相对于**网络输入**的格距；单个数，或 `(sy, sx)`。所有 `xyxy` 都是连续的输入像素坐标，不是归一化框、中心宽高，也不是深层检测网格。输入图像经过缩放、填边或翻转时，标注必须已做相同变换。

```python
from dronebench.methods.esod import ObjSeeker, foreground_loss, foreground_targets, select_patches

seeker = ObjSeeker(in_channels)
features = stem(images)
logits = seeker(features)
target = foreground_targets(logits, boxes_xyxy, batch_indices, stride)
aux_loss = foreground_loss(logits, target)
route = target if warmup_training else logits.detach().sigmoid()
layout = select_patches(route, patch_hw=(ph, pw), threshold=0.5, algorithm="grid")
blocks = layout.crop(features)
local_gt, patch_ids, source_ids = layout.project_boxes(boxes_xyxy, batch_indices, stride)
```

- `features`：`[B,C,H,W]`，在全图只提取一次。`logits` 和 `target`：`[B,1,H,W]`，不含类别。`boxes_xyxy` 为 `[N,4]`，`batch_indices` 为 `[N]`，可空。类别/其他标注可用 `source_ids` 回取。
- `layout.offsets` 为整型 `[K,5]`，每行为 `[原图批号,x1,y1,x2,y2]`，表示浅层有效格子 `y1:y2, x1:x2`，右端、下端不含。`layout.feature_hw`、`layout.patch_hw`、`layout.batch_size` 记录原格图、网络块和原批次尺寸。`K` 是全批块数，不是图数。
- `layout.crop(features)` 真实取出 `[K,C,ph,pw]`，不 resize，可对特征反传。正常边界把块平移回图内；当地图本身小于指定块时才右/底补零，`offsets` 不含补零。`layout.crop(finer, scale=2)` 可取同一物理区域的较细特征，要求它与浅层**零点对齐、空间尺寸恰为整数倍**。不把已经整图算好的深层特征伪称为省下的计算。
- `layout.project_boxes(..., clip=False)` 以原图编号和框中心是否落在半开有效窗口内作分配。返回块内像素 `local_gt`、块号 `patch_ids`、原始标注号 `source_ids`。若两个块含同一中心，就各有一份标注；默认保留完整框尺寸，即块内坐标可为负、可越过块边。`clip=True` 是不同的训练对照，会截短标注。
- 块检测器自行按**各检测层**步长解码为块内输入像素。`global_xyxy, image_ids = layout.restore_boxes(local_xyxy, patch_ids, stride, image_hw=(input_h, input_w))` 给四条边加浅层原点 `(x1*sx,y1*sy)`，并返回所属原图。不传 `image_hw` 时只平移，不截断；传入时限制到网络输入边界，不截到块边，始终保持行序和长度。`layout.valid_centers(local_xy, patch_ids, stride)` 接受 `[M,2]` 块内像素点，可过滤补零位置上的 anchor/中心。映射后按**原图、类别**合并/NMS，再撤销外部 letterbox。分数、去重、全图有效内容区不是本模块的职责。

## 方法对应与实现选择

1. **找前景。** [图 2、§III-A/B、式 (1)–(2)](https://arxiv.org/html/2407.16424v2#S3.SS2)：共享浅层接逐通道大核卷积（默认 `13×13`）、BN、ReLU、`1×1` 单通道输出。前景图只表示「可能有物体」，不是计数图，也不预测类别或框。
2. **给浅层监督。** [式 (3)–(4)](https://arxiv.org/html/2407.16424v2#S3.E3)：每个框内高斯值 `exp(0.5 * log(tau) * (dx² + dy²))`，`dx,dy` 是到框中心的距离，分别除以半宽、半高；连续中心为 1，两轴都到边缘的角点为 `tau`（默认 0.5）。本实现在可见框内取每个格子上式的**最大值**，多个框取最大值；这是保住亚格子小框的离散化选择，不是作者预生成图再下采样。无交集或退化框忽略。可选 `instance_masks=[N,H,W]` 是与每个框对应、已对齐到浅层的 `[0,1]` 形状掩膜；该框内非空时相乘，空时退回该框的高斯。**不装 SAM 也可用**，此时是论文的 Gaussian 对照，不称作 Hybrid。`foreground_loss` 用软目标 sigmoid focal（`alpha=.25, gamma=2`）与逐图平滑 Dice，默认权重 `20:1`；在半精度输入下以浮点 32 位累加。
3. **择块。** [§III-C、算法 1/2、式 (5)](https://arxiv.org/html/2407.16424v2#S3.SS3)：概率 `>= threshold` 且为 `3×3` 局部极大值的格子代表候选中心，平台上的并列极大值均保留。`algorithm="greedy"` 对应算法 1：`9×9` 池化统计激活密度，大者先作为中心，以固定大小初始化块，把左上角移到窗内激活的最小 x/y（再限界），跳过已覆盖中心。`algorithm="grid"` 对应算法 2：从规则候选网格中保留含极大值的块，并行作同一调整。网格右/底不整除时也参与选取；调整后**仅合并完全相同的块**，有部分重叠但可能包含不同目标的块不丢弃。论文未规定去重 IoU，本实现不引入额外阈值。
4. **训练与推理。** 根据 [§III-D 末段](https://arxiv.org/html/2407.16424v2#S3.SS4)，预热期可用标注前景图选块，之后（包括推理）改用预测概率；整个训练期间仍监督 `logits`。预热长度、辅助 loss 相对检测 loss 的权重由适配层控制并记录。离散选择不反传；检测梯度沿真实特征切片回到共享浅层，前景梯度从辅助 loss 回到浅层。若无候选，`K=0`：训练仍计算前景 loss，推理返回该图无候选，不兜底整图、不借标注。某张图为空不影响同批其他图；全批为空时直接跳过后半段。分布式训练需由训练器处理空分支的未用参数；极小块要留意 BN 有效样本数。
5. **和论文完整系统的边界。** [§III-D、式 (6)](https://arxiv.org/html/2407.16424v2#S3.E6) 还在块**内部**只对候选空间位置执行稀疏检测头。这里可用 `foreground_peaks(probabilities)` 取得布尔中心图，但**没有实现或宣称运行逐点稀疏卷积、ViT token 裁剪**。样例在选中块上运行普通卷积；下游可以比较密集块头和另接的空间选点后端。择块、择层（DM-EFS）、逐点查询（QueryDet）不共享一个开关。

块高宽以浅层格为单位；论文用近似 `H/k × W/k`。这里不把数据集、通道、分辨率、`k`、步长或设备写死。在有多次下采样和上采样拼接的后半段，选能整除其相对下采样倍数的 `ph,pw`。块原点不强制对齐更粗的全图网格；后续网络是在块内重新采样，不能期待边缘或所有位置与整图卷积数值相等。

**资源与测量。** 浅层和前景头仍计算整图。块批的输入存储与 `K*C*ph*pw` 成正比；重叠部分会重复计算。`grid` 在输入设备上批量求候选、平移并去重；`greedy` 为可读、可核对的算法 1 参考路径，把掩膜移到 CPU 做顺序选择。在线生成框监督也有与框数有关的调度开销，外部可缓存对齐的监督。验收应在同设备、相同输入/精度/后处理下测 batch=1 端到端延迟、峰值内存和检测指标；只记录块数不是测速。

## 实验出处（作者结果，不是本目录的成绩）

Kai Liu 等，*ESOD: Efficient Small Object Detection on High-Resolution Images*，[IEEE TIP 34 (2025), 183–195](https://doi.org/10.1109/TIP.2024.3501853)，阅读版本 [arXiv:2407.16424v2](https://arxiv.org/html/2407.16424v2)。[§IV-C](https://arxiv.org/html/2407.16424v2#S4.SS3) 默认中型 YOLOv5 骨干和解耦头、两张 V100、50 epoch、batch size 8；VisDrone、UAVDT、TinyPerson 的输入长边分别为 1536、1280、2048。[表 IV](https://arxiv.org/html/2407.16424v2#S4.T4) 中同在高分辨率且不启用稀疏头时，不切块为 AP 38.1 / 412.2 GFLOPs，Ada 为 38.1 / 232.9，Sim 为 37.8 / 236.9；[表 V](https://arxiv.org/html/2407.16424v2#S4.T5) 单独比较 Gaussian/SAM/Hybrid，[表 VII](https://arxiv.org/html/2407.16424v2#S4.T7) 比较不同 `k`。这些是特定协议结果，不能作为本实现或新基线的 AP、加速比。

## 作者代码核对与许可证

核对 [alibaba/esod 固定提交 `bde3571bd7db697e441eef0278cd425e888ea026`](https://github.com/alibaba/esod/tree/bde3571bd7db697e441eef0278cd425e888ea026)，2025-06-06；仓库根 [LICENSE](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/LICENSE) 为 GNU GPL v3，论文 arXiv HTML 标为 CC BY 4.0。本目录按论文自行实现，无作者源码、权重、数据或第三方 SAM 代码拷贝。

| 已读固定版本文件 | 核对内容及与本提取的区别 |
| --- | --- |
| [模型配置](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/models/cfg/esod/visdrone_yolov5m.yaml)、[Segmenter 和还原](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/models/yolo.py) | `DWConv`、单通道 `Segmenter`、`HeatMapParser` 插入浅层；检测头用块偏移解码。这里把格距、投影和还原独立暴露。 |
| [HeatMapParser](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/models/common.py#L313-L560) | 发布版 `forward` **训练走均匀全切**，评估走 `ada_slicer_fast`；快速版按两侧空白选位移且 NMS 被注释。这里训练/推理均可用正文算法，使用正文向右/下的调整，显式去重。 |
| [掩膜制作](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/scripts/data_prepare.py)、[离散高斯](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/utils/general.py#L779-L798) | 作者在图像尺度准备标签，含 SAM 质量判断、离散高斯截断、部分数据集按类/面积加权。这里按式 (3)–(4) 在浅层格生成，不复刻那些启发式。 |
| [监督和块内标注](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/utils/loss.py#L248-L347)、[训练入口](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/train.py#L337-L346) | 作者 `compute_loss_seg` 实际启用可加权 BCE，focal/Dice 调用被注释；训练入口有预热传 GT 掩膜分支。这里按正文保留 `20 × focal + Dice`，块归属采用半开窗口而非特定 anchor 的半格偏移。 |
| [逐点位置与检测头](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/models/yolo.py#L62-L100)、[作者稀疏算子](https://github.com/alibaba/esod/blob/bde3571bd7db697e441eef0278cd425e888ea026/models/spconv.py) | 解释了作者如何复用中心并扩邻域。本提取未移植这些算子，块头计算和原文完整系统不能混报。 |

## 跑通接口

在仓库根目录运行：

```bash
/home/user/work/yolo11n-env/bin/python -m dronebench.methods.esod.example
/home/user/work/yolo11n-env/bin/python -m pytest -q -c /dev/null -p no:cacheprovider dronebench/methods/esod/test_esod.py
```

[最小样例](./example.py) 在合成数据上训练一次共享浅层、前景头和「每块一个框」的小卷积分支，再**不传标注**推理并还原框。该玩具头仅演示接口，不支持一块多目标；输出在 NMS 前，不是可用权重或性能验证。[测试](./test_esod.py) 另核对多图、空图、非整除矩形、越界补零、坐标往返、重叠标签和联合梯度。

本机实测：Python 3.13.5、PyTorch 2.9.1+cpu、pytest 9.2.0、`OMP_NUM_THREADS=1`，上述测试 **10 项通过**。样例在固定种子 7 下预热选 2 块，联合损失 11.581；不使用标注的推理选 12 块，恢复 12 个 NMS 前示范框，覆盖两个输入图号。这是通路验证，不是检测精度或延迟成绩。
