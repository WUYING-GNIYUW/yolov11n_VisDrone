# QueryDet：先找大概位置，再在细网格上算

细网格有助于分辨相邻的小结构，但在每个空白位置都跑一遍预测头也很贵。QueryDet 在粗网格上问“这附近有没有值得细看的东西？”，不要求此时就能分清类别或画准边界。回答“有”的格子传给细网格；细网格只在这些位置报告结果。还可以从中挑点再往下传。这是**选空间位置**，不是选择整张特征层，也不是把图像裁块后重跑骨干。

## 输入怎样变成输出

输入是同一幅图或同一批图的多层特征 `B×C×H×W`，层间必须知道格子的对应关系。要省的是**细层预测头**的计算；粗层查询、已经生成的骨干和多尺度特征不会因为这套代码变稀疏。

| 步骤 | 做什么 | 留给下一步的东西 |
| --- | --- | --- |
| 空间选点 | `select_queries` 在起始层读取单通道查询分数，只保留 `sigmoid(logit) > threshold`；`map_children` 把入选父格投到下一层。 | 要出预测的细层中心 `keys[K,3]`。它们不是框，更不是最终分类分数。 |
| 上下文扩展 | `expand_context` 给每个中心加一圈邻居，将重叠位置取并集。半径为 2 即单点最多 5×5 格。 | 可供卷积取值和计算的 `active[M,3]`；邻居不是额外的检测中心。 |
| 稀疏执行 | `SparseQueryPlan.run` 从原特征只取 `active` 行，查找同图、同层的核邻居；缺失邻居给零。在这些行上复用传入卷积的权重及偏置，只返回 `keys` 对应的行。 | `K×C_out`，顺序与 `keys` 一致。分类、框回归、查询头可共用同一张坐标计划。 |
| 级联 | 若还有更细层，对当前 `keys` 的查询 logit 再过阈值，再传子格。无点即无下游预测，不回退成全图。 | 各细层的独立预测行和坐标；最后一层不用再算查询。 |

**坐标契约。** 所有点用 `int64` 的 `(batch, y, x)`，网格从零计数；`xyxy` 框则是连续像素坐标、横轴在前。原文式 (1) 的 2 倍映射是 `(b,y,x) → (b,2y+i,2x+j)`，`i,j∈{0,1}`。丢弃越过实际 `[0,H)×[0,W)` 的子格，不钳位复制、不从行尾绕到下一行。比如父格 `(0,1,2)` 对应 3×5 细层的右下角，四个候选只剩 `(0,2,4)`；半径 1 时需要计算的邻居含自身共有 4 个，但只对这一个中心报告预测。映射和扩展均去重、按 `(b,y,x)` 排序，各图互不借值。`SparseQueryPlan` 可接收任意顺序的有效 `keys`，但每个 key 必须在 `active` 里。

默认层间同原点、整倍步长。`cascade_plans` 检查 `H_coarse=ceil(H_fine/scale)`（宽同理）；允许末行、末列不足一整组，不用对坐标插值。`scale` 是整数步长比，2 对应原式；其他整数是扩展。若中间做了裁切、错位填充或仿射变换，先由适配层把点变到同一原点。这里的边界是张量边界；每图有效区小于填充张量时，调用方可在扩展前剔除无效 keys，并从 `active` 中剔除无效邻居。

## 训练与推理不要混用

- **训练用稠密图。** 在每个将要发出查询的层对整张特征运行 `QueryHead`。每图传入当前缩放、填充后的 `N_i×4` 像素框（无框传 `empty(0,4)`）。`query_targets` 先按 `lo ≤ sqrt(w*h) < hi` 选框；非有限坐标、非正宽高不参与。格子中心是 `((x+0.5)*stride_x,(y+0.5)*stride_y)`，找离它最近的合格框中心。没有这种框，全图为负。
- **两种半径有不同依据。** 论文 §3.2–3.3、式 (2)–(3) 描述固定距离门槛：传 `radius_px`，当最近距离严格小于该像素半径才为正。复现文中某一层时，需将尺寸和距离门槛换到同一像素尺度。作者代码则先取最近框，再以**该框**的 `sqrt(w*h)*radius_factor` 判正；这里省略 `radius_px` 即走这条路，不等于“所有不同半径圆盘取并集”。`radius_px` 和 `radius_factor` 都不是推理用的上下文格数。目标在 FP32（或 FP64）中计算、返回与 logits 相同的形状及 dtype；网格分段计算到所有合格框的距离，不保留整批大距离矩阵。
- **损失。** `query_focal_loss` 是二元 sigmoid focal loss 的全图均值：正例权重 `alpha`、负例 `1-alpha`，再乘 `(1-p_t)^gamma`，其中 `p_t` 是真值一方的概率。每层权重在外部相乘再相加。原文式 (4)–(5) 还保留原任务的分类/回归监督，并讨论新增高分辨率层带来的层权重变化。本模块**不**改检测损失、样本分配和解码；要测试层权重，把它作为单独变量。训练不拿硬阈值裁掉难例。
- **推理不用真值。** 起始查询层稠密预测，更细的查询头和任务头才按点执行。起始或更粗的常规预测由宿主保留。输出是原网格上的预测值，解码、反缩放、合并重复预测由调用方处理。注意 `plan.active` 在这里是卷积支持坐标；给任务解码器的有效输出名单是 `plan.keys`。

## 可复用接口与代价

```python
from dronebench.methods.querydet import (
    QueryHead, SparseQueryPlan, cascade_plans, expand_context, map_children,
    query_focal_loss, query_targets, select_queries,
)

logits = coarse_query(coarse_feature)
targets = query_targets(logits, boxes_xyxy, stride=16, scale_range=(0, 64))
loss_query = query_focal_loss(logits, targets, alpha=0.25, gamma=1.3)

parents = select_queries(logits, threshold=0.15)
hw = fine_feature.shape[-2:]
keys = map_children(parents, hw, scale=2)
active = expand_context(keys, hw, radius=2)
plan = SparseQueryPlan(keys, active, (fine_feature.shape[0], *hw))
rows = plan.run(task_head, fine_feature, chunk_size=256)
```

- `QueryHead(in_channels, hidden_channels=None, depth=4)`：`depth` 个 3×3 卷积和 ReLU，再接 3×3 单通道卷积；层数、通道和是否跨层共享由调用方定。`stride` 可为标量或 `(sy,sx)`；`scale_range`、训练正例范围、focal 参数不依赖类别或数据集。推理时，阈值越高通常点越少，但被错过的父格不可能靠它的子格补回来。
- `cascade_plans(features, query_heads, *, threshold=.15, radius=2, scales=None)` 是上面步骤的循环：输入从粗到细，`L` 张特征配 `L-1` 个查询头，返回每个细层的 plan；第 `i` 个 plan 对应 `features[i+1]`。想逐层指定不同阈值或加入其他选择条件，可以用上面的原语，下一步只调用 `select_queries(plan.run(next_query, fine_feature), threshold, plan.keys)`。
- 后端支持单层或嵌套 `nn.Sequential`：原生 `Conv2d`（奇数核、步长 1、保持尺寸的零填充，包括分组、深度可分组和空洞卷积）、使用固定运行统计的 `BatchNorm2d.eval()`、`ReLU/LeakyReLU/SiLU/Identity`。本实现不猜自定义模块内部的数据流，也不把训练态 BN、空间归一化、池化或注意力悄悄改成稠密推理。不支持的头会报错。调用推理时用 `eval()` 和 `torch.no_grad()`。
- 设层上有 `BHW` 格，卷积实际算 `M=len(plan.active)` 个输出，不是只算 `K=len(plan.keys)` 个。对核面积 `A`，每层乘加约为 `M*A*Cin*Cout/groups`；同一 plan 为相同核和空洞率复用 `M×A` 邻居表。按 `chunk_size` 个输出位置分批收集一个核窗口/点，再做无填充卷积得到每窗一个输出；1×1 非分组卷积直接对行做线性变换。**没有先在完整细网格上做卷积再乘 mask。** 只有数值校验用稠密的逐层置零参考。上下文比头的感受野小，数值可不同于完整稠密卷积；上下文覆盖整个感受野时，中心输出应与同权重稠密头一致。
- 邻居查找、去重、收集、分批、仍然稠密的特征生成及宿主解码都耗时。若 `M` 接近 `BHW` 或预测头本来就很便宜，真实耗时可能**更慢**。点执行是可检验的计算路径，不是速度承诺。

## 什么时候值得试，以及怎样证伪

| 可观测条件 | 用什么接口或标注检验 | 哪种结果会推翻预期 |
| --- | --- | --- |
| 粗层未必能分类，但能区分“附近有无小结构” | 在固定尺度分桶的同一组标注上，扫查询阈值，记父格命中率及最终 keys 对目标中心的覆盖率。 | 阈值稍升就大量漏中心；或不得不查询几乎整图才保住召回。 |
| 细层预测头占用可观时间，目标分布不是满图都是 | 每图每层分别记 `K/BHW`、`M/BHW`，在同一机器上比较稠密头、稠密后筛点、真点执行。对相同输入预热、同步计时，报告 batch=1 端到端延迟和分位数。 | keys 很少但邻域并集几乎铺满；或查表/后处理时间大于省下的卷积时间。 |
| 可用有限邻域做局部判断，并能在不同层追踪同一位置 | 对同一批 keys 改半径，比较稠密参考的中心预测差、任务指标和时间。分别统计边缘、拥挤区、空图。 | 必须看到全图才能区分，或层间错位导致系统性漏边。此时加更多查询层也解决不了该问题。 |

跨任务尝试（不是论文结果）：例如在工业表面检查中，粗图能标出疑似细裂纹附近的格子时，可让细层只在这些格子回归端点；检查端点召回，而不只数点。在边缘稀疏的分割任务中，可以保留完整粗分割，仅在疑似边界处预测细分辨率修正；没被选中的内部仍需粗结果填充。对于遍布整图的细纹或必须看整幅图才能定性的对象，先量覆盖率及头部耗时再决定，不预设会有收益。

## 实验出处：只作参照，不是本工程的结果

[Yang、Huang、Wang，CVPR 2022，pp. 13668–13677](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)。[正文](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf) §3.2、图 3、式 (1) 给级联；§3.3、式 (2)–(5) 给监督；§4.4 表 4–6 比较起始层、执行方式及上下文。表 5 将真正的稀疏查询与裁补丁、完整卷积后筛点分开比较。

表 2 是作者在 **VisDrone 验证集**上的检测实验，主要模型为 RetinaNet、ResNet-50/FPN，实验设置见 §4.1（2080 Ti，中间特征层及训练设置与本模块无绑定）。

| 表 2 的行 | AP | FPS |
| --- | ---: | ---: |
| RetinaNet | 26.21 | 2.63 |
| QueryDet 高分辨率、不做 CSQ | 28.35 | 1.16 |
| 同一高分辨率模型、做 CSQ | 28.32 | 2.75 |

比较稀疏执行应看后两行：作者报告速度从 1.16 到 2.75 FPS，AP 从 28.35 到 28.32，不是说打开一个查询开关就比任何基线快 2.3 倍。表 1 是 COCO，不能替代表 2。这些数字都不是本参考后端、其他宿主或任意组合的测量。

## 溯源、许可和实现差异

核过的[作者仓库固定提交 `feebf218d53d59ba054132dfa6ef84159f793967`](https://github.com/ChenhongyiYang/QueryDet-PyTorch/tree/feebf218d53d59ba054132dfa6ef84159f793967)；[上游根许可证](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/LICENSE) 是 MIT，Copyright (c) 2022 Chenhongyi Yang。部分检测器文件另有 Facebook 版权头。本目录按上述论文公式、算法和可核查行为独立写成，不拷入作者检测框架、权重、评测代码或 `spconv`；上游的 MIT 标识不代替本项目的[根许可证](../../../LICENSE)。

| 对照文件（固定提交） | 对照点及本实现的取舍 |
| --- | --- |
| [qinfer.py](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/models/querydet/qinfer.py) | 作者以 `SubMConv2d` 计算上下文并取查询中心，该快照稀疏路径要求 batch=1。本实现以 PyTorch 坐标表查邻居，从头到尾保留批编号；不构造 spconv 模块、不把稀疏输出转回整张图，不宣称达到其 CUDA 吞吐。 |
| [det_head.py](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/models/querydet/det_head.py)、[detector.py](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/models/querydet/detector.py) | 对照查询卷积、密集训练、查询 focal loss、稠密/稀疏检测头分工。这里把监督与执行分开；可传宿主的任务头而不绑定 RetinaNet 锚框、类别、NMS。 |
| [utils.py](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/utils/utils.py)、[anchor_gen.py](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/utils/anchor_gen.py) | 对照 `sqrt(area)`、最近中心距离及步长网格；查询锚点的半格偏移由上条检测器显式设为 0.5。这里输入为张量，不要求 Detectron2 的 Boxes 类型。 |
| [VisDrone 训练配置](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/configs/visdrone/querydet_train.yaml)、[推理配置](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/configs/visdrone/querydet_test.yaml)、[通用默认值](https://github.com/ChenhongyiYang/QueryDet-PyTorch/blob/feebf218d53d59ba054132dfa6ef84159f793967/configs/custom_config.py) | 上游示例从 P4 查询 P3、再查 P2；P3/P4 训练筛选 `[0,32)`/`[0,64)` 像素目标，半径系数 1，查询 γ=1.3、层权重各 10。§4.1 写查询阈值 0.15，而该测试配置给 0.12、`QUERY_INFER: False`，并非加载配置就启用稀疏；通用上下文半径为 2。本模块默认 focal γ=2，可显式传 1.3；不硬编码任何实验的尺度范围或层权重。 |

## 最小复现

在仓库根目录执行；模块只依赖 PyTorch，测试另需 pytest，不用模型文件、数据下载或额外编译。示例用可读的合成信号做一次真实反向与更新，再演示两次投点，不代表训练后的准确率。

```bash
python -m dronebench.methods.querydet.example
OMP_NUM_THREADS=1 python -m pytest -q -c /dev/null -p no:cacheprovider dronebench/methods/querydet/test_querydet.py
```

[最小样例](example.py)、[测试](test_querydet.py)、[选点与监督](query.py)、[实际执行后端](sparse.py)、[导入接口](__init__.py)。测试包含实际卷积和反向、奇数边界、批隔离、空集、固定/随框变化的半径、两级查询、稀疏与受限支持的稠密数值参考，并用 CPU profiler 检查卷积真正收到的是选中位置的核窗口。

本地实跑：Python 3.13.5、PyTorch 2.9.1+cpu、pytest 9.2.0，`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，上述 7 项测试全部通过。样例两细层分别输出 4、4 个中心，预测头可用的上下文各为 16 格，两张网格分别有 63、252 格；末层任务头返回 `4×6`。这是执行路径的验证，不是端到端测速。
