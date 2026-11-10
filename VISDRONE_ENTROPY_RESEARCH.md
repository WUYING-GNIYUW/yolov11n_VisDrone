# YOLO11n VisDrone 定向检索与模块审查

## 检索边界

检索对象包括 2024--2026 年的 YOLO/VisDrone 研究、无人机小目标/密集/遮挡检测研究，以及局部上下文、特征解耦、空间调制、频域建模、边界保持、动态卷积和跨层交互。优先核对了 IEEE、CVF、Springer、ScienceDirect、Sensors/Electronics/Applied Sciences 的正式元数据、论文摘要或全文，以及 QueryDet 官方代码仓库。没有把搜索摘要中的未核验指标、代码或数据划分写成事实。

在目标仓库中逐一读取了全部 VisDrone 实验分支。已覆盖的改动包括：

- `exp/visdrone-yolo11n-antialias`：没有可用的模型差异；
- `exp/visdrone-yolo11n-sppf-residual`：SPPF 残差/门控；
- `exp/visdrone-yolo11n-c2psa-centered`：C2PSA 局部中心对比残差；
- `exp/visdrone-yolo11n-source-route`：P3 C3k2 来源分离与细节旁路；
- `codex/visdrone-c3k2-crp`：C3k2 中间分支投影抑制；
- `codex/visdrone-c2psa-local-global`：C2PSA 局部深度卷积与全局 PSA 门控融合；
- `codex/visdrone-neck-diff-fusion`：Neck 差异图门控补偿；
- `codex/visdrone-combined` 与 `feature/visdrone-yolo11n-local-edge`：上述改动组合及 Conv 局部边缘调制。

这些分支被视为已尝试或重复边界；本实现没有复用其算子、插入位置或数据流。

## 论文核验摘要

| 研究 | 核验到的核心机制 | 与本实现的关系 |
|---|---|---|
| QueryDet, CVPR 2022, DOI `10.1109/CVPR52688.2022.01330` | 高分辨率 P2 与级联稀疏查询，面向小目标 | 本实现不增加 P2、不改 Head、不做稀疏查询；仅调制 YOLO11 原有 PSA 的 value 聚合 |
| YOLO-MARS, Sensors 2025, DOI `10.3390/s25082534` | ERAC、PD-ASPP、SGCS-FPN、WIoU；论文摘要明确针对 UAV 小、密、遮挡目标 | 本实现不加入 ASPP、浅层新分支或新损失；作用层级是 Backbone P5 C2PSA 内部 |
| RFAG-YOLO, Sensors 2025, DOI `10.3390/s25072193` | 感受野注意、FasterNet 改造、尺度注意融合 | 本实现不引入新的感受野卷积或跨尺度融合；熵来自已有 self-attention 的 N×N 权重 |
| GLDS-YOLO, Electronics 2025, DOI `10.3390/electronics14193831` | YOLO11n 变体，摘要列出 GSA、LSKA、DCNv4、SMSDE | 本实现不使用 GSA/LSKA/DCNv4/多尺度检测头；只改现有 C2PSA 的计算顺序 |
| Enhanced YOLO11 for tiny object detection based on multi-scale information interaction and fusion in UAV aerial images, JCDE 2026 | YOLO11 基线上的多尺度信息交互与融合 | 本实现不增加新的跨层融合路径，输入只来自现有 C2PSA 的 attention 矩阵；插入位置和信息交互方式不同 |
| SPTD-YOLO, Applied Sciences 2026, DOI `10.3390/app16126062` | P2、SPDConv、动态大核与任务对齐检测头 | 本实现不增加 P2、动态卷积或 Head 解耦，不能归类为该路线的复现 |
| A Lightweight Fusion Strategy With Enhanced Interlayer Feature Correlation, IEEE TGRS 2024, DOI `10.1109/TGRS.2024.3457155` | 论文题名指向跨层相关性融合，全文机制未在本轮复现 | 本实现没有跨层输入，故不把它作为直接先例或效果依据 |
| Feature disentanglement in one-stage object detection, Pattern Recognition 2024, DOI `10.1016/j.patcog.2023.109878` | 特征解耦/任务冲突处理 | 本实现不拆分类与回归特征，也不改 Detect head；仅按空间注意分布置信度调制上下文 |

本轮没有核验到“YOLO11/VisDrone + 注意力熵调制 value 聚合”的公开实现。该结论只表示在上述定向来源和仓库分支中未查到，不构成对全部文献的首创声明。

## 候选相似性审查

| 候选设计 | 最接近的论文 | 相同部分 | 不同部分 | YOLO11/VisDrone 中是否已有 | 是否建议实现 |
|---|---|---|---|---|---|
| C2PSA 熵置信度调制 | YOLO-MARS、RFAG-YOLO；通用 self-attention 熵统计 | 都利用注意/上下文改善小目标背景竞争 | 本方案从原有 PSA 的 N×N attention 计算归一化熵 `H(p)`，只对 value 聚合按查询位置缩放；不新增分支、卷积、层或损失 | 未见公开实现；仓库分支已有的是局部中心、局部/全局双支路，不是熵调制 | **建议实现** |
| C3k2 中间分支正交/冗余抑制 | Feature disentanglement；`codex/visdrone-c3k2-crp` | 都估计分支相关性并抑制冗余 | 论文/旧分支已在 C3k2 内做投影抑制；核心算子、顺序和输入输出相同 | **已有** | **重复或弱创新，不实现** |
| Neck 差异引导双路补偿 | IEEE TGRS 2024；`codex/visdrone-neck-diff-fusion` | 都利用跨层差异/相关性调节融合 | 旧分支已计算绝对差异、空间 gate 并补偿后 concat；核心算子、Neck 位置和张量来源相同 | **已有** | **重复或弱创新，不实现** |
| C2PSA 局部/全局门控融合 | YOLO-MARS 的多级上下文；`codex/visdrone-c2psa-local-global` | 都混合局部与全局上下文 | 旧分支已在 C2PSA 内加 DWConv、池化 gate；与该候选的输入、顺序和交互基本相同 | **已有** | **重复或弱创新，不实现** |

## 选定机制

选定 C2PSA 熵置信度调制，因为它只改一个现有算子的数据流，和已验证失败分支的算子及插入位置差异最大。每个启用的 PSABlock 只增加一个零初始化标量，保证预训练迁移开始时与原始路径等价。

设 C2PSA 的 PSA 分支输入为 `b ∈ R^(B×C×H×W)`，Attention 得到每个 head 的权重 `A ∈ R^(B×h×N×N)`，其中 `N=H·W`，且 `A_{q,:}` 是查询位置 `q` 对所有 key 的概率分布。定义：

`H_q = -Σ_k A_{q,k} log(A_{q,k} + ε)`

`r_q = clip(1 - H_q / log(N), 0, 1)`

对 head 求平均后得到 `r ∈ R^(B×1×N)`，原始 value 聚合为 `V A^T`，修改为：

`Y_q = [1 + g·(2r_q - 1)] · (V A^T)_q + PE(V)_q`

其中 `g=tanh(γ)`，`γ` 是每个启用 PSABlock 的零初始化可学习标量，`PE` 是 YOLO11 Attention 已有的 depthwise 3×3 位置编码。低熵表示注意力集中，高熵表示注意力分散；模型学习 `g` 的方向和幅度来增强或抑制全局混合，同时保留原位置编码和 C2PSA 的旁路 `a`。`γ=0` 时严格退化为原始 Attention：

`C2PSA(x) = cv2(concat(a, PSA_entropy(b)))`。

输入输出均保持 `B×C1×H×W`，本实验 YOLO11n 的 P5 为 `B×256×20×20`（640 输入、10 类）；隐藏 PSA 分支为 `B×128×20×20`。每个启用的 PSABlock 增加 1 个标量参数，额外计算只包括 attention 概率的 `log`、求和和逐元素缩放。

## 最小消融

固定 VisDrone2019 的 train/val/test-dev 划分、640 输入、同一 YOLO11n 预训练权重、优化器、增强、epoch、batch 和随机种子，至少运行：

1. 原始 10 类 YOLO11n；
2. 仅在 P5 C2PSA 启用 `entropy_gate=True`；
3. 熵调制强度的无调制对照（当前实现的 `False` 路径）；
4. 仅替换熵调制位置编码/仅保留 `r_q` 统计的诊断对照。

报告 Precision、Recall、mAP50、mAP50:95，并按小目标尺寸、每图目标密度和遮挡等级分桶；不把未训练的配置写成精度提升。

