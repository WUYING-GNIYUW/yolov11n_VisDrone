# YOLO11n / VisDrone 定向检索与 C2PSA 结构改进审查

## 范围与已排除的本地方向

本 worktree 从主 checkout 的 `55989f76c1e85c192ba12cb3e533e3de842a2301` 建立。主 checkout 的既有修改未清理。已检查的本地实验分支为：

| 分支 | 已改模块/位置 | 本轮处理 |
|---|---|---|
| `exp/visdrone-yolo11n-sppf-residual` | Backbone P5 的 SPPF 输入残差增益 | 排除，不复用 SPPF/残差增益 |
| `exp/visdrone-yolo11n-source-route` | Neck P3 的 C3k2 来源分离与细节旁路 | 排除，不复用 C3k2 来源路由/旁路 |
| `exp/visdrone-yolo11n-antialias` | Backbone P2→P3 stride-2 Conv 前固定 3×3 低通 | 排除，不复用抗混叠/下采样滤波 |
| 已删除的 `exp/visdrone-yolo11n-combined` | 三者合并 | 已归档结果后删除 worktree 和分支 |

三合一训练结果归档于 `/root/home/2112406148/visdrone_combined_deleted_archive_20261105`。其最后一轮为 P/R/mAP50/mAP50-95 = `0.45476/0.32489/0.31822/0.17450`；这与不同 batch 的基准不能作为因果比较。

## 定向检索

已核验或在已有报告中保留原始元数据的近年研究：

* RFAG-YOLO，Wei and Wang，*Sensors* 25(7), 2193 (2025)，DOI `10.3390/s25072193`：YOLOv8/VisDrone；RFN 动态核、FasterNet 主干和 SAF 跨层尺度融合。它不是 YOLO11 的 C2PSA，也不是本实现的固定局部均值残差。
* YOLO-MARS，Zhang et al.，*Sensors* 25(8), 2534 (2025)，DOI `10.3390/s25082534`：YOLOv8n/VisDrone；ERAC、PD-ASPP、SGCS-FPN 和 WIoU，涉及密集、遮挡小目标，但改动在多级主干/Neck/损失。
* GLDS-YOLO，Ju et al.，*Electronics* 14(19), 3831 (2025)，DOI `10.3390/electronics14193831`：YOLO11n/VisDrone；GSA、LSKA、DCNv4、SMSDE。摘要确认了模块名和数据集，未把其具体内部实现冒充为当前算子。
* AED-YOLO11，Gong et al.，*Digital Signal Processing* (2025)，DOI `10.1016/j.dsp.2025.105411`：YOLO11 小目标检测，Crossref 元数据已核验；摘要/元数据不足以确认与本地 C2PSA 同构。
* LSF-YOLOv11，Wang，ISCAIT 2026，DOI `10.1109/iscait69154.2026.11477258`：题名指向 VisDrone 的轻量特征融合；公开元数据未给出可用于等价性判断的完整算子。
* QueryDet，Yang et al.，CVPR 2022，DOI `10.1109/cvpr52688.2022.01330`，官方 `QueryDet-PyTorch`：VisDrone 高分辨率稀疏查询；作用在高分辨率区域和检测计算分配，不改变 YOLO11n 的 C2PSA。
* *Salient object detection from low contrast images based on local contrast enhancing and non-local feature learning*，*The Visual Computer* (2021)，DOI `10.1007/s00371-020-01964-9`：检索到的局部对比/非局部特征相关 CV 先例。它处理显著性检测，不能被称为当前 YOLO11/VisDrone 实现。
* *Localization Uncertainty-Based Attention For Object Detection*，ICIP 2021，DOI `10.1109/icip42928.2021.9506031`：检索到的不确定性注意先例；没有证据表明其算子、顺序或插入层级等同于本实现。
* *Truly Shift-Invariant CNNs*，CVPR 2021，DOI `10.1109/CVPR46437.2021.00377`：抗混叠/平移稳定背景，已用于排除本地抗混叠分支，而不是本轮方案依据。

检索同时覆盖 YOLOv8/YOLOv10/YOLO11 与 VisDrone 研究、无人机小目标/密集/遮挡研究及局部上下文、特征解耦、空间调制、频域/边界保持、动态卷积和跨层交互关键词。对于只拿到题名或摘要的论文，只记录已证实的元数据和模块名，不推断未公开的内部算子、代码或实验口径。

## 候选相似性审查

| 候选设计 | 最接近的论文 | 相同部分 | 不同部分 | YOLO11/VisDrone 中是否已有 | 是否建议实现 |
|---|---|---|---|---|---|
| 在现有 P5 `C2PSA` 的 PSA 分支前做 3×3 局部均值中心化，并以零初始化标量残差调制 | 局部对比增强显著性检测；GLDS 的 LSKA（仅问题动机相近） | 都利用邻域对比或局部/上下文互补来减弱背景干扰 | 本实现只有固定 `avg_pool2d`、`b-local_mean` 和一个标量 `tanh(g)`；顺序是 `cv1→split→局部中心化→原PSA→cv2`；输入输出仍为同一 P5 张量；不加注意力、卷积、Neck 节点或 Head | 未在已核验的 YOLO11+VisDrone 资料中发现同一算子/顺序/层级；不能证明不存在公开实现 | **建议实现，称待验证的结构改进** |
| 用每样本通道均值/方差对白分支做可学习标准化后再进入 PSA | 特征解耦/去相关类检测工作（Crossref 检索到 *A feature binding model in computer vision for object detection*, DOI `10.1007/s11042-021-10702-9`） | 都改变通道统计，试图减少特征竞争 | 需要样本统计与额外缩放，核心是通道归一化而不是空间局部对比；与已有归一化/解耦方法边界不清 | 未证明 YOLO11/VisDrone 中同构，但算子重合风险高 | **不建议，重复或弱创新** |
| 在 C2PSA 全局响应旁再放局部卷积/窗口注意并学习混合权重 | Swin Transformer；GLDS 的 LSKA；已有 YOLO VisDrone 注意力改造 | 都在局部与全局交互间混合 | 会新增局部注意/卷积支路，运算顺序和交互拓扑接近已有工作；早期本地报告已将其判为高相似 | 已有多项 YOLO11/VisDrone 注意力方向，且容易退化成模块堆叠 | **不建议，重复或弱创新** |

候选一仍不能宣称原创；它只是在原有 C2PSA 的既有 PSA 交互前改变一个分支的局部统计，差异来自算子顺序、固定局部均值、P5 单点插入和零初始化，而不是新注意力范式。

## 实现说明

对输入 `x∈R^{B×256×20×20}`（640 输入的 YOLO11n P5），原 C2PSA 为：

\[
(a,b)=\operatorname{split}(\operatorname{Conv}_{1×1}(x)),\quad b'=\operatorname{PSA}(b),\quad y=\operatorname{Conv}_{1×1}([a,b']).
\]

本实现只在 PSA 前加入：

\[
\mu_3(b)=\operatorname{AvgPool}_{3×3}(b),\quad
\tilde b=b+\tanh(g)(b-\mu_3(b)),\quad g_0=0,
\]

然后以原来的 `PSA(\tilde b)` 和 `cv2([a,·])` 完成输出。P5 的隐藏通道为 128，所有输入输出张量形状不变；`local_center=False` 保持原始 C2PSA 行为。

理论上该分支针对局部背景相近、目标边缘弱和密集目标相互遮挡的情况，给 PSA 一个可学习强度的局部对比信号。副作用是可能放大纹理噪声或边缘，且固定平均池化对很小目标可能造成过度平滑；因此必须以 APsmall、密度/遮挡分桶召回和真实设备延迟验证。

候选模型的额外参数为 1 个标量；平均池化的理论开销约为 `9×128×20×20` 次 MAC（约 0.00046 GMAC，按 2 FLOPs/MAC 约 0.00092 GFLOPs），实际 kernel 延迟需在 CUDA 恢复后测量。最小消融为同一预训练权重、imgsz=640、相同 batch/epoch/seed 的 `C2PSA baseline` 对 `local_center=True`，再用 3 个 seed 报告 mAP50、mAP50-95、APsmall、密集/遮挡召回、参数量、GFLOPs 和 batch=1 延迟。

## 文件与验证

* `ultralytics/nn/modules/block.py`：扩展已有 `C2PSA`，没有新增注册类或改动 `tasks.py`。
* `ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-centered.yaml`：只给第 10 层 C2PSA 传入 `local_center=true`。
* `ultralytics/cfg/models/11/yolo11n-visdrone-baseline.yaml`：同拓扑基准对照。
* `ultralytics/cfg/datasets/VisDrone_local.yaml`：指向现有转换后的 VisDrone 数据。

已完成 CPU 验证：模型构建、416/640 dummy 前向、P3/P4/P5 尺寸匹配、反向传播和有限梯度；局部增益梯度为有限值。参数量为基准 `2,591,790`、候选 `2,591,791`。THOP 在 640 输入下两者均报告 `3,225,139,200` MAC，即 `6.4502784 GFLOPs`（其默认统计不计入该平均池化）；因此不把这个数当作精确增量。`git diff --check` 和 Python 编译检查通过。

在 CPU 单线程、640×640、batch=1、交错运行 50 次（各模型预热 5 次）的配对测量中，基准中位数为 `228.594 ms`，候选为 `229.968 ms`，差值 `+1.375 ms`。这是当前 CPU 环境的工程参考，不替代 CUDA 或目标设备延迟。

本环境当前 `torch.cuda.is_available()` 为 `False`，`nvidia-smi` 无法连接驱动；没有伪造 CUDA 延迟或 GPU 训练结果。
