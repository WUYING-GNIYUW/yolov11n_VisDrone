# YOLO11n / VisDrone 定向检索与实施前审查（2026-11-17）

## 结论与证据边界

最终选择两个独立实验：SPPF 的跨尺度互斥峰值选择、C2PSA 的局部值变化预算。
它们是有明确数据流差异、尚待训练验证的设计；不宣称首创、论文级创新已成立或 mAP 必然提升。
时间窗为 2023-11-17 至 2026-11-17，较早的通用算子作为先例另外列出。
基准是 origin/main 的 55989f76c1e85c192ba12cb3e533e3de842a2301，Ultralytics 8.4.198。

网页搜索工具未返回可读结果，实际证据来自 curl 直接读取 arXiv API/HTML、出版方登记的 Crossref
元数据、Springer 页面以及作者官方 GitHub 源码。摘要级核验不冒充全文核验。
GitHub 仓库检索仅覆盖仓库元数据；代码搜索受 API 限流，不能证明不存在同构公开实现。
部分宽泛 Crossref 查询返回 429；对关键 DOI 的独立读取成功。

早期 GRN/XCA 接口试作在最终创新性复核中被否决并移除：归一化/协方差算子和运算顺序与官方实现一致，
迁入 C3k2/C2PSA 或为旧 QKV 通道分组不能成为充分的机制创新。试作验证结果不算最终实现验证结果。
以下两项机制的比较与公式在最终模型代码写入前确定。

## 全部分支审计

2026-11-17 已 fetch origin/main，并从 GitHub branches API 核对全部 16 个真实分支；排除 origin/HEAD 别名。
本地两个实验分支与对应远端头相同。读取模块 diff、模型 YAML、已有审查记录；组合分支与单项分支交叉比较。
指标状态来自用户反馈，未重新训练核验。

| 分支（省略 origin/）                      | 提交      | 机制 / 位置                                         | 处理                                            |
| ----------------------------------------- | --------- | --------------------------------------------------- | ----------------------------------------------- |
| main                                      | 55989f76c | 原始模型                                            | 两个新分支的共同起点                            |
| codex/model-change-marker                 | 55989f76c | 与 main 相同                                        | 不凭名称虚构改动                                |
| codex/visdrone-lcr-c3k2                   | 55989f76c | 与 main 相同                                        | 同上                                            |
| codex/visdrone-lg-c2psa-window-global     | 55989f76c | 与 main 相同                                        | 同上                                            |
| exp/visdrone-yolo11n-antialias            | 55989f76c | 与 main 相同                                        | 同上                                            |
| exp/visdrone-yolo11n-sppf-residual        | 8e8aef0b1 | P5 SPPF 最终输出加输入旁路、通道 gain、激活选项     | 不重复输出残差门控                              |
| exp/visdrone-yolo11n-source-route         | d7e2d43c7 | Neck P3 C3k2 分组投影、来源分离及细节旁路           | 不重复来源拆分                                  |
| exp/visdrone-yolo11n-c2psa-centered       | 46ddb61eb | C2PSA 中 b-AvgPool3(b) 的可学习注入                 | 不重复局部高通残差                              |
| codex/visdrone-c3k2-crp                   | 6fb9bf3e9 | C3k2 中间分支减去在旁路上的通道投影                 | 不重复正交/去冗余投影                           |
| codex/visdrone-c2psa-local-global         | 4bd2c0cb5 | DWConv 与 PSA 经池化门控混合                        | 不重复局部/全局双分支                           |
| codex/visdrone-neck-diff-fusion           | dc0dd9923 | 两来源绝对差、sigmoid、双向加性补偿、concat         | 不重复差异门控                                  |
| codex/visdrone-combined                   | 27babcf3d | 上述 CRP、local/global、差异融合组合                | 不重复组合                                      |
| feature/visdrone-c2psa-entropy-confidence | 4501ba9c7 | query 注意力熵缩放聚合 value                        | 不重复熵置信门控                                |
| feature/visdrone-yolo11n-local-edge       | 1c585c1b9 | P3 stride-2 Conv 前的方向差分幅度调制               | 不重复边缘增强                                  |
| C2PSA-20261116                            | 749a3e972 | CSP 旁路决定 key 接收容量，列限流，剩余质量回流自身 | 用户反馈约 0.5% mAP50 提升；新方案不改 key 容量 |
| Concat-20261116                           | 106b8bddd | 横向相似度重组语义，2×2 相位零均值投影              | 用户反馈约 0.5% mAP50 提升；不重复均值守恒重组  |

核对入口：https://api.github.com/repos/WUYING-GNIYUW/yolov11n_VisDrone/branches?per_page=100

## 文献与公开实现核验

| ID  | 文献 / 来源                                                                                                             | 实际读取层级                       | 核验内容与影响                                                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| R1  | YOLO-MAG，Scientific Reports，2026-10-27，10.1038/s41598-026-76441-9                                                    | 出版方登记摘要                     | YOLO11+VisDrone；C3k2-MEIS、C2PSA-ACmix、双边门控 GAC-FPN；不再将边缘或局部/全局混合当创新                  |
| R2  | PSF-YOLO11，Measurement Science and Technology，2026-06-17，10.1088/1361-6501/ae71f9                                    | 出版方登记摘要                     | 并行多尺度卷积、小目标金字塔、共享权重融合；与候选的选择排斥/值域预算不同，不能据摘要排除全文中的所有相似性 |
| R3  | YOLO-MARS，Sensors，2025-04-17，10.3390/s25082534                                                                       | 出版方登记摘要                     | YOLOv8n+VisDrone，小/密/遮挡；ERAC、PD-ASPP、SGCS-FPN、WIoU；上下文扩展和浅层引导已存在                     |
| R4  | FDM-YOLO，2025，arXiv:2503.04452                                                                                        | 作者摘要                           | YOLOv8+VisDrone；Fast-C2f/PConv、P2/-P5、DySample、EMA；简单插入现成模块不实施                              |
| R5  | Small Object Detection in Drone Aerial Imagery with LAF-YOLOv10，2026-09-13，arXiv:2609.14560                           | 作者摘要和代码入口                 | PC-C2f、AG-FPN、P2/-P5、Wise-IoU；报告组合失败及预训练迁移问题，支持独立消融而非假定叠加有效                |
| R6  | YOLO-Drone: An Efficient Object Detection Approach Using the GhostHead Network for Drone Images，2025，arXiv:2511.10905 | 作者摘要                           | YOLOv11n+VisDrone 的 GhostHead；本次不修改 Head                                                             |
| R7  | ScopeMamba-YOLO，2026，arXiv:2609.10156                                                                                 | 作者摘要                           | 旁路零门控选择扫描、CGCM、SS-PAN、AMS、SA-DFL；不把“旁路+零初始化”本身当创新                                |
| R8  | Getting the Numbers Right—Modelling Multi-Class Object Counting in Dense and Varied Scenes，arXiv:2510.02213            | 作者摘要                           | 密集遮挡的类别密度估计、辅助分割，含 VisDrone；属于计数，不能用作检测模块涨点证据                           |
| R9  | Top-K Pooling with Patch Contrastive Learning for Weakly-Supervised Semantic Segmentation，arXiv:2310.09828             | HTML 正文 §3.2、公式               | 每类别最高 K 个 patch 分数取平均，patch→image；与 SPPF 候选最接近的选峰先例；首发早于本时间窗，作为通用先例 |
| R10 | YOLOV5s object detection based on Sim SPPF hybrid pooling，Optoelectronics Letters，2024，10.1007/s11801-024-3170-x     | Springer 书目信息；全文未取得      | SPPF 池化改造先例；只核验题名/日期，不根据题名虚构算子顺序或 VisDrone 使用情况                              |
| R11 | Elliptical Attention，arXiv:2406.13770                                                                                  | HTML 正文 §2–3、公式；官方代码入口 | Mahalanobis 度量及坐标变化估计作用于 QK 相似度；不同于候选在 AV 后约束 O-V                                  |
| R12 | Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation，arXiv:2403.16990                             | HTML 正文 §5、附录说明             | 文本/布局指导的注意掩码限制主体间传播；不把“bounded attention”名称作为候选创新，不使用框或语义 mask         |
| R13 | ConvNeXt V2 / GRN，CVPR 2023，arXiv:2301.00808                                                                          | 官方 models/utils.py               | 空间 L2 范数→通道均值归一→gamma/beta 残差；直接迁入 C3k2 判弱创新                                           |
| R14 | XCiT: Cross-Covariance Image Transformers，arXiv:2106.09681                                                             | 官方 xcit.py 的 XCA                | Q/K 空间归一、通道矩阵、temperature、乘 V；直接迁入 C2PSA 判弱创新                                          |
| R15 | Frequency-Aware Feature Fusion for Dense Image Prediction，TPAMI 2024，10.1109/TPAMI.2024.3449959                       | DOI 元数据及官方 FreqFusion.py     | ALPF/AHPF、相似度引导重采样；直接用来替换 Neck 不满足本次创新性门槛                                         |
| R16 | Detail-Preserving Pooling in Deep Networks，CVPR 2018，arXiv:1804.04076                                                 | 作者摘要                           | 自适应保细节池化；不是“首次保留小目标细节”的依据，候选不能声称池化保细节为新问题                            |

主要检索式包含 `VisDrone AND (YOLO11 OR YOLOv11 OR YOLO11n OR YOLOv8 OR YOLOv10)`、
`pooling AND (rank OR peak OR distinct OR without replacement)`、`attention AND (limiter OR value clipping OR trust region)`、
`polyphase`、`SPPF hybrid pooling`。另外读取 GRN、XCA 和 FreqFusion 作者代码。
GitHub 仓库查询 `YOLO11 VisDrone pooling`、`VisDrone attention limiter`、`YOLO exclusive pooling` 返回 0 个仓库；
这是索引字段范围内的结果，**不是全站代码搜索或不存在同构工作的证明**。

可复核来源：

- R1–R3、R10、R15：`https://api.crossref.org/works/<DOI>` 及对应 DOI 出版方页面。
- R4–R9、R11–R14、R16：`https://arxiv.org/abs/<ID>`；本次正文通过 `https://export.arxiv.org/html/<ID>` 读取。
- R5 作者代码：https://github.com/Mudassiruddin7/Small-Object-Detection-in-UAV-Imagery （仅确认入口，未全仓审计）。
- R11 作者代码：https://github.com/stefvk/Elliptical-Attention （正文给出的入口）。
- R13：https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
- R14：https://github.com/facebookresearch/xcit/blob/main/xcit.py
- R15：https://github.com/Linwei-Chen/FreqFusion/blob/main/FreqFusion.py

## 候选相似性审查表

| 候选设计                                         | 最接近的论文                               | 相同部分                                       | 不同部分                                                                                                        | YOLO11/VisDrone 中是否已有                                   | 是否建议实现                                   |
| ------------------------------------------------ | ------------------------------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| SPPF 跨尺度互斥峰值选择                          | R9 Top-K；R10 Sim SPPF；旧 SPPF 分支       | max/argmax、多个上下文响应、原 SPPF 窗口尺度   | 同一输出点的 5→9→13 窗口依次排除先前选中的空间来源；保持三个尺度输出，既不求 Top-K 平均，也不把空间分成固定环带 | 本仓库无该选择关系；公开定向检索未核验到同构实现，覆盖不完全 | 实现独立实验；需测内存/延迟和密集目标召回      |
| C2PSA 局部值变化预算                             | R11 Elliptical、R12 Bounded；旧容量/熵分支 | 使用原 QKV、限制无关上下文影响、保留自身 value | 先完成原 AV，计算每通道 3×3 value 极差，以所有通道共同限定 head 的 O-V 更新步长；保留更新方向                   | 本仓库无该值域约束；公开同构实现未核验到                     | 实现独立实验；需检验是否过度抑制有用全局上下文 |
| C3k2 直接 GRN / C2PSA 直接 XCA                   | R13 / R14                                  | 核心算子、顺序、通道归一和残差/乘 V 一致       | 主要是模型/插入位置以及适配原 QKV 的分组                                                                        | 通用 CV 已有；即便没有同构 YOLO11 仓库也不足以新颖           | **重复或弱创新，已移除试作，不交付**           |
| Neck 直接 DySample/FreqFusion、Conv 直接相位采样 | R4、R15、APS/LPS/TIPS 系列                 | 已有上采样/频率或相位聚合算子                  | 主要改变部署位置或组合方式                                                                                      | YOLO+VisDrone 已有相关应用                                   | **重复或弱创新，不实施**                       |

七项逐项比较：

| 维度       | SPPF 候选                                    | C2PSA 候选                                                |
| ---------- | -------------------------------------------- | --------------------------------------------------------- |
| 核心算子   | 嵌套支持域、max/argmax、按索引排除已选来源   | 原 softmax/AV、局部 max-min、跨通道最小可行比例           |
| 顺序       | cv1→展开原特征→小到大选择并排除→concat→cv2   | QKV→原空间注意→O-V→按 V 的局部变化预算限幅→原 PE/proj/FFN |
| 输入输出层 | Backbone P5 第 9 层，B×256×H/32×W/32         | Backbone P5 第 10 层，B×256×H/32×W/32                     |
| 插入位置   | 修改原 SPPF 内部池化；不加层、不加 Head      | 修改原 C2PSA 所拥有的 Attention 聚合；不改 Neck/Head      |
| 信息交互   | 较小尺度已选来源影响较大尺度的可选集合       | 同一 query/head 内各 value 通道共同决定一个步长           |
| 问题       | 强峰重复占据多个池化分支，挤占其他上下文来源 | 全局聚合导致的更新幅度与本地可观察变化不匹配              |
| 公开同构   | 有限检索未核验到，不能证明首创               | 同左；与 key 容量上限有明确不同的受约束对象               |

## 方案一：SPPF 跨尺度互斥峰值选择

原始 `z=cv1(x)`，`p1=M5(z)`，`p2=M5(p1)`，`p3=M5(p2)`；有效窗口为 5、9、13。
若一个强峰同时支配这些窗口，同一个来源会出现在三个分支中。

修改为对每个 batch/channel/输出位置 i，从原始 z 的嵌套邻域选取：

`S0(i)=empty; r_t=1+t*(k-1)`

`j_t(i)=argmax_{j in N_{r_t}(i) minus S_{t-1}(i)} z_j`

`e_t(i)=z_{j_t(i)}; S_t(i)=S_{t-1}(i) union {j_t(i)}`

`y=cv2(concat(z,e1,e2,e3))`。

这是按尺度顺序的贪心选择，不是求最大权重的全局匹配。只约束同一通道、同一输出点的池化分支，
不是不同输出点之间的 NMS，也不保证选中不同实例。相等响应采用 PyTorch max 的第一个索引；
排除的是空间来源而非数值，边界 padding 不参与。极小特征图耗尽有效来源时以中心值补位，保证有限输出。
第一尺度与原 5×5 max 一致，常量特征保持常量。

640 输入时 x/y 为 B×256×20×20，z/e_t 为 B×128×20×20，concat 为 B×512×20×20。
所有原始参数名和形状保持一致，无新可训练参数，但启用后函数不等价于原模型。
展开 13×13 原特征需 8,652,800 个元素/样本，FP32 约 33 MiB，仅此一项就需要实测开销；
不能因参数量不变就声称轻量或加速。假设有利于密集/遮挡场景的次强证据，但也可能选择背景噪声。

## 方案二：C2PSA 局部值变化预算

原始 C2PSA：`x→cv1→(a,b); b→PSABlock; concat(a,b)→cv2`。
保留上述路径以及原 attention `A=softmax(Q^T K/sqrt(d))`、`O=V A^T`。
只在 O 加位置编码和 proj 前修改更新步长。

对每个 value 通道 c、位置 i 定义邻域动态范围：

`R_ci=max_{j in N3(i)} V_cj - min_{j in N3(i)} V_cj >= 0`

`Delta_ci=O_ci-V_ci`

`rho_ci=1 if Delta_ci=0 else R_ci/max(abs(Delta_ci),epsilon)`

`alpha_hi=min(1, min_{c in head h} rho_ci)`

`O'_ci=V_ci+alpha_hi*Delta_ci`。

然后仍执行原 `PE(V)`、`proj`、残差、FFN。统计与限幅用 FP32，结果回到原 dtype。
数学上每个通道 `|O'-V| <= R`，且同一 head 的更新向量方向不变（alpha 非负）；
全局 O 与自身 V 之间做凸组合。**这不保证 O' 位于原 3×3 的 min/max 区间，也不保证 PE/proj 后最终输出受同一界约束。**
它限制每个 query 的值更新，不改变 key 列总接收量、attention 熵、QK 度量，不涉及 CSP 旁路生成容量。

640 输入时 x/y=B×256×20×20，a/b=B×128×20×20；n 模型实际一个 PSABlock、2 heads；
Q/K=B×2×32×400，V/O=B×2×64×400，A=B×2×400×400，alpha=B×2×1×400。
原权重形状全部保留，无新可训练参数；启用后不保证与原函数相等。
额外开销主要是两次 3×3 pooling 以及逐元素约束；不增加另一个 N×N 矩阵。
理论上可压制与本地弱小目标证据不相称的远处背景更新，也可能因某个通道过平而过度限幅，必须观测 alpha 分布。

## 实现与消融原则

分别使用 `SPPF-20261117` 和 `C2PSA-20261117`，均从同一 origin/main 出发；每个分支只启用一种机制。
直接扩展现有 SPPF/C2PSA/Attention 的可选路径，不新增注册类、解析器分支、依赖或训练循环。
旧 checkpoint 通过类级默认关闭值继续走原路径；默认路径需与原提交的真实模型及已有权重比较。

固定 10 类、数据划分、640 分辨率、预训练来源、训练轮数、优化器、增强、batch 和随机种子。
至少比较原始基线、仅 SPPF、仅 C2PSA，以及用户已有两个有效分支；之后才考虑组合。
建议三随机种子；报告 P/R、mAP50、mAP50:95、参数量、实测时延/显存，并按小目标、密度、遮挡分桶。
约 0.5% 的单次结果不能自动排除随机波动，本次不替用户已有测量重作统计结论。
两项均作用于 P5，无法恢复已经完全丢失于下采样的目标。

验证结果单独写入 `VISDRONE_VALIDATION_20261117.md`，不把随机张量 smoke test 当成 VisDrone 精度实验证据。
