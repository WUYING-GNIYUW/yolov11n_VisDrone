# YOLO11n / VisDrone 定向创新性审查（2026-11-16）

## 范围、证据等级与实施决定

时间窗：2023-11-16 至 2026-11-16；较早的通用方法仅作算子先例，不计入近三年 YOLO/VisDrone 论文。
基准：origin/main，55989f76c1e85c192ba12cb3e533e3de842a2301，Ultralytics 8.4.198。
在任何模型代码修改前完成本审查。先读取全部远端分支代码，再读论文摘要、可访问正文与官方实现。

网页检索工具未返回可读内容，改用 HTTPS 读取 arXiv API、CVF、出版商登记的 Crossref 元数据、官方 GitHub。
访问失败（arxiv.org TLS、Nature 证书吊销检查、部分 arXiv/Crossref 429）均不视为“没有相关研究”。
GitHub repository search 只搜索仓库元数据，不能替代全站代码搜索。以下“未核验到同构实现”均为有限检索结论。
**两项实现属于有明确算子差异的待验证设计，不是首创认证，不声明精度提升。**
用户已说明旧分支实验无效；本次读取其实现以排重，没有重新训练或独立复核旧分支的指标。

## 全部分支审计

已 git fetch origin，并用 GitHub branches API 核对，共 14 个真实分支（含 main，排除 origin/HEAD 别名）。

| 分支（省略 origin/）                      | 提交前缀  | 实际机制与位置                                       | 本次处理               |
| ----------------------------------------- | --------- | ---------------------------------------------------- | ---------------------- |
| main                                      | 55989f76c | 原始 YOLO11                                          | 唯一共同起点           |
| codex/model-change-marker                 | 55989f76c | 与 main 完全相同                                     | 无可审查模块差异       |
| codex/visdrone-lcr-c3k2                   | 55989f76c | 与 main 完全相同                                     | 不根据名字虚构历史机制 |
| codex/visdrone-lg-c2psa-window-global     | 55989f76c | 与 main 完全相同                                     | 同上                   |
| exp/visdrone-yolo11n-antialias            | 55989f76c | 与 main 完全相同                                     | 同上                   |
| exp/visdrone-yolo11n-sppf-residual        | 8e8aef0b1 | P5 SPPF 输出加按通道缩放的原输入，兼有激活选项       | 排除残差/门控复用      |
| exp/visdrone-yolo11n-source-route         | d7e2d43c7 | Neck P3 C3k2 分组投影分离上采样与横向来源，细节旁路  | 排除来源分离重做       |
| exp/visdrone-yolo11n-c2psa-centered       | 46ddb61eb | P5 PSA 前 b + tanh(g)·(b-AvgPool3(b))                | 排除局部对比残差重做   |
| codex/visdrone-c3k2-crp                   | 6fb9bf3e9 | P3 C3k2 中间输出减去其在第一分支上的通道投影         | 排除去冗余投影         |
| codex/visdrone-c2psa-local-global         | 4bd2c0cb5 | P5 C2PSA 的 DWConv 与 PSA 输出经全局池化通道门控混合 | 排除局部/全局门控      |
| codex/visdrone-neck-diff-fusion           | dc0dd9923 | P3 两来源绝对差→1×1 Conv→sigmoid→双向加性补偿→concat | 排除差分门控补偿       |
| codex/visdrone-combined                   | 27babcf3d | CRP + local/global PSA + difference concat           | 排除同组合             |
| feature/visdrone-c2psa-entropy-confidence | 4501ba9c7 | 按 query 注意力熵缩放聚合后的 value                  | 排除 query 熵置信缩放  |
| feature/visdrone-yolo11n-local-edge       | 1c585c1b9 | P3 stride-2 Conv 前用方向差分边缘能量调制输入        | 排除边缘增强重做       |

审计材料：每个分支相对 origin/main 的模块 diff、模型 YAML；同时读取 source-route、entropy、combined 分支的研究记录。
远端核对入口：https://api.github.com/repos/WUYING-GNIYUW/yolov11n_VisDrone/branches?per_page=100

## 定向文献及最接近先例

| 编号 | 方法、年份、出处                                                                                                                                                                 | 本次核验层级                        | 核验到的机制 / 与候选的关系                                                                              |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| R1   | Frequency-aware Feature Fusion for Dense Image Prediction，TPAMI 2024，10.1109/TPAMI.2024.3449959                                                                                | 官方 README 与 FreqFusion.py        | ALPF 内容自适应低通、局部相似度引导偏移重采样、AHPF 高频增强。与 Neck 双来源融合最接近；不是 ECCV 论文   |
| R2   | FeatUp: A Model-Agnostic Framework for Features at Any Resolution，ICLR 2024，arXiv:2403.10516                                                                                   | 官方 README 与 featup/upsamplers.py | 由图像引导的学习式联合双边上采样 JBU、range/spatial kernel、局部动态加权；未见下面的逐单元均值硬投影     |
| R3   | Sinkformers: Transformers with Doubly Stochastic Attention，AISTATS 2022，arXiv:2110.11773                                                                                       | arXiv 摘要与官方实现说明            | 交替行列 Sinkhorn 归一化，将 attention 变为双随机矩阵；本次容量上限不是替换成 Sinkhorn                   |
| R4   | OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation，ECCV 2024，arXiv:2403.14183                                                                          | arXiv 元数据/摘要                   | 多文本提示与图像像素的 Sinkhorn 跨模态匹配；提醒不能把传输归一化换名                                     |
| R5   | Story2Board: A Training-Free Approach for Expressive Storyboard Generation，2025，arXiv:2508.09983                                                                               | arXiv 正文§3.2、官方仓库身份        | 双面板 token 的 min(Aij,Aji) 互惠分数与 value 混合；据此否决最初的“互惠注意”候选                         |
| R6   | Learning to Upsample by Learning to Sample / DySample，ICCV 2023                                                                                                                 | CVF 正式页面                        | 学习偏移的点采样；早于时间窗，作为一般算子先例。直接搬到 Neck 判弱创新                                   |
| R7   | YOLO-MAG: an edge-preserving and bilateral dual-gated YOLO11 network for small object detection in UAV aerial imagery，Scientific Reports 2026-10-27，10.1038/s41598-026-76441-9 | 出版方登记摘要，正文访问失败        | YOLO11 + VisDrone；C3k2-MEIS、C2PSA-ACmix、双边门控 GAC-FPN。边缘/局部全局混合已有明确先例               |
| R8   | PSF-YOLO11: an enhanced YOLO11 framework for small object detection in UAV imagery，Measurement Science and Technology 2026-06-17，10.1088/1361-6501/ae71f9                      | 出版方登记摘要                      | VisDrone2019；并行多尺度卷积、小目标金字塔、共享权重融合。本次查到正式版本，旧分支只列 SSRN 的状态已过时 |
| R9   | DASSF: Dynamic-Attention Scale-Sequence Fusion for Aerial Object Detection，2024，arXiv:2406.12285                                                                               | arXiv 摘要                          | YOLOv8 + VisDrone；动态尺度序列融合、额外小目标头、DyHead；涉及密集/重叠目标                             |
| R10  | A lightweight model FDM-YOLO for small target improvement based on YOLOv8，2025，arXiv:2503.04452                                                                                | arXiv 摘要                          | VisDrone；Fast-C2f/PConv、P2/P5调整、DySample、EMA。直接换卷积/上采样不是新机制                          |
| R11  | Small Object Detection in Drone Aerial Imagery with LAF-YOLOv10，2026-09-13，arXiv:2609.14560                                                                                    | arXiv 摘要                          | PC-C2f、AG-FPN、P2替P5、Wise-IoU 的组合及失败分析；必须关注预训练迁移与模块交互，不能假定叠加必涨点      |
| R12  | Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection，2026-11-05，arXiv:2611.07994                                                         | arXiv 摘要                          | 跨尺度可变形融合、椭圆引导监督、全尺度 query 交互；涉及 VisDrone，但非 YOLO，不直接移植                  |
| R13  | Getting the Numbers Right—Modelling Multi-Class Object Counting in Dense and Varied Scenes，2025/2026，arXiv:2510.02213                                                          | arXiv 摘要                          | 遮挡/密集场景的密度估计与辅助类别分割；包含 VisDrone 和 YOLO11 对照，是计数研究，不能当检测模块提升证据  |

完整论文指标未作本地结果使用；摘要不足以核验全部运算顺序的研究不作为“已排除同构”的证明。

来源与官方代码：

- R1 https://github.com/Linwei-Chen/FreqFusion/blob/main/FreqFusion.py ；https://doi.org/10.1109/TPAMI.2024.3449959
- R2 https://github.com/mhamilton723/FeatUp/blob/main/featup/upsamplers.py ；https://arxiv.org/abs/2403.10516
- R3 https://github.com/michaelsdr/sinkformers ；https://arxiv.org/abs/2110.11773
- R4 https://arxiv.org/abs/2403.14183
- R5 https://export.arxiv.org/html/2508.09983 ；https://github.com/DavidDinkevich/Story2Board
- R6 https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html
- R7 https://doi.org/10.1038/s41598-026-76441-9
- R8 https://doi.org/10.1088/1361-6501/ae71f9
- R9 https://arxiv.org/abs/2406.12285
- R10 https://arxiv.org/abs/2503.04452
- R11 https://arxiv.org/abs/2609.14560 ；摘要链接作者代码 https://github.com/Mudassiruddin7/Small-Object-Detection-in-UAV-Imagery ，本次未读取代码
- R12 https://arxiv.org/abs/2611.07994
- R13 https://arxiv.org/abs/2510.02213

成功查询包括：arXiv all:VisDrone、all:VisDrone AND all:YOLOv8、all:VisDrone AND all:YOLO11、
ti:Sinkformers、reciprocal attention AND cat:cs.CV、attention AND Sinkhorn AND vision；
Crossref YOLO11 UAV small object detection、YOLOv10 VisDrone（时间窗过滤）；
GitHub repository search：YOLO11 VisDrone FreqFusion、VisDrone FreqFusion、YOLO VisDrone Sinkhorn、
YOLO11 VisDrone reciprocal、VisDrone conservative fusion、YOLO VisDrone capacity attention（均返回0条仓库元数据）。
追加 capacity/conservative/capped attention 的三个 arXiv 精确检索分别 429/429/超时，不能解读为0条。
既未宣称上述候选没有公开先例，也未把未取得全文的论文当成已经完成公式级排重。
实施后的补充复核成功查询 ti:attention AND abs:capacity AND cat:cs.CV，返回167条，读取前8条；
最相关 AVQ-Attention（arXiv:2607.12789）根据注意重要性分配VQ码本容量，与key接收上限及query回流不同。
该补充不改变实施前审查结论，也不构成穷尽检索。

## 候选相似性审查表

| 候选设计                                                                | 最接近的论文                                       | 相同部分                                       | 不同部分                                                                                                | YOLO11/VisDrone 中是否已有                                  | 是否建议实现                                           |
| ----------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| C2PSA 旁路引导 key 容量截断与自保持                                     | R3 Sinkformers、R4 OTSeg；旧 entropy/centered 分支 | 约束注意分布，仍用 QKV；局部统计算子并非新发明 | 单次按列总接收量截断；容量来自 CSP 的另一分支；截断量回到 query 自身，既不交替归一化也不强制列和=1      | 未核验到完整同构实现；存在 attention 融合论文和旧同位置改动 | 实现为独立实验；创新点限于容量来源、竞争与回流的数据流 |
| Neck Concat 单元守恒的相似度重组                                        | R1 FreqFusion、R2 FeatUp/JBU、R6 DySample          | 横向细节引导语义重采样，局部相似度/加权聚合    | 对4个子像素共同计算9邻域重组，再硬投影到2×2零均值子空间；所有通道粗单元均值保持；无预测offset/ALPF/AHPF | 未核验到完整同构实现；旧差分融合已存在，但没有该约束        | 实现为独立实验；不是声称双边滤波本身创新               |
| C2PSA min(Aij,Aji) 互惠 value 混合                                      | R5 Story2Board                                     | 核心 min 双向注意和 value 混合相同             | 换成检测及C2PSA，不足以构成新机制                                                                       | 本仓库未见；通用CV已经有                                    | **重复或弱创新，不实现**                               |
| C2PSA 直接 Sinkhorn 替换 softmax                                        | R3、R4                                             | 核心迭代归一化相同                             | 插入位置变化                                                                                            | YOLO同构未核验，不影响已有算子判断                          | **重复或弱创新，不实现**                               |
| Neck 直接加入 DySample/FreqFusion/JBU                                   | R1、R2、R6、R10                                    | 核心重采样相同                                 | 模型/插入位置变化                                                                                       | YOLOv8/VisDrone已有DySample使用                             | **重复或弱创新，不实现**                               |
| C3k2去冗余、来源分离；SPPF残差；Conv边缘；C2PSA局部全局/局部中心/熵门控 | 全部分支审计及R7                                   | 与旧实现相同核心算子、顺序及位置               | 换名称或参数不构成差异                                                                                  | 本仓库已有，用户已判实验无效                                | **重复或弱创新，不实现**                               |

七项检查的结果（每项独立检查，不能仅凭名称）：

| 维度                    | C2PSA 容量设计                                                    | Concat 守恒设计                                             |
| ----------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------- |
| 核心算子                | softmax→列总量→容量截断→对角回流；不是Sinkhorn，也不是query熵缩放 | 余弦相似度→9邻域加权→4相位去均值；不是偏移采样或差异门控    |
| 运算顺序                | a分支生成容量；b分支QKV；在VA聚合前限流、聚合时补自身value        | nearest语义与横向输入；恢复粗网格；重组；投影；残差；concat |
| 输入/输出层             | P5 C2PSA；a、b是同尺度不同CSP分支                                 | P4语义升至P3 + Backbone P3横向，层15                        |
| Backbone/Neck/Head      | Backbone层10，Detect不变                                          | Neck层15，Detect不变                                        |
| 信息交互                | 旁路→key预算；所有query共同竞争同一key；未用容量回到各自value     | 横向→子像素分配；粗邻居→细语义；同一粗单元4相位共同约束     |
| 解决问题                | 强背景key被大量query反复接收的潜在语义淹没                        | 重复上采样语义定位模糊与融合后的粗语义漂移                  |
| 公开YOLO11+VisDrone同构 | 有限检索未核验到；全文覆盖不完全                                  | 有限检索未核验到；不能证明不存在                            |

## 实施方案一：C2PSA 接收容量约束

分支 C2PSA-20261116。只给原 C2PSA 增加 capacity_attention=False 的可选参数，继续使用原 Attention/PSABlock，
不新增模块类，不改变注册和 parser，不增加第三方依赖。

原始：x→cv1→(a,b)，b→PSABlock(s)，concat(a,b)→cv2。
修改：a额外产生空间容量，b的每个原PSABlock使用该容量限流；a本身仍进入原拼接。

设 N=HW，A=softmax(QᵀK/√d)，A 的行是query、列是key。
旁路的局部变化 d_j=mean_c |a_cj-AvgPool3(a)_cj|，μ=mean_j d_j。
令 u_j=1/2+d_j/(d_j+μ+ε)，c_j=u_j/mean(u)，于是 mean(c)=1，且 c>0；
常量旁路时 c=1。它只是局部变化代理，不能保证标出目标，纹理背景也可能获较大容量。

m_j=Σ_i A_ij，s_j=min(1,c_j/max(m_j,ε))，
B_ij=A_ij s_j，r_i=1-Σ_j B_ij，
T=B+diag(r)，O_i=Σ_j B_ij V_j+r_i V_i。

T行和为1；B的每列和≤c_j，表示全局聚合的接收上限。
**T并非双随机；加回对角后列和上界是c_j+1，不能声称仍≤c_j。**
保持原PE(V)、输出proj和FFN。FP32计算容量与权重统计，聚合回输入dtype。
这是非迭代限流+自身回流，和简单全局value缩放不同；没有新可训练参数。
开启时不保证与原函数相等；关闭时应保持原函数和参数形状完全一致。

640输入：C2PSA x/y=B×256×20×20；a/b=B×128×20×20；2 heads；
Q/K=B×2×32×400，V=B×2×64×400，A/B=B×2×400×400，c=B×1×1×400。
不增加分辨率，因此不能恢复已经在下采样中彻底丢失的目标。

必要消融：原始；仅容量设计；固定c=1（移除旁路引导）；去自保持（仅诊断）。
记录强背景key接收总量、目标/背景响应、小/密集/遮挡目标召回；变化代理可能选中纹理背景，需实测。

## 实施方案二：Neck Concat 单元守恒重组

分支 Concat-20261116。给原 Concat 增加 conservative=False 可选参数，只在P3的层15启用，
保留原层14 nearest upsample以及全部层号/通道。它需要两路同批次、同空间尺寸、偶数H/W；可不同通道。
默认Concat路径不增加运算。

输入U=nearest(S)，L为横向细节。D=2×2平均下采样；将L分成每粗单元的4相位并按通道归一化得到q；
对D(L)做通道归一化，取粗网格3×3邻域得到k。边缘用replicate。
w_ipj=softmax_j(4·cos(q_ip,k_ij))，j∈9邻域，p∈4子像素。
R_ip=Σ_j w_ipj·D(U)_ij，
Δ_ip=R_ip-(1/4)Σ_p R_ip，
U'=U+tanh(g)·PixelShuffle(Δ)，Y=concat(U',L)。

因此D(U')=D(U)，而常量U产生Δ=0：可改变同一粗格内的细分布，不能凭横向纹理凭空增加该格的语义均值。
与FeatUp/JBU的共同点必须承认：都使用局部引导相似度重组；
差异在相位共同约束、硬均值投影、来源和融合方式。与FreqFusion相比无自适应低/高通生成器和偏移预测。
g零初始化，增加1个参数，初始严格退化为原Concat；g的一阶梯度应非零，可启动学习。
有符号投影可能产生超出局部值域的更新，因此不声称保持凸包或必然锐化边界。

640输入：U/L各B×128×80×80；D(U)/D(L)各B×128×40×40；
q=B×128×4×40×40，k=B×128×9×40×40，w=B×4×9×40×40；
Y=B×256×80×80；层16输出仍B×64×80×80。
实现使用粗网格unfold和连续布局的batched matmul，避免构建完整高分辨率9邻居value副本；仍需实测延时。

必要消融：原始；仅守恒设计；相同重组去掉零均值投影（区分先例式重组和约束收益）；
固定均匀权重（该投影使更新为0，是诊断，不是新模型）。
零均值性质应在非零g下检验，不能仅验证零初始化。

## 验证与训练口径

两分支从同一main独立建立，不叠加，以便归因。提供10类、固定n规模的baseline和单模块YAML。
现有参数键/形状全部保留；需要检查真实模型构建、640/矩形前向、真实检测loss反向、参数迁移、
融合BN后输出、checkpoint往返，以及上述数值约束。验证记录在同目录验证报告中。

本机检测到RTX 5060，但可用Python环境的PyTorch为CPU构建；不能将nvidia-smi可见GPU写成CUDA验证成功。
当前核查的 F:/datasets 只有COCO2017，未找到本任务的本地VisDrone配置/权重。
不自动下载数据、不用COCO替代VisDrone、不启动完整精度训练。完整训练需在同一VisDrone数据、
同一预训练YOLO11n权重、imgsz、batch、优化器、增强、epoch、种子0/1/2上比较baseline和单模块。
报告mAP50、mAP50:95、P/R、参数、实测延时以及按尺寸/密度/遮挡的分桶结果。
这里的算子约束和工程通过仅支持实现正确性，不能支持检测精度提升结论。
