# C2PSA 两种约束的比较与组合（2026-11-17）

## 修改前结论

两者作用位置相同，但约束对象和统计来源不同；都增强自身 value 保留，因此存在功能重叠。
本次按用户要求组合已实现的两项机制，不把组合包装成第三项独立创新，也不假设训练收益相加。
用户反馈 C2PSA-20261116 的 mAP50 约提升 0.5%；C2PSA-20261117 尚无完整 VisDrone 训练结果。

代码对照：C2PSA-20261116 的 `749a3e972`，C2PSA-20261117 的 `33b364b49`。
新分支 C2PSA-combined-20261117 从 origin/main 的 `55989f76c` 建立，保留原两个实验分支。

| 维度           | C2PSA-20261116                                     | C2PSA-20261117                                |
| -------------- | -------------------------------------------------- | --------------------------------------------- |
| 核心算子       | 注意矩阵按 key 列总量截断，剩余权重回流 query 自身 | 按局部 value 极差收缩聚合更新                 |
| 顺序           | softmax → 列容量限制 → value 聚合和自身回流        | softmax → value 聚合 → 变化幅度限制           |
| 统计来源       | CSP 旁路 a 的局部对比度                            | 注意分支 V 的 3×3 极差                        |
| 信息交互       | 所有 query 共同竞争同一 key 的预算                 | 同一 query/head 的所有 value 通道共同限定步长 |
| 约束对象       | 聚合矩阵的列和                                     | 聚合特征相对于自身 value 的变化               |
| 插入位置及形状 | Backbone P5 第 10 层，B×256×20×20（640 输入）      | 相同                                          |
| 目标           | 减少一个强背景位置被全图反复使用                   | 减少相对局部变化过大的全局更新                |
| 重叠与风险     | 截断后向自身回流                                   | 也向自身收缩，组合可能过度抑制有用上下文      |

两项并非互相替代：列和合规仍可能把差异很大的 value 混入某 query；更新很小也不能保证各 key 的累计使用量均衡。

## 相似性审查与证据边界

本次复用并重读原两分支的定向检索报告和实际代码。追加网页查询
`attention value update limiter local range self attention gating`、`VisDrone capacity attention`
未返回可读取的结果，不将此记为没有相关论文。

| 候选设计       | 最接近的方法                                                   | 相同部分                     | 不同部分                                               | YOLO11/VisDrone 中是否已有                   | 是否建议实现                               |
| -------------- | -------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------ | -------------------------------------------- | ------------------------------------------ |
| 容量约束       | 本仓库 C2PSA-20261116；Sinkformers、OTSeg                      | 本次直接保留旧分支实现       | 没有重新声称此部分创新                                 | 本仓库已有；用户反馈提升                     | 复用                                       |
| 局部值变化预算 | 本仓库 C2PSA-20261117；Elliptical Attention、Bounded Attention | 本次直接保留旧分支限幅实现   | 与 Elliptical 的 QK 度量、Bounded 的主体掩码不同       | 本仓库已有；尚未验证精度收益                 | 复用并独立开关                             |
| 两约束串联     | 上述两个本仓库分支                                             | 使用相同算子、统计和插入位置 | 限幅对象改为容量约束后的聚合结果；保证两类约束同时成立 | 本次代码对照中未有该组合；未作全网新颖性断言 | 按用户要求作为组合消融实现，不作为独立创新 |

既有来源：Sinkformers/OTSeg 的核验记录见 C2PSA-20261116 的 VISDRONE_RESEARCH_20261116.md；
Elliptical Attention（arXiv:2406.13770）、Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation
（arXiv:2403.16990）的全文对照见 C2PSA-20261117 的 VISDRONE_RESEARCH_20261117.md。

## 组合数据流与公式

沿用原 C2PSA：`x → cv1 → (a,b)`。a 提供容量并继续参与最终拼接；b 仅通过原有 PSABlock。
在同一个 Attention 内执行 `QKV → softmax → 容量截断 → value 聚合和自身回流 → 局部值变化限幅 → PE/proj`，
再执行原残差、FFN、concat、cv2。不串联两个完整 C2PSA，不增加可训练参数。

以下省略 batch/head，A 的 i 为 query、j 为 key：

`A = softmax(Q^T K / sqrt(d))`

`d_j = mean_c |a_cj - AvgPool3(a)_cj|; mu = mean_j d_j`

`u_j = 0.5 + d_j / (d_j + mu + 1e-6); c_j = u_j / mean_j u_j`

`s_j = min(1, c_j / max(sum_i A_ij, 1e-6)); B_ij = A_ij s_j`

`r_i = max(0, 1 - sum_j B_ij); Ocap_i = sum_j B_ij V_j + r_i V_i`

第二项约束必须基于 Ocap 重新计算，不能沿用原始 O 的步长：

`R_ci = max_{j in N3(i)} V_cj - min_{j in N3(i)} V_cj; Delta_ci = Ocap_ci - V_ci`

`rho_ci = 1 if Delta_ci == 0 else R_ci / max(abs(Delta_ci), 1e-6)`

`alpha_hi = min(1, min_{c in head h} rho_ci); Ocombined_ci = V_ci + alpha_hi Delta_ci`

因此每通道 `|Ocombined-V| <= R`，且同 head 的更新方向保持不变。这个界只在 PE/proj 前成立。
忽略浮点误差，有等价有效注意矩阵：

`Tcombined = diag(alpha) B + diag(1 - alpha * rowsum(B))`

它非负且行和为 1。由于 `0 <= alpha <= 1`，`sum_i alpha_i B_ij <= sum_i B_ij <= c_j`，
第二步不会扩大容量受限的聚合流量。**自身回流另占对角项，最终 Tcombined 的列和不保证 <= c_j**；
也不保证 Ocombined 位于 3×3 局部 min/max 区间。

640 输入时输入输出 B×256×20×20，a/b 为 B×128×20×20；2 heads、每 head 64 个 value 通道；
Q/K 为 B×2×32×400，V 为 B×2×64×400，容量为 B×1×1×400，步长为 B×2×1×400。
组合仍位于 P5，不能恢复已经在下采样中完全丢失的目标。

## 配置与消融

保持 20261116 的参数位置，新增 range_limit 放在最后；两个布尔量都关闭时保留原路径。
YAML 中 C2PSA 的参数为 `[1024, 0.5, capacity_attention, range_limit]`，重复次数仍由原 parser 处理。
旧两个分支的三项参数 YAML 第三个值含义不同，不能直接互换；本分支给出的四份配置已明确区分。

| 配置后缀       | capacity_attention | range_limit | 用途          |
| -------------- | ------------------ | ----------- | ------------- |
| baseline       | False              | False       | 原始基线      |
| capacity       | True               | False       | 复现 20261116 |
| c2psa-range    | False              | True        | 复现 20261117 |
| c2psa-combined | True               | True        | 组合实验      |

四份 YAML 拓扑一致，均为 nc=10、scale=n。原生格式没有继承机制，保留完整配置以支持独立、可复现训练；
不增加配置生成器、parser 分支或新模块类。原有 Python owner block.py 容纳全部组合逻辑。
验证需覆盖单项与旧分支真实输出逐值一致、组合约束、真实检测损失反向、预训练迁移和导出。
完整训练必须固定数据、预训练来源、imgsz、batch、增强、优化器、epoch 和种子；至少比较四组。
建议记录 alpha 分布和小目标召回，以检查双重收缩是否导致负收益。不会自动启动完整训练。
