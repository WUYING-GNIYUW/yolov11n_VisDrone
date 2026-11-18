# YOLO11n / VisDrone 重构组合实验（2026-11-18）

分支：`C2PSA-SPPF-Concat-20261118`。
训练配置：`ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-sppf-concat.yaml`。

## 问题与设计假设

用户报告 `SPPF-20261117`、`C2PSA-20261116`、`Concat-20261116` 分别训练时 mAP50 约提高 6%，
而串联顺序为 `SPPF→C2PSA` 的三模块分支未超过有效单项。本仓库没有这些训练的 `results.csv`、权重或日志，
无法独立计算增益、置信区间或判断训练是否以 epoch、step 还是 wall-clock 对齐；这里把该反馈作为实验前提，不当成本地复现结果。

实现前的工作假设是：SPPF 的互斥峰值选择改变了 P5 特征的空间对比度；后接的 C2PSA 又从自身 CSP 旁路估计
key 容量，故其容量控制可能基于已被池化改写的对比度。两种 P5 改动依次作用于同一特征会产生负交互。
这只是从数据流提出的可检验假设，不是已由训练曲线证明的故障原因。

因此重构 P5 的次序为：

```text
P5 C3k2 → capacity C2PSA → distinct-pool SPPF → 原 Neck / P5 lateral
```

C2PSA 先读取未经过 SPPF 的高层特征，其旁路对比度在池化选峰之前计算；SPPF 再对注意力更新过的表征
做多个空间范围的互斥选峰。P3 的 Concat 仍在原第 15 层，并继续以零增益起步。和已失败分支的唯一结构变量
是两个 P5 模块的顺序；算子实现、输入输出通道、检测头及预训练参数形状均相同，因而能直接检验此假设。

这不是保证提升的架构结论，也不是新算子或论文新颖性主张。最接近的实证参考是 Quratulain Nayeem 等人的
[Small Object Detection in Drone Aerial Imagery with LAF-YOLOv10](https://arxiv.org/abs/2609.14560)
(2026-09-13)。该研究在 VisDrone 上以 3 个随机种子评估 YOLOv10n 组合，并用逐组件消融与预训练权重迁移检查
定位负交互；其 PC-C2f、AG-FPN、P2/P5 检测头替换和 Wise-IoU 与本分支的 C2PSA、SPPF、Concat 算子及 P5
顺序实验不同。两者共同点是直接验证模块组合，而不假定单项增益可相加。故本分支只作为受控新实验，最终判断
依据同协议训练，而不是模块单项增益的相加。

## 模块作用与张量形状

| 模块             | 新位置                        | 操作及预期作用                                                                               | 640 输入通道/形状                       |
| ---------------- | ----------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------- |
| C2PSA 容量注意力 | Backbone 第 9 层，SPPF 之前   | CSP 旁路的局部对比度生成 key 容量；限制注意力流入集中到少数 key，并将剩余质量返回 query 自身 | 输入/输出 B×256×20×20；内部注意力 N=400 |
| SPPF 互斥池化    | Backbone 第 10 层，C2PSA 之后 | 5/9/13 窗口按次序选择最大响应，并排除当前位置已被小窗口选中的空间索引                        | 输入/输出 B×256×20×20                   |
| Concat 保守重组  | Neck 第 15 层                 | P3 lateral 生成 3×3 相似度以重排上采样语义；每个 2×2 单元的更新零均值，初始化增益为 0        | 两路 B×128×80×80 → B×256×80×80          |

完整 P3/P4/P5 检测输入仍为 `B×64×80×80`、`B×128×40×40`、`B×256×20×20`；10 类解码输出为
`B×14×8400`。两个 P5 模块均保持 256 通道，层 10 仍是 Neck 和 Detect 使用的 P5 特征，其他层引用不变。

算子实现直接复用源分支 `SPPF-20261117`、`C2PSA-20261116`、`Concat-20261116`，未复制出新模块类或更改其内部算子；
本分支仅调整 P5 上 SPPF 与 C2PSA 的先后次序。

## 训练验证方案

需将以下模型和失败的 `SPPF-C2PSA-Concat-20261118` 旧组合、三项各自有效单项及原始基线放在完全相同的训练协议下：

- 同一官方权重、VisDrone train/val 划分、输入分辨率、增强和预处理；
- 相同 optimizer、LR schedule、有效 batch、总 optimizer steps 和验证间隔；
- 相同随机种子配对训练，至少 3 个 seed；如果比较目标是训练效率，再单列 wall-clock，不改变 step 数；
- 报告每个 seed 的最佳及末轮 mAP50、mAP50:95、precision/recall、速度、峰值显存；优先读保存预测做逐类和漏检/误检分析。

SPPF 会降低每秒 step 数，若将所有模型限制为相同时长，会让其见到更少优化步数。因此准确率主比较必须匹配训练 step 数，
同时单列时间和吞吐。沿用用户原来的同一 VisDrone 标签 YAML 和数据根路径。

运行示例：

```python
import torch

from ultralytics import YOLO

weights = "yolo11n.pt"
model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-sppf-concat.yaml").load(weights)
source = YOLO(weights)
with torch.no_grad():
    model.model.model[9].load_state_dict(source.model.model[10].state_dict(), strict=True)
    model.model.model[10].load_state_dict(source.model.model[9].state_dict(), strict=True)
model.train(
    data="your_local_VisDrone.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    seed=0,
    device=0,
    project="runs/visdrone_c2psa_sppf_concat_20261118",
    name="capacity_before_distinct_pool_seed0",
)
```

样例参数仅供调用；请替换为产生那三个 +6% 结果时实际使用的参数。由于两层 P5 模块换序，通用 loader 会按层号
尝试匹配并可能错配 SPPF/C2PSA 权重，因此示例显式将源 C2PSA 权重加载到新层 9、源 SPPF 加载到新层 10。

Ultralytics 用已经加载的 `model` 作为 Trainer 权重，并按目标 YAML 重建同一结构；本地验证会逐值核对重建后的
模块顺序和权重。`Concat.transport_gain` 是唯一新增参数，来源 checkpoint 中没有对应值，保留初始化 0。

## 验证记录

| 检查                                          | 结果                                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------- |
| 构建后层顺序和三个开关                        | 第 9 层 C2PSA(capacity)，第 10 层 SPPF(distinct)，第 15 层 Concat(conservative)     |
| 交换层权重后预训练 Backbone/Neck 张量逐值比较 | 378/378 匹配官方 yolo11n.pt；未交换前的通用加载只报告 451/500，因此不可省略显式映射 |
| 相同预训练权重下，与 `SPPF→C2PSA` 旧顺序比较  | 检测张量形状相同，输出不同，证实新配置检验的是执行顺序变量                          |
| 640×640 / 384×640 前向                        | 输出分别为 1×14×8400、1×14×5040，均有限                                             |
| 关键层 640 形状                               | C2PSA、SPPF 均 B×256×20×20；Concat B×256×80×80                                      |
| 合成标注 v8DetectionLoss 反向                 | 256 个可训练参数梯度均有限；C2PSA、SPPF、Concat 的关键梯度非零                      |
| 按目标 YAML 重建训练模型                      | 层顺序、三个开关、已交换的权重保持                                                  |
| 完整模型 checkpoint 往返                      | 前向逐元素一致                                                                      |
| Concat 增益为 0.25 的 TorchScript 导出及重载  | 通过 rtol=atol=1e-4，输出 1×14×336                                                  |

PyTorch 工程检查使用 Python 3.12.2 / 2.15.0+cpu、Intel Xeon E3-1270 v6、4 线程；可用 PyTorch 没有 CUDA。
原始三种算子的坐标选择、注意力预算、零增益与均值守恒合同已在源分支验证，本分支 AST 与三个源模块实现逐项相同。
本工作区没有找到 VisDrone 数据集、训练日志或训练权重，因此这里不报告 mAP 结果，也不宣称新顺序一定提升指标。

检查脚本与导出文件保存在忽略目录 `.worktrees/_validation_20261118/reordered.py`、`reordered.json`、
`reordered.torchscript`，可在本 worktree 重跑：

```powershell
& 'F:/DE/Python/Python312/python.exe' ../_validation_20261118/reordered.py
```
