# YOLO11n / VisDrone 三模块组合（2026-11-18）

分支：`SPPF-C2PSA-Concat-20261118`。
工作目录：`C:/Users/Administrator/yolov11n_VisDrone/.worktrees/SPPF-C2PSA-Concat-20261118`。
训练入口：`ultralytics/cfg/models/11/yolo11n-visdrone-combined.yaml`，10 类、n 规模。

## 组合来源与边界

从 `origin/main` 的 `55989f76c1e85c192ba12cb3e533e3de842a2301` 建立独立 worktree，逐项复用以下分支的模块实现。
本次是已有机制的可消融组合，不提出新算子或新颖性主张。用户此前反馈 C2PSA、Concat 单项各约有 0.5% mAP50
提升；该反馈没有在本次重新测量，不能相加为组合收益。

| 模块来源          | 核对的源提交                               | 原位置与用途                                           |
| ----------------- | ------------------------------------------ | ------------------------------------------------------ |
| `SPPF-20261117`   | `809c94906e47f2ce218cf5b9bb19de56188de4fb` | Backbone 第 9 层：嵌套窗口排除先前选中的空间来源       |
| `C2PSA-20261116`  | `749a3e9728c7b1e2725d1f894fac548f6b1e58cf` | Backbone 第 10 层：CSP 旁路生成 key 容量，限制聚合集中 |
| `Concat-20261116` | `106b8bddde43ad08ea1db06049ffc9e029d4436f` | Neck 第 15 层：横向特征引导语义重组，保持 2×2 均值     |

实施前的文献与全部历史分支审查保留在对应来源分支的 `VISDRONE_RESEARCH_20261116.md` 和
`VISDRONE_RESEARCH_20261117.md`。本组合使用容量版 C2PSA；`C2PSA-20261117` 的 value 范围约束不在本次范围内。

## 数据流与互补关系

顺序为 `P5 C3k2 → SPPF → C2PSA → 原 P4/P3 上采样路径 → P3 Concat → 原 C3k2/检测头`。
保持原通道数、层索引、P3/P4/P5 检测尺度以及所有已有卷积参数的键和形状。
SPPF、C2PSA 不添加可训练参数；Concat 添加一个初值为 0 的 `transport_gain`。

1. **SPPF**：令 `z=cv1(x)`。在每个输出位置、每个通道的 5、9、13 嵌套窗口内，依次选取最大响应，
   并排除该位置之前尺度已选过的空间索引，最后执行 `cv2(concat(z,p1,p2,p3))`。
   理论作用是减少强峰重复占据三个上下文分支，为次强响应保留位置；不保证不同峰属于不同目标。
2. **C2PSA**：由 `cv1(x)` 分出的旁路 `a` 计算局部对比度，再产生均值为 1 的 key 容量 `c_j`。
   对原行归一化注意力 `A` 执行 `B_ij=A_ij min(1,c_j/sum_i A_ij)`，
   `r_i=1-sum_j B_ij`，`O_i=sum_j B_ij V_j+r_i V_i`，随后保留原 PE、投影、残差和 FFN。
   理论作用是限制少数 key 主导跨位置聚合，未分配权重回流自身。
   容量上界约束的是 `B` 的列和；加入自身回流后的完整矩阵不具有同一个列容量上界。
3. **Concat**：P3 横向特征的四个子像素分别与粗格 3×3 邻居算相似度，用权重重组语义值 `R`。
   对每个 2×2 单元执行 `U=R-mean_phase(R)`，输出
   `concat(S+tanh(g)U,L)`，其中 `S` 是最近邻上采样语义，`L` 是横向特征，`g` 是可训练标量。
   `mean_phase(U)=0`，因此保持传入语义的 2×2 均值；`g=0` 时是普通 Concat，梯度仍可更新 `g`。

三者依次作用于上下文来源选择、全局聚合分配和高分辨率语义分布，原算子的计算顺序保持一致。
Concat 的零初始化使三模块模型初始输出与同权重的“仅 SPPF+C2PSA”相同，降低新增融合对初始化的扰动。
这不意味着整个组合与原 YOLO11n 相同，SPPF 和容量注意力在启用后已经改变输出。
SPPF 也可能引入背景峰，容量也可能限制有用上下文；叠加是否有效须由训练消融决定。

| 640×640 输入时的位置    | 输入                                   | 输出           |
| ----------------------- | -------------------------------------- | -------------- |
| SPPF / C2PSA            | B×256×20×20                            | B×256×20×20    |
| 第 15 层 Concat         | 两路 B×128×80×80                       | B×256×80×80    |
| Detect 的 P3/P4/P5 输入 | B×64×80×80 / B×128×40×40 / B×256×20×20 | 解码 B×14×8400 |

## 训练与消融

在新 worktree 根目录、现有 CUDA 训练环境中运行。沿用之前有效实验的完整训练参数，仅替换模型 YAML，
并从同一份官方预训练权重初始化。下面是调用示例；数据路径、epochs、batch、优化器及增强需要与已有实验一致。

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-combined.yaml").load("yolo11n.pt")
model.train(
    data="your_local_VisDrone.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    seed=0,
    device=0,
    project="runs/visdrone_combined_20261118",
    name="sppf_capacity_conservative_seed0",
)
```

使用目标 YAML 先构建再 `.load(...)`，保证组合结构真正启用。直接 `YOLO("旧分支/best.pt")` 会沿用旧模型结构。
如从某个已有实验权重继续微调，也用目标 YAML 加载该权重，并将该初始化条件单独报告；不混合三个独立训练的
完整 state_dict。新组合训练不使用旧实验的 `resume=True`。

三个开关均为原类的可选参数，默认关闭；上述组合 YAML 显式全部启用。
在该 YAML 的副本中只修改以下三个布尔值，即可在同一个分支复现各消融：

- SPPF：Backbone 第 9 层的 `[1024, 5, 3, False, distinct_pool]` 最后一项。
- C2PSA：Backbone 第 10 层的 `[1024, 0.5, capacity_attention]` 最后一项。
- Concat：整体第 15 层（`head[4]`）的 `[1, conservative]` 最后一项。

| 方案                   | distinct_pool | capacity_attention | conservative |
| ---------------------- | ------------- | ------------------ | ------------ |
| 基线                   | False         | False              | False        |
| 仅 SPPF                | True          | False              | False        |
| 仅 C2PSA               | False         | True               | False        |
| 仅 Concat              | False         | False              | True         |
| C2PSA + Concat         | False         | True               | True         |
| SPPF + C2PSA           | True          | True               | False        |
| SPPF + Concat          | True          | False              | True         |
| 三模块组合（交付配置） | True          | True               | True         |

优先比较基线、两个已反馈有效的单项、C2PSA+Concat、三模块组合；其余组合用于定位交互来源。
固定数据划分、预训练来源、分辨率、有效 batch、训练步数、增强、优化器和评估配置，至少复核相同的 0/1/2 种子。
同时记录 mAP50、mAP50:95、P/R、时延和峰值显存。如果显存要求减小 batch，各对照组使用相同的有效 batch。
这些是消融条件，尚未在本机运行完整 VisDrone 训练。

SPPF 的 13×13 unfold 在 640 输入、FP32 时仅展开张量就约 33 MiB/样本，反向与其他中间张量还需额外空间。
Concat 在 P3 进行相似度重组也有开销；参数量几乎不变不能说明推理或训练成本不变。

## 实现与验证记录

模型代码只扩展已有 `ultralytics/nn/modules/block.py` 和 `ultralytics/nn/modules/conv.py`，复用源分支的实现。
另增加一个明确启用三项机制的模型 YAML 和本说明；不新增注册类、模型解析器分支、依赖或训练循环。
三项原有机制合计超过 50 行，但本次没有复制为平行模块或增加新的组合逻辑，组合由原网络连接和三个开关完成。

草稿 PR #6 的自动格式化提交 `97a230deb` 另改动了 `ultralytics/cfg/__init__.py` 的等价条件写法，以及
`ultralytics/engine/model.py`、`ultralytics/optim/muon.py`、`ultralytics/utils/ops.py` 的 docstring 换行。
已拉取并复核这些机器人改动；三个模块及模型 YAML 没有被改变。

数值检查使用 Python 3.14.3 / PyTorch 2.15.0+cpu；官方权重和导出使用 Python 3.12.2 / 同版 PyTorch。
CPU 为 Intel Xeon E3-1270 v6，固定 4 线程。当前检查环境 `torch.cuda.is_available()` 为 False。

| 实际检查                                                         | 结果                                                              |
| ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| SPPF、Attention、PSABlock、C2PSA、Concat 的 AST 与指定源分支比较 | 一致                                                              |
| 全部开关关闭，与原 main 模型及其旧 pickle 前向比较               | 逐元素一致                                                        |
| 单独启用 SPPF / C2PSA，与真实源分支的 128×192、384×640 输出比较  | 逐元素一致                                                        |
| 单独启用非零增益 Concat，与源分支输出比较                        | 最大绝对差 1.53e-5 / 7.63e-6，满足 rtol=atol=1e-6；不宣称逐位一致 |
| 三个源分支的 pickle 在组合代码下重载                             | 与各自保存的源输出逐元素一致                                      |
| 三模块初始输出与同权重 SPPF+C2PSA 对比                           | 逐元素一致                                                        |
| 640×640、384×640 完整前向                                        | 有限输出，分别为 1×14×8400、1×14×5040                             |
| 同 10 类基线状态迁移                                             | 499 个已有张量全部加载，仅多出 Concat gain                        |
| 官方 yolo11n.pt 经公共 YOLO.load 加载                            | 493/500；378 个 Backbone/Neck 张量全部逐值一致                    |
| 从已加载模型的 YAML 重建并迁移权重                               | 三项开关和参数量保持                                              |
| 真实 v8DetectionLoss + SGD，gain=0 / 0.25                        | 两次均有 256 个有限梯度张量，三个模块的关键梯度非零               |
| 检测损失对零初始化 gain 的梯度绝对值                             | 约 4.12e-4，SGD 后 gain 确实更新                                  |
| BN 融合对比 / 保存重载                                           | 融合最大差 1.53e-5；重载逐元素一致                                |
| CPU bfloat16 autocast                                            | 有限输出                                                          |
| 非零 gain=0.25 的 TorchScript 实际导出、重载与推理（128）        | 通过 rtol=atol=1e-4；最大绝对差 1.83e-4，输出 1×14×336            |
| 两个模块文件的 docstring 渲染                                    | 无缺失类型                                                        |

SPPF 用独立坐标枚举核对来源排除，并覆盖常量、边界和来源耗尽情形。C2PSA 验证有界聚合列容量与行总质量。
Concat 复用原分支的算子检查：非零增益下 2×2 均值误差不超过 2.39e-7，横向分支保持不变，零增益能收到梯度。
在整个三模块模型中也用 hook 观察实际输入输出，确认非零增益确实改变了语义分布，同时保持传入的粗格均值。

直接由 10 类 YAML 构建时，基线 **2,591,790** 参数，组合 **2,591,791** 参数。
公共加载器恢复官方分类塔隐藏宽度后，二者分别为 **2,607,070** 和 **2,607,071**；比较口径保持一致。

下面为同一进程中串行测量的 FP32、batch=1、640 前向：每个模型预热 2 次，再取 7 次的中位数。
Concat 增益设为 0.25，未通过跳过零增益路径隐去其开销；包含解码，不包含图像处理或 NMS。

| 模型                  | CPU 前向中位数 |
| --------------------- | -------------- |
| 原始基线              | 89.81 ms       |
| C2PSA + Concat        | 118.77 ms      |
| SPPF + C2PSA + Concat | 179.09 ms      |

这是本机短测，不能推导 GPU FPS、训练时长或 VisDrone 精度。完整 VisDrone 训练、CUDA AMP/DDP、
ONNX/TensorRT/CoreML 转换和 GPU 峰值显存没有验证，也没有生成组合模型的 mAP 指标。

本地复核材料位于忽略目录 `.worktrees/_validation_20261118/`：`verify.py`、`pretrained.py`、三个源分支输出、
`combined.json`、`pretrained.json` 和实际导出的 `combined.torchscript`。沿用原实验的验证逻辑，不增加跟踪的测试脚手架。
在组合 worktree 运行：

```powershell
python ../_validation_20261118/verify.py combined
& 'F:/DE/Python/Python312/python.exe' ../_validation_20261118/pretrained.py
```

格式检查使用仓库工作流的 Ruff 0.14.1 完整规则及 ignore 列表、Prettier 3.9.9 与 `git diff --check`。
本地检查和远端自动审查分开报告；本文的工程验证不代表远端 CI 已通过。
