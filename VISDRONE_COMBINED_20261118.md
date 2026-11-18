# YOLO11n / VisDrone 容量版 C2PSA 消融（2026-11-18）

分支：`VisDrone-capacity-ablation-20261118`。
工作目录：`C:/Users/Administrator/yolov11n_VisDrone/.worktrees/VisDrone-capacity-ablation-20261118`。
训练入口：下表八个独立 YAML，均为 10 类、n 规模，顺序固定为 SPPF→C2PSA。

## 组合来源与边界

从 `origin/main` 的 `55989f76c1e85c192ba12cb3e533e3de842a2301` 建立独立 worktree，逐项复用以下分支的模块实现。
再快进复用 `SPPF-C2PSA-Concat-20261118` 的 `c8ac0f3404844a536cadb70b50662caeb9daf820`。
本次增加七个明确开关的 YAML，并让八组配置显式保留 YOLO11 的 SPPF 输入激活。
三个优化的池化选择、容量分配和语义重组公式不变；SPPF 新增一个末尾的可选 `act` 参数。
用户本次反馈三个单项各有约 0.3%–0.5% mAP50 提升，两个三项组合均未超过基线；没有在本机重新测量。
本分支用于定位交互，不保证组合增益。

| 模块来源          | 核对的源提交                               | 原位置与用途                                           |
| ----------------- | ------------------------------------------ | ------------------------------------------------------ |
| `SPPF-20261117`   | `809c94906e47f2ce218cf5b9bb19de56188de4fb` | Backbone 第 9 层：嵌套窗口排除先前选中的空间来源       |
| `C2PSA-20261116`  | `749a3e9728c7b1e2725d1f894fac548f6b1e58cf` | Backbone 第 10 层：CSP 旁路生成 key 容量，限制聚合集中 |
| `Concat-20261116` | `106b8bddde43ad08ea1db06049ffc9e029d4436f` | Neck 第 15 层：横向特征引导语义重组，保持 2×2 均值     |

实施前的文献与全部历史分支审查保留在对应来源分支的 `VISDRONE_RESEARCH_20261116.md` 和
`VISDRONE_RESEARCH_20261117.md`。本组合使用容量版 C2PSA；`C2PSA-20261117` 的 value 范围约束不在本次范围内。

## 对照中必须固定的 SPPF 激活

本次完整模型验证发现：现有解析器只对短 SPPF 参数列表恢复 YOLO11 的输入 SiLU；旧实验为了传入
`distinct_pool` 而扩展参数列表，会让 `cv1.act` 保持为 Identity。即使把三个优化开关全部关闭，
这种长参数配置也不等于原始 YOLO11。激活没有 state_dict 张量，权重迁移数量无法检出这个差异。

本分支在 SPPF 所属构造函数中增加末尾参数 `act=False`，八组 YAML 均显式传入 True：
`[1024, 5, 3, False, distinct_pool, True]`。旧短参数 YOLO11、默认 YOLO26 和旧 checkpoint 的行为不变；
新消融的八组都使用 SiLU，只改变三个优化开关，不另改解析器或训练循环。

因此，新三项配置与旧 `SPPF-C2PSA-Concat-20261118` 的激活行为有意不同；旧 SPPF 单项也包含这个混杂变量。
首次运行应在本分支统一复核基线和三个单项，再与两两、三项结果比较，不能直接把旧指标填进新消融表。
若要复现旧 SPPF/三项配置，可将末尾 act 改为 False，但必须单独标记初始化结构，不能混入本表的 SiLU 对照。

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

所有配置位于 `ultralytics/cfg/models/11/`，文件名均为 `yolo11n-visdrone-<后缀>.yaml`。
三个布尔值依次控制第 9 层 SPPF 的 `distinct_pool`、第 10 层 C2PSA 的 `capacity_attention`、
第 15 层 Concat 的 `conservative`，不交换层号。

| 方案                | 文件名后缀              | distinct_pool | capacity_attention | conservative |
| ------------------- | ----------------------- | ------------- | ------------------ | ------------ |
| 基线                | `baseline`              | False         | False              | False        |
| 仅 SPPF             | `sppf-distinct`         | True          | False              | False        |
| 仅容量 C2PSA        | `capacity`              | False         | True               | False        |
| 仅 Concat           | `conservative`          | False         | False              | True         |
| SPPF + 容量 C2PSA   | `sppf-capacity`         | True          | True               | False        |
| 容量 C2PSA + Concat | `capacity-conservative` | False         | True               | True         |
| SPPF + Concat       | `sppf-conservative`     | True          | False              | True         |
| 三项全开            | `combined`              | True          | True               | True         |

已有的 `C2PSA-SPPF-20261118` 和 `C2PSA-Concat-20261118` 分支使用 20261117 的 value 范围约束版本，
不能替代本表的容量版两两消融。普通 SPPF/C2PSA 在开关为 False 时仍保留在网络中。

在本 worktree 根目录、原 CUDA 环境中运行，先确认 `ultralytics.__file__` 指向本 worktree。
将下例保存为训练脚本后执行；Windows 的多进程数据加载需要保留 main 保护。
把旧基线的 `args.yaml` 和统一预训练权重改为实际绝对路径。数据路径也应能从本 worktree 正确解析。
样例复用已有训练参数，不另外指定 epochs、优化器或增强；首次运行包含完整八组。

```python
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds

if __name__ == "__main__":
    train_args = YAML.load("C:/your_runs/baseline/args.yaml")
    weights = "C:/your_weights/yolo11n.pt"
    for key in ("model", "pretrained", "resume", "cfg", "project", "name", "save_dir", "exist_ok", "seed"):
        train_args.pop(key, None)
    variants = (
        "baseline", "sppf-distinct", "capacity", "conservative",
        "sppf-capacity", "capacity-conservative", "sppf-conservative", "combined",
    )
    for seed in (0, 1, 2):
        for variant in variants:
            init_seeds(seed, deterministic=train_args.get("deterministic", True))
            cfg = Path("ultralytics/cfg/models/11") / f"yolo11n-visdrone-{variant}.yaml"
            model = YOLO(cfg).load(weights)
            layers = model.model.model
            print(variant, seed, layers[9].distinct_pool, layers[10].capacity_attention, layers[15].conservative)
            model.train(
                **train_args,
                seed=seed,
                resume=False,
                project="runs/visdrone_capacity_ablation",
                name=f"{variant}_seed{seed}",
                exist_ok=False,
            )
```

已有本分支同协议的基线、单项和三项结果后，可把 `variants` 缩成三个两两组合，避免重复训练。
每次循环重新构建并加载同一权重，不接着上一组训练。
在模型构建之前设种子，是为了让 80 类预训练权重转 10 类时随机初始化的检测头也配对一致。
本分支保持原层序，无须执行换序分支的第 9/10 层权重交换。

使用目标 YAML 先构建再 `.load(...)`，保证组合结构真正启用。直接 `YOLO("旧分支/best.pt")` 会沿用旧模型结构。
如从某个已有实验权重继续微调，也用目标 YAML 加载该权重，并将该初始化条件单独报告；不混合三个独立训练的
完整 state_dict。新组合训练不使用旧实验的 `resume=True`。

固定数据划分、预训练来源、分辨率、实际 batch、训练步数、增强、优化器和评估配置，至少复核相同的 0/1/2 种子。
检查旧参数的 `batch` 为固定正整数，`time` 未限制训练时长，并记录实际停止轮数；不要让自动 batch、
提前停止或时间上限悄悄改变比较条件。确需改 batch 时，所有对照组一起调整并重新训练。

每组保留 `args.yaml`、`results.csv`、`weights/best.pt` 和 `weights/last.pt`。
本仓库 `best.pt` 按 mAP50:95 选取，比较时统一选用 best.pt 指标、末轮指标或逐轮最高 mAP50。
对相同 seed 计算相对基线的差值，再报告均值和各次结果。若单项提升、SPPF+容量 C2PSA 不提升，
优先检查 P5 两项交互；若该组合提升而三项不提升，再检查 Concat 与上游语义的交互。

训练后可直接核对 checkpoint 中保存的开关和 Concat 增益：

```python
from ultralytics import YOLO

layers = YOLO("runs/visdrone_capacity_ablation/capacity-conservative_seed0/weights/best.pt").model.model
print("SPPF / capacity / Concat:", layers[9].distinct_pool, layers[10].capacity_attention, layers[15].conservative)
if layers[15].conservative:
    print("Concat tanh(g):", layers[15].transport_gain.detach().tanh().item())
```

开关和增益只能确认结构及参数状态，不能单独证明涨点原因。C2PSA 限流强度、SPPF 次峰是否对应目标，
以及 Concat 实际更新幅度，仍需结合真实验证图像和逐类误检/漏检分析。

SPPF 的 13×13 unfold 在 640 输入、FP32 时仅展开张量就约 33 MiB/样本，反向与其他中间张量还需额外空间。
Concat 在 P3 进行相似度重组也有开销；参数量几乎不变不能说明推理或训练成本不变。

## 实现与工程验证

复用源分支的三个优化实现；新增七个 YAML 采用现有配置解析器，标准训练入口和 checkpoint 会记录完整结构。
没有增加模型注册类、训练参数、依赖或训练循环。SPPF 的末尾 `act` 参数用于消除已复现的激活混杂，
比修改通用解析器的长度判断或在训练脚本中临时替换激活更明确，并能在 Trainer 重建模型时保留。

本次使用 Python 3.12.2 / PyTorch 2.15.0+cpu，固定 4 个 CPU 线程和种子。
工程验证使用本地官方 yolo11n.pt 和合成输入，不代表 VisDrone 精度。

| 检查                                             | 结果                                                          |
| ------------------------------------------------ | ------------------------------------------------------------- |
| 八组 YAML 归一化三个优化开关后比较               | 完全相同，SPPF 的 act 始终为 True                             |
| 全关闭基线对标准 YOLO11n，同权重、640 输入       | 输出逐元素一致                                                |
| 八组同种子初始化                                 | 499 个公共状态张量逐值一致，启用 Concat 时仅新增零初始化 gain |
| 官方预训练 Backbone/Neck 迁移                    | 每组 378/378 张量逐值一致，无需换层映射                       |
| 按 Trainer 路径重建八组模型                      | 三个开关、SiLU 和所有状态张量保持一致                         |
| 八组 640×640 / 384×640 前向                      | 输出有限，形状分别为 1×14×8400 / 1×14×5040                    |
| 三个两两组合的检测损失反向                       | 255 / 256 / 256 个有限梯度张量，关键模块梯度均非零            |
| 两个含 Concat 的两两组合                         | gain=0 能更新，gain=0.25 确实改变输出                         |
| 三个两两组合的 checkpoint 保存重载               | 输出逐元素一致，包括非零 Concat 增益                          |
| 四组零增益 Concat 与对应无 Concat 优化组         | 输出逐元素一致                                                |
| 原默认 SPPF 与显式 act=True                      | 分别为 Identity / SiLU                                        |
| act=False 重建旧 SPPF 单项并加载源分支记录       | 两种输入尺寸的输出均逐元素一致                                |
| 容量注意力、Concat、SPPF 池化与 forward 算子 AST | 与来源实现一致；SPPF 只增加构造时的激活选择                   |

公共加载后，不含 Concat 优化的四组为 2,607,070 参数，含 Concat 优化的四组为 2,607,071 参数。
本地结果保存于忽略目录 `.worktrees/_validation_capacity_ablation_20261118/validation.json`。
工作区没有用户实验的数据集或训练日志；本机 PyTorch 为 CPU 构建，没有运行完整 VisDrone 训练。
