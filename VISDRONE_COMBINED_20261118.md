# S-P-20261118：SPPF + 容量版 C2PSA

本分支固定运行一个实验，训练入口为 `ultralytics/cfg/models/11/yolo11n-visdrone.yaml`。
类别数为 10，规模为 n，层序为 SPPF → C2PSA；Concat 优化仅作用于第 15 层 P3 融合。
S 表示 SPPF，P 表示 C2PSA-20261116 的容量版，C 表示 Concat-20261116。
无需手动修改优化开关，标准训练器会保留 YAML 中的配置。

## 本分支设置

| 设置                       | 固定值 |
| -------------------------- | ------ |
| SPPF `distinct_pool`       | True   |
| C2PSA `capacity_attention` | True   |
| P3 Concat `conservative`   | False  |
| SPPF `cv1.act`             | SiLU   |
| SPPF `cv2.act`             | SiLU   |

本分支使用与标准 YOLO11 相同的 SPPF 输入激活。
`False` 表示保留普通模块，并非删除该层。此处 Identity 只指 `SPPF.cv1.act`。

## 五个独立分支

五个分支都使用相同的训练入口文件名；Git 分支决定具体实验。

| 分支                | 实验                | S/P/C 开关        | SPPF 输入激活 |
| ------------------- | ------------------- | ----------------- | ------------- |
| S-Identity-20261118 | 原始池化 + Identity | False/False/False | Identity      |
| S-SiLU-20261118     | 优化池化 + SiLU     | True/False/False  | SiLU          |
| S-P-20261118        | SPPF + 容量 C2PSA   | True/True/False   | SiLU          |
| P-C-20261118        | 容量 C2PSA + Concat | False/True/True   | SiLU          |
| S-C-20261118        | SPPF + Concat       | True/False/True   | SiLU          |

## 交给服务器 Codex 的指令

> 请在分支 `S-P-20261118` 上开展 VisDrone 消融训练。先读取 AGENTS.md 和本说明，
> 使用 `ultralytics/cfg/models/11/yolo11n-visdrone.yaml` 构建模型，再加载与此前基线相同的初始预训练权重。
> 复用服务器上基线的数据划分、训练参数和评估协议；在构建模型前设置相同种子。
> 先核对当前分支、实际导入代码位置、三个优化开关和 SPPF 激活，再开始训练。
> 本次只运行当前分支的一个实验，不循环运行八组配置。不要从旧 best.pt 直接构建模型或恢复旧实验。
> 输出目录使用当前分支名和种子，并记录实际训练轮数、args.yaml、results.csv、best.pt、last.pt。

若服务器已有本地改动或运行中的实验，应为此分支建立独立 worktree，保留原实验状态。
从本分支目录启动 Python，并检查 `ultralytics.__file__` 确实指向该目录。
运行所需的数据、原基线 args.yaml、初始权重和 CUDA 环境沿用服务器现有资源；本仓库不硬编码它们的位置。

以下示例仅供 Codex 适配服务器路径，使用原基线的种子运行一次。

```python
from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds

if __name__ == "__main__":
    train_args = YAML.load("/absolute/path/to/baseline/args.yaml")
    weights = "/absolute/path/to/yolo11n.pt"
    train_args["data"] = "/absolute/path/to/VisDrone.yaml"
    seed = int(train_args.pop("seed", 0))
    for key in ("model", "pretrained", "resume", "cfg", "project", "name", "save_dir", "exist_ok"):
        train_args.pop(key, None)
    init_seeds(seed, deterministic=train_args.get("deterministic", True))
    model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone.yaml").load(weights)
    layers = model.model.model
    print(layers[9].distinct_pool, layers[10].capacity_attention, layers[15].conservative)
    print("SPPF cv1 activation:", layers[9].cv1.act)
    model.train(
        **train_args,
        seed=seed,
        resume=False,
        project="runs/visdrone_branch_ablation",
        name=f"S-P-20261118_seed{seed}",
        exist_ok=False,
    )
```

固定正整数 batch、实际训练步数和停止规则；统一处理自动 batch、time 上限和早停，避免额外变量。
复核随机波动时，各分支配对运行同一组种子，例如 0、1、2。
新分支实验从共同初始权重开始；已完成结果不要覆盖，重复运行时明确新目录名。
本分支只有一个 VisDrone 实验 YAML；`yolo11n.yaml` 仍是原始标准模型，不是本实验入口。
`YOLO("yolo11n.pt")` 直接采用权重文件中的结构，不能替代先构建目标 YAML 再 `.load(...)`。

## 怎样比较实验

激活与池化的四种对照如下。新建的两个 S 分支补齐缺少的两格。

| 池化     | SiLU                  | Identity                   |
| -------- | --------------------- | -------------------------- |
| 原始池化 | main 标准 YOLO11 基线 | S-Identity-20261118        |
| 优化池化 | S-SiLU-20261118       | SPPF-20261117 的旧优化配置 |

在同一列比较池化效果，在同一行比较激活效果。
不能只用 S-Identity 与 S-SiLU 相减来归因，因为它们同时改变了池化和激活。
旧指标可在训练结构、初始化、数据和评估协议可核对一致时复用；否则补跑对应对照。
main 的标准 YOLO11 配置本来就是 SiLU，不需要因本次发现而修改 main。

三个两两组合统一使用 SiLU，并使用同一容量版 C2PSA。
将其与同条件的 main、S-SiLU、C2PSA-20261116、Concat-20261116 对照，观察哪个组合开始失去收益。
旧 `SPPF-C2PSA-Concat-20261118` 使用 Identity，不能直接充当此处 SiLU 组合的三项对照；
若需要三项 SiLU 对照，已有共享分支 `VisDrone-capacity-ablation-20261118` 的
`yolo11n-visdrone-combined.yaml` 可用。旧 value 范围约束版 C2PSA 的两两分支不能替代容量版。

统一选用 best.pt 指标、末轮指标或其他预先确定的规则；本仓库 best.pt 按 mAP50:95 选取。
保存每个 seed 的结果，先计算同 seed 相对基线的差值，再汇总均值和波动。
训练后用本分支加载输出 checkpoint，核对三个开关和激活；含 Concat 时另记录
`layers[15].transport_gain.detach().tanh().item()`。
开关和增益只表示结构及参数状态，实际特征更新和目标收益仍需验证图像与逐类指标分析。

## 实现来源与验证边界

本分支从 origin/main 建立隔离工作目录，快进复用已验证的共享实现
`VisDrone-capacity-ablation-20261118@cbb9e26d7090de12930d08381d6da163ab1c1211`。
三个算子的来源分别为 SPPF-20261117、C2PSA-20261116、Concat-20261116。
本次只固定实验 YAML 并简化分支说明，不修改共享算子或训练循环。

SPPF 优化在 5/9/13 嵌套窗口排除已选空间来源；容量 C2PSA 限制 key 聚合并把剩余权重回流自身；
Concat 用横向特征引导语义重组并保持 2×2 均值。Concat 的新增 gain 初值为零，但可以获得梯度。
五个分支均已独立通过：固定入口构建、公共初始权重与参考配置一致、训练器重建、
640×640 和 384×640 前向、真实检测损失反向、checkpoint 保存重载。
输出与对应参考配置逐元素一致；两个含 Concat 的组合在 gain=0 时均获得非零增益梯度。
验证记录保存在本地忽略目录 `.worktrees/_validation_five_branches_20261118/`。
本地验证使用 CPU 和合成输入，不代表完整 VisDrone 训练或实际 mAP 收益。
