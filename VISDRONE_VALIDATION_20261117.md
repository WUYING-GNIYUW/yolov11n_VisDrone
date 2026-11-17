# YOLO11n / VisDrone 实现与验证（2026-11-17）

## 交付位置与使用

两个实验均在指定项目目录内部的独立 worktree，基于同一个 origin/main，主 checkout 的 main 保持干净。

| 分支           | 工作目录                  | 模型配置                                                      |
| -------------- | ------------------------- | ------------------------------------------------------------- |
| SPPF-20261117  | .worktrees/SPPF-20261117  | ultralytics/cfg/models/11/yolo11n-visdrone-sppf-distinct.yaml |
| C2PSA-20261117 | .worktrees/C2PSA-20261117 | ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-range.yaml   |

每个分支包含同一个 `yolo11n-visdrone-baseline.yaml`，用于 10 类独立消融。
SPPF 在 Backbone 层 9 启用 `distinct_pool=True`；C2PSA 在 Backbone 层 10 启用 `range_limit=True`。
先分别训练两个实验，不据此假定组合有效。建议先跑 C2PSA，因为本次 SPPF 的 CPU 开销明显。

训练前在对应 worktree 根目录运行 Python，确认导入路径来自该 worktree。将下面数据路径替换为你现有的
本地 VisDrone YAML，并使用你已有的训练协议、预训练来源、CUDA Python 环境；本次未启动完整训练。

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-range.yaml").load("yolo11n.pt")
model.train(data="your_local_VisDrone.yaml", imgsz=640, epochs=100, batch=16, seed=0, device=0)
```

SPPF 分支将模型路径改为 `yolo11n-visdrone-sppf-distinct.yaml`；基线改为 `yolo11n-visdrone-baseline.yaml`。
上述超参数只是调用示例；正式比较必须沿用你已有结果的完整协议，并保留原始/两个新实验的结果。
两个变体都不是零初始化恒等改动，训练前行为会改变。

## 实际修改文件

每个分支仅修改一个已有 Python 文件：`ultralytics/nn/modules/block.py`。

- SPPF：在原类内部增加可选嵌套窗口选峰路径，后一个尺度排除前面已经使用的空间来源；输出通道和原卷积不变。
- C2PSA：原 C2PSA 启用其 Attention 的值更新预算；统计局部 value 极差，以一个 head 共享步长收缩 O-V。

每个分支另外增加基线 YAML、单模块 YAML、本验证报告和 `VISDRONE_RESEARCH_20261117.md`。
不增加模块注册、解析器分支、依赖或训练循环。接口文档放在已有 docstrings；没有新增公开类，参考页桩不变。
原生模型 YAML 没有继承语法，因此两份完整 YAML 用于可复现实验，不为减少配置行数添加新的解析抽象。
验证脚本复用此前实验的真实模型/损失验证方式，作为本地检查材料保存在忽略的 `.worktrees/_validation_20261117/`，
不向项目测试目录增加脚手架；这里检查的是新算子的数值约束和完整模型接口。

## 已完成验证

环境：Intel Xeon E3-1270 v6，4 CPU 线程；主要数值验证为 Python 3.14.3 / PyTorch 2.15.0+cpu，
官方权重与最终 TorchScript 检查使用 Python 3.12.2 / PyTorch 2.15.0+cpu。
系统可识别 RTX 5060，但现有 PyTorch 环境没有 CUDA。另一套 PyTorch 2.5.1 环境缺少 cv2，
因此未完成旧版本端到端检查，也未宣称验证了 PyTorch 1.8 / Python 3.8 的最低支持组合。

| 检查                                         | SPPF                             | C2PSA              |
| -------------------------------------------- | -------------------------------- | ------------------ |
| 关闭变体时与真实 main 提交生成的模型逐值比较 | 完全一致                         | 完全一致           |
| main 生成的旧 pickle 在新代码下前向          | 完全一致                         | 完全一致           |
| 同 nc 基线 state_dict 严格加载               | 499/499                          | 499/499            |
| 官方 yolo11n.pt 经公共 YOLO.load 加载        | 493/499；backbone 378 个全部一致 | 同左               |
| 640×640 完整推理、384×640 矩形输入           | 通过                             | 通过               |
| 合成标注上的真实 v8DetectionLoss、SGD 步骤   | 通过；255 个参数梯度有限         | 同左               |
| Conv/BN 融合输出比较                         | 最大绝对差 7.63e-6               | 7.63e-6            |
| 新模型保存/重载                              | 完全一致                         | 完全一致           |
| CPU bfloat16 自动混合精度前向                | 有限输出                         | 有限输出           |
| TorchScript 实际导出、重载、输出比较（128）  | 通过                             | 通过               |
| SDPA 替代前向路径                            | 不涉及                           | 与常规路径容差一致 |
| 原 docstring 渲染                            | 无缺失类型                       | 无缺失类型         |

SPPF 的独立坐标枚举验证确认按空间索引排除，且第一尺度与原 5×5 max pooling 完全相同。
三个不同位置的峰值 9/8/7，在原始池化会重复选到 9 的测试中心，修改后依次取得 9/8/7。
覆盖负常量、边界、1×1、1×2 和奇数矩形；候选耗尽时回到中心值，NaN 不会被作为 padding 吞掉。

C2PSA 的值更新上界实测最大数值误差为 4.77e-7；同 head 各通道保持共同非负步长。
覆盖常量、单位置特征、零更新、梯度、以及“某个通道无更新时不冻结其他通道”。
该约束只针对加 PE/proj 前的 O'-V，不声称最终输出被局部 min/max 夹住。

640 输入的共同形状：SPPF/C2PSA 是 B×256×20×20；三个 Detect 输入为
B×64×80×80、B×128×40×40、B×256×20×20，解码输出 B×14×8400。

## 参数和性能

同为 10 类、直接从 YAML 构建时，原始与两个变体都是 **2,591,790 参数**；没有新增可训练参数。
走现有公共预训练加载器后，本次三者均为 **2,607,070 参数**，不能把加载前后计数混用为模块开销。
原加载器报告的 493/499 来自 80→10 类迁移；本次重点检查所有 378 个 backbone 张量完全相同。

下表为随机输入 batch=1、640、CPU FP32、预热两次后七次前向的中位数，基线与变体在各自进程中配对测量。
这不是 VisDrone 端到端吞吐或 GPU benchmark，不包含读图/NMS；不同实验进程的绝对时间不可直接排名。

| 配对实验 | 原始基线  | 变体      | 解释                                              |
| -------- | --------- | --------- | ------------------------------------------------- |
| SPPF     | 91.58 ms  | 157.94 ms | 约增加 72.5%，不能称为轻量加速改动                |
| C2PSA    | 103.39 ms | 103.20 ms | 本次差异约 0.2%，属于小样本测量噪声，不能声称加速 |

SPPF 仅 13×13 unfold 张量在 640/FP32 时就约 33 MiB/样本，mask/scatter/反向还需额外内存。
本次未测 CUDA 显存峰值；实际训练可能需要减小 batch，比较指标时须考虑 batch 改动的影响。
C2PSA 不增加 N×N 注意矩阵数量，只加局部统计及逐元素预算。未用参数量或常见 FLOPs 计数器掩盖内存算子开销。

## 可复核材料与边界

本地 `.worktrees/_validation_20261117/verify.py` 以当前 worktree 为导入根目录，
基线模式在主 checkout 运行，变体模式分别在对应 worktree 运行：

```powershell
# 主 checkout，仅产生忽略目录中的验证材料
python .worktrees/_validation_20261117/verify.py baseline

# 对应 worktree
python ../_validation_20261117/verify.py sppf-distinct
python ../_validation_20261117/verify.py c2psa-range
```

同目录的 `pretrained.py`、两份算子 JSON、两份预训练 JSON 与各变体 `.torchscript` 保留实际结果。
官方预训练权重读取自 Ultralytics assets 的 v8.3.0/yolo11n.pt；没有使用未训练输出编造检测指标。
TorchScript 在当前 PyTorch 有弃用提示，测试仍成功；ONNX/TensorRT/CoreML 实际转换、CUDA AMP/DDP
和完整 VisDrone 精度训练均未验证。SDPA 路径检查不等于已经完成 CoreML 导出。

格式要求为 Ruff 0.14.1 的 `--extend-select F,I,D,UP,RUF,FA`、同版本 format、Prettier 3.9.9
以及 `git diff --check`。最终变更保持每个分支一个模块消融；本报告的结论限于接口、数值和实测 CPU 开销。
