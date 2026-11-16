# YOLO11n / VisDrone 实现与验证（2026-11-16）

## 交付位置

主项目 C:/Users/Administrator/yolov11n_VisDrone 的 main 保持干净。
遵循项目 AGENTS.md 的独立 worktree 要求，在指定项目目录内部实现：

| 分支            | 工作目录                   | 启用的原模块        | 单模块配置                                                   |
| --------------- | -------------------------- | ------------------- | ------------------------------------------------------------ |
| C2PSA-20261116  | .worktrees/C2PSA-20261116  | Backbone 层10 C2PSA | ultralytics/cfg/models/11/yolo11n-visdrone-capacity.yaml     |
| Concat-20261116 | .worktrees/Concat-20261116 | Neck 层15 Concat    | ultralytics/cfg/models/11/yolo11n-visdrone-conservative.yaml |

两个分支均从 55989f76c1e85c192ba12cb3e533e3de842a2301 独立建立；不互相叠加。
不存在第三个“组合已提升”分支。没有推送远端，没有完整 VisDrone 精度训练。

## 实际修改文件

C2PSA 分支修改 ultralytics/nn/modules/block.py：

- Attention 接收可选容量，计算每个 key 的入流上限与 query 自保持权重；
- PSABlock 透传容量；
- C2PSA 从原旁路 a 估计容量，只在 capacity_attention=True 时启用。
- 保留原类名、参数键和 parser；未添加模型类、模型注册或依赖。

Concat 分支修改 ultralytics/nn/modules/conv.py：

- 原 Concat 增加 conservative 选项；
- 横向特征生成粗格3×3相似度，重组上采样语义的4个子像素，并投影到单元零均值子空间；
- 使用连续布局 batched matmul，避免按空间位置反复复制小矩阵；
- 增加1个零初始化 transport_gain 标量。

每个分支另外包含：

- ultralytics/cfg/models/11/yolo11n-visdrone-baseline.yaml：同一10类n规模基准；
- 上表列出的单模块模型 YAML；
- scripts/verify_visdrone_20261116.py：可重跑的数值约束与真实检测loss验证；
- VISDRONE_RESEARCH_20261116.md：实施前文献审查、全部旧分支、相似性表、数据流/公式/形状；
- 本验证报告。

本机 .git/info/exclude 加入 /.worktrees/，仅用于避免主 checkout 把嵌套工作树识别为未跟踪内容。
这不修改仓库的 .gitignore 或模型。

## 环境及方法

实测解释器 F:/DE/Python/Python312/python.exe，Python 3.12.2，PyTorch 2.15.0+cpu，4 CPU线程。
本次CPU推理测试为batch1、640×640、FP32、未融合整模型前向（包含解码，不含预处理/NMS）。
每项计时2次预热、9次重复取中位数，两个分支的最终计时串行运行。
短测有系统噪声，不能推导GPU FPS、训练时长、显存或部署端性能。

采用随机初始化权重与合成输入/标签，只验证工程路径。没有使用真实预训练checkpoint或VisDrone样本。
80类到10类测试使用随机权重的原始80类YOLO11，验证BaseModel.load的键形状迁移；不是预训练精度测试。
本机可见RTX5060，但可用PyTorch不是CUDA构建；CUDA/FP16、DDP、ONNX/TensorRT均未验证。

## 实测结果

| 项目                            | C2PSA容量分支 |           Concat守恒分支 |
| ------------------------------- | ------------: | -----------------------: |
| 10类原始基准参数                |     2,591,790 |                2,591,790 |
| 改进参数                        |     2,591,790 |                2,591,791 |
| 从10类基准完整迁移的状态张量    |       499/499 | 499/499（另有1个新gain） |
| 80类来源迁移后参数              |     2,607,070 |                2,607,071 |
| 80类来源迁移的骨干/Neck状态张量 |       378/378 |                  378/378 |
| 640前向形状                     |     1×14×8400 |                1×14×8400 |
| 384×640矩形前向形状             |     1×14×5040 |                1×14×5040 |
| 有限梯度参数张量                |           255 |                      256 |
| BN融合后最大绝对误差（64×96）   |       7.63e-6 |          0.0（本次运行） |
| 完整模型内存序列化往返          |    逐元素一致 |               逐元素一致 |
| CPU bfloat16 autocast前向       |      有限输出 |                 有限输出 |
| 单独运行CPU基准前向中位数       |      95.02 ms |                 91.58 ms |
| 单独运行CPU改进前向中位数       |      97.59 ms |                117.29 ms |
| CPU profiler支持算子的基准FLOPs | 6,383,145,600 |            6,383,145,600 |
| CPU profiler支持算子的改进FLOPs | 6,383,569,200 |            6,413,513,601 |

Profiler只计其支持的算术算子，不完整计入pooling、softmax、归一化、内存复制等，不能作为完整GFLOPs。
参数量的10类从头构建口径和80类权重迁移口径不同：原仓库load会恢复来源分类塔的隐藏宽度。
不能以相同参数量推断相同速度；尤其Concat有约28%的CPU前向开销，应在精度增益和GPU实测之后判断取舍。
这两个改进目前都没有mAP50、mAP50:95、APsmall提升证据。

## 数值与训练路径检查

C2PSA：

- 构造高集中度attention，核对所有key的有界部分入流≤容量；
- 每行有界权重与自保持权重之和为1；容量本身与logits均收到有限且非零梯度；
- 均匀注意与单token边界输入；
- 双PSABlock、奇数/矩形空间尺寸、1×1空间输入；
- 关闭选项时还原原始C2PSA数据流。

Concat：

- 零gain与原始concat、整模型初始输出逐元素相同；
- 非零gain=0.8时，2×2均值最大误差1.1920928955078125e-7；
- 非零gain确实改变语义子像素，横向来源保持逐元素相同；
- 常量语义、零横向向量、2×2最小尺寸、不同通道来源；
- 奇数空间尺寸明确报错（实际YOLO stride32对齐输入产生偶数P3尺寸）；
- 零gain梯度约-7.56827；只对语义输出反传时横向引导仍有非零梯度；
- 检测loss对gain的梯度约5.03e-5，验证它能通过真实损失开始学习。

两项整模型检查：

- 使用仓库真实v8DetectionLoss（box/class/DFL、TAL）和合成标签反传、执行一次SGD更新；
- 包含640以及矩形前向，P3/P4/P5输出接口保持；
- 更新后检查BN融合、内存torch.save/load完整模型往返；
- 默认原始yolo11.yaml在main与两个分支上，固定种子、64×96输入的输出SHA256完全一致：
  f4e43ac560a928cb6811ab4d8b1c271e41e44e68ff997f9e34a6ac022b8803ac。
- Ruff 0.14.1，check --extend-select F,I,D,UP,RUF,FA 通过；git diff --check通过。
- 验证脚本因需优先导入当前worktree，对E402作单一文件级豁免；没有吞掉数值或模型断言。

## 复现验证

在对应worktree根目录执行（脚本也会显式把自己的根目录放到sys.path首位）：

```powershell
# C2PSA-20261116
& 'F:\DE\Python\Python312\python.exe' -B scripts\verify_visdrone_20261116.py capacity

# Concat-20261116
& 'F:\DE\Python\Python312\python.exe' -B scripts\verify_visdrone_20261116.py conservative
```

训练时也必须从相应worktree导入，不能让已安装的另一份Ultralytics覆盖本实现。
以下为模板，替换实际存在的本地数据YAML和官方预训练权重路径后运行：

```python
from ultralytics import YOLO
import ultralytics

print(ultralytics.__file__)  # 应位于选定的 .worktrees/<分支>/ultralytics
model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-capacity.yaml")
# Concat分支改为 yolo11n-visdrone-conservative.yaml；控制组用 yolo11n-visdrone-baseline.yaml。
model.load("实际的本地/yolo11n.pt")
model.train(
    data="实际的本地/VisDrone_local.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    seed=0,
    deterministic=True,
    workers=4,
    device=0,  # 必须使用已经验证支持该GPU的CUDA版PyTorch环境
    project="runs/visdrone_20261116",
    name="capacity_seed0",  # 每个方案/随机种子采用不同名称
)
```

这是消融模板，不是经过调参的最佳配置。baseline和每个单模块必须使用相同权重、数据划分、
输入尺寸、训练步数、有效batch、优化器/增强设置，并至少重复种子0/1/2。
设备容量不够时统一调整各方案batch，不只调整单个方案。
VisDrone遮挡分桶需要保留原始标注的遮挡字段；普通YOLO文本标签本身不包含它。
