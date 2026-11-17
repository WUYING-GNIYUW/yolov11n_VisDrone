# C2PSA 组合实验验证（2026-11-17）

分支：`C2PSA-combined-20261117`。
工作目录：`C:/Users/Administrator/yolov11n_VisDrone/.worktrees/C2PSA-combined-20261117`。
主 checkout 与两个原 C2PSA 分支未修改。组合推导和相似性比较见 `VISDRONE_RESEARCH_20261117.md`。

## 修改文件与配置

模块机制仅修改已有 `ultralytics/nn/modules/block.py` 的 Attention、PSABlock、C2PSA。
不新增模块类、依赖、parser 分支或训练循环；容量约束与局部值变化限幅共用原 QKV 和输出投影。

`ultralytics/cfg/models/11/` 中提供以下四份 nc=10、scale=n 配置，均保持原拓扑：

| 文件                                 | 容量限制 | 局部值变化限幅 |
| ------------------------------------ | -------- | -------------- |
| yolo11n-visdrone-baseline.yaml       | 关       | 关             |
| yolo11n-visdrone-capacity.yaml       | 开       | 关             |
| yolo11n-visdrone-c2psa-range.yaml    | 关       | 开             |
| yolo11n-visdrone-c2psa-combined.yaml | 开       | 开             |

由于两个旧分支曾用相同位置的布尔量表达不同机制，必须使用本分支的配置。
本分支参数顺序是 `[1024, 0.5, capacity_attention, range_limit]`；不直接复制旧 range YAML。
若要用旧分支训练出的权重初始化组合，应先用组合 YAML 建模，再调用 `.load(旧权重路径)`。
旧 checkpoint 的直接前向已验证，但直接对旧 range checkpoint 调用 train/resume 可能按其内嵌三项 YAML
重建成不同机制，因此本次不把旧分支 checkpoint 的恢复训练列为兼容功能。

在此 worktree 根目录及你现有的 CUDA Python 环境中训练，保持既有训练协议不变。例如：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-visdrone-c2psa-combined.yaml").load("yolo11n.pt")
model.train(data="your_local_VisDrone.yaml", imgsz=640, epochs=100, batch=16, seed=0, device=0)
```

以上数据路径和超参数只是调用示例。本次未启动完整训练，不把随机张量验证当作检测精度证据。
完整消融应在同一数据、预训练来源、训练协议及种子上比较四组。

## 本地验证结果

Intel Xeon E3-1270 v6，4 CPU 线程；算子及训练步骤验证为 Python 3.14.3 / PyTorch 2.15.0+cpu，
官方权重和 TorchScript 验证为 Python 3.12.2 / PyTorch 2.15.0+cpu。

| 检查                                             | 结果                                                      |
| ------------------------------------------------ | --------------------------------------------------------- |
| 两项关闭，与真实 main 模型输出及旧 pickle 比较   | 逐值完全一致                                              |
| 仅容量限制，与 C2PSA-20261116 的真实模型输出比较 | 128×192、384×640 逐值完全一致                             |
| 仅局部限幅，与 C2PSA-20261117 的真实模型输出比较 | 128×192、384×640 逐值完全一致                             |
| 上述两个旧分支的 pickle 在组合代码下运行         | 与各自原输出完全一致                                      |
| 组合后的聚合流量列和                             | 不超过旧容量预算                                          |
| 等价注意矩阵（包含自身回流）                     | 非负、行和为 1，与组合输出在容差内一致                    |
| 组合的局部变化界                                 | 构造样例最大越界 0；通用限幅随机样例误差 4.77e-7          |
| 非退化限幅样例                                   | 同 head 共享步长，实测范围约 0.377～1，确实触发第二项约束 |
| 同 nc 原模型 state_dict 严格迁移                 | 499/499 张量                                              |
| 官方 yolo11n.pt 公共加载接口                     | 493/499；全部 378 个 backbone 张量保持一致                |
| 完整前向                                         | 640×640 和 384×640 通过                                   |
| 合成标注上的真实 v8DetectionLoss、反向及 SGD     | 255 个参数梯度有限；关键 QKV 梯度非零                     |
| Conv/BN 融合                                     | 最大输出绝对差 7.63e-6                                    |
| 新模型保存/重载                                  | 完全一致                                                  |
| CPU bfloat16 前向                                | 输出有限                                                  |
| SDPA 路径与容量模式的导出分支选择                | 通过；不等于 CoreML 实际转换通过                          |
| TorchScript 导出、重载及输出比较（128）          | 通过                                                      |
| 原 docstring 渲染                                | 无缺失类型                                                |

YAML 直接构建时四组均为 2,591,790 参数；本次公共预训练加载后基线和组合均为 2,607,070 参数。
组合没有新增可训练参数，加载器前后计数不能混作模块增量。

一次配对 CPU FP32 测量：batch=1、640 输入、2 次预热和 7 次测量，中位数为
原始基线 99.37 ms、组合 102.04 ms，约增加 2.7%。样本量小，不代表 GPU 或数据集端到端吞吐。
未做 CUDA/FP16/DDP、ONNX/TensorRT/CoreML 转换、最低 PyTorch 版本或完整 VisDrone 精度实验。

## 可复核材料

复用项目内忽略目录 `.worktrees/_validation_20261117/` 的验证脚本，未增加仓库测试脚手架。
先在两个旧 worktree 保存真实输出，再在组合 worktree 比较，避免只将新实现与自身比较：

```powershell
# C2PSA-20261116 worktree
python ../_validation_20261117/verify.py capture-capacity

# C2PSA-20261117 worktree
python ../_validation_20261117/verify.py capture-range

# C2PSA-combined-20261117 worktree
python ../_validation_20261117/verify.py c2psa-combined
& 'F:/DE/Python/Python312/python.exe' ../_validation_20261117/pretrained.py c2psa-combined
```

结果保存为 `c2psa-combined-2.15.0.json`、`pretrained-c2psa-combined.json`，
并保留两个旧分支的捕获文件及组合 TorchScript 文件。
格式检查使用 Ruff 0.14.1（扩展 F,I,D,UP,RUF,FA）、Prettier 3.9.9 和 `git diff --check`。

上述结果支持组合实现正确，不证明收益可以相加。两项均倾向自身保留，
必须通过四组完整训练判断组合增益，以及是否因过度收缩导致小目标召回下降。
