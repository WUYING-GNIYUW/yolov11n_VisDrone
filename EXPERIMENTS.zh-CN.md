# YOLO11n 五方法实验：从一项跑到全组合

这份工作副本回答两个不同的问题：**“机制能不能接上？”已有工程测试；“接上后对 VisDrone 有没有用？”需要足预算实训。** [32 项清单](experiment_records/matrix.csv)含全关、五个单项及其余 26 个组合。正式分数、增益和速度不从论文移植，也不从小样本外推。

基线仅使用用户仓库 [main 的固定提交 `55989f7`](https://github.com/WUYING-GNIYUW/yolov11n_VisDrone/tree/55989f76c1e85c192ba12cb3e533e3de842a2301)。[原 YOLO11 配置](ultralytics/cfg/models/11/yolo11.yaml)、[TAL](ultralytics/utils/tal.py)、[原始损失](ultralytics/utils/loss.py)等基线源码未改。全关构建的就是原生 `DetectionModel`；开关及其组合只经 [实验模型](dronebench/model.py)、[实验损失](dronebench/loss.py)和[原训练器的子类](dronebench/engine.py)接入。交付是 PyTorch 参考实现，不是五篇论文的官方 YOLO 权重。

**本次验收：**106 项测试通过；32/32 种开关组合都完成带标签反传、真实图片训练、检查点重载及独立验证。32 组均以同一源码、同一数据和一次实际优化器更新完成；每组验证桶分母都是 39 / 52 / 33。下文的批量推理延迟也已实测，但这些权重只用 4 张训练图更新过，不能据此给方法排名。

## 先跑起来

从**解压后的项目根目录**执行命令，确保导入本地的 `ultralytics`，不要另装同名 PyPI 框架替代本仓库。每个方法包只需 PyTorch 即可从项目根独立导入，不必构建 YOLO。

**本机已有环境**（只测了单 CPU）：

```bash
cd /home/user/work/yolo11n-five
source /home/user/work/yolo11n-env/bin/activate
python -m dronebench --help
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m dronebench smoke --id 11111 --imgsz 96
```

**新机器**：推荐 64 位 Linux、Python 3.13 和匹配的 torch/torchvision。本次环境为 Python 3.13.5、torch 2.9.1+cpu、torchvision 0.24.1+cpu；[实测依赖版本](experiment_requirements.txt)和基线 [pyproject](pyproject.toml) 分开。CPU 安装示例：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
python -m pip install -r experiment_requirements.txt
python -c 'import ultralytics; print(ultralytics.__file__)'
```

GPU 请先按[官方 PyTorch 安装指引](https://pytorch.org/get-started/locally/)安装与驱动匹配的 torch/vision，再装其余依赖。正式命令接受 `--device 0` 或 `--device cpu`；`auto` 表示本机自动选一台设备。**本交付只承诺单设备、标准 PyTorch eager 检测路径**，未验证 CUDA、导出、DDP、TTA 或断点续训。只加载可信本地 `.pt`。未随包附带数据、权重或虚拟环境。

### 正式训练、验证、复测

`VisDrone.yaml` 指本地基线[数据配置](ultralytics/cfg/datasets/VisDrone.yaml)。可用其下载并转换正式数据（约 2 GB），也可用 `--data /绝对路径/数据.yaml` 指向已按相同规则转换的正式 train/val。检查 `images/train` 为 6,471 图、`images/val` 为 548 图；不要用由 val 拆出的工程数据作正式训练。

```bash
python -m dronebench train --id 00000 --data VisDrone.yaml
python -m dronebench train --id 10000 --data VisDrone.yaml
python -m dronebench train --id 11111 --data VisDrone.yaml
```

这三条分别运行基线、DM-EFS 单项、全开。推荐先只跑其中一条。每次训练会打印**实际 best 权重和对应 val 命令**；`best.pt` 的选择仍由原生 fitness 决定。以下示例的 `$BEST` 是该次训练打印的权重路径，`$VAL_IMAGES` 是同一份整图验证图片目录：

```bash
python -m dronebench val --weights "$BEST" --data VisDrone.yaml --imgsz 640 --batch 8 --device 0
python -m dronebench.evaluation.latency --weights "$BEST" --images "$VAL_IMAGES" --imgsz 640 --device 0 --warmup 10 --runs 100 --output "$(dirname "$(dirname "$BEST")")/latency.json"
python -m dronebench summarize --runs runs/five --output measured.csv
```

默认对照处方：从零初始化、100 epoch、640、batch 8、seed 0、SGD、FP32、不开 early stopping，其他训练超参见每次运行保存的 `args.yaml`。**这不是用户原先训练的设置**。如有可信的 stock YOLO11n 初始权重，给各组相同的 `--init /path/to/init.pt`（先装载原头，再添加 P2 和各方法自身的先验）。调整预算、学习率或增强时，建议先重新生成整队命令，而不是手工只改一行：

```bash
python -m dronebench matrix --project runs/vis960 --data VisDrone.yaml --imgsz 960 --epochs 100 --seed 42 --device 0 --output queue960.csv
python -m dronebench matrix --hparams '{"lr0":0.005,"mosaic":0.5}' --output queue-lr005.csv
```

`matrix` **只生成** CSV，不自动启动 32 次训练。每行 `train_command`、`val_command` 可以独立执行。`id` 按文本读取，保留前导零；`run` 名含五位开关、分辨率、种子、预算和配置摘要。已有同名 run 会拒绝启动，而不是覆盖或默默另存后去验证旧权重；复测请换 `--project` 或改种子。不同方法的内存需求不同；遇 OOM 请对整个对照队列显式改用同一较小 batch，不把自动降档的训练混入比较。

## 五项分别改变什么

ID 顺序始终是 **DM-EFS / ESOD / QueryDet / RFLA / UGS**。每行第一个链接是可独立导入的方法卡、通用实现和样例；第二个链接是另一位实现者完成的 YOLO11n 适配、单项 JSON、参数及测试。论文模型、数据集、上游代码提交/许可证、没有可核实官方代码的情况，均在各卡的“实验出处/来源”中单列。

| 开关与单项 | 可观测的问题与可检验假设 | 在这个 YOLO11n 上的变化 |
| --- | --- | --- |
| `dm_efs` · `10000` · [方法](dronebench/methods/dm_efs/README.md) / [适配](dronebench/adapters/dm_efs/README.md) | 浅层仍有细节，但目标短边接近 P3 间距；不同目标尺寸是否真的偏好不同层？看逐尺寸投票、端点误差和固定桶召回。 | 加真实 P2/4 预测层，保留原 P3–P5 权重；训练全开，学每图尺度端点，并用每 GT 中心格的逐层定位 IoU 投票；推理按图只执行允许的检测头。PAN 全算，不声称节省其 FLOPs。 |
| `esod` · `01000` · [方法](dronebench/methods/esod/README.md) / [适配](dronebench/adapters/esod/README.md) | 前景集中在少量区域，浅层信号可能足以定位它们。先查块对 GT 的覆盖、重叠和边界漏检，再看端到端时间。 | P2 后的 ObjSeeker 选**特征块**；深层骨干、颈、头只在所选块上运行；暖启可用训练标签，验证/推理不读 GT。空块绕过深层，重叠预测在原图统一 NMS。 |
| `querydet` · `00100` · [方法](dronebench/methods/querydet/README.md) / [适配](dronebench/adapters/querydet/README.md) | 粗层标出的热点能否覆盖细层需要预测的位置？应分别量保留的预测点和它们占用的上下文，而不是只数非零输出。 | 粗到细传**空间位置**，训练稠密监督；推理细层按坐标及邻域执行使用实际头权重的 PyTorch 稀疏卷积，兼容输出仍写回原网格。未省特征生成，参考后端可能更慢。 |
| `rfla` · `00010` · [方法](dronebench/methods/rfla/README.md) / [适配](dronebench/adapters/rfla/README.md) | 小框里没有网格中心，另一种候选几何能否减少拿不到正样本的 GT？检查每 GT 计数、冲突、框外正例和误检。 | 用感受野与框的分布相似度、两轮排名替换 TAL。YOLO 默认半径 `2×stride` 是可改的假设，不是实测有效感受野。记录逐 GT 正例、框外数和未匹配数；无额外推理头。 |
| `ugs` · `00001` · [方法](dronebench/methods/ugs/README.md) / [适配](dronebench/adapters/ugs/README.md) | 定位目标是否集中在零附近，细小偏移是否伴随梯度波动？应测饱和/负边距比例，并检查降熵是否真的降低定位误差。 | 用有符号非均匀支持点的 two-hot 分类和物理期望替换 DFL；训练有熵约束、用熵梯度扰动特征并固定干净分配重跑回归头。推理只走一次干净分支。保留 CIoU，初始化避免对称网格解出零面积。 |

例如要把分配方法拿去别的检测框架，只需 `from dronebench.methods.rfla import hierarchical_assign`，给它同一坐标系的候选点、感受野半径和有效 GT，不需加载本仓库的 YOLO 类。各卡还给出分割、关键点或其他任务尝试时**需要重新定义的接口和失败条件**，不限定在论文数据集，也没有预先筛掉任何方法。

### 组合顺序与必须分开的对照

- **整图 stem → ESOD 块 → 块内骨干和 PAN → DM-EFS 层 → QueryDet 位置 → DFL/UGS 解码 → 整图还原、NMS**。无 ESOD 时整个输入就是一张图；有 ESOD 时尺度预测、分配和查询都在当前块上进行。训练先按干净候选做 TAL/RFLA，再对这些正例做 UGS。`order` 是配置的一部分；未实现的次序直接报错，不会伪装成同一个实验。[张量与坐标契约](dronebench/CONTRACT.md)记录了具体接口。
- P2 使 TAL 自动最短候选边从 **16 px 变成 8 px**。比较 [只加 P2、自动下限](experiment_configs/efs_only_auto.json)、[只加 P2、固定 16](experiment_configs/efs_only_fixed16.json) 与[完整 DM-EFS、固定 16](experiment_configs/dm_p2_fixed16.json)：例如 `python -m dronebench train --config experiment_configs/dm_p2_fixed16.json --data VisDrone.yaml`。RFLA 不读取这个下限。
- RFLA 可能选中框外点。保留原 GT 做 IoU；原 DFL 对负边距截到 0，UGS 则在同一格点单位保留负值，只有超出支持点范围才饱和。[同域均匀](experiment_configs/ugs_signed_uniform.json)、[仅非均匀](experiment_configs/ugs_nonuniform_only.json)、[再加熵](experiment_configs/ugs_nonuniform_um.json)和完整 UGS 可逐项对照。**同域均匀不是原生 DFL**。编码和解码不共享定义的实验不能解释为分配器差异。
- 跨 640/960 比较用同组原图 GT，不能随当前输入重新定义“小框”。若 DM-EFS 的目标尺寸超出默认票表 `max_size=640`，会饱和；如要扩表，把改动列为独立配置维度，不混入只改变输入分辨率的对照。

## 评测、结果和使用边界

- 正式 val 是原生 YOLO mAP50–95：NMS `conf=.001, iou=.7, max_det=500`。附加召回用 GT 原图短边乘 `640 / 原图长边`，分 `[0,8)`、`[8,16)`、`[16,∞)`；在一张图上统一做 `conf≥.25`、最多 300 个检测、按置信度排序的同类一对一 `IoU≥.5` 匹配，再按 GT 桶计入 TP。空桶未定义，不当 0。[协议与延迟边界](dronebench/evaluation/README.md)说明了 letterbox 还原、边界框和检测上限。
- 原 val 的 548 图、38,759 有效框分为 **17,003 / 13,641 / 8,115**。这是标签核对，不是模型召回。此处没有用官方 ignore-region 对预测做额外处理；若需对齐官方 AP/AR，应另外用[官方 VisDrone 评测](https://aiskyeye.com/evaluate/object-detection-2022/)，不用本表冒充官方分数。
- 每次训练留存处方、源代码/初始化/数据哈希、真实 batch、epoch 和 optimizer 更新数、日志与检查点。开启方法时另存 `routing.jsonl`，其计数是**每轮最后一个训练 batch**，不是数据集总量。`val` 将每次测量单独保存；只有预算已完成、同一最佳权重、源码、分辨率、数据与 split 一致的验证才回填该 run 的正式结果。`summarize` 不读 smoke 指标。
- 正式主矩阵的 `delta_mAP = m(S) − m(全关)`；若单项齐全，`interaction_mAP = delta_mAP − Σᵢ[m(单项 i)−m(全关)]`，均用 0–1 原单位。二项时是差中之差；更多方法时是相对单项加和的**总非加和项**，不分离各阶效应。缺参照留空。只比较相同预算、种子、实际更新数、初始化、数据内容、设备/软件及评测协议的 run；非默认子消融另组，不能自动借用主矩阵基线。不同种子独立成组，不将一次差异宣称为稳定增益。
- 延迟按 batch=1，从本地图像读取/解码、预处理、设备搬运、推理和 NMS 到 CPU 框，预热后同步计时，记录 p50/p95、权重/图片摘要与设备线程。**只在同 `latency_group` 间比较**。训练步数、FLOPs、候选个数都不是实测延迟。

## 工程验收与重跑

| 检查 | 对应证据与作用 |
| --- | --- |
| 全关回归 | [逐张量比较](experiment_records/baseline-regression.json)：同种子分别构建原生与全关模型，逐字节比较初始状态、训练输出及 loss、梯度、一次 SGD 后状态、验证解码和 NMS。 |
| 组合构建/反传/推理 | [32 项张量冒烟与子对照](experiment_records/tensor-smoke.json)：B=2，96 px，记录真实更新、各层 active 数、局部/原图网格、分配和裁块信息；额外强制稀疏路由覆盖非平凡交互。 |
| 真实数据、训练器与检查点 | [逐项端到端验收](experiment_records/real-smoke.json)：官方 val 的不重合序列拆出 4 图/295 框训练与 2 图/124 框验证，独立运行每项 1 epoch，重载权重并验证原图及三桶 GT 数。不能用这些训练权重或分数作正式比较。 |
| 批量推理和计时 | [32 项逐次 CPU 耗时](experiment_records/latency-smoke.json)：各自重载上行的 best 权重，i5-8250U、单线程、FP32、batch=1、96×96，3 次热身、10 次计时；两张图交替。320 次请求均未输出置信度达 0.25 的框，不代表训练后路由、NMS 负载或正式 640 延迟。 |
| 数值、边界与测量工具 | [106 项测试输出](experiment_records/test-output.txt)；[测量工具烟测](dronebench/evaluation/experiment_records/metric-smoke.json)另外记录 640 输入的 CPU 延迟管线及 640/960 分桶一致性，不是训练好的网络性能。 |

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
python -m pytest -q -c /dev/null --noconftest --import-mode=importlib -p no:cacheprovider dronebench/methods dronebench/adapters dronebench/evaluation/tests dronebench/checks
python -m dronebench.checks.test_regression --output /tmp/all-off.json
python -m dronebench smoke --all --output /tmp/tensor-smoke.json
python -m dronebench.evaluation.smoke_data --zip /path/to/VisDrone2019-DET-val.zip --out /tmp/drone-smoke-data
PYTHON_CPU_COUNT=2 python -m dronebench.e2e_smoke --data /tmp/drone-smoke-data/smoke.yaml --project /tmp/drone-smoke-runs --output /tmp/real-smoke.json
PYTHON_CPU_COUNT=2 python -m dronebench.checks.latency_smoke --project /tmp/drone-smoke-runs --images /tmp/drone-smoke-data/images/val --output /tmp/latency-smoke.json
```

给 pytest 设置 `VISDRONE_VAL_ZIP=/path/to/VisDrone2019-DET-val.zip` 可同时执行两项依赖原始标注包的测试。夹具和训练输出目录须为新目录。离线开关用于本地冒烟；需要下载正式数据时先取消离线开关。项目和新增代码沿用仓库 [AGPL-3.0 许可证](LICENSE)。
