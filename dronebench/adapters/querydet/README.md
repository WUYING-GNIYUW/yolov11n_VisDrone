# QueryDet 接入 YOLO11n：只把细层检测头按位置执行

在固定基线 `55989f7` 上，单项训练命令是：

```bash
python -m dronebench train --id 00100 --data VisDrone.yaml
```

`00000` 是全关基线。要调整 QueryDet，使用[单项配置](<single.json>)，把 `--id 00100` 换成 `--config dronebench/adapters/querydet/single.json`。`querydet` 是顶层布尔值；子参数只放在 `options.querydet`。未知子键在构造时即报错，不会变成无效果的实验。训练结束后可直接执行终端打印的验证命令；例如令 `BEST_PT` 指向本次训练的最佳权重：

```bash
python -m dronebench val --weights "$BEST_PT" --data VisDrone.yaml --imgsz 640
```

纯 QueryDet **不新增 P2、不改输入分辨率、不换 TAL/DFL、也不分块**。公共入口的组合顺序见[接入契约](<../../CONTRACT.md>)。

## 输入、训练与输出

[适配器](<adapter.py>) 接收同原点、细→粗的特征及步长，一般为 P3/8、P4/16、P5/32；DM-EFS 接入 P2/4 后也支持四层。单项的 TAL 默认候选下限仍是 16 px；若另加 P2，自动值变为 8 px，可用公共 `tal_min: 16` 把这项变化单独控制。相邻步长需为 ≥2 的整数比，粗格覆盖细格，边界按实际 `H,W` 裁掉。`QueryDetAdapter(channels, strides, options)` 持有各非最细层上独立的单通道查询头，**不持有、复制或冻结 Detect 权重**。训练/评估时让 adapter 与真实 Detect 一起切换模式。

| 调用 | 实际行为 |
| --- | --- |
| `raw = adapter(head, feats, level_mask)`，训练 | 调用真实 Detect 的 `forward_head` 在所有层整图算框与分类；不按查询阈值裁掉难例。各非最细层完整算查询 logits。`level_mask[B,L]` 只标记各图/块的检测资格，不影响查询监督。公共损失在分配候选和分类 BCE 两处使用 `raw["active"]`。 |
| `adapter.aux_loss(raw, batch, hw)`，训练 | 用 `hw=(H,W)` 把 `bboxes` 的局部归一化 `xywh` 换为局部像素 `xyxy`，按 `batch_idx` 分组。无须 `img`，不区分类别；某图无框则全负。复用 `query_targets` 与 `query_focal_loss`；每层取格点均值，再对所有查询层等权平均，乘 `loss_weight`。返回未乘 batch 大小的标量。 |
| `raw = adapter(head, feats, level_mask)`，评估 | 最粗层（默认 P5）对所有样本整图算查询。`sigmoid(q)>threshold` 的父格将键传到细层；细层只在上一级给的键上产生新的查询，不能从上下文邻格另开检测或查询。对最终可检测的键，稀疏运行 Detect 的 `cv2` 和 `cv3`。阈值不读取真值。 |
| 解码与验证 | 原 DFL 可直接 `head._inference(raw)`；使用 UGS 时改由其解码器读取同布局的回归 logits。`aux_loss` 对**稀疏评估输出**返回同设备零标量，不为填写验证损失而重新计算稠密查询图；这个零不表示预测准确。 |

原网格没有重排：`raw["feats"]` 是输入特征；`boxes[B,4R,N]`、`scores[B,nc,N]` 为原始 logits，`N=ΣH_lW_l`，先按细→粗层序、每层按行优先（`y*W+x`）。`active[B,N]` 为输出是否可用的布尔掩码。训练保留稠密数值；推理的未激活位置框通道填 0、分类 logits 填 −1000。**占位框不是检测框**；先按 `active` 屏蔽，再使用置信度。若某图无查询，细层保持空集，不回退整图；允许同一 batch 的图有不同结果。

与 ESOD 组合时，`B` 是当前局部块数，`hw` 是块的像素尺寸，层形状、查询位置、预测框全是同一块的坐标。适配器不读取整图、不自己加块偏移；原图还原、跨块 NMS 由宿主完成。零块时宿主在进入深层以前短路。

## 两种位置，分开跟踪

实现直接调用[通用级联与稀疏执行](<../../methods/querydet/sparse.py>)和[通用查询监督](<../../methods/querydet/query.py>)。这里只把 stock `Conv`/`DWConv` 外壳展开为其当前持有的 `conv, [bn], act`。未融合时用真实 BN 的评估统计、SiLU 和分组权重，融合后直接用带融合偏置的卷积。没有单独的稀疏权重副本，不把整张细图送入 Detect 卷积再抹掉结果。

推理的 `raw["query_state"]["levels"]` 按细→粗排列。每级四个字段都是零起始的 `int64 (b,y,x)` 坐标：

| 字段 | 数什么 |
| --- | --- |
| `keys` | 当前收到的候选格；最粗层为整张起始图。非最细层只从这些位置的输出中挑选可继续下传的父格。 |
| `context` | 为 `keys` 扩展、裁边、合并重叠后的邻域（含键）；它是计划中的卷积支持，不是额外检测。除最细层外，它也用于查询头。 |
| `det_keys` | 在 `keys` 中，所属图的本层 `level_mask` 为真的位置；只在这里写框与类别，即 `active` 的来源。 |
| `det_context` | 该层检测头实际计算的空间支持。统计预测数用 `len(det_keys)`，统计每层卷积处理的格点数用 `len(det_context)`。 |

例如[路由测试](<test_adapter.py>)的四层、双图示例中，P2→P5 的候选数为 `[4,4,8,24]`，可检测数是 `[4,0,4,12]`。P3 虽不做检测，仍传给 P2 四个键；P2 半径 2 的检测上下文共有 **36** 格而不是 4 格。该测试还对照真实稠密卷积与解码值，并用 forward hook 检查只有最粗层的已启用图进入整图检测卷积。

**没省掉的计算：** 训练时仍是稠密骨干、PAN、检测和新增的查询监督；推理也已生成全部传入的特征，最粗查询是稠密的，最粗检测对选中图整层执行。即使某层不做检测，该层的查询仍可能占用上下文。还要付出索引去重、邻域收集和写回完整网格的代价；兼容的 DFL/UGS 解码及 NMS 输入并未压缩。上下文若几乎铺满，PyTorch 点执行可能比优化的稠密卷积**更慢**。半径不足以覆盖头的感受野时，中心预测也可能不同；未选邻居在子流形卷积中按零处理。

## 子参数与对照

| 子键 | 缺省值与含义 |
| --- | --- |
| `threshold`, `context_radius` | `0.15`（仅评估，严格大于）、`2`（格，含中心最多 5×5）。阈值为 1 时不会有任何细层键。 |
| `hidden_channels`, `depth` | `32`、`4`；每个查询头有 4 个 3×3 隐层和一个 3×3 单通道输出。`hidden_channels: null` 按输入宽度；`depth: 0` 只保留输出卷积。 |
| `scale_ranges` | 缺省每个查询层为 `[0,4*stride)`；也可依非最细层的细→粗顺序传 `[lo,hi]` 列表，按 `lo ≤ sqrt(w*h) < hi` 筛选像素框。三层单项默认就是 P4 的 `[0,64)`、P5 的 `[0,128)`。这些是 YOLO 适配选择，并非 TAL 的候选边长参数。 |
| `radius_factor`, `radius_px` | `1.0`、`null`。训练正例半径默认取最近合格框的 `sqrt(w*h)*radius_factor`；指定 `radius_px` 则改用固定的像素半径，不能同时给非默认系数。这两个量都不是推理的上下文半径。 |
| `alpha`, `gamma`, `loss_weight` | `0.25`、`2.0`、`1.0`，依次为二元 focal 参数与多层查询均值的乘数。原分类/定位各 loss 权重保持不动。 |

来源是 [QueryDet，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) 图 3、§3.2–3.3。原论文和作者 VisDrone 设置（RetinaNet/FPN、从 P4 查向 P2）的实验出处、与官方代码的差异由[方法卡](<../../methods/querydet/README.md>)单列；这里从实际提供的最粗层开始，沿用 YOLO 分类/回归损失、宽度及网格，不是原配置重跑。

**可检验的假设：** 若粗层能标出大部分需要细查的区域，而大多数细层邻域可省，才值得预期速度与召回的折中。用同一批图、同一固定尺寸分桶的 GT、同一台设备和预热条件，分别量各层 `det_keys`/`det_context` 占比、漏掉的 GT 中心、最终召回及 batch=1 端到端延迟。与 `00000` 和低阈值近似全查的对照分开；不能从点数或论文的 FPS 推算本模型增益。

## 可复核的 CPU 冒烟

在仓库根目录：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/user/work/yolo11n-env/bin/python -m pytest -q -c /dev/null -p no:cacheprovider dronebench/adapters/querydet/test_adapter.py
```

本机 PyTorch 2.9.1+cpu 实测 **7 项通过**：真实 YOLO11n Detect、TAL/DFL 反向、局部矩形图和空标注、双图四层路由/空查询、融合前后解码，以及 640px 默认 CPU 前向。`00100` 公共模型的合成冒烟另跑了一次 96px、batch=2 的 SGD 更新：查询损失 `0.10842`，P4/P5 查询输出层权重梯度范数 `0.21095/0.16323`，评估解码 `2×14×189`；融合后 640px 解码 `1×14×8400`，均为有限值。该种子和初始化下默认阈值通过了全部格点，**不是加速证据**。

公共训练与验证 CLI 也用 4 张训练、2 张验证的工程夹具实跑：从零训练 1 个 epoch，随后独立加载最佳权重，验证命令读完 2 张图。复核时令 `SMOKE_DATA` 指向按[夹具脚本](<../../evaluation/smoke_data.py>)生成的数据 YAML、`RUNS` 指向临时输出目录，训练结束使用它打印的 `val` 命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m dronebench train --id 00100 --data "$SMOKE_DATA" --imgsz 96 --batch 2 --epochs 1 --workers 0 --device cpu --kind smoke --project "$RUNS"
```

夹具来自官方 val，只用于检查执行链。这些不是完整 VisDrone 训练、精度或延迟结果。
