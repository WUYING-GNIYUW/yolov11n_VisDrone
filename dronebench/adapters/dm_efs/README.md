# DM-EFS → YOLO11n：增加 P2，按图选择检测层

单项 ID 为 `10000`。从固定 main `55989f76c1e85c192ba12cb3e533e3de842a2301` 的 YOLO11n 构建；没有 `dm_efs` 开关时不构建此适配器，五项全关的 `00000` 是原基线。[论文机制与出处](../../methods/dm_efs/README.md)和 [YOLO 接入代码](adapter.py)分列，以下是本实验的改动，不是作者 YOLOv7 的逐层复现。

```bash
python -m dronebench train --id 10000 --data VisDrone.yaml
```

[单项 JSON](single.json) 写出相同的子选项默认值。做消融时复制配置，只改一个选项，用 `--config PATH` 代替 `--id 10000`。验证使用训练产出的权重，不重新建票表：

```bash
python -m dronebench val --weights /path/to/run/weights/best.pt --data VisDrone.yaml
```

## 哪些张量变了

- **EFS**：用 [骨干第 2 层](../../../ultralytics/cfg/models/11/yolo11.yaml#L22) 的真实 P2（64 通道，stride 4），而非上采样 P3。`expand_features` 把它前插在 PAN 的第 16/19/22 层输出前，形成通道 `[64,64,128,256]`、步长 `[4,8,16,32]` 的细→粗列表。P2 直接接新增的回归/分类分支；不把 P2 再反向混入原 PAN，也不替换整座颈。640×640 时，网格由 8,400 个位置增至 34,000 个。
- **保留基线参数**：`extend_head` 只为 P2 新建一个 stock Detect 分支，用 stride 4 初始化该分支的偏置/BN，原有 P3–P5 的模块对象、参数、BN 和 DFL 均保留。前插会使三个支路在 ModuleList 中的索引后移一位；公共构建器先装载所给的 **stock** 初始权重，然后才扩头。DM-EFS 自己不会重置旧权重；组合里的 UGS 定位先验是另一项可追踪的改变。
- **DFM**：尺寸控制器读真实骨干索引 **2/4/6/10** 的 P2–P5，不把 PAN 输出冒充骨干输入；在后续 Detect 之前产生尺寸端点与层门。独立调用可不传 `backbone_channels` 和 `backbone_feats`，此时控制器只用真实 `shallow`。EFS 关而 DFM 开时，P2 仍可供控制器看，**检测**候选只有原来的 P3–P5。
- **ESOD 组合**：进入 `route` 的每一行已是一个块。尺寸、中心 cell、预测框、标注都在该块的本地输入像素坐标中；两个来自同一原图的块可以有不同的门。块间标签裁剪、重叠、恢复和原图 NMS 不由 DFM 负责。一个 GT 若被保留在多个块里，每个块分别投票。没有块时公共层不进入此适配器。

```python
from dronebench.adapters.dm_efs import DMEFSAdapter

adapter = DMEFSAdapter(shallow_channels, channels, strides, options, backbone_channels=backbone_channels)
adapter.extend_head(head)
feats, level_mask, state = adapter.route(shallow, native_feats, backbone_feats=backbone_feats)
size_loss = adapter.aux_loss(state, local_batch, (h, w), boxes_px, feats, update_votes=adapter.training)
```

## 训练：端点和逐图投票是两件事

1. 全层参与原检测监督。`local_batch` 用 `batch_idx` 分组，把增强后输入上的归一化 `bboxes`（xywh）转为本地 xyxy 像素，排除非有限/非正面积框。有效框的尺寸是 `floor(sqrt(width*height))`，夹到 `[2,max_size]`；不是固定 640 评测分桶所用的短边。每个非空图分别取最小/最大尺寸，监督 `[B,2,max_size-1]` 的两个 softmax 分类器；两项 focal 相加、按非空图平均，再乘 `loss_weight`。空图不监督尺寸，全空批返回可反传的零。
2. **给算法 1 的每个 GT、每一层确定一个预测框**：取 GT 中心所在 cell，列/行分别为 `floor(cx/stride)`、`floor(cy/stride)`，夹到网格边界；落在分界线选右/下。按层拼接偏移和行优先 `row*W+col`，从公共 decoder 的 **实际** `[B,N,4]` 像素 xyxy `boxes_px` 取框；无 UGS 时是 DFL 框，有 UGS 时是 clean UGS 框。DM-EFS 不另行解码，也不看分类分数、TAL/RFLA 正样本或 NMS。多个 GT 允许共享 cell。这是 anchor-free 接入的配对约定，不是原论文指定的唯一实现：它衡量中心 cell 定位能力，会漏掉邻近 cell 的好预测；不会因为某层有更多位置而对全层取最优 IoU。
3. 对一图 `G` 个 GT 仅收集 `[G,L,4]` 对应框，`paired_box_ious` 得到 `[G,L]`。`IoU >= iou_threshold` 才是命中；同图同层同尺寸即便多个 GT 命中仍只加 **1**，逐图跨轮累计。此处无 `[G,N]` 或 `[B,G,N]`，每图配对工作量为 `O(G·L)`，不改变公共检测损失/分配器本身的显存需求。
4. 票不参与梯度。`dm.codebook.votes` 是持久的 `[L,max_size-1]` **int64** buffer（默认 20,448 字节），半精度权重或保存/加载 `state_dict` 不会把它变成浮点数。只有 `adapter.training and update_votes` 才更新；验证 loss 可以使用标签算端点损失，但不会投票，推理门也不读验证标签。[公共单设备训练器](../../engine.py) 在每个 epoch 结束、验证/存档之前把累计整数票原样拷入 EMA；不能对票做指数移动平均。

## 推理：式 6 请求了什么，实际执行了什么

`raw["dm_state"]` 保存下列本地行级状态；门均为 `[B,L]` bool，列按细→粗。尺寸类目的最高票层胜出；无票尺寸沿尺寸轴用最近有票尺寸补齐，尺寸等距取较小者，层等票取较细者。整表没有成功票时全开，不假定“细层一定适合小目标”。

| 字段 | 读法 |
| --- | --- |
| `size_range` | 每图/块由两个 logits argmax 得到的有序端点 `[B,2]`（输入像素）；DFM 关时为 `None`。 |
| `head_requested` | 式 6 只开两端尺寸所对应层的**并集**，相同时只开一层，而不是开两个端点间的所有层。 |
| `neck_requested` | 式 6 从最细层开到**最大尺寸**的获胜层；票表不强制单调。训练时两个 requested 字段仅为当前批投票前的诊断，不据此裁剪。 |
| `neck_executed` | 全 `True`：本适配保留完整骨干/原 PAN 和 P2 直连，表示各尺度的特征都已得到，**不宣称省了颈部计算**。 |
| `head_enabled` / `raw["level_mask"]` | 训练全开；推理传给公共检测头的门。DM-EFS 单项里，`gated_detect` 只把该层被选行送入 `cv2/cv3`，零行的整层不调用。 |
| `head_executed` | 公共头执行后，将 `raw["active"]` 按各层 `H*W` 切分，对位置取 `any(1)`。这是每个输入真正有活动检测点的层；与 QueryDet 组合时可少于 `head_enabled`。 |

公共输出仍保留原网格布局，未激活位置不参与候选（分类概率清零），不是先稠密跑完检测卷积再乘零。是否执行 QueryDet 的空间查询载体是另一个问题。全颈、控制器、分派/回填和解码都有成本；不能用门的稀疏率代替 batch=1 端到端实测。

## 子选项与对照

`options.dm_efs` 仅接受下列键；未知键报错：

| 键 | 默认 | 含义 |
| --- | --- | --- |
| `efs`、`dfm` | 均 `true` | 可独立关闭。仅 EFS：无尺寸预测器/票，`size_loss=0`，P2–P5 全开；仅 DFM：学原三层的尺寸票。 |
| `max_size` | `640` | 像素尺寸类别上界，整数 ≥2；改变它会改变端点头和票表形状，不会改变输入分辨率。切到 960 时需明确是保持 640 以饱和处理，还是另训 960 桶。 |
| `iou_threshold` | `0.5` | 闭区间 `[0,1]`；设 0 连零重叠都可投票，不适合作为“定位可靠”的正式对照。 |
| `loss_weight`、`gamma` | `0.04`、`2.0` | 两端 focal 的系数和调制指数，均为有限非负数。 |
| `projection_channels`、`hidden_channels` | `16`、`64` | 骨干每层 1×1 投影通道数和共享/分支 MLP 隐元数，正整数。 |

`tal_min="auto"` 时 TAL 候选边长下限取第二个检测步长，从原三层的 16 变为扩头后的 8 px。要隔离这一变量，用 `{"dm_efs":true,"tal_min":16}`，`tal_min` **在顶层**。RFLA 组合不使用 TAL 下限。可先对比全关→仅 EFS 在同一尺寸桶的召回，再对比仅 EFS→EFS+DFM 的误检、召回、实测延迟，并同时检查端点误差、空票/补表情况和实际层门。

## 实测与边界

```bash
OMP_NUM_THREADS=1 /home/user/work/yolo11n-env/bin/python -m pytest -q dronebench/adapters/dm_efs/test_adapter.py
```

- Python 3.13.5、PyTorch 2.9.1+cpu，[五项适配测试](test_adapter.py)通过。stock P3–P5 在扩头前后对同一特征的 logits 逐元素完全相等；两子开关独立，未知键报错。2×96×128 单项训练构建出 1,020 个位置，真实反向到 P2 分支、四个分类分支、骨干和尺寸控制器；两图各投一票，验证后仍是两票。
- 两行分别要求 P2/P4 时，真实 box/cls 卷积各只在第 0/2 层各收到一行，第 1/3 层零次调用；真实公共模型验证时 PAN 第 11–22 层全部被调用。另测单图去重、跨图累计、精确阈值、空图、带梯度端点 loss、int64 `state_dict` 序列化往返；UGS 同一组定位 logits 在 UGS 解码过阈值、在 stock DFL 不过阈值时，实测记的是 UGS 的票。
- 公共 `train --id 10000` 和独立 `val --weights ...` 均已实跑：4 张合成训练图、2 张独立合成验证图，96×96、batch=2、1 epoch。重载 EMA 保存出的 `best.pt`，四层真实骨干输入通道为 `(64,128,128,256)`，票表仍是 int64：P2 的 60 px 桶恰为 4 票；验证图没有计入。这是用于检查训练器/存档的工程冒烟，不是 VisDrone 检测结果。
- 单独压力冒烟：960×1280 局部网格 102,000 个位置、8,192 个同尺寸 GT；各层中心 cell 均成功时，总数恰为 4 票，而非按 GT 数累加。完整 VisDrone 训练、mAP、固定分桶召回、GPU 和端到端延迟未在此适配测试中测量。
