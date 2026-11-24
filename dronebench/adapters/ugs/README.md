# UGS 单项：在 YOLO11n 的干净正样本上稳定定位分布

基线为 main `55989f76c1e85c192ba12cb3e533e3de842a2301`。`00001` 只开 UGS，`00000` 走原模型和原 DFL。激活共享环境后在仓库根运行：

```bash
source /home/user/work/yolo11n-env/bin/activate
python -m dronebench train --id 00001 --data VisDrone.yaml
```

同一单项的显式配置见 [single.json](./single.json)。子消融请复制它、修改 `options.ugs`，以 `--config PATH` 替代 `--id 00001`。训练结束后，用实际打印出的权重路径运行 `python -m dronebench val --weights PATH --data VisDrone.yaml`。论文机制、适用条件、来源及与原公式的解释差异见[阶段一方法卡](../../methods/ugs/README.md)；本层是 YOLO11n 的实验适配，不是作者的 YOLO 实现。

## 同一套训练、分配与推理几何

1. [模型](../../model.py)提供各层干净特征、`stride[N,1]` 及中心格点 `anchor[N,2]`，从细到粗、每层按行展开，中心偏移为 0.5。原 `Detect.cv2` 的输出仍为 `[B,4R,N]`，转成 `[B,N,4,R]` 后每个坐标有 `R=head.reg_max` 个 logits，次序为 `l,t,r,b`。不新增层，不改分类头、分类目标、候选位置或图像／块坐标原点。
2. 支持点由阶段一 `make_support` 生成，作为适配器持久 FP32 buffer `support[R]` 注册进模型；设备跟随模型迁移，转半精度模型时不量化网格。默认 `R=16`、`α=15` **格**、`β=2/15` 每格。`u_i=2i/(R−1)−1`，`s_i=sign(u_i)·α·expm1(αβ|u_i|)/expm1(αβ)`；靠近零的点更密，覆盖 `[−α,+α]`。同一个 `support` 用于标签插值和解码，不用 stock `head.dfl` 的索引期望。
3. 对 `p=(x,y)` 与**原 GT 像素 xyxy 除以该位置的 stride** 后的框，目标是 `(x−x1, y−y1, x2−x, y2−y)`。TAL 放宽极小框候选范围、或启用 RFLA 时，框外格点仍是合法正样本：**保留负边距，不先截零、不丢弃、不移动点或 GT**。只在 two-hot CE 编码时把超出 `[−α,+α]` 的单边饱和到对应端点；域内按相邻真实支持点的物理间距插值，等于端点不算越界。CIoU 始终对原 GT，而非编码后的饱和框。
4. 干净／验证／推理共用 [UGSAdapter.decode](./adapter.py)：各边取 `d=Σ softmax(z)_i·s_i`，还原 `(x−l, y−t, x+r, y+b)` 格。负单边不意味着框非法，例如 `p=(0.5,0.5)`、`d=(−0.5,0.5,1.5,0.5)` 在 stride=8 是像素框 `[8,0,16,8]`。只看总宽 `l+r` 和总高 `t+b`：若某轴的解码跨度 `≤4·eps·α`（`eps` 为期望计算类型的机器精度），在**该轴中点折叠到零**；另一轴不动。不取绝对值、不交换端点、不构造伪正面积，不裁到图内。TAL 用这份框乘 stride 做干净匹配，CIoU 用同一格单位框；公共推理将非正宽／高候选的类别分数置零，再作块还原／NMS。
5. signed 对称网格配原先各箱相等的 bias，零特征会解出零面积，可能使 TAL 质量权重全零。因此在模型创建、可选 P2 已插入后，[initialize_head](./adapter.py) 把**所有** `cv2[i][-1].bias` 四组都初始化为 `−|8s_i/α−1|`。先验偏向 `+α/8`，零输入分布有正的期望边距，分类 bias 和回归卷积权重不动。这只防止对称初始化的确定性退化，不保证每个 GT 都被分配。有 stock 初始权重时也重设定位 bias，原 DFL logits 不能被当成相同的坐标分布。

## 各损失与第二次头前向

[公共 criterion](../../loss.py) 只用干净预测做一次 TAL 或 RFLA 分配，传入 `fg`、`target_boxes`、`target_scores`；本层既不调用分配器，也不在扰动分支重选框。对同一组正样本，`w_j=Σ_c target_scores[j,c]/max(Σ target_scores,1)`；四边的 CE／熵先按边求均值，再按 `w_j` 求和。

| 项 | 如何算 |
| --- | --- |
| `box_loss` | `Σ w_j(1−CIoU)`；公共 criterion 乘 `hyp.box`。退化轴已折叠，可参与这个有限的损失；CE 仍监督四边。 |
| `ugs_loc_loss` | 阶段一 `localization_ce`：真实相邻支持点 two-hot 的交叉熵；公共 criterion 乘 `hyp.dfl`，不再加原 DFL。 |
| `ugs_um_loss` | `λ·H_clean`，`H` 是固定正样本处逐坐标的 Shannon 熵（自然对数），不是类别熵。 |
| `ugs_ur_loss` | 干净 `H_clean` 对每层特征按图取梯度，按该图该层的 L2 范数归一，加入停梯度增量 `ρg/‖g‖₂`（零梯度不加噪）；复用原 `cv2`，在**相同的位置和权重**上计算 `γ·H_perturbed`。不加整图特征 L2 项，不按熵值另做 top-k。 |

只有 `refine=True`、适配器和头均在训练、autograd 开启、有 FG、UR 有正权重时才复算。第二次只执行回归塔：在 `try/finally` 内临时关闭原来就在训练且跟踪统计的 BN 的 `track_running_stats`，故仍用当前 batch 的均值／方差做归一化，不更新 running mean、variance 或 `num_batches_tracked`，并恢复原标志；本来冻结或不跟踪的 BN 不变。增量不建二阶图，原特征和共享定位头保留外层梯度。只对 FG 的输出取熵，不等于把特征增量硬裁到 FG 格点：卷积感受野、PAN 的层间依赖及训练 BN 的跨样本统计都会传播梯度。关闭 UM 不阻止 UR 获取扰动方向；只设 `rho=0` 仍会复算并再加一项干净熵，不能替代关闭 UR。

验证 loss 可报干净 CE、CIoU、UM，UR 为零；标签不参与推理，不触发求扰动梯度或第二次头前向。无 FG 时四项都是与回归 logits 连图的零，不执行第二次前向，`backward()` 得零梯度而不是断图。与 ESOD 组合时一块是一个局部样本；若根本没有块，公共模型直接跳过后续层。与 QueryDet 组合时验证／推理仅 `active` 点有效；UGS 不是额外的空间或层选择器。

## 可复现实验与诊断

| `options.ugs` | 默认 | 含义 |
| --- | --- | --- |
| `nonuniform` | `true` | 单独关闭 IN：改用**同一 signed 域**的等宽点（β=0 的极限）。这不等于 stock 的 `0…R−1`。 |
| `um`, `ur` | 均 `true` | 分别关闭干净熵损失、扰动与第二次回归。 |
| `alpha`, `beta` | `R−1`, `2/alpha` | signed 支持半径与形状；单位分别是格、每格。支持点必须有限且可区分。 |
| `lambda_um`, `gamma_ur`, `rho` | 各 `0.5` | 两项熵的外层非负权重、每图每层扰动的非负半径。 |

未知键报错。建议固定数据、初始化方案、分配器与预算，对比 `00000`、signed 等宽仅 CE（`nonuniform=false,um=false,ur=false`，仍保留 CIoU）、signed 非均匀仅 CE（仅打开 `nonuniform`）、单加 UM、单加 UR、最后两者都加。原 DFL 与 signed 等宽同时改变了支持域、端点及先验，不应把二者的差异都算成 IN 的收益。

`ugs_target_coords` 是 FG 边总数，`ugs_negative_coords` 和 `ugs_clipped_low_coords`／`ugs_clipped_high_coords` 分别数负边、两端越界的边；可据前者作分母算饱和率。`ugs_outside_fg` 数至少一个负边的正样本，`ugs_degenerate_fg` 与 `ugs_degenerate_active` 数至少一轴折叠的 FG／有效候选。`ugs_refined` 指示实际是否走第二次前向，`ugs_weighted_clean_entropy` 只在计算了干净熵时记录，未乘 λ；FG 与各图 active 的分母由公共 criterion 记录。看召回／定位误差时同时看这些计数，避免把尾部饱和或无配误作熵损失问题。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -c /dev/null --noconftest -p no:cacheprovider -q \
    dronebench/adapters/ugs/test_adapter.py dronebench/methods/ugs/test_ugs.py
```

[适配测试](./test_adapter.py) 覆盖四级头的先验与网格持久化、signed 框外和两端裁剪、共享几何与零分过滤、独立消融、固定 FG 的真实扰动和梯度、第二次 BN 的 batch 归一与零状态更新、空 FG 和无梯度路径，以及真实 YOLO11n 的训练反传与验证。共享环境（PyTorch 2.9.1+cpu）上述两套共 **24 passed**。

另已实跑公共 `smoke --id 00001 --imgsz 96`（含一次 SGD、eval 解码和 NMS）。真实入口也以 `--kind smoke --epochs 1 --batch 2 --imgsz 96 --device cpu --workers 0 --seed 17` 跑通：两张训练、两张验证图均从本机官方 val 压缩包选取，四个序列互异，**不能当正式训练／验证划分**。训练批有 94 个 FG，其中 61 个在原 GT 外，82/376 条边为负，两端越界和预测退化均为 0，UR 确实执行。保存的最佳权重可由独立 `val` 重载，处理 2 张图／116 个 GT。另在合成批实测 RFLA+UGS（`00011`）的 6 个框外 FG 与 RFLA 独立计数相同；P2+UGS（`10001`）四层 `[4,8,16,32]` 均可前向、反传。

**局限：**仅针对本快照 YOLO11n one-to-many Detect；非零半径 UR 需要干净头输入特征本就可求导（冻结整个特征提取器时应关闭 UR，或在干净前向前显式保留特征梯度）。UR 是一次额外的特征反传和稠密回归头前向，需要更多显存，不是推理加速。固定格单位支持半径让粗层的像素量程更大；非均匀网格可能牺牲长边精度。以上均为工程／数值验证，不代表正式 VisDrone AP、固定尺寸召回、GPU 延迟或方法收益。
