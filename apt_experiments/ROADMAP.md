# APT 方案实验推进报告

## 1. 报告目的

本报告基于当前项目代码与 `EXPERIMENTS.md` 中的已有结果，规划下一阶段围绕 APT
（Adaptive Patch Tokenization）方案的实现、验证和改进工作。

项目最终要回答的问题是：

> Vision Transformer 是否可以根据图像局部信息量，自适应减少输入 Transformer 的
> patch/token 数量，并在尽量不降低分类精度的前提下获得真实的计算收益？

当前 MAE + Router 路线已经提供了一个较完整的学习型参照，但现有 APT 实验尚不足以
支持“APT 有效”或“APT 无效”的结论。下一阶段应先建立可信、统一的实验协议，再补齐
APT Selection、APT Merge 和真正的多尺度自适应 tokenization 实验。

---

## 1.1 精简执行协议（当前最高优先级）

从 2026-06-12 起，本节覆盖本文后续章节中更重的测试矩阵、全组合消融和阶段验收要求。
项目只回答一个核心问题：

> APT 能否在分类精度基本不下降的前提下，降低 ViT 的真实计算资源消耗？

### 唯一核心验收指标

一个 APT 配置只有同时满足以下两项，才记为“有效”：

1. **精度保持**：相对同数据集、同训练协议的 Full ViT，最终测试准确率下降不超过
   `1.0` 个百分点。若下降在 `1.0-2.0` 个百分点之间，只记为“精度换效率”，不得表述为
   保持精度。
2. **真实资源下降**：在同一 GPU、相同 batch size、相同 AMP 设置下，包含 entropy
   计算和 token 组装开销后，至少满足一项：
   - 端到端延迟下降不少于 `10%`；
   - 吞吐量提高不少于 `10%`；
   - 峰值显存下降不少于 `10%`。

平均真实 token 数减少不少于 `20%` 仅作为候选筛选条件，不能单独证明项目目标成立。

### 精简阶段

| 阶段 | 环境 | 必做工作 | 通过条件 |
|:-----|:----:|:---------|:---------|
| L0 最小正确性 | CPU | import、单 batch forward、一次 backward、同图 batch 一致性 | Selection、Merge、Hierarchical 均可运行且无明显 mask/shape 错误 |
| L1 无训练筛选 | CPU | 每个数据集抽样统计 entropy 与 token 数 | 为 75%、60%、50% token 预算各给出候选阈值 |
| L2 短训练筛选 | GPU | 单 seed、最多 10 epochs，只跑 Selection/Merge/Hierarchical 候选 | 保留验证精度最好且 token 至少减少 20% 的少量配置 |
| L3 正式评估 | GPU | 先跑 CIFAR-100 与一个高分辨率数据集，每个候选 2 seeds | 依据“精度保持 + 真实资源下降”作结论 |
| L4 扩展验证 | GPU，可选 | 仅当 L3 成功或结论不稳定时增加数据集、seed、消融 | 不影响核心结论时不执行 |

### 明确取消或降级的工作

- 不再要求每次修改都运行完整单元测试；只运行受影响的最小 smoke test。
- 不在 GPU 筛选前比较所有 aggregation、位置编码、entropy 特征和动态 batching 组合。
- 不要求四个数据集全部完成后才形成结论；先用 CIFAR-100 加 Oxford Pets 验证。
- Food-101、DTD、多 seed、完整消融均降为 L4 可选扩展。
- 不要求短训练达到收敛；短训练只用于淘汰明显无效配置。
- 不重复验证已通过且未受代码修改影响的模块。

### Agent 执行限制

1. CPU 上禁止完整训练，训练 smoke test 最多 1 个 mini-batch。
2. 当前代码未修改核心 tokenization/mask 逻辑时，不重复运行全部测试。
3. GPU 队列先执行最小矩阵，只有候选接近精度门槛才扩大实验。
4. 一旦 L3 已能明确回答核心问题，停止新增无关实验。

---

## 2. 当前进展判断

### 2.1 已经完成的工作

1. 已建立 ViT-B/16 全量微调基线，并在 CIFAR-100、Oxford Pets、Food-101、DTD 等
   数据集上积累了精度结果。
2. 已完成 MAE + Router 的主要实验，验证了固定保留 75% 和 50% patch 时的精度变化。
3. 已实现基于像素熵的 APT Selection：
   - 在 16x16 patch 粒度上计算熵；
   - 高于阈值的 patch 被保留；
   - 低于阈值的 patch 被直接丢弃。
4. 已实现 APT Merge 的初版：
   - 以 2x2 个 16x16 patch 为一个 32x32 block；
   - 低熵 block 由 4 个 token 平均合并为 1 个 token；
   - 高熵 block 保持 4 个独立 token。
5. 已具备 checkpoint、自动续训、评估、吞吐量统计等基本训练设施。

### 2.2 已有 APT 结果的有效边界

| 实验 | 当前状态 | 可以说明 | 不能说明 |
|:-----|:---------|:---------|:---------|
| APT Selection | CIFAR-100，约 50 epochs，单阈值 | 当前配置在 CIFAR-100 上弱于 MAE + Router | Selection 在其他数据集上无效；50 epochs 已充分收敛 |
| APT Merge | CIFAR-100，约 8 epochs，单阈值 | 激进合并会显著压缩 token，早期精度较低 | Merge 的最终精度；阈值 5.5 是否合理；多尺度合并是否有效 |
| APT 跨数据集 | 基本缺失 | 无 | 无法判断数据分辨率、纹理依赖与 APT 收益的关系 |
| APT 效率 | 仅有初步统计 | token 数减少可能带来潜在收益 | 实际 FLOPs、端到端延迟、熵计算开销和动态 padding 开销 |

因此，当前最重要的结论不是“APT 精度差”，而是：

> 现有实验只验证了两个不充分训练、单数据集、单阈值的近似实现，证据不足以评价 APT
> 路线本身。

### 2.3 当前实现与完整 APT 思路的差距

原始 APT 的重点是根据局部信息量建立层次化 patch 划分，以细 patch 表示复杂区域，
以粗 patch 表示简单区域。当前项目仍存在以下差距：

1. `apt_experiments/train_apt_patch_selection.py` 的 `--multi_scale` 分支最终仍回退到普通 threshold
   selection，并未形成真正的多尺度 token 序列。
2. Selection 是“丢弃低熵区域”，不是“用更粗粒度 patch 表示低熵区域”，只能作为
   APT 风格的消融基线。
3. Merge 当前只支持 16x16 和 32x32 两级，而且使用 embedding 平均池化，尚未比较
   resize、卷积聚合、可学习聚合等方式。
4. 动态长度 batch 通过补零对齐，但 Transformer attention 并未使用严格的 key padding
   mask；补零 token 仍可能参与当前 block 的注意力计算。
5. 当前记录的 token 数是 batch 内最大长度 `K_max`，不是每张图的平均实际长度，可能
   高估有效 token 数，也会掩盖动态 batching 的效率问题。

---

## 3. 推进原则

后续所有实验遵循以下原则：

1. **先保证实现正确，再扩大训练规模。**
2. **精度和计算必须同时报告。** 只减少 token 而没有真实加速，不算完成项目目标。
3. **在相同计算预算下比较。** APT、MAE + Router、随机选择和固定 token baseline
   应尽量匹配平均 token 数。
4. **区分方法验证与超参数搜索。** 测试集只用于最终报告，不参与阈值或 epoch 选择。
5. **优先使用高分辨率数据集验证 APT。** CIFAR-100 保留为低分辨率对照，不作为
   APT 有效性的唯一依据。
6. **所有结论附带实验条件。** 包括数据划分、seed、epoch、阈值、平均 token 数、
   GPU、batch size、精度和延迟。

---

## 4. 核心研究问题

下一阶段围绕五个问题组织实验：

1. 熵是否能稳定区分应使用细粒度 patch 和粗粒度 patch 的区域？
2. Selection 与 Merge 在相同 token 预算下，谁的精度更高？
3. APT 是否在高分辨率、背景较多的数据集上比 CIFAR-100 更有优势？
4. 多尺度层次划分是否优于当前固定 16/32 两尺度合并？
5. 加上熵计算、动态组装和 padding 后，APT 是否仍能获得真实端到端加速？

主要假设如下：

- H1：APT 在 Oxford Pets、Food-101 等原生高分辨率数据集上优于其在 CIFAR-100
  上的表现。
- H2：相同 token 数下，Merge 会优于直接 Selection，因为低熵区域的信息没有被完全
  丢弃。
- H3：固定全局熵阈值不具备跨数据集泛化性，按目标 token 预算校准阈值会更稳定。
- H4：如果动态序列仍采用 batch 内 padding，理论 token 减少不会等比例转化为吞吐量
  提升。

---

## 5. 算力分工与 Agent 自动执行协议

### 5.1 当前算力边界

当前本地环境只有 CPU，不适合进行 ViT-B/16 的完整训练、5 至 10 epochs 短训练、
多 seed 复现或最终吞吐量评估。本地算力应集中用于实现、测试、数据统计和实验编排。

建议在满足 GPU Gate 前不租用 GPU。理想状态是租到 GPU 后无需继续修改核心实验逻辑，
只需同步代码、准备数据并启动已生成的实验队列。

### 5.2 CPU 与 GPU 任务划分

| 工作类型 | 本地 CPU | 租用 GPU |
|:---------|:--------:|:--------:|
| 阅读代码、修复实现、编写测试 | 执行 | 可执行但不应占用租赁时间 |
| 熵分布和 token 预算统计 | 执行，可抽样 | 可选全量复核 |
| 1 至 2 个 mini-batch smoke test | 执行 | 执行 |
| ViT-B 短训练筛选 | 禁止 | 执行 |
| 完整训练和多 seed 实验 | 禁止 | 执行 |
| 最终 GPU 延迟、吞吐量和显存测试 | 禁止作为正式结果 | 执行 |
| 结果汇总、绘图和文档更新 | 执行 | 训练期间可同步执行 |

### 5.3 本地推进终点：GPU Gate

Agent 在本地应自动推进到以下条件全部满足：

- [x] L0 最小正确性检查通过。
- [x] L1 四个数据集的无训练扫描完成，候选阈值已映射到目标 token 预算。
- [x] Selection、Merge 和层次化 APT 可完成最小 forward/backward smoke test。
- [x] 核心动态 mask 与同图 batch 一致性已通过。
- [x] 已生成 6 个任务的精简 GPU 队列，并确认三个训练入口支持自动续训。
- [x] 已在 `apt_experiments/GPU_WORKFLOW.md` 记录 GPU 运行顺序、显存建议和数据同步清单。

**当前状态：GPU Gate 已到达。后续完整训练必须在 CUDA 环境执行。**

达到 GPU Gate 后，本地 Agent 必须停止任何完整训练尝试，并输出：

1. GPU 待执行实验清单；
2. 可直接运行的命令或批处理脚本；
3. 数据与 checkpoint 同步清单；
4. 预计时间和显存需求；
5. 尚未解决但不阻塞 GPU 训练的风险。

### 5.4 Agent 通用执行规则

当用户要求“继续推进”“执行下一阶段”或“按报告自动执行”时，Agent 应遵循：

1. 读取本报告和当前 Git 状态，保护已有用户改动。
2. 检查下方执行看板，选择第一个未完成且前置条件已满足的任务。
3. 先更新执行计划，再实施代码、测试或实验。
4. 每完成一个任务，立即更新对应复选框和产出记录。
5. CPU 环境中不得启动超过 1 个 mini-batch 的训练 smoke test。
6. 需要 CUDA 的任务若当前无 CUDA，则标记为 `WAITING_GPU`，不得用 CPU 硬跑。
7. 仅当修改了 tokenization、mask、数据划分或 checkpoint 逻辑时，重新运行受影响的最小 smoke test。
8. 实验失败时先自动诊断并重试可恢复问题；配置或方法问题不得通过无依据改参掩盖。
9. 不使用 test set 选择阈值、epoch 或模型。
10. 阶段结束时提交：改动文件、测试结果、产出路径、遗留风险和下一阶段入口。

### 5.5 状态标记约定

Agent 更新任务状态时使用以下标记：

- `[ ] TODO`：尚未执行；
- `[-] IN_PROGRESS`：正在执行；
- `[x] DONE`：已经通过验收；
- `[!] BLOCKED`：存在实现或数据阻塞；
- `[G] WAITING_GPU`：本地准备完成，等待 GPU；
- `[S] SKIPPED`：经记录理由后跳过。

同一阶段最多只能有一个 `IN_PROGRESS` 任务。只有验收标准全部满足后，阶段才能标记为
`DONE`。

### 5.6 全流程执行看板

| 顺序 | 阶段 | 主要环境 | 当前状态 | Agent 推进条件 |
|:----:|:-----|:--------:|:---------|:---------------|
| 0 | L0：最小正确性 | CPU | `[x] DONE` | 已完成，不重复全量测试 |
| 1A | L1：无训练预扫描 | CPU | `[x] DONE` | 四个数据集扫描已完成 |
| 1B | 阶段 1：短训练筛选 | GPU | `[G] WAITING_GPU` | GPU Gate 完成 |
| 2 | L3：Selection 正式评估 | GPU | `[G] WAITING_GPU` | L2 保留候选后执行 |
| 3I | L0：Merge/层次化实现 | CPU | `[x] DONE` | 已完成最小实现与 smoke test |
| 3T | L3：Merge/层次化正式评估 | GPU | `[G] WAITING_GPU` | L2 保留候选后执行 |
| 4 | L3：统一真实资源评估 | GPU | `[G] WAITING_GPU` | 与正式精度评估使用同一候选 |
| 5 | L4：消融与分析 | CPU + GPU | `[S] SKIPPED` | 仅在核心结论不稳定时恢复 |
| 6 | L4：扩展复现 | GPU + CPU | `[S] SKIPPED` | 仅对成功配置追加数据集或 seed |

### 5.7 推荐的 Agent 启动口令

用户可以使用以下简短指令驱动全流程：

```text
按 apt_experiments/ROADMAP.md 自动执行下一项可用任务。
```

```text
从阶段 0 开始自动推进，直到 GPU Gate；不得在 CPU 上启动完整训练。
```

```text
当前已进入 GPU 环境，检查 CUDA 后从阶段 1B 继续执行。
```

```text
汇总当前执行看板、已完成产出、阻塞项和下一条可运行命令。
```

Agent 收到上述指令后不得只给计划；只要任务可在当前环境执行，就应直接完成实现、验证、
状态更新和产出记录。只有到达 GPU Gate、缺少数据/凭据或出现无法自动解决的阻塞时才停止。

---

## 6. 分阶段推进计划

### 阶段 0：实验协议与实现可信度修正

#### 阶段目标

建立可以支撑后续大规模训练的统一实验底座。该阶段未通过前，不启动完整 APT 实验矩阵。

#### Agent 自动执行任务卡

**执行环境：CPU；允许自动执行全部任务。**

- [ ] TODO：盘点 Selection、Merge、数据加载、评估和 checkpoint 代码路径。
- [ ] TODO：抽取统一 entropy 模块，消除两套熵实现的参数分叉。
- [ ] TODO：修复 Normalize 前后数据流，保证熵基于正确像素值计算。
- [ ] TODO：修复动态序列 padding 对 attention 的影响，或实现 token 数分桶方案。
- [ ] TODO：实现每图真实 token 与 padded token 的统计。
- [ ] TODO：建立固定 validation split，移除 CIFAR-100 的 val/test 混用。
- [ ] TODO：建立统一配置、日志和结果 JSON/CSV。
- [ ] TODO：添加纯色、噪声、棋盘格和不同 batch 组合的自动测试。
- [ ] TODO：在 CPU 上运行模块导入、单元测试和最多 2 个 mini-batch 的 smoke test。
- [ ] TODO：记录修改后旧 checkpoint 是否兼容；不兼容时标记需要重新训练。

**Agent 停止条件：**

- 测试失败且连续修复后仍无法满足 batch 一致性；
- 修改会破坏现有 checkpoint，但尚未记录迁移或重训方案；
- 任务开始依赖完整 ViT-B 训练。

#### 主要任务

1. 修正熵计算输入：
   - 最优方案是在图像 Normalize 前计算熵；
   - 备选方案是按各数据集实际 mean/std 精确反归一化；
   - 禁止继续统一使用 `mean=std=0.5` 近似反归一化。
2. 统一 Selection 和 Merge 的熵定义：
   - 灰度转换方式；
   - histogram bins；
   - log 底数；
   - padding 规则；
   - threshold 的记录单位。
3. 修正动态序列 mask：
   - padding token 不得作为 attention 的 key/value；
   - 或按相同 token 数分桶组 batch，减少 padding；
   - 验证同一图像单独推理与混合 batch 推理输出一致。
4. 修正 token 统计：
   - 报告每张图真实 token 数；
   - 报告 mean、std、P50、P90、min、max；
   - 同时记录 padded sequence length。
5. 统一验证协议：
   - CIFAR-100 从训练集划分固定 validation set；
   - test set 只在最佳配置确定后评估一次；
   - Oxford Pets、Food-101 使用固定 seed 的 train/val/test 划分。
6. 统一效率测试：
   - 记录 APT 预处理耗时和 Transformer 耗时；
   - 至少测试 batch size 1、32、128；
   - 固定 warmup 次数、测试 batch 数、AMP 状态和硬件。
7. 建立统一结果文件，例如每次实验保存：

```json
{
  "method": "apt_merge",
  "dataset": "oxford_pets",
  "seed": 42,
  "threshold": 4.5,
  "best_val_acc": 0.0,
  "test_acc": 0.0,
  "avg_real_tokens": 0.0,
  "avg_padded_tokens": 0.0,
  "latency_ms_bs1": 0.0,
  "throughput_bs32": 0.0,
  "peak_memory_mb": 0.0
}
```

#### 验收标准

- 熵输入可以通过人工构造图验证：纯色区域低熵、噪声区域高熵。
- 单图推理与不同 batch 组合中的同图输出误差小于 `1e-5`。
- 实际 token 统计与模型输入序列长度完全一致。
- validation 与 test 不再混用。
- 同一 checkpoint 连续三次效率测试波动不超过 5%。

#### 阶段产出

- 修正后的 APT 公共模块；
- APT 单元测试和 smoke test；
- 统一实验配置模板；
- 统一结果 JSON/CSV 格式。

---

### 阶段 1：阈值与 token 预算预扫描

#### 阶段目标

在完整训练前确定合理的阈值范围，避免对每个数据集盲目使用 `threshold=5.5`。

#### Agent 自动执行任务卡

**CPU 子阶段 1A：允许自动执行。**

- [ ] TODO：为每个数据集抽样 500 至 2000 张 validation 图像。
- [ ] TODO：生成 Selection 和 Merge 的 entropy 分布。
- [ ] TODO：扫描阈值并计算 mean/std/P50/P90/min/max token。
- [ ] TODO：反求 75%、60%、50%、37.5% token 预算对应的候选阈值。
- [ ] TODO：保存扫描 CSV/JSON，并绘制 threshold-token 曲线。
- [ ] TODO：生成 GPU 短训练配置，但不在 CPU 上启动训练。

**GPU 子阶段 1B：达到 GPU Gate 后自动执行。**

- [G] WAITING_GPU：每个候选预算运行 5 至 10 epochs 短训练。
- [G] WAITING_GPU：根据 validation 精度和 token 预算筛选完整训练配置。
- [G] WAITING_GPU：将失败配置、OOM 和异常 token 分布写入实验记录。

**Agent 推进门槛：**

- CPU 只完成统计时，本阶段标记为 `PARTIAL_DONE/WAITING_GPU`；
- 只有 GPU 短训练筛选完成后，才能自动进入阶段 2 和阶段 3T。

#### 实验数据集

- CIFAR-100：低分辨率对照；
- Oxford Pets：高分辨率、小规模；
- Food-101：高分辨率、大规模；
- DTD：纹理敏感压力测试。

#### 预扫描内容

1. 仅在 validation set 统计不同阈值下的 token 分布。
2. 为 Selection 和 Merge 分别寻找约 75%、60%、50%、37.5% 四档平均 token 预算。
3. 每档选择 1 至 2 个阈值进行 5 至 10 epochs 短训练。
4. 绘制：
   - threshold - 平均 token 曲线；
   - threshold - token 方差曲线；
   - 短训练精度 - token 曲线；
   - 不同数据集的熵分布直方图。

#### 建议预算矩阵

| 方法 | 目标平均 token 比例 |
|:-----|:-------------------:|
| APT Selection | 75%、60%、50% |
| APT Merge | 75%、60%、50%、37.5% |
| Random Selection | 与以上预算逐一匹配 |
| Fixed Uniform Merge | 与 APT Merge 的 token 数匹配 |

#### 验收标准

- 每个数据集至少找到三个可稳定复现的 token 预算点。
- 不同 batch 的平均 token 比例波动可解释且记录完整。
- 淘汰明显失效的阈值，不进入完整训练阶段。

---

### 阶段 2：补齐 APT Selection 完整基线

#### 阶段目标

把 Selection 从“单数据集初步结果”提升为可用于论文或课程报告的完整基线。

#### Agent 自动执行任务卡

**执行环境：GPU；本地 CPU 仅生成配置和验证命令。**

- [G] WAITING_GPU：按阶段 1B 选出的预算点创建完整训练队列。
- [G] WAITING_GPU：先执行单 seed 全矩阵，并启用自动续训。
- [G] WAITING_GPU：自动收集 best validation checkpoint，不使用 test 选模型。
- [G] WAITING_GPU：仅对 Pareto 候选配置补齐 3 seeds。
- [G] WAITING_GPU：运行与 token 数匹配的 Random Selection 和固定规则基线。
- [G] WAITING_GPU：最终一次性运行 test，并保存结果 JSON。
- [G] WAITING_GPU：生成 Selection accuracy-token 曲线和跨数据集表格。

**Agent 停止条件：**

- validation 曲线明显未收敛时，不得直接发布 test 结果；
- token 预算偏离目标超过 5% 时，返回阶段 1 重新校准；
- 多 seed 配置只对 Pareto 候选执行，避免无效消耗 GPU。

#### 完整实验

| 数据集 | 建议训练轮数 | 预算点 |
|:------|:------------:|:------:|
| CIFAR-100 | 100 | 75%、60%、50% |
| Oxford Pets | 100 | 75%、60%、50% |
| Food-101 | 30 或与 baseline 一致 | 75%、60%、50% |
| DTD | 100 或与 baseline 一致 | 75%、60%、50% |

每个关键配置至少运行 3 个 seeds。算力不足时，先单 seed 完成全部矩阵，再对 Pareto
前沿配置补 3 seeds。

#### 必须比较的基线

- Full ViT-B/16；
- Random Selection；
- 按中心区域或固定规则选择；
- MAE + Router 50% 和 75%；
- APT Selection；
- 可选：降采样模型，但需明确其改变了输入分辨率。

#### 验收标准

- 所有数据集完成至少两个有效 token 预算点。
- 训练曲线达到平台期，不以固定 epoch 数替代收敛判断。
- 报告 mean +/- std，而不是单次最佳结果。
- 能回答 Selection 是否显著优于相同 token 数的随机丢弃。

---

### 阶段 3：完成 APT Merge 与多尺度实现

#### 阶段目标

完成当前只训练约 8 epochs 的 Merge 实验，并将实现从“两尺度平均池化原型”推进到真正
可研究的多尺度自适应 tokenization。

#### Agent 自动执行任务卡

**CPU 实现子阶段 3I：允许在阶段 0 完成后自动执行。**

- [ ] TODO：修复或重构当前两尺度 Merge，使区域、位置和尺度映射可追踪。
- [ ] TODO：实现真正的 hierarchical APT，不再回退到 Selection。
- [ ] TODO：实现至少 Average Pool 和一种非平均聚合方式。
- [ ] TODO：实现位置编码与 scale encoding 的可配置切换。
- [ ] TODO：添加 token 区域覆盖、无重叠/无遗漏和 batch 一致性测试。
- [ ] TODO：使用小输入或最多 2 个 mini-batch 完成 forward/backward smoke test。
- [ ] TODO：生成阶段 3T 的 GPU 配置矩阵。

**GPU 训练子阶段 3T：达到 GPU Gate 后自动执行。**

- [G] WAITING_GPU：完成当前两尺度 Merge 的正式重训，不直接沿用不兼容旧 checkpoint。
- [G] WAITING_GPU：运行 Oxford Pets、Food-101、DTD 和 CIFAR-100。
- [G] WAITING_GPU：按相同 token 预算比较 Selection、Fixed Merge 和 Hierarchical APT。
- [G] WAITING_GPU：筛选位置编码和聚合方式，淘汰明显劣势配置。
- [G] WAITING_GPU：只对 Pareto 候选补多 seed。

**Agent 推进门槛：**

- 阶段 3I 可以在本地提前完成；
- 阶段 3T 必须等待阶段 1B 给出合理阈值或 token 预算；
- 多尺度区域覆盖测试未通过时，禁止启动 GPU 训练。

#### 阶段 3A：完成当前两尺度 Merge

1. 完成 CIFAR-100 全训练周期。
2. 在 Oxford Pets、Food-101、DTD 上训练。
3. 在 75%、60%、50%、37.5% token 预算下比较。
4. 与 Selection、Random、Fixed Uniform Merge 做同预算比较。
5. 比较三种位置编码：
   - 4 个原位置编码平均；
   - 从 14x14 插值到 7x7；
   - 使用 block 中心位置编码。

#### 阶段 3B：实现层次化多尺度 APT

建议将图像划分组织为 quadtree 或递归 block：

1. 高熵区域继续细分；
2. 低熵区域保留为粗粒度 token；
3. 支持至少 16x16、32x32 两级；
4. 在尺寸允许时扩展到 16x16、32x32、64x64；
5. 每个最终叶节点对应一个 token，而不是简单丢弃区域。

为了公平比较：

- 224x224 实验先采用 16/32 两级；
- 需要 16/32/64 时，可增加 256x256 独立实验组；
- 256x256 结果不得直接与 224x224 baseline 混为同一组。

#### 阶段 3C：比较 token 聚合方式

| 聚合方式 | 说明 |
|:---------|:-----|
| Average Pool | 当前最低成本基线 |
| Pixel Resize + Patch Embed | 先将粗 block 缩放到统一尺寸，再使用共享 embedding |
| Separate Conv Embed | 不同 patch 尺度使用不同卷积投影 |
| Shared Embed + Scale Encoding | 共享投影并增加 patch scale embedding |
| Lightweight Learnable Aggregator | 使用小 MLP/卷积聚合 4 个细 token |

#### 验收标准

- 当前 Merge 至少完成与 Selection 相同的数据集和训练周期。
- 多尺度实现不再回退到 Selection。
- 每个粗 token 都能映射回明确的原图区域和尺度。
- Merge 在至少一个数据集、一个 token 预算下显著优于同预算 Selection 或固定合并；
  如果没有，也应形成可信的负结论。

---

### 阶段 4：精度与计算量的统一对比

#### 阶段目标

形成项目最重要的 accuracy-efficiency Pareto 曲线。

#### Agent 自动执行任务卡

**执行环境：GPU benchmark + CPU 汇总。**

- [G] WAITING_GPU：锁定阶段 2 和 3 的 checkpoint，不再边测边改模型逻辑。
- [G] WAITING_GPU：在同一 GPU、软件环境和 benchmark 脚本下测试所有方法。
- [G] WAITING_GPU：分别测量 entropy/tokenization、Transformer 和端到端耗时。
- [G] WAITING_GPU：测试 batch size 1、32、128；OOM 时记录最大可用 batch。
- [G] WAITING_GPU：记录 FLOPs/MACs、真实 token、padded token、显存和吞吐量。
- [ ] TODO：在 CPU 上自动汇总结果，生成 Pareto 图和对比表。
- [ ] TODO：标记理论计算下降但 wall-clock 未加速的配置。

**Agent 停止条件：**

- 不同方法的 benchmark 条件不一致；
- 测量没有包含 APT 预处理开销；
- 连续三次测量波动超过 5% 且未定位原因。

#### 统一比较方法

1. Full ViT；
2. Downsample ViT；
3. Random Selection；
4. APT Selection；
5. APT Merge；
6. Hierarchical APT；
7. MAE + Router；
8. 可选的学习型 Patch Merging。

#### 统一指标

**精度指标**

- Best validation accuracy；
- Final test accuracy；
- 多 seed mean +/- std；
- 相对 Full ViT 的精度差。

**计算指标**

- 每图平均真实 token 数；
- padding 后平均 token 数；
- 模型 FLOPs/MACs；
- APT 熵计算与 token 组装开销；
- batch size 1 延迟；
- batch size 32/128 吞吐量；
- 峰值 GPU 显存；
- 参数量。

**统计图表**

- Test accuracy - average token；
- Test accuracy - measured latency；
- Test accuracy - throughput；
- Token 数分布箱线图；
- 每类样本的 token 分布。

#### 注意事项

Transformer 计算量并不与 token 数严格线性。Attention 部分近似随 token 数平方变化，
MLP 部分近似线性变化，因此不能再用“减少 50% token 等于减少 50% FLOPs”作为最终
结论，必须使用实际分析工具和真实硬件测量。

#### 验收标准

- 至少在三个数据集上形成完整 Pareto 曲线。
- 所有方法包含预处理开销后的端到端效率结果。
- 明确指出哪些方法只减少理论计算，哪些方法带来了真实 wall-clock 加速。

---

### 阶段 5：消融实验与机制分析

#### 阶段目标

解释 APT 在什么条件下有效，以及精度损失来自哪里。

#### Agent 自动执行任务卡

**执行环境：CPU 分析为主，GPU 只训练必要消融。**

- [ ] TODO：根据阶段 4 结果自动选择少量 Pareto 候选，不做全排列消融。
- [ ] TODO：生成熵图、patch 划分、正确/错误样本和类别 token 分布。
- [ ] TODO：分析 token 数与类别、图像复杂度、置信度和错误率的关系。
- [G] WAITING_GPU：仅运行能够回答明确机制问题的消融训练。
- [ ] TODO：为每项消融记录假设、控制变量、结果和结论。
- [ ] TODO：淘汰没有改善精度、效率或解释力的复杂组件。

**Agent 推进门槛：**

- 每个 GPU 消融必须在启动前写明研究问题；
- 同时变化多个因素的实验不得用于单因素结论；
- 阶段结束时必须形成推荐配置和明确负结论。

#### 消融维度

1. 熵特征：
   - 灰度 Shannon entropy；
   - RGB 分通道 entropy；
   - Laplacian variance；
   - edge density；
   - entropy + edge 的组合评分。
2. histogram bins：32、64、128、256。
3. 阈值策略：
   - 全局固定阈值；
   - 每图分位数；
   - 目标 token 预算；
   - 按数据集校准阈值。
4. patch 尺度：16/32、16/32/64。
5. 聚合方式：平均、插值、卷积、轻量可学习聚合。
6. 位置与尺度编码。
7. 动态 batch 策略：
   - 直接 padding；
   - token 数分桶；
   - packed sequence 或 nested tensor。

#### 可视化

- 原图、熵图和最终 patch 划分叠加图；
- 正确与错误分类样本对照；
- 不同类别的平均 token 数；
- 细粒度纹理区域是否被保留；
- Selection 丢弃区域与 Merge 粗粒度区域的差异。

#### 验收标准

- 能解释至少一个主要精度损失来源。
- 能确认熵值是否真的与分类所需信息一致。
- 能给出推荐配置，而不是只报告大量独立结果。

---

### 阶段 6：最终复现与结论整理

#### 阶段目标

将最优配置复现为最终结果，并更新项目文档。

#### Agent 自动执行任务卡

**执行环境：GPU 复现 + CPU 整理。**

- [G] WAITING_GPU：冻结代码版本、依赖版本、数据划分和随机种子。
- [G] WAITING_GPU：对最终候选执行 3 seeds 独立训练或复现。
- [G] WAITING_GPU：使用统一脚本完成最终 test 和效率 benchmark。
- [ ] TODO：校验所有表格数字可追溯到结果 JSON 和 checkpoint。
- [ ] TODO：更新 `EXPERIMENTS.md`、最终结果表和复现命令。
- [ ] TODO：生成最终图表与 patch 可视化。
- [ ] TODO：将 preliminary 结果与正式结果明确分区。
- [ ] TODO：输出项目结论、限制和后续工作。

**Agent 完成门槛：**

- 最终数字均可追溯；
- 没有使用 test set 调参；
- 同一指标的测量口径一致；
- 文档中的结论不超出实验覆盖范围。

#### 最终复现实验

1. 对每个数据集选取：
   - Full ViT；
   - MAE + Router 最优配置；
   - APT Selection Pareto 最优配置；
   - APT Merge/Hierarchical APT Pareto 最优配置。
2. 每个最终配置运行 3 个独立 seeds。
3. 使用同一硬件和同一 benchmark 脚本重新测试效率。
4. 固化 checkpoint、参数文件、日志和结果 JSON。

#### 最终结论应回答

1. 是否能在精度下降不超过 1% 时减少计算？
2. 如果不能保持 1%，在 2% 或 3% 精度预算下可减少多少 token 和实际延迟？
3. APT 相比 MAE + Router 的优势是精度、速度、无需额外训练，还是实现简单？
4. 哪些数据集适合 APT，哪些数据集不适合？
5. Selection、Merge 和层次化 APT 中哪一种最值得保留？

#### 最终产出

- 更新后的 `EXPERIMENTS.md`；
- 最终结果汇总表；
- accuracy-efficiency 曲线；
- patch 划分可视化；
- 可复现训练与评估命令；
- 最优模型 checkpoint 和配置文件。

---

## 7. 推荐执行优先级

### P0：本地 CPU 立即执行

1. 修复反归一化和熵输入。
2. 修复动态 padding attention。
3. 修复真实 token 统计。
4. 建立独立 validation split。
5. 完成阈值与 token 预算的无训练扫描。
6. 完成 Selection、Merge 和 hierarchical APT 的实现与 smoke test。
7. 生成自动训练队列、结果汇总和续训工具。
8. 判断已有 Merge checkpoint 是否兼容；不在 CPU 上继续训练。

### P1：租用 GPU 后完成可信基线

1. 完成候选阈值的短训练筛选。
2. 完成 Selection 的跨数据集实验。
3. 完成当前 Merge 和 hierarchical APT 的完整训练。
4. 补齐 Random 和 Fixed Merge 同预算对照。

### P2：GPU 候选实验与方法改进

1. 比较多种 token 聚合和位置编码。
2. 优化动态 batch 和实际推理速度。
3. 只对 Pareto 候选补多 seed 与必要消融。

### P3：形成最终报告

1. 多 seed 复现；
2. Pareto 曲线；
3. 消融与可视化；
4. 更新项目结论。

---

## 8. 阶段依赖关系

```text
阶段 0：正确性与协议
    |
    v
阶段 1A：CPU 阈值/预算扫描 ----+
    |                           |
    +--> 阶段 3I：CPU 方法实现 -+
                                |
                                v
                           GPU Gate
                                |
                                v
                     阶段 1B：GPU 短训练筛选
    |
    +------------------+
    v                  v
阶段 2：Selection      阶段 3T：Merge/多尺度训练
    |                  |
    +--------+---------+
             v
      阶段 4：统一精度-效率比较
             |
             v
      阶段 5：消融与机制分析
             |
             v
      阶段 6：最终复现与文档整理
```

任何阶段发现实现正确性问题，都应返回阶段 0 修正，而不是继续扩大训练规模。

---

## 9. 风险与应对

| 风险 | 影响 | 应对方式 |
|:-----|:-----|:---------|
| 动态 token 需要 padding | token 减少但没有真实加速 | 分桶、packed sequence、单图延迟单独报告 |
| 固定阈值跨数据集失效 | 结果不可比较 | 使用目标 token 预算或每图分位数 |
| CIFAR-100 熵区分度低 | 错误否定 APT | 以高分辨率数据集作为主要验证 |
| 完整矩阵算力过高 | 实验周期失控 | 先短训练筛选，再完整训练 Pareto 候选 |
| test 被用于选模型 | 结果偏乐观 | 固定 validation split，最后一次测试 |
| 不同实现的熵 bins 不一致 | 阈值失去可比性 | 抽取统一 entropy 模块 |
| 平均池化破坏细节 | Merge 精度下降 | 比较 resize、conv 和可学习聚合 |
| 旧 checkpoint 与修复后实现不兼容 | 无法直接续训 | 保留旧结果为 preliminary，修复后重新建立正式基线 |

---

## 10. 项目成功判据

项目不应只以“token 变少”作为成功。建议采用三级成功标准：

### 最低目标

- APT 在至少三个数据集上完成可信、可复现的实验；
- 形成精度与效率 Pareto 曲线；
- 明确当前方案有效或无效的条件。

### 预期目标

- 在至少一个高分辨率数据集上，减少 25% 以上平均 token；
- test accuracy 下降不超过 1% 至 2%；
- 端到端吞吐量有可测量提升。

### 理想目标

- 在多个数据集上，以不超过 1% 的精度下降减少 25% 至 50% token；
- 实际延迟和显存同步下降；
- APT 不需要 MAE、教师模型或额外蒸馏，训练流程明显简单于 MAE + Router。

---

## 11. 当前阶段结论

下一步不应直接根据 CIFAR-100 的 50-epoch Selection 和约 8-epoch Merge 结果评价 APT。
合理的推进顺序是：

1. 在本地 CPU 修复实现与评估协议；
2. 在本地完成跨数据集阈值/token 预算的无训练扫描；
3. 在本地完成 Selection、Merge、hierarchical APT 和自动实验工具；
4. 到达 GPU Gate 后租用 GPU，先做短训练筛选；
5. 在 GPU 上补齐 Selection 和 Merge 的完整训练；
6. 在相同 token 预算下与 MAE + Router 等方法比较；
7. 最终用真实延迟、吞吐量和多 seed 精度回答项目问题。

只有完成以上链路，项目才能可靠地判断：APT 是否能够在保持精度的同时减少 ViT 的
实际计算量，以及它相对于 MAE + Router 的真正优势在哪里。

## 参考资料

- 项目实验记录：`EXPERIMENTS.md`
- APT 论文：[Accelerating Vision Transformers with Adaptive Patch Sizes](https://arxiv.org/abs/2510.18091)
- APT 项目页面：[OpenReview submission](https://openreview.net/forum?id=04dNQ1m17o)
