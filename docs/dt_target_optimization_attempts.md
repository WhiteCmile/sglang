# DT V3 Target 侧优化：全部尝试、局部收益、端到端结果与原因

更新：2026-11-25。范围：Target 侧（调度、控制面、host 路径、verify、feature source），以及为定位 Target 问题而做的诊断。
环境：7T2D（H800_3：3 个 TP2 Target + 2 个 Draft；H800_4：4 个 TP2 Target），Qwen3-14B Target，FlashMTP-v2 Draft（block 8）。
来源：接手前的结论见 H800_4 `/home/lishuhuan/Target侧优化实验总结_2026-11-23.md`；接手后的逐项记录、原始数据路径见本目录 `README.md`。本文是二者的汇总，并按「局部有没有改善 → 端到端为什么没跟上」重新组织。

## 0. 先读：三个阶段的测量口径不同

| 阶段 | 口径 | 噪声 | 能分辨的效果 |
|---|---|---|---|
| A. 接手前（≤11-23） | 7T2D，确定性推理，每格 128–384 请求、单次冷启动，固定并发下的墙钟吞吐 | 冷启动间约 9%，短窗口相邻两轮可差 27%；ABBA 里 baseline 自身漂移 9.7–18.8% | 只能看出 >10% 的效果 |
| B. 重测 harness（11-24） | 3T2D 单机（并发 21）或 7T2D，非确定性，每格丢弃热身 + 2048–6144 请求长窗口，多次冷启动交错；exact-token 门禁单独跑（确定性模式） | 3T2D 1–2%；7T2D 6–7% | 3T2D 约 3%，7T2D 约 8% |
| C. 新指标（11-24 起，学长决定） | **TPOT p90 ≤ 30 ms 时的最大负载**；7T2D、PROFILE=0、每档 2 次冷启动，448/512 等档插值 | 同 7T2D；trace 本身在高负载下有 10–15% 开销，所以按新指标评估一律不开 trace | 约 2–3%（两档插值后） |

当前基线（shm+S1+S3，7T2D）：SLO 交点约 **并发 494–502、20.1–20.7k tok/s**。
关键背景数字（并发 384–448，带 trace，绝对值偏大）：每轮约 42–45 ms = verify GPU 约 10–11 + Draft 往返约 20–27 + Target host 侧等待约 10（GPU 结束→派发 2.3、候选到达→freeze 5.0–5.7、freeze→GPU 起跑 2.7）。每次 verify 平均只接受约 1.97 个 token。

## 1. 总表

判定：✅ 有效、保留；➖ 中性；❌ 负收益或已否定；🔍 诊断（不改行为）。

| # | 改动 | 做法 | 局部（目标段）变化 | 端到端 | 判定 |
|---|---|---|---|---|---|
| A1 | READY refresh | 选批前再调用一次完整 `poll()` | 未测局部 | −4.0%，复审 −13.9%，p95 变差 | ❌ |
| A2 | Adaptive READY hold / deadline-slack | GPU 结果 ready 或预估 slack 不足时提前结束 10 ms hold | 波数增加 | −9.3%；deadline/slack 版 −27.2%，接受长度 −2.8% | ❌ |
| A3 / B3 / C2 | Early result（`DT_TARGET_EARLY_READY_RESULT`） | poll 已证明 copy_done 时，在选批前处理上一波结果 | 重测：copy_done→开始处理本来只有 7 µs，没有可提前的空间；目标段「提交返回→下一轮派发」两臂相同（8.69–8.72 ms） | 旧 +2.5%（漂移 9.7%）；重测 +1.1%（噪声内）；SLO 交点 **−3.4%** | ❌ |
| A4 / B9 | 关 late-freeze | 不等 GPU 剩余时间，提前 freeze | 波更小 | 旧 −7.7%；重测 3T2D −0.8%；SLO 512 点 p90 30.3/30.9（基线 30.05/30.11） | ❌（保留 late-freeze） |
| A5 / B8 | READY hold=0 | 去掉最多 10 ms 的合批等待 | 波数 2.5 万→7.0 万，Target GPU 忙 41%→88% | 旧 −13.3%；重测 7T2D@48 **−8.1%** | ❌（保留 hold） |
| A6 | source slots=8 | 增加 feature source 槽 | source gate 在诊断里**从未触发**（reserve failure 0，高水位 3–4/24） | 初筛 +2.8% → 复测 −6.64% | ❌ |
| A7 | page64 分页 source pool | 64 行 extent，逻辑 lease 与物理 extent 解耦 | 同上，容量从不是瓶颈 | −1.2% | ➖ |
| A8 | early + slots8 / page64 / 三项组合 | 叠加 | 各单项局部都是 0 | −4.88% / −3.99% / −1.03% | ❌ |
| A9 | #281 P0 | 逐跳埋点 | 拆出 descriptor→dispatch 7.6 ms、candidate→freeze p90 10.7 ms、freeze→submit 2.9 ms | — | 🔍 |
| A10 | #281 P1 | 更早发布给 Draft | 完整 P1 未落地：FlashMTP 的 bonus/seq_len/eligibility 在 park 前不权威 | 无 | — |
| A11 / B6 | #281 P2 预留目的地（`DT_FLASHMTP_PRE_RESERVE_DEST`） | launch 时为整波预留候选目的地，park 时复用 | 局部未单独测（省 park 时一小段 allocator） | 旧 +1.26% / −7.17%；重测 3T2D **−0.2%** | ➖ |
| A12 / B5 | #281 P3 控制面拆分（`DT_TARGET_P3_CONTROL_SPLIT`） | READY 前只观察 native 完成，控制面挪到 launch 之后 | 3T2D：poll 0.519→0.476 ms，候选→freeze 1.68→1.60 ms（−0.08） | 旧 −14.65%（漂移 18.8%）；重测 **−0.2%** | ➖ |
| B1 | harness 修复 | 端口移出临时端口段；长窗口、交错冷启动、单独门禁 | 噪声 9% → 1–2%（3T2D） | — | ✅（测量） |
| B2 | 关确定性推理（测量口径） | 性能 A/B 改为非确定性，门禁另跑 | verify −1.7 ms（约 6 行）到 −3.1 ms（约 20 行） | 7T2D@48 +8.1%，3T2D +10.3%，7T2D@384 +7.5%，7T2D@768 **≈0** | ✅（口径；生产本来不开确定性） |
| B4 | 合并 READY H2D（`DT_TARGET_FUSED_READY_H2D`，已 revert） | build 内多次小 H2D 合成一次 | build 0.534→0.522 ms（−0.012） | −0.5% | ❌ |
| B7 | 策略：hold 5/20 ms | 改 `DT_TARGET_BATCH_WAIT_MS` | 5 ms：691 次到期截断，波更碎 | 3T2D：5 ms −2.3%，20 ms −0.6% | ❌（10 ms 最优） |
| B10 | shm 控制通道（`DT_TARGET_SHM_CONTROL`） | poll 控制包从 gloo 两次 TCP 广播 + pickle 改为同机共享内存双缓冲 | TP 广播 0.20→0.04 ms，候选到达→freeze 0.81→0.56 ms，poll 0.51→0.31 ms | 3T2D **+5.3%**；SLO 交点 **0** | ✅ 低负载 / ➖ 新指标 |
| B11 | S1 精简 sampling 构造（`DT_TARGET_LEAN_READY`） | 统一贪心参数时复用缓存的 device 张量切片 | sampling 0.116→0.047 ms | 3T2D +0.5%；SLO **0** | ➖ |
| B12 | S3 精简 poll 扫描（`DT_TARGET_LEAN_POLL`） | 只扫可能匹配的 arena 槽，加无锁预过滤 | poll 间隔 0.49→0.39 ms，候选到达→freeze 0.99→0.89 ms | 3T2D +1.6%；SLO **0** | ➖ |
| C1 | shm+S1+S3 组合在新指标下 | 320–640 扫描 | 384 点：TP 广播 0.285→0.066 ms，候选到达→snapshot 1.64→1.45 ms | 交点基线 498 / 20.39k，组合 497 / 20.39k，**0** | ➖ |
| C3 | 组批策略扫描 | conc 512：B=24/32、T=5 ms、adaptive hold、关 late-freeze | — | 默认（B16/T10）最好；B24 p90 30.9–33.1、B32 31.4–31.8（基线 30.1） | ❌（默认已最优） |
| C4 | verify CUDA-graph 档位（`DT_TARGET_CUDA_GRAPH_BS`） | 避开 cuBLAS gate_up GEMM 的坏形状（M=320/216/192） | bs40 档 15.28 ms → 改 pad 到 41 档；bench：33–40 行 11.2→7.5–8.6 ms | 交点 498→507（+4%），但几乎全来自基线一格偏低；去掉后 **≤1%**；门禁 PASS | ➖（无风险，可保留） |
| C5 | Target FP8 权重（`DT_TARGET_QUANTIZATION=fp8`） | 在线 FP8 量化 bf16 权重，Draft 不变 | 同负载 TPOT p90 −1.5–2 ms；接受长度 1.97→2.03 | 交点 494/20.1k → 539/21.5k（**+9% 并发、+7% 吞吐**）；token 一致率 25%；**GSM8K −1.3 pt（显著）**，MMLU 不变 | ✅ 性能 / ⚠ 改精度，待决定 |
| C6 | ready record 发布时序 | publisher 线程埋点 + 跨机时钟对齐 | Target event 完成→发出 0.018 ms，比 Draft 提交 pull 早 10.5 ms | — | 🔍（Target 无问题） |
| C7 | launch 迭代拆分 | run_batch 收尾、park 分段、select 细分 | 见 §4 | — | 🔍（进行中） |

## 2. 接手前（A 组）：做法、结果与原因

接手前的 A/B 都用 128–384 请求、单次冷启动，baseline 自身漂移 9.7–18.8%，所以 **±10% 以内的结论都不可信**。重测（B 组）后结论如下：

- **READY refresh（A1，❌）**：第二次完整 `poll()` 每圈都要付 TP 广播、Coordinator 收消息、cleanup 等全部控制面开销，而候选恰好在两次 poll 之间到达、并改变本轮成员的概率很低。**开销大过收益的地方**：每圈固定多付一次 poll（当时约 0.5 ms），收益只在少数圈出现。
- **Adaptive hold / deadline-slack（A2，❌）**：hold 的作用不是消除 GPU 空闲，而是把即将返回的行凑成大波。提前释放产生更多小波：Target 每波 verify 至少约 7 ms 固定成本，Draft 每批 proposal 约 2.2 ms 固定成本，两边一起变慢。接受长度 −2.8%，说明请求相位也被改变了。
- **Early result（A3/B3/C2，❌）**：重测发现目标段没有可提前的空间：copy_done 被看到后 7 µs 就开始处理。新指标下反而 −3.4%：同一线程里把约 1 ms 的结果处理挪到 select 之前，下一波的 launch 就晚约 1 ms。**开销**：另一条 lane 的 freeze→GPU 起跑变晚，抵消并超过了本 lane 早派发的收益。同一线程内调整顺序是零和的。
- **关 late-freeze / hold=0（A4/A5，❌）**：两者都让波变小。hold=0 的机制已拆清：Target 波数 ×2.8、GPU 忙 41%→88%；Draft 每批 3.4→2.2 行，执行占用 71%→107%（饱和），每行 GPU 658→980 µs；coordinator→Draft 接手 1.97→2.49 ms；每轮 23.6→26.1 ms。**开销**：两侧每批的固定成本被更多批次放大。
- **slots8 / page64 / 组合（A6–A8）**：诊断证明 source gate 从未触发（reserve failure 0，高水位 3–4/24，因 gate 被挡的 launch 为 0）。**没有局部收益**，只增加 bookkeeping，端到端为负或中性是必然的。
- **P2（A11/B6，➖）**：省下的是 park 时一小段 allocator 工作，但 launch 时要整波预留、回滚、取消不合格行，同一线程上的 bookkeeping 并没有减少，目的地的持有期也被拉长。重测 −0.2%，旧的 −7.17% 是漂移。
- **P3（A12/B5，➖）**：局部确实动了：poll −0.04 ms，候选→freeze −0.08 ms，但只有 0.04–0.08 ms。原因是控制面工作只是从 READY 前挪到 launch 后，**仍在同一个主线程里**，而且在处理上一波结果之前执行，并没有真正异步。旧的 −14.65% 是漂移（baseline 自身 −18.8%）。

## 3. 接手后（B、C 组）：做法、局部收益，以及端到端没跟上的原因

### 3.1 低负载有效、高负载归零：shm、S1、S3（B10–B12，C1）
- **局部**：shm 让 TP 广播 0.20→0.04 ms；S1 让 sampling 0.116→0.047 ms；S3 让 poll 间隔 0.49→0.39 ms。3T2D@21 上合计 +7.5%，门禁全部 PASS。
- **低负载为什么被放大**：3T2D@21 每轮约 17 ms，GPU 两波之间空闲约 8 ms，host 路径完全暴露在关键路径上；而 poll 每轮要做很多次（结果处理、hold、候选到达），每次 poll 省下的开销会被乘上次数。
- **SLO 交点为什么归零**：并发约 500 时每轮约 45 ms，这几项合计只省约 0.2–0.3 ms（<1%），而占大头的等待都没动：GPU 结束→派发 2.3 ms、候选到达→freeze 5 ms、Draft 往返 20–27 ms。更关键的是，高负载下主线程的瓶颈变成了 **launch 迭代本身**：一次约 6.85 ms 回不到 poll（select 1.4 + run_batch 2.2 + 结果处理 1.0 + poll 0.7 + 其他）。poll 变快，并不会让主线程更早回到 poll。

### 3.2 微合并无效（B4）
- 局部只省 0.012 ms。每个小张量操作 15–25 µs 是 torch 调度和 Python 开销，与拷贝次数无关；合并成一次 [3,n] H2D（0.065 ms）并不比一次单字段 H2D（0.048 ms）便宜多少。要省时间，必须去掉整段 Python 路径，或把它移出主线程。

### 3.3 关确定性：拐点前有效、拐点后被 Draft 吸收（B2）
- 局部：verify −1.7 ms（约 6 行）到 −3.1 ms（约 20 行）；拐点前每轮省下的时间 1:1 转成容量（7T2D@384 +7.5%，每轮 −4.4 ms，Draft 往返不变）。
- 并发 768（过拐点）时 ≈0：Target 省下约 6 ms，其中 4.2 ms 被 Draft 往返吃掉（coordinator 等 Draft 接手 32 ms）。**开销**：Target 更快 → 单位时间交给 Draft 的任务更多 → Draft 单线程 host 已饱和，排队更长。这就是「一处的时间降下去，别处积压反过来抵消收益」的典型例子。SLO 交点（约 500）在拐点附近，所以 Target 提速只有部分能转化为收益。

### 3.4 组批策略（C3）
- B 调大反而更差：verify 每多一行只多约 0.11 ms（1 行 6.8 ms，32 行 10.4 ms），大波几乎省不到 GPU 时间，却要多等候选凑批，等待直接计入 TPOT。T 调小、adaptive hold 又会产生更多小波（同 A2）。默认 B=16 / T=10 是两者的平衡点（#255 的调参在交点处依然成立）。

### 3.5 graph 档位（C4）
- 根因已确认：cuBLAS 给 MLP 的 gate_up GEMM（每 rank M×5120 × 5120×17408）在 M=320 时选了差 kernel（147 µs/层 对 M=328 的 100 µs/层），40 层累计约 2 ms；M=216、184/192 同理。
- 局部：33–40 行的波每波省约 2 ms。端到端 ≤1%：这类波在交点处约占 8%，平均到每轮只省约 0.15 ms。

### 3.6 FP8（C5）：唯一推动交点的项，但改变精度
- 局部：同负载 TPOT p90 低 1.5–2 ms（448：25.9/26.5 对 29.3/28.1；512：28.6/28.9 对 30.7/30.4）。原因是 verify 固定成本主要是每波读一遍约 14 GB/卡的 bf16 权重，FP8 把读取量减半。
- 端到端：交点 +7% 吞吐。负载继续往上加时收益缩小（576 时吞吐只高约 2.5%），符合拐点处被 Draft 吸收的规律。
- 精度：贪心解码下 token 一致率 25%（平均在第 58 个 token 分叉）；GSM8K 94.47%→93.18%（不一致 29/12 题，bf16 自身两次运行只有 1/3 题不一致，McNemar p=0.012），MMLU 78.71%→78.78%（无差别）。是否接受由用户或学长决定。

### 3.7 诊断：TPOT 尾部来自 sys-4，而 sys-4 的惩罚在 Draft 侧（C6）
- p90 以上的请求中 95–99% 来自 sys-4 的 4 个 Target，它们的平均 TPOT 比 sys-3 高约 3 ms。
- sys-4 每轮多出约 6.6 ms：拉 hidden +2.5、等进入 proposal 批 +1.8、候选推回记账 +2.3。
- 跨机时钟对齐后：Target 的 ready record 在 event 完成后 0.018 ms 就发出，比 Draft 提交 pull 早 10.5 ms；候选推回时，Target 的原生 observer 也比 Draft 记下 push complete 早约 2.6 ms。多出来的约 5 ms 全部在 Draft 进程的 RDMA pull 路径上（Python 循环 `sleep(50 µs)` 等 ready、READ 后的 fence 和拷贝、交回主循环），不在 Target。

## 4. 为什么局部收益常常到不了端到端：五条规律

1. **测量噪声比效果大**（A 组）：单次冷启动 9%、baseline 漂移最高 18.8%。接手前的 −14.65%、−7.17%、+2.5%、−7.7% 等结论大多是噪声。
2. **固定开销在低负载被放大，在高负载被稀释**（shm/S1/S3）：低负载时 host 路径暴露在关键路径上，又按 poll 次数累乘；高负载时每轮被 Draft 往返和 verify 主导，host 的固定开销占比很小。
3. **同一线程内调整顺序是零和的**（early result、P3）：主线程串行做观察、选批提交、上一波结果处理和派发，把一件事提前，另一件就推后。真正的收益要靠把工作移出主线程（GPU 发布、原生线程），或者让它整段消失。
4. **Target 更快会让 Draft 更忙**（关确定性@768、FP8@576）：交点在 Draft 单线程 host 的拐点附近，Target 省下的时间有一部分会变成 Draft 前面的排队。
5. **两侧的批次固定成本互相耦合**（hold=0、关 late-freeze、adaptive hold、B 调大）：Target verify ≥6.8 ms/波，Draft proposal 约 2.2 ms/批。波变小，两边的固定成本都被放大；波变大，又要多等候选凑批。默认策略已经接近平衡点。

## 5. 进行中：Target 主线程结构性改造（用户 2026-11-25 批准三步）

目标：让主线程不再卡住请求。按依赖关系，执行顺序调整为第 2 步 → 第 3 步 → 第 1 步（原因：p90 由 sys-4 决定，而第 1 步只有连远程 ring 一起改成原生发送才会对 sys-4 生效，改动最大）。

- **第 2 步：缩短 launch 迭代**。首次追踪（`results/20261125-182003-loop-trace-7t2d`，带逐行 p0 事件，数值偏大）中主线程最重的几段：
  - pivot 发布 `stage_overlap_pivot_copy` 0.85 ms：在 run_batch 内，推迟主线程回到 poll；疑点是每波新建跨进程 CUDA event 并导出 IPC 句柄，以及逐行构造 descriptor；
  - park 里的 `queue_flush` 0.47 ms：把任务写进 ring 或交给 Coordinator；
  - select 中 build（0.55 ms）和 update_running（0.26 ms）之外的部分，约 0.67 ms；
  - 结果处理中 park 之外的部分，0.27 ms。

  去掉逐行事件后的干净追踪在排队（`queue/28-loop-trace2-7t2d`）。拿到干净数据后按耗时从大到小处理：per-slot event 复用、descriptor 批量构造、flush 向量化。
- **第 3 步**：候选到达 → freeze 中由主线程忙造成的部分（约 1.5 ms），预计随第 2 步缩小，再单独处理剩余部分。
- **第 1 步**：FlashMTP 的 Draft 任务改由 GPU 在 launch 时直接发布（复用 DFlash 的 §7.1/Q15b kernel，每请求 1 行 pivot），Draft 侧 `_accept_ring_record` 构造 FlashMTP 形状的任务；sys-4 的远程 ring 改为 kernel 先写本地暂存、再由 host 回调发 RDMA。预期消掉每轮「GPU 结束 → 派发」2.3 ms（p90 约 5 ms）。

每一步都按新指标验收：7T2D 448/512、PROFILE=0、每档 2 次冷启动、512 档跑 exact-token 门禁；GPU 使用遵守共用规矩（每次持锁一个计划、≤1 小时，经 CI gpu-lease 等 GPU 空闲，结束时按进程组清理）。

## 6. 代码与数据索引

- 接手前：early `de585f9`、adaptive hold `13c0542`、READY refresh `dd41fae`、paged allocator `b03f57f` / `7eb9efc`、#281 P0 `f28805c` / `ea3d7fd`、P2 `1e3bc66` / `2d7faf4`、P3 `9d80488`；结果在 `/data/lishuhuan/dt-v3-pilot/{target-ab,target-combo,source-metrics,issue281-p2,issue281-p3}-results/`。
- 接手后代码：`/data/lishuhuan/dt-v3-remeasure`（分支 `v3_target_lsh/remeasure-instrumentation`：埋点、fused H2D 已 revert、shm `27ef43feb`）；`/data/lishuhuan/dt-v3-lean`（分支 `v3_target_lsh/lean-ready`：S1 `16a237d05`、S3 `c85af64d1`、FP8 开关 `be290ed5f`、graph 档位开关 `e1a62349e`、ready publisher 埋点 `b2ded8d2c`、run_batch 收尾埋点 `6f74ec8dd`）。
- 接手后数据：本目录 `results/<stamp>/`，逐项见 `README.md`。分析脚本：`slo_interp.py`、`tpot.py`、`loop_iter.py`、`loop_tail.py`、`hostpath.py`、`draft.py`、`ready_join.py`、`bench_verify_shapes.py`、`pick_buckets.py`、`eval_quality.py`。
