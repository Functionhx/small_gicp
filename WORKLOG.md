# 工作日志（cuda-x86）

## 2026-08-17 · 第 1 天：设计定稿

**已完成**
- 深读 upstream：热点路径（GICP 因子/KdTree/LM/降采样/协方差）、VGICP 引擎语义（`search_offsets=1` 中心体素单探）、`IncrementalVoxelMap::insert` 为串行热点。
- 环境确认：RTX 4070 16GB (sm_89) / CUDA 12.4 / Ryzen 9 9950X；本地 KITTI 仅 object3d 离散帧，需下载 odometry seq 00 子集（622MB）。
- 决策（用户确认）：GICP+VGICP 双引擎同时做；交付为本仓库分支内 GPU-native 实现（无兼容层）；KITTI 00 + 本地合成双基准；双层精度验收；C++/CUDA 项目；A/B/C 三架构全部参数化实测对比。
- Git：fork 至 Functionhx/small_gicp；`cuda-x86`（worktree 开发）+ `cuda-jetson` 分支已推送；验证后设 default（挂起任务 #7）。
- 设计文档：`docs/superpowers/specs/2026-08-17-small-gicp-cuda-design.md`，第 1/2 段经用户逐段确认，第 3 段由用户授权自主定稿。

**关键设计要点**
- 降采样 radix sort 输出（排序体素桶+CSR）即空间索引，kd-tree 退役。
- 全 GPU 常驻 fp32 + 归约/求解 fp64；LM 控制在 CPU；CUDA Graph 固化迭代。
- 上游 LM/终止/拒绝准则逐常数移植，保证精度对齐。

**下一步**：M0 脚手架（CMake CUDA + GTest + KITTI/bin+ply 读取 + 空 harness）。

**挂起决策**：无。

## 2026-08-17 · T1 完成：CUDA 脚手架

- 顶层 `BUILD_CUDA` 门控 + `cuda/` 子项目；`sgc` 静态库（cudart 公共链接）。
- `sgc::GpuBuffer<T>`（RAII/移动/上传下载）+ `SGC_CHECK` + `launch_fill_kernel`。
- smoke_test 2 用例通过；无 CUDA 默认配置不受影响。
- 坑位记录：(1) worktree 内构建源路径是 `..` 而非 `../..`；(2) `enable_testing()` 必须在顶层调用；(3) .cpp 测试链 CUDA 头需 `CUDA::cudart`；(4) `<<<>>>` 只能在 .cu，host 包装命名 `launch_*`。
- 决策：执行方式为会话内联（GPU 编译-运行-调试循环紧）。

## 2026-08-17 · T2 完成：IO + GpuCloud

- `sgc::io::read_ply`（移植 upstream，修正了任意 stride 读取）+ `read_kitti_bin`。
- `sgc::GpuCloud`：points(float4)/covs(9f)/keys(u64) 三缓冲 + 宿主互转助手。
- io_test 2 用例通过（data/target.ply 69088 点，4 属性 stride=4）。
- 坑位：file(GLOB) 配置期求值，新增源文件要重跑 cmake。

## 2026-08-17 · T3 完成：GPU 降采样

- `voxel_key.hpp`（fast_floor/21bit×3 打包/解包，host+device）+ `Downsampler`（keys → CUB radix sort → run flags → ExclusiveSum/CSR → 每桶一 block fp64 树归约质心）。
- ParityWithUpstream 通过：桶数、key 升序、质心 1e-4 内与 upstream 一致（data/target.ply, 69088→N 点 @0.25m）。
- 坑位：(1) 上游 voxelgrid_sampling 对 vector<Vector4f> 输出需显式 OutputPointCloud=small_gicp::PointCloud；(2) CUB 输出缓冲（reduce_out_）必须显式分配。

## 2026-08-17 · T4 完成：协方差估计

- 内核：自适应扩张壳精确 kNN（早停界 = 到当前立方体边界的最近距离）+ 稀疏点第二遍暴力补全 + Jacobi 3x3 特征分解 + `cov = I − 0.999·n·nᵀ`。
- Parity：median<1e-4、p99<1e-3、p99.9<5e-3、max 1.8e-2（孤立 tie 离群）；确定性测试通过（逐位一致）。
- **调试记录（重要）**：
  1. 初版固定 5³ 邻域 top-20 → 中位误差 3.4e-2。CPU 复现隔离证明：邻居集合相同时数学 100% 正确（bad_with_same_set=0），错误全来自集合差异——KITTI 户外第 20 近邻常在 ±0.5m 外。
  2. 早停界推导：查询点在自身体素内分数坐标 f，越出 s-cube 的最小世界距离 = (s + min_axis(f,1−f))·leaf。
  3. 二次修复：n==k 但壳耗尽未触发早停的集合也不精确 → 高位 bit 标记未终止，pass-2 暴力重算（修掉 p99 0.76→1e-3）。
  4. 残差 ~0.1% 点为 k 边界等距 tie（kd-tree 与排序插入选点不同），影响 ≤2e-3，无害。
- 结论：协方差搜索从此**精确**等价 kd-tree kNN（除 tie），无需策略参数。

## 2026-08-17 · T5 完成：NN 策略

- `nn_query.cuh`（device 模板，voxel3/voxel5 立方探查 + ExactBF 线性全扫）+ `nn_search.hpp/cu`（宿主批查 API）。
- 一致率（data/target.ply @0.25m，60k 点，kd-tree 参考）：ExactBF 全等（same-index>99.9%）；Voxel3 <1% 不一致；Voxel5 <0.2%。
- ExactBF 60k×60k 在 4070 上 ~0.1s——作为对照/验收路径完全可用。

## 2026-08-17 · T6 完成：线性化 + 确定性归约

- 融合内核（每源点一线程）：T·p → NN(策略) → M=(Ct+R·Cs·Rᵀ)⁻¹ → J=[R·skew(p)|−R] → H(上三角21项)/b/e + CorrCache（target_idx+M）。
- 确定性归约：43 值 warp shuffle fp32 → lane0 fp64 定槽写 partial → CPU 按序 double 终和 → H 对称化。
- 对拍（合成对，T 偏离最优 0.15m/0.5°）：ExactBF 全部 <0.1%；Voxel5 <0.5%；inlier 数一致；100 连跑逐位一致。
- 调试记录：(1) T 在最优解时 ref_b≈0 导致相对误差虚高——测试改为偏置 T 使梯度有量级；(2) 内点计数曾只统计 lane 0 自身点（1/32 症状 0.0314），`__ballot_sync+__popc` 修复。

## 2026-08-17 · T7 完成：LM + GicpGpu 引擎（含重大性能工程）

**功能**：`sgc::GicpGpu::align`——LM 循环逐行移植 upstream（λ₀=1e-3/×10/外20/内10/LDLT/右乘 se3_exp/终止准则），`se3_exp` 精确移植。合成对端到端：与 upstream 位姿差 <1cm/0.3°（真实验收门），inlier 数 ±2%。

**新增基础设施**：`VoxelHashIndex`（ulonglong2 单加载槽 + splitmix64 + atomicCAS 线性探测）建在 GpuCloud 上，协方差壳搜索/NN/配准内核全部受益。

**性能攻坚记录（本日主要时间投入，KITTI 35k 体素帧）**：
| 方案 | 协方差耗时 |
|---|---|
| 固定 5³ 邻域（精度错） | 11ms |
| 自适应壳+二分探查 | 38ms |
| 壳+哈希(两加载槽) MAX_SHELL=10 | 52ms |
| 壳+哈希 MAX_SHELL=8 无暴力回退 | 10.2ms |
| **壳+哈希 MAX_SHELL=4（当前）** | **2.4ms** |
- 根因链：(1) KITTI 半稀疏（环间距 0.5-1m），大量点 20NN 在 1-2m 外；(2) 深层壳探查是随机访存延迟尾部（6.5% 点烧 ~1242 探查/点）；(3) N² 暴力在 35k 点 = ~400GB 流量不可行。
- 20NN 壳分布实测：63% 在壳3内、76.5% 壳4、93.5% 壳8。**决策 MAX_SHELL=4**：76.5% 精确集合，其余用部分邻居（≥5 即有效平面），精度由 T11 KITTI 门裁决。
- 小云（≤8192 点）保留 warp-轮转暴力精确路径（单测严格对拍依赖它）。
- 分项（暖态）：read 0.2-0.5ms / H2D 0.23ms / downsample 0.7-1.2ms / covariance 1.5-4.1ms（随密度）。
- 每迭代 align ~0.3-0.8ms（35k 点 Voxel5）。

**遗留（T12）**：深层壳尾部延迟用 warp 协作探查根治；kernel launch 间隙用 CUDA Graph。

**测试**：17/17（含 eval_error 自洽性检查、逐位确定性、ExactBF/Voxel5 线性化对拍、双引擎 align 对拍）。

## 2026-08-17 · T8 完成：VoxelHashMap + LRU

- `VoxelHashMap`：开放寻址（slot=表位）、排序 run 串行累加 + 每体素每帧一次原子（跨帧确定性）、finalize 写独立 mean/cov、LRU horizon/cycle 移植、倍增 grow 全载荷迁移。
- Parity：与 upstream GaussianVoxelMap 双帧插入对拍——体素数一致、mean<1e-3、cov<2e-2（fp32 原子和 vs double）；确定性测试逐位通过。
- **调试教训（1.5h 排查）**：对拍失败根因是测试 dump 函数 `download(..., cap)` 少乘 9（只填前 1/9，槽位靠后的体素全零）。GPU 实现自始正确。排查路径：sum_cov 直读→finalize 直读→槽位 TRACE→同一循环双路对质→定位 dump 笔误。过程中顺手修掉两个真 bug：槽累加数组未清零（依赖 cudaMalloc 零页）、grow() 丢载荷。

## 2026-08-17 · T9 完成：VgicpGpu 引擎 + 越界 bug 根治

- `VgicpGpu::align`：中心体素单探（与 upstream `set_search_offsets(1)` 同义）+ 同款 LM 循环 + CorrCache 误差重评。合成对与 upstream 对拍通过（<1cm/0.3°）。
- **重大 bug 修复（影响 GICP 与 VGICP 两引擎）**：尾块中 `i≥num_source` 的线程仍执行 warp 归约并写 `partials[warp_id*43]`——缓冲按 ceil(n/32) 分配而实际发射 ceil(n/256)×8 个 warp → 越界写 ~2.4KB 到相邻缓冲。症状随分配布局漂移：slot=0 幽灵对应、M 缓存 NaN、结果非确定。修复=按发射 warp 数分配（两处）。
- **调试方法论收获**：症状"矛盾"（e 有限但缓存 NaN）+ 非确定性 ⇒ 立即上 compute-sanitizer，比继续打补丁快一个量级。inv3 加了行列式下限保护（防御 NaN 输入）。
- 数据集状态（用户询问）：KITTI 00 尚未下载（T11 执行）；本地 object3d 约 6GB 为会话前已有。
- 测试：20/20，sanitizer 0 错误。

## 2026-08-17 · T10 完成：odometry_gpu 对比 harness

- CLI：`--exec full-gpu|hybrid|cpu` × `--engine gicp|vgicp` × `--nn voxel3|voxel5|exact-bf` + 参数 + `--traj/--report`；KITTI 目录模式 + `--synth` 合成漂移序列模式；计时口径=upstream（降采样+预处理+配准，I/O 在外）。
- **合成序列验收层 1 通过**：两引擎 100% 帧在 1cm/0.3° 内（GICP max 0.00cm / VGICP max 0.03cm，vs upstream 串行 double）。
- 稀疏 ply（6k 点）：cpu 11.4ms vs full-gpu 4.6ms（2.5×；9950X 单线程太强，Orin 上差距将拉大）。
- 稠密 KITTI 帧（无关场景最坏情况，满 20 迭代）：**GICP 17×**（191.6→11.3ms p50）、**VGICP 10×**（67.2→6.8ms p50）。
- hybrid 在 x86 上比 cpu 慢（CPU kdtree+covs 与 GPU covs 双算）——符合预期，Orin 上才有意义；已在代码注释说明。
- 首帧 CUDA 上下文预热 ~130ms，报告看 p50/min。

## 2026-08-17 · T11 进行中：真实帧精度攻关（重大）

**发现**：真实 KITTI 帧上 GPU 与 CPU 轨迹差达 0.45-0.9m/帧（合成对却 100% 达标）。逐层隔离：
1. NN 窗口（voxel5 ±0.5m 装不下 ~1-2m 初始位移）→ 加自适应二次扩张（未命中/超界→9³ 重探）。GICP 差 0.455→0.365m，**非主因**。
2. **协方差近似（MAX_SHELL=4 部分邻居集）是根因**：强制精确路径后单帧差坍缩至 3-9mm/0.002-0.02°。
3. 工程化精确协方差：**warp 协作壳搜索**（每点一 warp，壳偏移跨 lane 分摊隐藏延迟、轮式候选合并、lane0 字典序插入、早停界）——精确且 38ms→**4.99ms**。修复两个实现 bug：lane 每 shell 只留一个候选会丢点（改轮式全合并）；壳耗尽但未触发早停被误标精确（一律交暴力补全）。
4. 结果：8 帧轨迹 APE GICP 0.64→**0.071m**、VGICP 2.77→**0.223m**；单帧 ~9mm（=fp32×20 次未收敛迭代累积，合成收敛对 100% 在 1cm 内）。exact-bf 与 voxel5+扩张结果逐位一致 → NN 已非瓶颈。
5. 挂起：VGICP 残差 18cm/帧 疑来自大云（>8192）地图体素的 2.6% 部分协方差——KITTI 全序列 APE/RPE 相对指标裁决。

**网络**：Google Drive 下载持续失败（代理 SSL/网络中断，重试循环挂后台）。object3d 相邻帧确认为同场景连续帧，先用其做真实序列基准。

## 2026-08-18 · T11：真实序列基准 + 下载状态

- 网络：换 7897 代理后连通，但 Google Drive 文件配额墙（"many accesses"），每 30 分钟重试中；**备选：用户浏览器直接下载**放 `~/datasets/kitti/odometry/KITTI00.tar.gz`。SemanticKITTI=80GB 过大、HF 无该子集镜像、Kaggle 需 key。
- object3d 帧 0-59 验证为同场景连续序列，60 帧真实基准（p50 ms/帧）：
  | 引擎 | CPU | GPU | 加速 | 60帧轨迹差 |
  |---|---|---|---|---|
  | GICP | 150.7 | 33.4 | 4.5× | APE 0.53m（~0.9cm/帧，门内） |
  | VGICP | 78.4 | 8.9 | 8.8× | APE 1.80m（~3cm/帧，开放项） |
- MAX_SHELL 12→16 试验：协方差 4.99→9.29ms，VGICP APE 不变（1.83）→ 深壳非瓶颈，已回退 12。VGICP 残差疑为 fp32 地图累积反馈放大，待全序列 APE/RPE 相对指标裁决。
- GPU 吞吐：GICP 27.5 fps、VGICP 91.5 fps（4070，含最坏情况满迭代）。

## 2026-08-18 · T12：性能调优

1. **NN/线性化分离 + warp 协作批量 NN**：独立内核每查询一 warp（5³ 窗口归约 → 超阈值才扩张到剪枝后的 9³ 球，剪枝基准取 max(best, max_dist²)），线性化内核读预算结果。期间修复关键 bug：批量 NN 漏乘 T（用源坐标查询）——8ms 假快精度崩，修后精确一致。
2. 逐体素剪枝（体素最近距离² ≥ 参考距离² 跳过）。
3. 60 帧真实序列 GICP：27.75→**16.33ms**（p50），vs CPU 150.7ms = **9.2×**；APE 与逐线程版逐位一致（0.532m）。
4. 结论：NN 内核提速后协方差（5.3ms）成为最大单项；launch 开销占比小，CUDA Graph 低 ROI 暂缓；流重叠留 Orin 移植期。

## 2026-08-18 · T13：报告与收尾

- BENCHMARK_GPU.md：60 帧真实序列 GICP 9.2×/VGICP 8.7×、单帧精度表、内核级对拍、流水线分解、复现步骤。README 增加 GPU 构建节。
- 待办（用户协助）：KITTI00.tar.gz（622MB）浏览器下载放 ~/datasets/kitti/odometry/ → 正式 APE/RPE 验收 → 若过再设 default 分支（用户原条件"验证完成后"）。
- jetson 移植要点已写入设计文档 §11。

## 2026-08-18 · T13 完成态

- 最终验证：20/20 测试、合成验收 30/30=100%（1cm/0.3°）、审计通过（我方 13 提交作者/签名合规、零 Co-Authored-By；上游历史自带 5 处非我方）。
- 已推送 origin/cuda-x86（335ed04）。default 分支切换按用户条件挂起（待 KITTI00 验收，任务 #7）。

## 2026-08-18 · T11/T13 完成：官方数据验收通过，全线绿灯

- **数据突破**：上游 Google Drive 子集已死链（404）；改用官方 avg-kitti S3 桶 + HTTP Range 远程 zip 抽取（`scripts/fetch_kitti00_range.py`，免注册免 80GB 下载，按帧精准抽取）+ 官方 GT 位姿（data_odometry_poses.zip）。
- **正式验收（KITTI-00 官方 100 帧）**：GICP APE/RPE 相对差 +0.01%/+0.00%、VGICP +0.00%/+0.00% —— **双引擎 PASS（门限 5%）**。速度：GICP 43.9→6.0ms（**7.3×**，136fps）、VGICP 34.4→5.6ms（**6.1×**，143fps）。
- object3d 上 VGICP ~3cm/帧 残差确认为无关场景最坏情况的 fp32 放大；官方数据下四位小数一致，撤销该开放项。
- 验收四层全过：合成 100%、KITTI ≤0.01%、内核 20/20、确定性逐位。执行用户指令：设 default 分支。

## 2026-08-18 · cuPCL 对比（用户点题，诚实口径）

- 用户指路 `NVIDIA-AI-IOT/cuPCL` 的 `x86_64_lib` 分支（x86 预编译 .so，sm_86 cubin 经 CUDA 小版本二进制兼容在 sm_89 实测可跑）。
- **公平对比**（同数据同链式策略，降采样计入计时）：cuPCL 最优配置=CPU降采样+cuICP **9.44ms**（APE 51.4m，trimmed point-to-plane）；其 cuFilter GPU 降采样在 0.25m 分辨率 ~275ms（参数与官方 demo 一致）反成瓶颈；**我们 6.01ms（1.6×）且与 upstream GICP 位姿 ≤0.01% 一致**。
- 撤回先前"700×"说法（原始未降采样输入对 cuICP 是无效工况）——用户判断正确。
- 结论入 BENCHMARK_GPU.md；bench 源码入库 cuda/bench/cupcl_bench.cpp。

## 2026-08-18 · 代码侧收尾（用户确认维持 fork 分支形态）

- CI：`.github/workflows/cuda-build.yml`——GPU-less runner 上的 sm_89 + sm_87 双架构编译验证；`gtest_discover_tests` 改 `DISCOVERY_MODE PRE_TEST`（无 GPU 环境构建不执行二进制）。
- cupcl_bench 补 cuPCL 库获取与构建说明（不 vendor 其 .so，指路官方仓库）。
- README 增加文档导览图。
- 打 tag `v1.0.0-gpu`（论文工件引用锚点，可配 Zenodo DOI）。
