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
