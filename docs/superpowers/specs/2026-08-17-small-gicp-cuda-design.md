# small_gicp GPU 加速设计（cuda-x86 / cuda-jetson）

日期：2026-08-17
分支：cuda-x86（开发/验证），cuda-jetson（移植）
上游参考：koide3/small_gicp v1.0.1 (aea1313)

## 1. 目标与约束

- **终极目标**：Jetson Orin NX（aarch64 + Ampere sm_87 + 统一内存）上极致的连续帧点云配准速度；先在 x86_64 + RTX 4070 (sm_89) + CUDA 12.4 上开发验证。
- **唯一硬指标**：端到端单帧延迟（含降采样、预处理、传输、迭代、同步），面向 LiDAR/SLAM 连续帧场景；配准结果与 upstream 可靠性对齐。
- **形式**：本仓库分支内全新 GPU-native 实现（`cuda/` 目录），以 upstream 为算法参考与 baseline，不做无 GPU 兼容、不做兼容接口。
- **架构选型不做二选一**：A（全 GPU 体素桶管线）/ B（精确 NN 路径）/ C（CPU 预处理+GPU 迭代混合）全部做成运行时参数，实测对比。

## 2. 验收标准（双层）

1. **合成序列层**：本地帧施加已知扰动（平移 0.1-1.0m、旋转 0.5°-3°，≥3 组量级），GPU 与 upstream CPU double 逐帧对比，平移差 <1cm 且旋转差 <0.3° 的帧占比 ≥99%。
2. **KITTI 00 层**：APE/RPE 与 upstream 同参数结果相差 <5%（GICP 与 VGICP 两引擎）。
3. **内核级单测**（GTest，逐项 vs upstream CPU 参考）：
   - 降采样：体素集合与质心一致（质心差 <1e-4 m）
   - 协方差：Frobenius 相对差 <1e-3（典型值）
   - NN：voxel3/voxel5 vs 精确 kd-tree，内点对一致率 ≥99%（exact-bf 全等）
   - 因子：固定对应关系下 H/b/e 相对差 <1e-4
   - 归约：固定 launch config 下 100 次运行逐位一致
4. **稳定性**：KITTI 连跑 5 次，轨迹最大差异 <1e-3 m（确定性归约 + 可选确定性插入）。

## 3. 代码布局

```
cuda/
├── include/sgc/
│   ├── core/       # GpuBuffer / Stream / Graph RAII, 类型定义, 错误检查
│   ├── voxel/      # 体素 key 打包、排序桶索引(CSR)、开放寻址哈希、增量高斯体素地图
│   ├── preproc/    # 降采样、法线/协方差估计
│   ├── search/     # NN 策略: voxel3 / voxel5 / exact-bf
│   ├── factor/     # GICP 因子内核 (fp32 数学 + double 归约接口)
│   └── reg/        # RegistrationGpu 引擎 + LM CPU 控制器
├── src/            # .cu 内核（与 include 一一对应）
├── test/           # GTest 逐内核 vs upstream 参考
└── bench/          # odometry_gpu 对比 harness
```

CMake：顶层新 option `BUILD_CUDA`（依赖 CUDA toolkit，`CMAKE_CUDA_ARCHITECTURES` 可配，默认本机 native）。upstream 代码路径零改动，baseline 与 GPU 实现同二进制链接。

## 4. 运行时策略参数

| 参数 | 取值 | 对应方案 |
|---|---|---|
| `exec` | `full-gpu` / `hybrid` / `cpu` | A / C / upstream baseline |
| `nn` | `voxel3` / `voxel5` / `exact-bf` | A / A+ / B（仅 GICP 引擎；VGICP 恒为中心体素单探） |
| `engine` | `gicp` / `vgicp` | scan-to-scan / scan-to-model |
| `det_insert` | bool（默认 on） | 插入前按 key 排序保证确定性 |

CLI 与 config 一处定义，bench/单测/引擎共享。

## 5. 设备端数据结构

| 结构 | 布局 | 说明 |
|---|---|---|
| 点 | `float4 xyz1` SoA | 16B 合并访问；fp32 |
| 协方差 | `float[9]` 3×3 | 对齐 upstream 左上 3×3 用法 |
| 体素 key | `uint64`（3×21bit） | fast_floor 语义移植 |
| 桶索引 | 排序 unique keys + CSR | 降采样输出即索引 |
| 增量地图 | 开放寻址哈希 2^n | {key, state, Σpt f32×4, Σcov f32×9, count, lru} |

内存：预分配 arena，帧间零 cudaMalloc；可选 cudaMallocAsync。

## 6. 内核流水线

**预处理**：keys → CUB radix sort → unique/CSR → 分段质心 → 协方差（邻桶候选 + 线程内 top-20 排序插入；`exact-bf` 可切全量暴力 kNN）。

**配准迭代**（CUDA Graph 固化，外层≤20）：
1. 融合内核/源点：`T·p` → NN（策略分派）→ RCR=Ct+T·Cs·Tᵀ → 3×3 闭式求逆 → H=JᵀMJ, b, e
2. 确定性归约：warp shuffle fp32 → lane fp64 累加 → per-warp partial → CPU double 终和
3. D2H 43 double → CPU LDLT `(H+λI)δ=-b` → 收敛判断 → H2D T
4. LM 内层：缓存 target_index/mahalanobis 的轻量误差重评内核

**NN 分派**：voxel3=27 桶、voxel5=125 桶、exact-bf=分块全扫；VGICP=中心体素单探（与 upstream `set_search_offsets(1)` 精确同义）。

**VGICP 插入**：变换帧点 → key → atomicCAS 占位 → atomicAdd Σpt/Σcov → finalize；LRU 标记-压缩（horizon=100/cycle=10 语义移植）。

## 7. 数值精度策略

存储 fp32；因子数学 fp32；归约 partial GPU fp64 + 终和 CPU double；LM 求解 CPU double（Eigen LDLT）。LM 常数逐项移植：λ₀=1e-3、×10、外层≤20、内层≤10；终止准则 1e-3 m / 0.1°。fp64 仅占总 FLOP ~0.1%。

## 8. 对比 harness 与指标

`cuda/bench/odometry_gpu`：同一二进制内链接 upstream CPU 引擎与 sgc GPU 引擎；计时口径与 upstream `benchmark_odom.hpp` 完全一致（每帧 = 降采样+预处理+配准，I/O 在外）。输出 KITTI 格式轨迹 + 延迟统计（mean/p50/p95/max）+ `--report json`。

- 合成模式：`--synth` 施加已知扰动，输出双层验收统计。
- APE/RPE：python 评估脚本（优先 evo，退化自写 ~100 行 KITTI devkit 式）。
- 资源：GPU mem/util 采样线程 + RSS + 可选 Nsight Systems。

## 9. 里程碑

| 里程碑 | 内容 | 出口条件 |
|---|---|---|
| M0 | 脚手架：CMake/架构/GTest/数据读取/WORKLOG | 空测试跑通 |
| M1 | 体素排序 + 降采样 | 单测：桶集合与质心 vs upstream 一致 |
| M2 | 协方差 + NN 策略 | 单测：协方差/NN 一致率达标 |
| M3 | 因子 + 归约 + LM + GICP 引擎 | 合成对（仓库 ply + object3d 采样帧）端到端收敛，因子/归约单测全绿 |
| M4 | KITTI harness + 双方数字 | 验收层 1+2（GICP） |
| M5 | VGICP 增量地图 + 引擎 | 验收层 1+2（VGICP） |
| M6 | 调优：Graph/流重叠/融合/扫描表 | 延迟预算达成，扫描对比表 |
| M7 | 报告 BENCHMARK_GPU.md、最终验证、推送、设默认分支、jetson 移植要点 | 全部验收门通过 |

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 体素邻域 NN 近似超差 | 一致率单测量化；超 1% 则 voxel5/精确回退（已是参数） |
| 协方差 kNN 近似差 | 同上；exact-bf 对照定位差异来源 |
| fp32 数值漂移 | 归约/求解已 fp64；验收层把关 |
| 哈希原子累加非确定 | det_insert 排序插入；稳定性测试 5 连跑 |
| 4070 上 GPU 化收益被强 CPU 掩盖 | 报告同时给 hybrid/cpu 数据；以 Orin 预期为准绳 |
| KITTI 00 下载受限 | 代理下载 622MB 子集；本地 object3d 合成对先行开发 |

## 11. Jetson Orin NX 移植要点（cuda-jetson 分支）

- `CMAKE_CUDA_ARCHITECTURES=87`；JetPack 6 / CUDA 12.x；代码禁用 x86 专属 intrinsics。
- 统一内存：数据本就 GPU 常驻，H2D 仅原始帧一次（Orin 上近零成本）；必要时 cudaMallocManaged 消最后回传。
- A78AE CPU 弱：`hybrid` 模式预计劣势扩大，`full-gpu` 为默认。
- 功耗约束：记录 perf/W 对比，`nvpmodel`/jetson_clocks 固频测。
