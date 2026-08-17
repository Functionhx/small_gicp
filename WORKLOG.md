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
