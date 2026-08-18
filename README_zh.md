# small_gicp-cuda（中文说明）

**GPU 加速的 GICP / VGICP，面向实时 LiDAR 配准。**

[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)
[![tested on sm_87 · sm_89](https://img.shields.io/badge/tested-sm__87%20%C2%B7%20sm__89-blueviolet.svg)](BENCHMARK_GPU.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[English](README.md)

<p align="center">
  <img src="docs/assets/hero.png" alt="GPU 上的真实 KITTI LiDAR 配准" width="896">
</p>

[small_gicp](https://github.com/koide3/small_gicp) 的 CUDA 原生重写：完整配准管线——降采样、
精确协方差 kNN、对应点搜索、LM 迭代——围绕 GPU 空间原语而非 CPU kd-tree 重建，并保持
upstream 的优化目标与精度。

**快 7.5×** · **162 FPS** · **轨迹偏差 ≤0.22%** · **GPU 显存 358 MiB** · **支持 Jetson Orin**

任意 CUDA 12 NVIDIA GPU 均可运行，桌面独显或 Jetson 模组。无 kd-tree。无 CPU 回退。
无精度折损。

## 性能

<p align="center">
  <img src="docs/assets/benchmark.png" alt="CPU 与 GPU 端到端延迟对比" width="896">
</p>

计时为**每帧端到端**：H2D 上传、降采样、协方差估计、对应点搜索、完整 LM 迭代至收敛——
没有任何环节被排除。完整 4541 帧序列同样成立：GICP 45.2 → **6.0 ms**（7.5×，162 fps），
VGICP 35.5 → **5.7 ms**（6.2×，170 fps）。同等输入下对比 NVIDIA cuPCL（cuICP）：
**快 1.5–1.65×**，且提供 cuPCL 没有的分布到分布 GICP/VGICP 目标函数
（[协议](BENCHMARK_GPU.md)）。

## 端到端里程计

<p align="center">
  <img src="docs/assets/kitti00_trajectory.png" alt="KITTI-00 全序列轨迹：真值 vs upstream CPU vs GPU" width="896">
</p>

这不是微基准。KITTI-00 完整序列，无回环检测的链式帧间配准：

| | |
|---|---|
| 序列 | KITTI-00，官方 S3 数据 + 真值——**4,541 帧**，3.7 km |
| GICP 轨迹 vs upstream CPU | 全程偏差 **0.22%** |
| VGICP 轨迹 vs upstream CPU | 全程偏差 **1.45%** |
| 确定性 | 5 次重跑 → 轨迹文件字节级相同 |
| 跨架构 | Orin（sm_87）≡ 桌面（sm_89），APE = 0.0000 m |
| 持续资源 | GPU 显存 358 MiB（LRU 封顶）、利用率均值 86% |

两个实现对真值的漂移方式一致（RPE(400) 2.908 vs 2.907 deg/km）——无回环的链式逐对
配准本来如此；GPU 复现紧跟原版，不改善也不劣化。

## 为什么上 GPU？

<p align="center">
  <img src="docs/assets/kdtree-vs-gpu.svg" alt="CPU kd-tree 管线 vs GPU 体素桶重设计" width="896">
</p>

kd-tree 对单线程 CPU 做延迟 bound 的查询接近最优，但对 GPU 是错误的形状：数千线程想要
对扁平结构的合并、无分支访存——而不是在树上发散地追指针。所以本项目**不是"把 CPU
代码翻译成 CUDA"**：空间索引、内存布局、邻域搜索与归约策略全部重设计，让管线在数据
结构层面适配 GPU 的执行模型。

## 核心思想

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="一次 radix sort 同时服务协方差 kNN、对应点 NN 与体素地图" width="896">
</p>

**一次排序，服务所有工作负载。** 降采样器产出的排序体素桶不是副产品——它*就是*协方差
估计、对应点搜索、VGICP 体素地图共同通过 O(1) 哈希探测读取的空间索引。

## 精度：没有捷径

速度不是用近似换的：

- **相同目标函数**——GICP 分布到分布因子、VGICP 体素地图因子，upstream 数学原封不动
- **精确协方差 kNN**——扩张壳上的可证明早停界；近似邻域试过并被否决（真实序列漂移放大 50×）
- **相同优化器**——LM 常数（λ₀=1e-3、×10、1e-3 m / 0.1°）逐行移植；6×6 求解在主机端以 double 完成
- **20 个内核对拍测试**同时构建两种实现并逐阶段断言一致：降采样、协方差、NN、H/b/e、体素地图、端到端
- **实测**：100 帧对 upstream APE/RPE ≤0.01%；4541 帧全程 0.22% / 1.45%

## small_gicp vs small_gicp-cuda

| | small_gicp | small_gicp-cuda |
|---|---|---|
| 执行 | CPU（可选 OpenMP/TBB） | NVIDIA GPU（CUDA 12），独显或 Jetson |
| 空间索引 | 每云一棵 kd-tree，每帧重建 | 排序体素桶 + O(1) 哈希，由降采样顺带构建 |
| 邻域搜索 | 逐点树遍历 | warp 协作 GPU 搜索，可证精确 |
| 数值精度 | 全程 double | fp32 存储/因子 + fp64 归约 + double 求解 |
| 确定性 | 受线程调度影响 | 相同输入 → 相同输出字节，跨运行跨架构 |
| 引擎 | ICP / Plane-ICP / GICP / VGICP | GICP + VGICP |
| 周边能力 | PCL 适配、ROS 桥、Python 绑定 | 暂无——upstream CPU 代码留在树内作对拍基线 |
| 目标硬件 | 任意 CPU | NVIDIA GPU / Jetson Orin |

upstream CPU 实现原样保留在树内（`include/small_gicp`、`src/`），因为对拍套件需要它：
本仓库的精度主张就是"与 upstream 一致"，逐内核验证。`master` 分支跟踪上游基线。

## 内部实现

- **`sgc::GpuCloud`**——`float4` 质心存点、9 个紧凑浮点存协方差，降采样后按体素 key 排序：后续每个内核读到的都是合并、空间连贯的内存。
- **体素化**——fast-floor + 3×21 位 key 打包进一个 `uint64`；CUB `DeviceRadixSort` 排序，exclusive scan 得桶边界。不存在独立的索引构建。
- **体素哈希**——2 的幂开放寻址 + splitmix64；每槽一个 `ulonglong2`（key + value 单次 16 B 加载），`~0ull` 标空。纯算术探测，无发散。
- **协方差 kNN**——每点一个 warp；lane 逐壳推进，候选经 `__shfl_down_sync` 锦标赛合并，距离界证明无改进即停；稀疏区域 warp 级暴力。协方差用 upstream 的特征分解 + 特征值替换模型（平面状 (1e-3, 1, 1)）。
- **对应点搜索**——每次 LM 迭代一个批量 NN 内核：每变换源点探测 5³ 体素窗口，逐体素剪枝扩张直到最近可证。当前变换在内核内完成。
- **归约与确定性**——每点 H/b/e 经 warp shuffle 以 fp32 累加；lane 0 把 fp64 partial 写入按发射 warp 固定的槽位；主机按序 double 求和。固定 launch 配置 → 每次运行、每个架构输出相同字节。

## 谁适合用？

需要配准压进 10 ms 的 LiDAR SLAM / LIO 管线 · Jetson 级边缘硬件上的自动驾驶与机器人
感知 · 被 CPU 邻域搜索卡住的三维重建 · 想看数据结构重设计案例的 CUDA 工程师。

> 不要加速那棵树。删掉它。

## 快速开始

```bash
git clone https://github.com/Functionhx/small_gicp.git && cd small_gicp
cmake -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # Jetson Orin 加 -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
ctest --test-dir build          # 20 个内核对拍测试（需要 GPU）
./build/cuda/odometry_gpu <velodyne目录> --exec full-gpu --engine gicp
```

依赖：CUDA 12.x 工具链、Eigen3、CMake ≥ 3.18。

## 用法

```cpp
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/reg/gicp.hpp>

sgc::GpuCloud target = sgc::GpuCloud::from_host(target_points);   // std::vector<Eigen::Vector4f>
sgc::Downsampler downsampler;
downsampler.run(target, target_points.size(), 0.25);              // 体素桶 + 哈希索引
sgc::estimate_covariances(target, 0.25f, 20);                     // 精确 kNN-20 协方差

sgc::GicpGpu reg;                                                 // LM 常数 = upstream
auto result = reg.align(target, source, init_T, 0.25f);
```

VGICP 帧到地图：`sgc::VoxelHashMap`（带 LRU 的增量高斯体素地图）+
`sgc::VgicpGpu::align(...)`。带运行时策略切换的基准工具：
`cuda/bench/odometry_gpu`
（`--exec full-gpu|hybrid|cpu --engine gicp|vgicp --nn voxel3|voxel5|exact-bf`）。

## Jetson

已在 Jetson Orin NX（JetPack 6、CUDA 12.2、sm_87）设备端构建并验证：20/20 对拍测试通过，
GICP 118.3 → **16.5 ms**，VGICP 102.3 → **14.2 ms**，轨迹与桌面 GPU 逐位一致。部署指南：
[JETSON.md](JETSON.md)（cuda-jetson 分支）。

## 文档

[BENCHMARK_GPU.md](BENCHMARK_GPU.md) 结果、方法论、消融、cuPCL 对比与复现命令（本页
图表脚本：`scripts/make_readme_figures.py`）· [WORKLOG.md](WORKLOG.md) 工程日志（含失败
方案）· [docs/superpowers/specs](docs/superpowers/specs/) 设计文档 ·
[README_upstream.md](README_upstream.md) 原 CPU 库 README。

## 致谢与许可

本仓库是 [koide3/small_gicp](https://github.com/koide3/small_gicp)（Kenji Koide，AIST，
MIT 许可）的衍生作品，许可随上游继承。使用本工作时请引用上游 JOSS 论文（算法）与本
仓库（CUDA 实现）。
