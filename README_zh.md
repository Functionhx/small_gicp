# small_gicp-cuda（中文说明）

**让点云配准跑满帧率，任意 NVIDIA GPU**——从桌面独显到 Jetson Orin 模组。
[English](README.md)

GICP 与 VGICP 的 GPU 从零重写，精度契约原封不动：输出轨迹与 CPU 原版逐帧吻合到
≤0.01%。

> 比 CPU 原版在完整 KITTI 环路上快 **7.5×** · 持续 **162 fps** · 重跑**逐位一致** ·
> 跨 GPU 架构确定性（sm_87 ≡ sm_89）· GPU 显存恒定 **358 MiB** · 比 NVIDIA 自家
> cuPCL cuICP 快 **1.5×**

无 kd-tree。无 CPU 回退。无精度折损。

## 一帧的完整旅程

12 万点 LiDAR 帧进，位姿出——降采样、精确 kNN-20 协方差估计、完整 LM 迭代全部计入
计时：

| | upstream（CPU） | 本库（GPU） | |
|---|---|---|---|
| RTX 4070 + 锐龙 9950X，GICP | 43.9 ms | **6.0 ms** | **7.3×** |
| RTX 4070 + 锐龙 9950X，VGICP | 34.4 ms | **5.6 ms** | **6.1×** |
| Jetson Orin NX，GICP | 118.3 ms | **16.5 ms** | **7.2×** |
| Jetson Orin NX，VGICP | 102.3 ms | **14.2 ms** | **7.2×** |

长序列同样站得住。KITTI-00 完整序列（4541 帧，3.7 km 环路，无回环检测的链式帧间
配准）：

| 引擎 | upstream CPU | 本库 | 加速比 | fps | 轨迹与 upstream 对比 |
|---|---|---|---|---|---|
| GICP  | 45.2 ms | **6.0 ms** | 7.5× | 162 | 4541 帧全程相差 0.22% |
| VGICP | 35.5 ms | **5.7 ms** | 6.2× | 170 | 4541 帧全程相差 1.45% |

## 核心思路：让 kd-tree 退休

upstream 的管线为每帧点云建一棵 kd-tree，协方差 kNN 查它——之后每次迭代的每个点的
对应点搜索再查它。本库直接删掉这个数据结构：降采样器产出的 radix 排序体素桶**本身
就是空间索引**，后续所有阶段都通过 O(1) 哈希探测读取它：

```
12万点 ─▶ 体素 key ─▶ 一次 radix sort ──┬─▶ 桶质心              （降采样）
                                        ├─▶ 扩张壳 kNN          （协方差，精确）
                                        ├─▶ 5³ 窗口 + 自适应 NN （对应点搜索）
                                        └─▶ O(1) 哈希插入/探测 （增量体素地图）
```

一次排序，全家受益。在此之上：

- **warp 协作精确搜索**。协方差 kNN 以严格的早停界逐壳扩张——结果是精确的，不做固定
  半径妥协；稀疏区域回退 warp 级暴力搜索。
- **能在 GPU 上活下来的数值方案**。带宽敏感处用 fp32 存储与因子计算，相消敏感处用
  fp64 warp shuffle 归约，最终 LM 在 CPU 上以 double 求解——LM 常数逐行移植自 upstream。
- **确定性即特性**。固定 launch 配置 + 有序归约，4541 帧序列连跑五次轨迹文件字节级
  相同，Orin 与桌面 GPU 输出**同样的字节**（跨设备 APE = 0.0000 m）。多线程 CPU 代码
  给不了这个承诺；这里可以。

## 与原版 small_gicp 的区别

| | small_gicp（上游） | small_gicp-cuda |
|---|---|---|
| 目标平台 | CPU，header-only（可选 OpenMP/TBB） | **必须 NVIDIA GPU**，独显或 Jetson |
| 空间索引 | 每云一棵 kd-tree | 排序体素桶 + O(1) 哈希——**全程无 kd-tree** |
| 协方差 kNN | kd-tree 精确 kNN | warp 协作扩张壳（精确，带早停界） |
| 对应点 | 每次迭代 kd-tree NN | warp 协作批量 NN + 逐体素剪枝 |
| 数值精度 | 全程 double | fp32 计算 + fp64 归约 + CPU double LM |
| 确定性 | 受线程调度影响 | 逐次运行、跨架构**逐位一致** |
| 引擎 | ICP / Plane-ICP / GICP / VGICP | GICP + VGICP（`sgc::GicpGpu`、`sgc::VgicpGpu`） |
| 周边能力 | PCL 适配、ROS 桥、Python 绑定 | 暂无——CPU 参考实现保留在树内作对拍基线 |
| 验证 | 上游测试套件 | 上游原样保留 + **20 个内核对拍测试**逐阶段断言两种实现一致 |

上游 CPU 实现保留在树内（`include/small_gicp`、`src/`）是有意为之：本仓库的精度主张
就是"与 upstream 一致"，对拍套件按内核逐项验证的正是这件事。`master` 分支跟踪上游
基线。

与 NVIDIA cuPCL（cuICP）的对比：同等输入下快 1.5–1.65×，且提供 cuPCL 没有的
分布到分布 GICP/VGICP 目标函数——完整对比协议见 [BENCHMARK_GPU.md](BENCHMARK_GPU.md)。

## 快速开始

```bash
# 依赖：CUDA 12.x 工具链、Eigen3、CMake >= 3.18
cmake -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=87]  # 87 = Jetson Orin
cmake --build build -j$(nproc)
ctest --test-dir build          # 20 个内核对拍测试（需要 GPU）
./build/cuda/odometry_gpu <velodyne目录> --exec full-gpu --engine gicp
```

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

sgc::GicpGpu reg;                                                 // LM 常数与 upstream 一致
auto result = reg.align(target, source, init_T, 0.25f);
```

VGICP 帧到地图：`sgc::VoxelHashMap`（带 LRU 的增量高斯体素地图）+
`sgc::VgicpGpu::align(...)`。带运行时策略切换的基准工具：
`cuda/bench/odometry_gpu`
（`--exec full-gpu|hybrid|cpu --engine gicp|vgicp --nn voxel3|voxel5|exact-bf`）。

## 文档

[BENCHMARK_GPU.md](BENCHMARK_GPU.md) 结果与方法论 · [JETSON.md](JETSON.md)（cuda-jetson
分支）设备部署 · [WORKLOG.md](WORKLOG.md) 工程日志 ·
[docs/superpowers/specs](docs/superpowers/specs/) 设计文档 ·
[README_upstream.md](README_upstream.md) 原 CPU 库 README。

## 致谢与许可

本仓库是 [koide3/small_gicp](https://github.com/koide3/small_gicp)（Kenji Koide，AIST，
MIT 许可）的衍生作品，许可随上游继承。使用本工作时请引用上游 JOSS 论文（算法）与本
仓库（CUDA 实现）。
