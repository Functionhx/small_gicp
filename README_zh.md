# small_gicp-cuda（中文说明）

**CUDA 专属的点云配准库，面向所有 NVIDIA GPU**——桌面独显与 Jetson 嵌入式模块。
[English](README.md)

GICP（帧到帧）与 VGICP（帧到地图）的 GPU 原生重实现：与 CPU 参考实现保持逐位一致的
精度，单帧延迟毫秒级。

## 结果

| 设备 | 引擎 | CPU 基线（upstream 串行） | 本库（GPU） | 加速比 | GPU 与 CPU 精度差 |
|---|---|---|---|---|---|
| RTX 4070 + 锐龙 9950X | GICP | 43.9 ms | **6.0 ms** | 7.3× | APE/RPE 差 ≤ 0.01% |
| RTX 4070 + 锐龙 9950X | VGICP | 34.4 ms | **5.6 ms** | 6.1× | ≤ 0.01% |
| Jetson Orin NX（MAXN） | GICP | 118.3 ms | **16.5 ms** | 7.2× | 100 帧累计 1.2 cm |
| Jetson Orin NX（MAXN） | VGICP | 102.3 ms | **14.2 ms** | 7.2× | 100 帧累计 0.2 cm |

KITTI 里程计序列 00（官方数据 + 官方真值），100 帧；计时含降采样 + 协方差估计 + 配准
全流程。完整方法论见 [BENCHMARK_GPU.md](BENCHMARK_GPU.md)（含与 NVIDIA cuPCL cuICP
的公平对比：同等输入下快 1.6×）。

跨架构确定性：GPU 管线在 **sm_87 与 sm_89 上输出逐位相同的轨迹**（跨设备 APE = 0.0000 m）。

## 与原版 small_gicp 的区别

| | small_gicp（上游） | small_gicp-cuda（本仓库） |
|---|---|---|
| 目标平台 | CPU（header-only，可选 OpenMP/TBB） | **必须 NVIDIA GPU**（独显或 Jetson），无 CPU 回退 |
| 空间索引 | 每云建 kd-tree（nanoflann 风格） | **降采样自身的 radix 排序体素桶 + O(1) 哈希**——一趟排序同时服务降采样、协方差 kNN、对应点搜索与增量体素地图；全程无 kd-tree |
| 协方差 kNN | kd-tree 精确 kNN | warp 协作扩张壳搜索 + 严格早停界（精确），稀疏云用暴力补全 |
| 对应点搜索 | 每次迭代 kd-tree NN | warp 协作批量 NN：5³ 窗口 + 自适应扩张（逐体素剪枝） |
| 数值精度 | 全程 double | fp32 存储/因子计算 + **fp64 warp 归约** + CPU 端 double LM 求解 |
| 确定性 | 多线程 CPU 归约受调度影响 | 固定 launch 配置 + 有序归约：**逐次运行、跨 SM 架构逐位一致** |
| 引擎 | 模板化 ICP / Plane-ICP / GICP / VGICP | GICP + VGICP（`sgc::GicpGpu`、`sgc::VgicpGpu`），LM 常数逐行移植自上游 |
| 周边能力 | PCL 适配、ROS 桥、Python 绑定 | 暂无——CPU 参考实现保留在树内作为对拍基线 |
| 验证 | 上游测试套件 | 上游代码原样保留 + **20 个内核级对拍测试**（同时构建两种实现并逐项断言一致：降采样、协方差、NN、H/b/e、体素地图、端到端） |

上游 CPU 实现原样保留在树内（`include/small_gicp`、`src/`），正是因为对拍测试套件需要
它：本仓库的精度主张就是"与 upstream 一致"，并按内核逐项验证。`master` 分支跟踪上游基线。

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

VGICP 帧到地图：`sgc::VoxelHashMap`（带 LRU 的增量高斯体素地图）+ `sgc::VgicpGpu::align(...)`。
带运行时策略切换的基准工具：`cuda/bench/odometry_gpu`
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
