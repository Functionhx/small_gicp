# small_gicp-cuda（中文说明）

**GPU 加速的 GICP / VGICP，面向实时 LiDAR 配准。**
[English](README.md)

> [small_gicp](https://github.com/koide3/small_gicp) 的 CUDA 原生重写——把点云配准从 CPU
> kd-tree 迁移到 GPU 体素原语，同时保持上游的优化目标与精度。

**快 7.5×** · **162 FPS** · **轨迹偏差 ≤0.22%** · **GPU 显存 358 MiB** · **支持 Jetson Orin**

任意 CUDA 12 NVIDIA GPU 均可运行——桌面独显或 Jetson 模组。无 CPU 回退。

## 改了什么？

- **CUDA 原生 GICP / VGICP**——完整管线（降采样 → 协方差 → LM 迭代）全部在 GPU 上
- **无 kd-tree**——一次 radix 排序的体素结构替换管线中的每一棵树
- **GPU 体素哈希**——O(1) 开放寻址探测支撑邻域搜索与增量体素地图
- **精确协方差 kNN**——warp 协作搜索 + 严格早停界，不做固定半径近似
- **确定性执行**——相同输入 → 相同输出字节，逐次运行、跨 GPU 架构均成立
- **与 upstream 逐行对拍**——相同目标函数、相同 LM 常数，20 个内核级测试验证

## 为什么会有这个项目

small_gicp 是优秀的 CPU 库。但它的管线围绕一种 GPU 用不好的数据结构构建：

```text
点云
  ↓
KD-tree 构建                 ← 串行、指针追逐、分支密集
  ↓
协方差 kNN                    ← 每点一次树遍历
  ↓
对应点 NN                     ← 每点、每次 LM 迭代都要树遍历
  ↓
LM 优化
```

kd-tree 对单线程 CPU 做延迟 bound 的查询接近最优；但对 GPU 是错误的形状：数千线程
想要对扁平结构的合并、无分支访存——而不是在树上发散地追指针。

所以本项目**不是"把 CPU 代码翻译成 CUDA"**，而是重新设计空间索引、内存布局、邻域
搜索与归约策略，让 GICP 管线在**数据结构层面**适配 GPU 的执行模型。

## 核心思想：替换掉 kd-tree

```text
        CPU small_gicp                    small_gicp-cuda
        ──────────────                    ───────────────

    点云                                点云
      ↓                                   ↓
    KD-tree 构建                        体素化
      ↓                                   ↓
    ├── kNN → 协方差          一次 radix sort → 体素桶 + 哈希
      ↓                                   ↓
    ├── NN → 对应点          ┌── 协方差 kNN（扩张壳）
      ↓                      ├── 对应点 NN（窗口 + 自适应）
    LM 优化                   └── 增量体素地图（VGICP）
                                           ↓
                                    warp 协作内核
                                           ↓
                                       GICP / VGICP
```

**一次排序，服务所有工作负载。** 降采样器产出的排序体素桶不是副产品——它*就是*
空间索引：协方差估计、对应点搜索、VGICP 体素地图全部通过 O(1) 哈希探测读取它。

## 性能

### 单帧

```text
12 万 LiDAR 点，KITTI-00

upstream CPU    ████████████████████████████████████   43.9 ms
small_gicp-cuda ██████                                 6.0 ms     7.3×
```

| 设备 | 引擎 | upstream CPU | 本库 | 加速比 | fps |
|---|---|---:|---:|---:|---:|
| RTX 4070 + 锐龙 9950X | GICP | 43.9 ms | **6.0 ms** | 7.3× | 136 |
| RTX 4070 + 锐龙 9950X | VGICP | 34.4 ms | **5.6 ms** | 6.1× | 143 |
| Jetson Orin NX（MAXN） | GICP | 118.3 ms | **16.5 ms** | 7.2× | 56 |
| Jetson Orin NX（MAXN） | VGICP | 102.3 ms | **14.2 ms** | 7.2× | 63 |

计时为**每帧端到端**：H2D 上传、降采样、协方差估计、对应点搜索、完整 LM 迭代至收敛。
没有任何环节被排除在外。

### 完整序列——不是微基准

```text
KITTI-00 · 4541 帧 · 3.7 km · 帧间里程计 · 无回环检测
```

| 引擎 | upstream CPU | 本库 | 加速比 | fps | 与 upstream 轨迹差 |
|---|---:|---:|---:|---:|---|
| GICP | 45.2 ms | **6.0 ms** | 7.5× | 162 | 4541 帧全程 0.22% |
| VGICP | 35.5 ms | **5.7 ms** | 6.2× | 170 | 4541 帧全程 1.45% |

GPU 实现不是只在单帧上快——在完整 3.7 km 序列上，链式轨迹与 upstream CPU 参考保持在
0.22%（GICP）/ 1.45%（VGICP）以内；同时两者对真值的漂移方式一致（RPE(400) 2.908 vs
2.907 deg/km——无回环的链式逐对配准在 3.7 km 路线上必然发散，upstream 与本库以相同
方式发散）。

整趟运行的持续资源占用：GPU 显存 **358 MiB**（恒定，LRU 封顶）、利用率均值 **86%**。

### 对比 NVIDIA cuPCL（cuICP）

同等输入（相同帧、降采样计入计时）下，cuPCL 最优配置 100 帧 9.44 ms/帧 vs 本库
**6.01 ms**；全序列 4541 帧 8.85 vs **6.00 ms**（p50）——**快 1.5–1.65×**，且提供
cuPCL 没有的分布到分布 GICP/VGICP 目标函数。完整协议与诚实的注意事项见
[BENCHMARK_GPU.md](BENCHMARK_GPU.md)。

## 精度

**没有精度捷径。** 速度不是用近似换的：

- **相同目标函数**。GICP 分布到分布因子、VGICP 体素地图因子——upstream 的数学，原封不动。
- **精确协方差 kNN**。邻居集合是精确 kNN-20，不是固定半径集合：warp 协作扩张壳 +
  可证明的早停界；稀疏区域暴力补全。（近似邻域试过并被否决——真实序列上漂移放大 50×。）
- **相同优化器行为**。LM 常数（λ₀=1e-3、×10 调度、1e-3 m / 0.1° 容差）逐行移植自
  upstream；6×6 求解在主机端以双精度完成。
- **逐内核验证**。20 个对拍测试同时构建两种实现并逐阶段断言一致：降采样、协方差、
  NN、H/b/e 累加、体素地图、端到端。
- **实测轨迹偏差**：100 帧对 upstream APE/RPE ≤0.01%；全序列 4541 帧 0.22% / 1.45%。
- **确定性**。全序列连跑五次，轨迹文件字节级相同；同一运行在 sm_87（Orin）与
  sm_89（桌面）上一致到 APE = 0.0000 m。多线程 CPU 归约给不了这个承诺。

## 内部实现

**1. GPU 点云表示**。`sgc::GpuCloud` 以 `float4` 质心存点（单次 16 B 加载）、9 个
紧凑浮点存协方差，降采样后按体素 key 排序布局——后续每个内核读到的都是空间连贯、
合并访存的内存。

**2. 体素化**。每点生成体素 key：fast-floor + 3×21 位打包进一个 `uint64`。CUB
`DeviceRadixSort` 按 key 排序，桶边界由 exclusive scan 得出。排序桶既是降采样结果
*也是*空间索引——没有独立的索引构建。

**3. 体素哈希**。2 的幂大小的开放寻址表，splitmix64 哈希。每槽一个 `ulonglong2`
（key + value 单次 16 B 加载），`~0ull` 标空。探测是纯算术——无树、无再平衡、无发散。

**4. 协方差估计**。每点一个 warp：lane 逐体素跨扩张壳推进，候选经 `__shfl_down_sync`
锦标赛合并；当严格的距离界证明未探索体素不可能改进 kNN 集合时壳循环停止。共享内存
存每 warp 候选列表；稀疏区域回退 warp 级暴力。协方差使用与 upstream 相同的特征分解
+ 特征值替换模型（平面状 (1e-3, 1, 1)）。

**5. 对应点搜索**。每次 LM 迭代一个批量 NN 内核：对每个变换后的源点在目标哈希中探测
5³ 体素窗口，再以逐体素剪枝自适应扩张直到结果可证明为最近。查询变换在内核内完成——
无往返。

**6. 归约与确定性**。每点 H/b/e 因子以 fp32 经 warp shuffle 累加；lane 0 把 fp64
partial 写入按发射 warp 固定的槽位；主机按顺序以 double 求和。固定 launch 配置 +
有序归约 = 每次运行、每个 GPU 架构输出相同字节。

## small_gicp vs small_gicp-cuda

| | small_gicp | small_gicp-cuda |
|---|---|---|
| 执行 | CPU（可选 OpenMP/TBB） | NVIDIA GPU（CUDA 12），独显或 Jetson |
| 空间索引 | 每云一棵 kd-tree，每帧重建 | 排序体素桶 + O(1) 哈希，由降采样顺带构建 |
| 邻域搜索 | 逐点树遍历 | warp 协作 GPU 搜索，可证精确 |
| 协方差估计 | CPU kd-tree kNN | GPU warp 协作扩张壳 kNN |
| 配准循环 | CPU 因子 + CPU LM | GPU 因子 + 有序 fp64 归约 + 主机 double LM |
| 数值精度 | 全程 double | fp32 存储/因子 + fp64 归约 + double 求解 |
| 确定性 | 受线程调度影响 | 相同输入 → 相同输出字节，跨运行跨架构 |
| 引擎 | ICP / Plane-ICP / GICP / VGICP | GICP + VGICP |
| 周边能力 | PCL 适配、ROS 桥、Python 绑定 | 暂无（upstream CPU 代码留在树内作对拍基线） |
| 目标硬件 | 任意 CPU | NVIDIA GPU / Jetson Orin |

upstream CPU 实现原样保留在树内（`include/small_gicp`、`src/`），因为对拍套件需要它：
本仓库的精度主张就是"与 upstream 一致"，逐内核验证。`master` 分支跟踪上游基线。

## 谁适合用？

- 需要扫描到扫描 / 扫描到地图配准压进 10 ms 以内的 LiDAR SLAM / LIO 管线
- 在 Jetson 级边缘硬件上做自动驾驶与机器人感知
- 被 CPU 邻域搜索卡住脖子的三维重建与建图工作负载
- 想找数据结构重设计案例的 CUDA 性能工程师

> 如果你的配准管线把毫秒花在等 CPU kd-tree 查询上，这个项目就是为你准备的。

## 设计哲学

> **不要加速那棵树。删掉它。**

"GPU 加速"一个 CPU 库的常见路径：

```text
CPU 算法 → 把每个操作 CUDA 化 → 跑分 → 调优
```

本项目走了另一条路：

```text
CPU 算法
    ↓
找出 GPU 用不了的结构（kd-tree）
    ↓
围绕 GPU 内存模型重设计数据结构
    ↓
围绕 warp 执行重设计搜索与归约
    ↓
重建管线，然后证明数值上什么都没变
```

每个阶段都与它所替换的 upstream CPU 实现对拍验证——对拍测试就是契约。

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

已在 Jetson Orin NX（JetPack 6、CUDA 12.2、sm_87）上设备端构建并验证：20/20 对拍测试
通过，GICP 118.3 → 16.5 ms，VGICP 102.3 → 14.2 ms，轨迹与桌面 GPU 逐位一致
（sm_87 ↔ sm_89 APE = 0.0000 m）。部署指南：[JETSON.md](JETSON.md)（cuda-jetson 分支）。

## 基准方法论

上述所有数字均来自 KITTI 官方里程计序列 00（avg-kitti S3 数据 + 官方真值）。协议细节、
消融（NN 策略、执行模式）、cuPCL 对比协议与复现命令见
**[BENCHMARK_GPU.md](BENCHMARK_GPU.md)**。含失败方案的工程日志见 [WORKLOG.md](WORKLOG.md)。

## 文档

[BENCHMARK_GPU.md](BENCHMARK_GPU.md) 结果与方法论 · [JETSON.md](JETSON.md) 设备部署 ·
[WORKLOG.md](WORKLOG.md) 工程日志 · [docs/superpowers/specs](docs/superpowers/specs/)
设计文档 · [README_upstream.md](README_upstream.md) 原 CPU 库 README。

## 致谢与许可

本仓库是 [koide3/small_gicp](https://github.com/koide3/small_gicp)（Kenji Koide，AIST，
MIT 许可）的衍生作品，许可随上游继承。使用本工作时请引用上游 JOSS 论文（算法）与本
仓库（CUDA 实现）。
