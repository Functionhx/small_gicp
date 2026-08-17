# small_gicp-cuda

**CUDA-exclusive point cloud registration for NVIDIA GPUs** — desktop dGPUs and Jetson modules.
[中文说明](README_zh.md)

GICP (scan-to-scan) and VGICP (scan-to-model) reimplemented GPU-native, with bit-level
fidelity to the CPU reference and single-frame latencies in the milliseconds.

## Results

| device | engine | CPU ref (upstream, serial) | this work (GPU) | speedup | GPU vs CPU accuracy |
|---|---|---|---|---|---|
| RTX 4070 + Ryzen 9950X | GICP | 43.9 ms | **6.0 ms** | 7.3x | APE/RPE diff <= 0.01% |
| RTX 4070 + Ryzen 9950X | VGICP | 34.4 ms | **5.6 ms** | 6.1x | <= 0.01% |
| Jetson Orin NX (MAXN) | GICP | 118.3 ms | **16.5 ms** | 7.2x | 1.2 cm over 100 frames |
| Jetson Orin NX (MAXN) | VGICP | 102.3 ms | **14.2 ms** | 7.2x | 0.2 cm over 100 frames |

Full KITTI-00 sequence (all 4541 frames, the complete 3.7 km loop): GICP 45.2 -> **6.0 ms**
(7.5x, 162 fps), VGICP 35.5 -> **5.7 ms** (6.2x, 170 fps); trajectories stay within 0.22%/1.45%
of upstream over the whole run, 5 reruns are bit-identical, GPU memory steady at 358 MiB.
Timing includes downsampling + covariance estimation + registration. Full methodology:
[BENCHMARK_GPU.md](BENCHMARK_GPU.md) (includes a fair comparison against NVIDIA cuPCL's cuICP:
1.5-1.65x faster at matched inputs).

Cross-architecture determinism: the GPU pipeline produces **bit-identical trajectories on
sm_87 and sm_89** (APE = 0.0000 m across devices).

## How this differs from the original small_gicp

| | small_gicp (upstream) | small_gicp-cuda (this repo) |
|---|---|---|
| Target | CPU (header-only, OpenMP/TBB optional) | **NVIDIA GPU required** (dGPU or Jetson), no CPU fallback |
| Spatial index | kd-tree (nanoflann-style) per cloud | **radix-sorted voxel buckets + O(1) hash** built by downsampling itself — one pass serves downsampling, covariance kNN, correspondence search, and the incremental voxel map; no kd-tree anywhere |
| Covariance kNN | kd-tree exact kNN | warp-cooperative expanding-shell search with a rigorous early-stop bound (exact), brute-force completion for sparse clouds |
| Correspondence | kd-tree NN per iteration | warp-cooperative batch NN: 5^3 window + adaptive expansion with per-voxel pruning |
| Precision | all double | fp32 storage/factor math + **fp64 warp reductions** and CPU double LM solve |
| Determinism | threaded CPU reductions are scheduling-dependent | fixed launch configs + ordered reductions: **run-to-run and cross-SM-architecture bit-identical** |
| Engines | ICP / Plane-ICP / GICP / VGICP via templates | GICP + VGICP (`sgc::GicpGpu`, `sgc::VgicpGpu`), LM constants ported line-by-line from upstream |
| Extras | PCL adapter, ROS bridge, Python bindings | none of those (yet) — the CPU reference stays in-tree as the parity baseline |
| Verification | upstream test suite | upstream kept unmodified + **20 kernel-parity tests** that build both implementations and assert agreement (downsampling, covariance, NN, H/b/e, voxel map, end-to-end) |

The upstream CPU implementation is kept unmodified in-tree (`include/small_gicp`, `src/`)
exactly because the parity suite needs it: this repo's accuracy claim is "identical to
upstream", tested kernel by kernel. The `master` branch tracks the upstream baseline.

## Quick start

```bash
# requires: CUDA 12.x toolkit, Eigen3, CMake >= 3.18
cmake -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=87]  # 87 = Jetson Orin
cmake --build build -j$(nproc)
ctest --test-dir build          # 20 kernel-parity tests (GPU required)
./build/cuda/odometry_gpu <velodyne_dir> --exec full-gpu --engine gicp
```

## Usage

```cpp
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/reg/gicp.hpp>

sgc::GpuCloud target = sgc::GpuCloud::from_host(target_points);   // std::vector<Eigen::Vector4f>
sgc::Downsampler downsampler;
downsampler.run(target, target_points.size(), 0.25);              // voxel buckets + hash index
sgc::estimate_covariances(target, 0.25f, 20);                     // exact kNN-20 covariance

sgc::GicpGpu reg;                                                 // LM constants = upstream
auto result = reg.align(target, source, init_T, 0.25f);
```

VGICP scan-to-model: `sgc::VoxelHashMap` (incremental Gaussian voxel map with LRU) +
`sgc::VgicpGpu::align(...)`. Benchmark harness with runtime-switchable policies:
`cuda/bench/odometry_gpu` (`--exec full-gpu|hybrid|cpu --engine gicp|vgicp --nn voxel3|voxel5|exact-bf`).

## Documentation

[BENCHMARK_GPU.md](BENCHMARK_GPU.md) results & methodology · [JETSON.md](JETSON.md) (cuda-jetson
branch) device deployment · [WORKLOG.md](WORKLOG.md) engineering log ·
[docs/superpowers/specs](docs/superpowers/specs/) design doc ·
[README_upstream.md](README_upstream.md) original CPU library README.

## Credits & license

Derivative of [koide3/small_gicp](https://github.com/koide3/small_gicp) by Kenji Koide (AIST),
MIT license inherited. If you use this work, please cite the upstream JOSS paper for the
algorithms and this repository for the CUDA implementation.
