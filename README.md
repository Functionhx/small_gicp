# small_gicp-cuda

**Point cloud registration at frame rate, on any NVIDIA GPU** — from a desktop dGPU to a
Jetson Orin module. [中文说明](README_zh.md)

GICP and VGICP, rewritten for the GPU from the kernel up — with the accuracy contract kept
intact: the output trajectory matches the CPU original to ≤0.01%, on every frame, every run.

> **7.5×** faster than the CPU original over a full KITTI loop · **162 fps** sustained ·
> **bit-identical** reruns · deterministic across GPU architectures (sm_87 ≡ sm_89) ·
> **358 MiB** steady GPU memory · **1.5×** faster than NVIDIA's own cuPCL cuICP

No kd-trees. No CPU fallback. No accuracy trade-off.

## One frame, end to end

A 120k-point LiDAR frame goes in, a pose comes out — downsampling, exact kNN-20 covariance
estimation, and the full LM loop all included in the timing:

| | upstream (CPU) | this work (GPU) | |
|---|---|---|---|
| RTX 4070 + Ryzen 9950X, GICP | 43.9 ms | **6.0 ms** | **7.3×** |
| RTX 4070 + Ryzen 9950X, VGICP | 34.4 ms | **5.6 ms** | **6.1×** |
| Jetson Orin NX, GICP | 118.3 ms | **16.5 ms** | **7.2×** |
| Jetson Orin NX, VGICP | 102.3 ms | **14.2 ms** | **7.2×** |

And it holds up over distance. The complete KITTI-00 sequence (4541 frames, the full 3.7 km
loop, chained frame-to-frame with no loop closure):

| engine | upstream CPU | this work | speedup | fps | trajectory vs upstream |
|---|---|---|---|---|---|
| GICP  | 45.2 ms | **6.0 ms** | 7.5× | 162 | within 0.22% over 4541 frames |
| VGICP | 35.5 ms | **5.7 ms** | 6.2× | 170 | within 1.45% over 4541 frames |

## The idea: retire the kd-tree

Upstream's pipeline builds a kd-tree per cloud, then queries it for covariance kNN — and
again for every correspondence, of every point, in every LM iteration. This library deletes
that data structure. The downsampler's radix-sorted voxel buckets *are* the spatial index,
and every later stage reads them through O(1) hash probes:

```
120k points ─▶ voxel keys ─▶ one radix sort ──┬─▶ bucket centroids          (downsampling)
                                              ├─▶ expanding-shell kNN      (covariances, exact)
                                              ├─▶ 5³-window + adaptive NN  (correspondences)
                                              └─▶ O(1) hash insert/probe   (incremental voxel map)
```

One sort pays for everything. On top of it:

- **Warp-cooperative exact search.** Covariance kNN expands voxel shells with a rigorous
  early-stop bound — exact results, no fixed-radius compromise. Sparse regions fall back to
  warp-level brute force.
- **Numerics that survive the GPU.** fp32 storage and factor math where bandwidth matters,
  fp64 warp-shuffle reductions where cancellation hurts, and the final LM solve in double on
  the CPU — LM constants ported line-by-line from upstream.
- **Determinism as a feature.** Fixed launch configs and ordered reductions mean five reruns
  of a 4541-frame sequence produce byte-identical trajectory files, and an Orin and a
  desktop GPU produce the *same* bytes (cross-device APE = 0.0000 m). Threaded CPU code
  can't promise that; this can.

## How it differs from the original small_gicp

| | small_gicp (upstream) | small_gicp-cuda |
|---|---|---|
| Target | CPU, header-only (OpenMP/TBB optional) | **NVIDIA GPU required**, dGPU or Jetson |
| Spatial index | kd-tree per cloud | sorted voxel buckets + O(1) hash — **no kd-tree anywhere** |
| Covariance kNN | kd-tree exact kNN | warp-cooperative expanding shells (exact, early-stop bound) |
| Correspondences | kd-tree NN per iteration | warp-cooperative batch NN with per-voxel pruning |
| Precision | all double | fp32 math + fp64 reductions + CPU double LM solve |
| Determinism | thread-scheduling dependent | run-to-run and cross-architecture **bit-identical** |
| Engines | ICP / Plane-ICP / GICP / VGICP | GICP + VGICP (`sgc::GicpGpu`, `sgc::VgicpGpu`) |
| Extras | PCL adapter, ROS bridge, Python bindings | not yet — CPU reference kept in-tree as parity baseline |
| Verification | upstream test suite | upstream unmodified + **20 kernel-parity tests** asserting both implementations agree, stage by stage |

The upstream CPU implementation stays in-tree (`include/small_gicp`, `src/`) on purpose: this
repo's accuracy claim is "identical to upstream", and the parity suite tests exactly that,
kernel by kernel. `master` tracks the upstream baseline.

For how this compares with NVIDIA's cuPCL (cuICP): 1.5–1.65× faster at matched inputs, while
also providing the distribution-to-distribution GICP/VGICP objectives cuPCL doesn't have —
full protocol in [BENCHMARK_GPU.md](BENCHMARK_GPU.md).

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
