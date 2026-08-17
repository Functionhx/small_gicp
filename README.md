# small_gicp-cuda

**CUDA-exclusive point cloud registration for NVIDIA GPUs** — desktop dGPUs and Jetson.
GICP (scan-to-scan) and VGICP (scan-to-model) reimplemented GPU-native, with bit-level
fidelity to the CPU reference and single-frame latencies in the milliseconds.

- **No CPU fallback.** An NVIDIA GPU (or Jetson module) is required.
- **Accuracy parity by construction**: every kernel is tested against the CPU reference
  implementation (kept in-tree under `include/small_gicp`); on official KITTI odometry data
  the full pipeline reproduces the CPU result to **<=0.01% APE/RPE**.
- **Speed**: 7.3x (GICP) / 6.1x (VGICP) over the single-thread CPU reference on an RTX 4070
  (6.0 / 5.6 msec per 120k-point frame end-to-end, including downsampling and covariance
  estimation); the gap widens on Jetson-class CPUs. See [BENCHMARK_GPU.md](BENCHMARK_GPU.md)
  for the full comparison, including a fair match-up against NVIDIA cuPCL's cuICP.
- **Deterministic**: fixed launch configurations and fp64 reductions give run-to-run
  bit-identical results — a property the CPU reference itself does not guarantee under
  thread scheduling.

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
downsampler.run(target, target_points.size(), 0.25);              // sorted voxel buckets + hash index
sgc::estimate_covariances(target, 0.25f, 20);                     // exact kNN-20 covariance (GPU)

sgc::GicpGpu reg;                                                 // LM constants = upstream
auto result = reg.align(target, source, init_T, 0.25f);           // RegistrationResult-equivalent
```

VGICP scan-to-model: `sgc::VoxelHashMap` (incremental Gaussian voxel map, LRU) +
`sgc::VgicpGpu::align(...)`. Benchmark harness with runtime-switchable policies:
`cuda/bench/odometry_gpu` (`--exec full-gpu|hybrid|cpu --engine gicp|vgicp --nn voxel3|voxel5|exact-bf`).

## Architecture (one paragraph)

Downsampling's radix-sorted voxel buckets double as the spatial index — no kd-trees anywhere.
O(1) hash probes power the covariance kNN (warp-cooperative shells with a rigorous early-stop
bound), correspondence search (adaptive window + pruned sphere expansion), and the incremental
voxel map. Per-point factor math runs in fp32, warp-shuffle reductions promote to fp64 with
fixed slot assignment, and the 6x6 LM solve runs on the CPU in double precision.

## Documentation

[BENCHMARK_GPU.md](BENCHMARK_GPU.md) results & methodology · [JETSON.md](JETSON.md) (cuda-jetson branch)
device deployment · [WORKLOG.md](WORKLOG.md) engineering log ·
[docs/superpowers/specs](docs/superpowers/specs/) design doc ·
[README_upstream.md](README_upstream.md) original CPU library README.

## Relation to upstream small_gicp

This repository is a CUDA-exclusive derivative of [koide3/small_gicp](https://github.com/koide3/small_gicp)
(MIT). The upstream CPU implementation is kept unmodified in-tree (`include/small_gicp`,
`src/`) as the accuracy reference: the parity test suite builds both and asserts agreement
kernel by kernel. The `master` branch tracks the upstream baseline this work started from.

## License

MIT (inherited from upstream small_gicp; bundled nanoflann/Sophus licenses apply as in upstream).
