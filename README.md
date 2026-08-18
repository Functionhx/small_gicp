# small_gicp-cuda

**GPU-accelerated GICP / VGICP for real-time LiDAR registration.**
[中文说明](README_zh.md)

> A CUDA-native rewrite of [small_gicp](https://github.com/koide3/small_gicp) — moving point
> cloud registration from CPU kd-trees to GPU voxel primitives, while preserving the upstream
> optimization objective and accuracy.

**7.5× faster** · **162 FPS** · **≤0.22% trajectory deviation** · **358 MiB GPU memory** · **Jetson Orin support**

Runs on any CUDA 12 NVIDIA GPU — desktop dGPU or Jetson module. No CPU fallback.

## What changed?

- **CUDA-native GICP / VGICP** — the full pipeline (downsampling → covariance → LM iterations) runs on the GPU
- **No kd-tree** — one radix-sorted voxel structure replaces every tree in the pipeline
- **GPU voxel hashing** — O(1) open-addressing probes power neighbor search and the incremental voxel map
- **Exact covariance kNN** — warp-cooperative search with a rigorous early-stop bound, no fixed-radius approximation
- **Deterministic execution** — same input → same output bytes, run to run and across GPU architectures
- **Line-for-line parity with upstream** — same objective, same LM constants, verified by 20 kernel-level tests

## Why this exists

small_gicp is an excellent CPU library. But its pipeline is built around a data structure the
GPU cannot use well:

```text
Point cloud
    ↓
KD-tree construction          ← serial, pointer-chasing, branch-heavy
    ↓
kNN for covariances           ← tree traversal per point
    ↓
NN per correspondence          ← tree traversal per point, per LM iteration
    ↓
LM optimization
```

A kd-tree is close to optimal for a single CPU thread doing latency-bound queries. It is the
wrong shape for a GPU: thousands of threads want coalesced, branch-free access to a flat
structure — not divergent pointer chasing down a tree.

So this project is not "CUDA-translate the CPU code". It redesigns the spatial indexing, the
memory layout, the neighbor search, and the reduction strategy so the GICP pipeline fits the
GPU's execution model at the data-structure level.

## The key idea: replace the kd-tree

```text
          CPU small_gicp                      small_gicp-cuda
          ──────────────                      ───────────────

    Point cloud                              Point cloud
        ↓                                        ↓
    KD-tree build                            Voxelization
        ↓                                        ↓
    ├── kNN → covariances          One radix sort → voxel buckets + hash
        ↓                                        ↓
    ├── NN → correspondences     ┌── covariance kNN (expanding shells)
        ↓                        ├── correspondence NN (windowed + adaptive)
    LM optimization              └── incremental voxel map (VGICP)
                                     ↓
                              Warp-cooperative kernels
                                     ↓
                                 GICP / VGICP
```

**One sort, every workload.** The downsampler's sorted voxel buckets are not a byproduct —
they *are* the spatial index that covariance estimation, correspondence search, and the
VGICP voxel map all read through O(1) hash probes.

## Performance

### One frame

```text
120K LiDAR points, KITTI-00

upstream CPU   ████████████████████████████████████   43.9 ms
small_gicp-cuda ██████                                 6.0 ms     7.3×
```

| device | engine | upstream CPU | this work | speedup | fps |
|---|---|---:|---:|---:|---:|
| RTX 4070 + Ryzen 9950X | GICP | 43.9 ms | **6.0 ms** | 7.3× | 136 |
| RTX 4070 + Ryzen 9950X | VGICP | 34.4 ms | **5.6 ms** | 6.1× | 143 |
| Jetson Orin NX (MAXN) | GICP | 118.3 ms | **16.5 ms** | 7.2× | 56 |
| Jetson Orin NX (MAXN) | VGICP | 102.3 ms | **14.2 ms** | 7.2× | 63 |

Timing is **end-to-end per frame**: H2D upload, downsampling, covariance estimation,
correspondence search, and the full LM loop to convergence. Nothing is excluded.

### Full sequence — not a microbenchmark

```text
KITTI-00 · 4541 frames · 3.7 km · frame-to-frame odometry · no loop closure
```

| engine | upstream CPU | this work | speedup | fps | trajectory vs upstream |
|---|---:|---:|---:|---:|---|
| GICP | 45.2 ms | **6.0 ms** | 7.5× | 162 | 0.22% over 4541 frames |
| VGICP | 35.5 ms | **5.7 ms** | 6.2× | 170 | 1.45% over 4541 frames |

The GPU implementation is not merely faster on one frame — over the entire 3.7 km sequence,
the chained trajectory stays within 0.22% (GICP) / 1.45% (VGICP) of the upstream CPU
reference, while both drift identically against ground truth (RPE(400) 2.908 vs 2.907
deg/km — chained pairwise registration without loop closure diverges on a 3.7 km route;
upstream and this repo diverge the same way).

Sustained resources over a full run: **358 MiB** GPU memory (constant, LRU-bounded),
**86%** mean GPU utilization.

### Against NVIDIA cuPCL (cuICP)

At matched inputs (same frames, downsampling inside the timed section), cuPCL's best
configuration runs 9.44 ms/frame vs **6.01 ms** here on 100 frames, and 8.85 vs **6.00 ms**
(p50) over the full 4541-frame sequence — **1.5–1.65× faster**, while also providing the
distribution-to-distribution GICP/VGICP objectives cuPCL does not have. Full protocol and
honest caveats: [BENCHMARK_GPU.md](BENCHMARK_GPU.md).

## Accuracy

**No accuracy shortcut.** Speed was not bought with approximation:

- **Same objective.** GICP distribution-to-distribution factors, VGICP voxel-map factors — the upstream math, unchanged.
- **Exact covariance kNN.** Neighbor sets are exact kNN-20, not fixed-radius sets: warp-cooperative expanding shells with a provable early-stop bound; brute-force completion where clouds are sparse. (Approximate neighborhoods were tried and rejected — they amplified drift 50× on real sequences.)
- **Same optimizer behavior.** LM constants (λ₀=1e-3, ×10 schedule, 1e-3 m / 0.1° tolerances) ported line-by-line from upstream; the 6×6 solve runs in double precision on the host.
- **Verified kernel by kernel.** 20 parity tests build both implementations and assert agreement stage by stage: downsampling, covariances, NN, H/b/e accumulations, voxel map, end-to-end.
- **Measured trajectory deviation**: ≤0.01% APE/RPE vs upstream on 100 frames; 0.22% / 1.45% over the full 4541-frame sequence.
- **Deterministic.** Five full-sequence reruns produce byte-identical trajectory files; the same run on sm_87 (Orin) and sm_89 (desktop) agrees to APE = 0.0000 m. Threaded CPU reductions cannot promise this.

## Under the hood

**1. GPU point cloud representation.** `sgc::GpuCloud` holds points as `float4` centroids
(one 16 B load) and covariances as 9 packed floats, both laid out in voxel-key order after
downsampling — every later kernel reads spatially coherent, coalesced memory.

**2. Voxelization.** Each point gets a voxel key: fast-floor + 3×21-bit packing into one
`uint64`. A CUB `DeviceRadixSort` orders points by key; bucket boundaries come from an
exclusive scan. The sorted buckets are the downsampling result *and* the spatial index —
there is no separate index build.

**3. Voxel hashing.** An open-addressing table sized to a power of two, keyed by
splitmix64. Each slot is one `ulonglong2` (key + value in a single 16 B load), empty marked
by `~0ull`. Probes are pure arithmetic — no tree, no rebalancing, no divergence.

**4. Covariance estimation.** One warp per point: lanes stride over the voxels of each
expanding shell, candidates merge through `__shfl_down_sync` tournaments, and the shell loop
stops when a rigorous distance bound proves no unexplored voxel can improve the kNN set.
Shared memory holds the per-warp candidate lists; sparse regions fall back to warp-level
brute force. Covariances use the same eigendecomposition + eigenvalue replacement model as
upstream (plane-like (1e-3, 1, 1)).

**5. Correspondence search.** Per LM iteration, a batch NN kernel probes the target's hash
for a 5³ voxel window around each transformed source point, then adaptively expands with
per-voxel pruning until the result is provably the nearest. Queries apply the current
transform inside the kernel — no round trips.

**6. Reductions and determinism.** Per-point H/b/e factors accumulate in fp32 through warp
shuffles; lane 0 writes fp64 partials to a fixed slot per launched warp; the host sums the
partials in order, in double. Fixed launch configurations + ordered reductions = the same
output bytes every run, on every GPU architecture.

## small_gicp vs small_gicp-cuda

| | small_gicp | small_gicp-cuda |
|---|---|---|
| Execution | CPU (OpenMP/TBB optional) | NVIDIA GPU (CUDA 12), dGPU or Jetson |
| Spatial index | kd-tree per cloud, rebuilt per frame | sorted voxel buckets + O(1) hash, built by downsampling |
| Neighbor search | per-point tree traversal | warp-cooperative GPU search, provably exact |
| Covariance estimation | kd-tree kNN on CPU | warp-coop expanding-shell kNN on GPU |
| Registration loop | CPU factors + CPU LM | GPU factors + ordered fp64 reduction + host double LM |
| Precision | double throughout | fp32 storage/factor math + fp64 reductions + double solve |
| Determinism | thread-scheduling dependent | same input → same output bytes, across runs and architectures |
| Engines | ICP / Plane-ICP / GICP / VGICP | GICP + VGICP |
| Extras | PCL adapter, ROS bridge, Python bindings | none yet (upstream CPU code stays in-tree as the parity baseline) |
| Target hardware | any CPU | NVIDIA GPU / Jetson Orin |

The upstream CPU implementation is kept unmodified in-tree (`include/small_gicp`, `src/`)
because the parity suite needs it: this repo's accuracy claim is "identical to upstream",
tested kernel by kernel. `master` tracks the upstream baseline.

## Who is this for?

- LiDAR SLAM / LIO pipelines that need scan-to-scan or scan-to-map registration under 10 ms
- Autonomous driving and robotics perception on Jetson-class edge hardware
- 3D reconstruction and mapping workloads bottlenecked on CPU neighbor search
- CUDA performance engineers looking for a case study in data-structure redesign

> If your registration pipeline spends its milliseconds waiting on CPU kd-tree queries, this project is for you.

## Design philosophy

> **Don't accelerate the tree. Remove the tree.**

The common approach to "GPU-accelerating" a CPU library:

```text
CPU algorithm → CUDA-ify each operation → benchmark → tune
```

This project took a different path:

```text
CPU algorithm
    ↓
Find the structure the GPU can't use (the kd-tree)
    ↓
Redesign the data structure around GPU memory model
    ↓
Redesign search and reduction around warp execution
    ↓
Rebuild the pipeline, then prove nothing changed numerically
```

Every stage is validated against the upstream CPU implementation it replaces — the parity
tests are the contract.

## Quick start

```bash
git clone https://github.com/Functionhx/small_gicp.git && cd small_gicp
cmake -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # add -DCMAKE_CUDA_ARCHITECTURES=87 for Jetson Orin
cmake --build build -j$(nproc)
ctest --test-dir build          # 20 kernel-parity tests (requires a GPU)
./build/cuda/odometry_gpu <velodyne_dir> --exec full-gpu --engine gicp
```

Requires: CUDA 12.x toolkit, Eigen3, CMake ≥ 3.18.

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
`cuda/bench/odometry_gpu`
(`--exec full-gpu|hybrid|cpu --engine gicp|vgicp --nn voxel3|voxel5|exact-bf`).

## Jetson

Built and validated on-device on Jetson Orin NX (JetPack 6, CUDA 12.2, sm_87): 20/20 parity
tests pass, GICP 118.3 → 16.5 ms, VGICP 102.3 → 14.2 ms, and trajectories bit-agree with the
desktop GPU (APE = 0.0000 m across sm_87 ↔ sm_89). Deployment guide: [JETSON.md](JETSON.md)
(cuda-jetson branch).

## Benchmark methodology

All numbers above come from the official KITTI odometry sequence 00 (avg-kitti S3 data +
official GT), official protocol details, ablations (NN strategies, exec modes), the cuPCL
comparison protocol, and reproduction commands: **[BENCHMARK_GPU.md](BENCHMARK_GPU.md)**.
The engineering log, including the failed approaches, is [WORKLOG.md](WORKLOG.md).

## Documentation

[BENCHMARK_GPU.md](BENCHMARK_GPU.md) results & methodology · [JETSON.md](JETSON.md) device
deployment · [WORKLOG.md](WORKLOG.md) engineering log ·
[docs/superpowers/specs](docs/superpowers/specs/) design doc ·
[README_upstream.md](README_upstream.md) original CPU library README.

## Credits & license

Derivative of [koide3/small_gicp](https://github.com/koide3/small_gicp) by Kenji Koide
(AIST), MIT license inherited. If you use this work, please cite the upstream JOSS paper for
the algorithms and this repository for the CUDA implementation.
