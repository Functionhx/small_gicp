# Jetson Orin deployment and validation (`cuda-jetson`)

This branch is validated directly on a 12-core Cortex-A78AE Jetson Orin device running
JetPack/L4T R36.3, CUDA 12.2, and GCC 11.4. Both the project and the external
FastVGICPCuda comparison were built as native `sm_87` cubins.

## Build on the device

```bash
sudo apt install libeigen3-dev libgtest-dev  # if missing
cmake -S . -B build_jetson \
  -DBUILD_CUDA=ON \
  -DBUILD_WITH_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build_jetson -j4
ctest --test-dir build_jetson --output-on-failure
```

Current device gate: **32/32 tests pass**. Compute Sanitizer memcheck reports zero errors
for downsampling, covariance/normal estimation, ICP/GICP registration, and voxel-map reuse.

## Jetson covariance profiles

Normal/covariance estimation uses an expanding voxel-shell kNN search. The shell cap is
explicit so a faster Jetson deployment cannot silently change the quality baseline:

- `--cov_max_shell 12`: **quality baseline** (default), matching the original search cap.
- `--cov_max_shell 8`: **Jetson balanced** profile. Sparse outliers use a shorter
  neighborhood search; validate this mode against application ground truth.
- `--cov_max_shell 6`: aggressive optional profile.
- `--cov_max_shell 4`: experimental only; not recommended because the 100-frame relative
  rotation difference exceeded the selected balanced-mode envelope.

Examples:

```bash
# Quality baseline
./build_jetson/cuda/odometry_gpu <velodyne_dir> \
  --exec full-gpu --engine gicp --cov_max_shell 12 --max_frames 100

# Jetson balanced throughput profile
./build_jetson/cuda/odometry_gpu <velodyne_dir> \
  --exec full-gpu --engine gicp --cov_max_shell 8 --max_frames 100

# Point-to-Plane and both VGICP policies use the same explicit preprocessing profile.
./build_jetson/cuda/odometry_gpu <velodyne_dir> \
  --exec full-gpu --engine plane_icp --cov_max_shell 8 --max_frames 100
./build_jetson/cuda/odometry_gpu <velodyne_dir> \
  --exec full-gpu --engine vgicp_s2s --cov_max_shell 8 --max_frames 100
```

ICP does not estimate normals/covariances, so changing `--cov_max_shell` has no effect on
its output or runtime.

## Performance under the live vehicle workload

The on-device acceptance deliberately left MPCC, LIO, YOLO, PointPillar, RViz, Xorg, and
the rest of the ROS stack running. MAXN and `jetson_clocks` were enabled. Each row below is
the median of ten 100-frame runs (99 timed frames) in a rotated-order block design; no slow
run was removed.

| engine | small_gicp 12T p50 | small_gicp 8T p50 | CUDA quality12 p50 | CUDA balanced8 p50 |
|---|---:|---:|---:|---:|
| ICP | 101.34 ms | 43.25 ms | **23.85 ms** | same as quality12 |
| Point-to-Plane ICP | 67.59 ms | 31.85 ms | **29.48 ms** | **22.18 ms** |
| GICP | 57.02 ms | 27.80 ms | **26.03 ms** | **18.11 ms** |
| VGICP scan-to-model | 58.24 ms | 33.72 ms | **23.16 ms** | **15.97 ms** |
| VGICP scan-to-scan | 52.38 ms | 27.10 ms | **22.79 ms** | **15.84 ms** |

Paired p50 speedups versus small_gicp 8T are 1.89x (ICP), 1.15x/1.50x
(Point-to-Plane quality/balanced), 1.10x/1.55x (GICP), 1.52x/2.18x (VGICP model),
and 1.22x/1.73x (VGICP s2s). Every quoted paired comparison won 10/10 blocks.

FastVGICPCuda only provides VGICP scan-to-scan. Its default CPU-KD-tree + GPU path was
42.87 ms p50 in the same ten blocks; project quality12/balanced8 were 22.79/15.84 ms,
or **1.98x/2.83x faster**.

## Numerical envelope

The quality12 optimizer changes are effectively numerical no-ops: versus the saved base
binary, maximum 100-frame differences stayed below 0.4 mm and 0.004 degrees. Relative to
the matching small_gicp 8T trajectory, quality12 remains within the existing implementation
parity envelope.

Balanced8 is an explicit approximate preprocessing profile. Relative to quality12 over the
same 100 frames, maximum differences were:

| engine | max translation | max rotation |
|---|---:|---:|
| Point-to-Plane ICP | 3.37 cm | 0.058 deg |
| GICP | 1.73 cm | 0.035 deg |
| VGICP scan-to-model | 1.35 cm | 0.034 deg |
| VGICP scan-to-scan | 2.75 cm | 0.048 deg |

These are implementation-to-implementation checks, not errors against KITTI ground truth.
Use quality12 when that distinction has not been evaluated for the application.

## Jetson-specific optimizations

- GPU second-stage reduction downloads 43 doubles for H/b/e and one double for error-only
  passes instead of all warp partials. GICP D2H fell from 48.1 MB to 0.035 MB per 20 frames.
- Capacity-retaining `GpuBuffer`, persistent source clouds, downsampler storage exchange,
  and double-buffered scan-to-scan voxel maps cut GICP `cudaMalloc/cudaFree` calls from
  453/453 to 67/67 per 20 profiled frames.
- One warp now computes one voxel centroid (eight voxels per block) instead of launching a
  mostly idle 256-thread block per voxel. Centroid time fell from about 2.48 to 1.29 ms/frame.
- The symmetric 3x3 smallest eigenvector uses a checked closed-form path, with Jacobi
  fallback for ill-conditioned eigenspaces.
- The SM87 covariance kernel uses four warps per block. Nsight Compute measured achieved
  occupancy at 60.9% for quality12 and 73.2% for balanced8 (57.9% before optimization).
- Transform staging uses fixed arrays, correspondence initialization stays on-device, and
  the downsampler keeps its final run boundary on-device.

The attempted 64-element warp-bitonic top-k merge was reverted: sparse LiDAR voxel hits made
the sorting network 12-15% slower despite passing every parity test. CUDA Graph capture was
also rejected for the current LM loop because each CPU-side accept/reject decision changes
the next lambda and transform; fixed unrolling would launch up to 200 candidate passes.

## Profiling and repeatability

```bash
sudo nvpmodel -m 0
sudo jetson_clocks

tegrastats --interval 500 > tegrastats.log &
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o gicp_quality12 \
  ./build_jetson/cuda/odometry_gpu <velodyne_dir> \
  --exec full-gpu --engine gicp --cov_max_shell 12 --max_frames 20

# Hardware counters on Jetson require root by default.
sudo ncu --set basic --kernel-name regex:covariance_shell_kernel --launch-count 1 \
  ./build_jetson/cuda/odometry_gpu <velodyne_dir> \
  --exec full-gpu --engine gicp --cov_max_shell 8 --max_frames 2
```

Do not compare Nsight-instrumented wall times with Release benchmark wall times. Nsight is
for attribution; the JSON benchmark reports are the performance source of record.
