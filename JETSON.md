# Jetson Orin NX Deployment (cuda-jetson branch)

This branch carries the same GPU implementation as `cuda-x86`, validated to cross-compile
cleanly for Orin (`sm_87`) with CUDA 12.x. The x86-side acceptance (KITTI-00 official data,
APE/RPE within 0.01% of upstream, 7.3x/6.1x end-to-end) is documented in `BENCHMARK_GPU.md`.

## Build on the device (JetPack 6.x, CUDA 12.x)

```bash
sudo apt install libeigen3-dev libgtest-dev  # if missing
cmake -B build_jetson -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build_jetson -j$(nproc)
ctest --test-dir build_jetson          # kernel parity tests (same 20 tests)
./build_jetson/cuda/odometry_gpu <velodyne_dir> --exec full-gpu --engine gicp
```

## Benchmark protocol on the device

1. Fix clocks for stable numbers:
   ```bash
   sudo nvpmodel -m 0            # MAX-N mode
   sudo jetson_clocks
   ```
2. Run the comparison matrix (`scripts/run_gpu_matrix.sh` style):
   ```bash
   for e in gicp vgicp; do
     ./build_jetson/cuda/odometry_gpu <velodyne> --exec cpu      --engine $e --max_frames 500 --traj /tmp/cpu_$e.txt
     ./build_jetson/cuda/odometry_gpu <velodyne> --exec full-gpu --engine $e --max_frames 500 --traj /tmp/gpu_$e.txt
     ./build_jetson/cuda/odometry_gpu <velodyne> --exec hybrid   --engine $e --max_frames 500   # CPU-vs-GPU split on A78AE
   done
   python3 scripts/eval_traj.py /tmp/cpu_gicp.txt /tmp/gpu_gicp.txt
   ```
3. Record power during the run: `sudo tegrastats --interval 1000 --logfile tegrastats.log`
   (report msec/frame and msec*W/frame).

## Device-specific expectations / TODO when hardware is available

- **Unified memory**: `GpuBuffer` allocations are already device-resident and the pipeline
  uploads each raw frame once (~1.9 MB); on Orin this transfer is nearly free. If profiling
  shows the final per-iteration D2H (43 doubles) mattering, switch those to mapped pinned
  memory — no algorithm change needed.
- **Weak A78AE cores**: the CPU baseline will be several times slower than the x86 numbers,
  and `exec=hybrid` should lose to `full-gpu` clearly. The LM solve (6x6 LDLT on CPU) is
  microseconds and stays on CPU.
- **Power envelope**: log `nvpmodel MAXN` vs `15W` modes; the GPU pipeline at ~6 msec/frame
  leaves headroom to downclock.
- If anything differs numerically from x86, first check fp32 contraction flags
  (`--fmad` defaults) before suspecting the kernels; the tests in `ctest` encode the exact
  upstream parity contracts.
