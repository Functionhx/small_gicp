// SPDX-License-Identifier: MIT
#include <sgc/voxel/downsample.hpp>

#include <cub/cub.cuh>

#include <vector>

#include <sgc/core/check.hpp>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

namespace {

constexpr int BLOCK = 256;

inline int grid_size(size_t n, int block = BLOCK) { return static_cast<int>((n + block - 1) / block); }

__global__ void compute_keys_kernel(const float4* pts, size_t n, float inv_leaf, unsigned long long* keys) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  keys[i] = voxel_key(pts[i], inv_leaf);
}

__global__ void iota_kernel(unsigned int* values, size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    values[i] = static_cast<unsigned int>(i);
  }
}

// A sorted point starts a new voxel run when its key differs from the previous one. Invalid keys never start runs.
__global__ void run_flags_kernel(const unsigned long long* sorted_keys, size_t n, unsigned int* flags) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const bool new_run = sorted_keys[i] != INVALID_KEY && (i == 0 || sorted_keys[i - 1] != sorted_keys[i]);
  flags[i] = new_run ? 1u : 0u;
}

// Scatter run start positions into the CSR array: starts[prefix[i]] = i for run starts.
__global__ void scatter_starts_kernel(const unsigned long long* sorted_keys, const unsigned int* flags, const unsigned int* prefix, size_t n, unsigned int* starts) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) {
    starts[prefix[i]] = static_cast<unsigned int>(i);
  }
}

__global__ void invalid_start_kernel(const unsigned long long* sorted_keys, size_t n, unsigned int* out) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && sorted_keys[i] == INVALID_KEY && (i == 0 || sorted_keys[i - 1] != INVALID_KEY)) {
    atomicMin(out, static_cast<unsigned int>(i));
  }
}

// One block per voxel. Per-thread strided fp64 accumulation + fixed shared-memory tree reduction.
__global__ void centroid_kernel(
  const float4* raw_points,
  const unsigned int* sorted_values,
  const unsigned long long* sorted_keys,
  const unsigned int* starts,
  unsigned int num_buckets,
  unsigned int invalid_start,
  float4* out_points,
  unsigned long long* out_keys) {
  const unsigned int bucket = blockIdx.x;
  if (bucket >= num_buckets) {
    return;
  }

  const unsigned int s = starts[bucket];
  const unsigned int e = (bucket + 1 < num_buckets) ? starts[bucket + 1] : invalid_start;

  __shared__ double sx[BLOCK], sy[BLOCK], sz[BLOCK], sw[BLOCK];

  double px = 0.0, py = 0.0, pz = 0.0, pw = 0.0;
  for (unsigned int i = s + threadIdx.x; i < e; i += BLOCK) {
    const float4 p = raw_points[sorted_values[i]];
    px += p.x;
    py += p.y;
    pz += p.z;
    pw += p.w;
  }
  sx[threadIdx.x] = px;
  sy[threadIdx.x] = py;
  sz[threadIdx.x] = pz;
  sw[threadIdx.x] = pw;
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx[threadIdx.x] += sx[threadIdx.x + stride];
      sy[threadIdx.x] += sy[threadIdx.x + stride];
      sz[threadIdx.x] += sz[threadIdx.x + stride];
      sw[threadIdx.x] += sw[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_points[bucket] = make_float4(static_cast<float>(sx[0] / sw[0]), static_cast<float>(sy[0] / sw[0]), static_cast<float>(sz[0] / sw[0]), 1.0f);
    out_keys[bucket] = sorted_keys[s];
  }
}

}  // namespace

void Downsampler::ensure_sizes(size_t n) {
  if (keys_in_.size() != n) {
    keys_in_.resize(n);
    values_in_.resize(n);
    sorted_keys_.resize(n);
    sorted_values_.resize(n);
    flags_.resize(n);
    prefix_.resize(n);
  }
  if (reduce_out_.size() != 1) {
    reduce_out_.resize(1);
  }
}

void Downsampler::run(GpuCloud& cloud, size_t num_raw, double leaf_size) {
  num_buckets_ = 0;
  if (num_raw == 0) {
    cloud.points.resize(0);
    cloud.keys.resize(0);
    return;
  }

  ensure_sizes(num_raw);
  const float inv_leaf = 1.0f / static_cast<float>(leaf_size);

  compute_keys_kernel<<<grid_size(num_raw), BLOCK>>>(cloud.points.raw(), num_raw, inv_leaf, keys_in_.raw());
  SGC_CHECK(cudaGetLastError());
  iota_kernel<<<grid_size(num_raw), BLOCK>>>(values_in_.raw(), num_raw);
  SGC_CHECK(cudaGetLastError());

  // Radix sort (key, value) pairs
  size_t sort_temp = 0, scan_temp = 0, reduce_temp = 0;
  SGC_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_temp, keys_in_.raw(), sorted_keys_.raw(), values_in_.raw(), sorted_values_.raw(), static_cast<int>(num_raw)));
  SGC_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp, flags_.raw(), prefix_.raw(), static_cast<int>(num_raw)));
  SGC_CHECK(cub::DeviceReduce::Sum(nullptr, reduce_temp, flags_.raw(), reduce_out_.raw(), static_cast<int>(num_raw)));
  const size_t temp_needed = std::max({sort_temp, scan_temp, reduce_temp});
  if (cub_temp_.size() < temp_needed) {
    cub_temp_.resize(temp_needed);
    cub_temp_size_ = temp_needed;
  }

  SGC_CHECK(cub::DeviceRadixSort::SortPairs(
    cub_temp_.raw(), cub_temp_size_, keys_in_.raw(), sorted_keys_.raw(), values_in_.raw(), sorted_values_.raw(), static_cast<int>(num_raw)));

  run_flags_kernel<<<grid_size(num_raw), BLOCK>>>(sorted_keys_.raw(), num_raw, flags_.raw());
  SGC_CHECK(cudaGetLastError());

  SGC_CHECK(cub::DeviceScan::ExclusiveSum(cub_temp_.raw(), cub_temp_size_, flags_.raw(), prefix_.raw(), static_cast<int>(num_raw)));
  SGC_CHECK(cub::DeviceReduce::Sum(cub_temp_.raw(), cub_temp_size_, flags_.raw(), reduce_out_.raw(), static_cast<int>(num_raw)));

  unsigned int num_buckets = 0;
  reduce_out_.download(&num_buckets, 1);
  num_buckets_ = num_buckets;
  if (num_buckets == 0) {
    cloud.points.resize(0);
    cloud.keys.resize(0);
    return;
  }

  invalid_start_.resize(1);
  unsigned int n_raw = static_cast<unsigned int>(num_raw);
  invalid_start_.upload(&n_raw, 1);
  invalid_start_kernel<<<grid_size(num_raw), BLOCK>>>(sorted_keys_.raw(), num_raw, invalid_start_.raw());
  SGC_CHECK(cudaGetLastError());
  unsigned int invalid_start = 0;
  invalid_start_.download(&invalid_start, 1);

  starts_.resize(num_buckets + 1);
  scatter_starts_kernel<<<grid_size(num_raw), BLOCK>>>(sorted_keys_.raw(), flags_.raw(), prefix_.raw(), num_raw, starts_.raw());
  SGC_CHECK(cudaGetLastError());
  SGC_CHECK(cudaMemcpy(starts_.raw() + num_buckets, &invalid_start, sizeof(unsigned int), cudaMemcpyHostToDevice));

  out_points_.resize(num_buckets);
  out_keys_.resize(num_buckets);
  centroid_kernel<<<static_cast<int>(num_buckets), BLOCK>>>(
    cloud.points.raw(), sorted_values_.raw(), sorted_keys_.raw(), starts_.raw(), num_buckets, invalid_start, out_points_.raw(), out_keys_.raw());
  SGC_CHECK(cudaGetLastError());

  cloud.points = std::move(out_points_);
  cloud.keys = std::move(out_keys_);
  out_points_ = GpuBuffer<float4>();
  out_keys_ = GpuBuffer<unsigned long long>();
}

}  // namespace sgc
