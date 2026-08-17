// SPDX-License-Identifier: MIT
#include <sgc/voxel/voxel_hash_map.hpp>

#include <cub/cub.cuh>

#include <vector>

#include <sgc/core/check.hpp>
#include <sgc/voxel/hash_index.cuh>

namespace sgc {

namespace {

constexpr int BLOCK = 256;
constexpr unsigned long long EMPTY = 0xFFFFFFFFFFFFFFFFull;

inline int grid_size(size_t n, int block = BLOCK) { return static_cast<int>((n + block - 1) / block); }

// Transform points and covariances by T (covariance: R * Cs * R^T).
__global__ void transform_kernel(const float4* pts, const float* covs, size_t n, const float* T, float4* t_pts, float* t_covs) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const float4 p = pts[i];
  const float4* T4 = reinterpret_cast<const float4*>(T);
  const float4 r0 = T4[0], r1 = T4[1], r2 = T4[2];
  t_pts[i] = make_float4(
    r0.x * p.x + r0.y * p.y + r0.z * p.z + r0.w, r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w, r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w, 1.0f);

  const float Cs[9] = {covs[i * 9 + 0], covs[i * 9 + 1], covs[i * 9 + 2], covs[i * 9 + 3], covs[i * 9 + 4], covs[i * 9 + 5], covs[i * 9 + 6], covs[i * 9 + 7], covs[i * 9 + 8]};
  const float R[9] = {r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z};
#pragma unroll
  for (int r = 0; r < 3; r++) {
#pragma unroll
    for (int c = 0; c < 3; c++) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 3; k++) {
#pragma unroll
        for (int l = 0; l < 3; l++) {
          s += R[r * 3 + k] * Cs[k * 3 + l] * R[c * 3 + l];
        }
      }
      t_covs[i * 9 + r * 3 + c] = s;
    }
  }
}

__global__ void compute_keys_kernel(const float4* pts, size_t n, float inv_leaf, unsigned long long* keys) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  keys[i] = voxel_key(pts[i], inv_leaf);
}

__global__ void iota_kernel(unsigned int* v, size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    v[i] = static_cast<unsigned int>(i);
  }
}

__global__ void run_flags_kernel(const unsigned long long* sorted_keys, size_t n, unsigned int* flags) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  flags[i] = (sorted_keys[i] != EMPTY && (i == 0 || sorted_keys[i - 1] != sorted_keys[i])) ? 1u : 0u;
}

// For each run (consecutive equal sorted keys): find or claim a table slot, record the slot id,
// and bump the LRU counter.
__global__ void claim_slots_kernel(
  const unsigned long long* sorted_keys,
  size_t n,
  unsigned int mask,
  unsigned long long* table_key,
  unsigned int* table_slot,
  unsigned int* slot_of_run,
  const unsigned int* run_prefix,
  unsigned long long* lru,
  unsigned int* live_count,
  unsigned long long lru_counter) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || sorted_keys[i] == EMPTY || !(i == 0 || sorted_keys[i - 1] != sorted_keys[i])) {
    return;
  }

  const unsigned long long key = sorted_keys[i];
  unsigned int pos = voxel_hash(key, mask);
  while (true) {
    const unsigned long long prev = atomicCAS(table_key + pos, EMPTY, key);
    if (prev == EMPTY) {
      atomicAdd(live_count, 1u);
      break;
    }
    if (prev == key) {
      break;
    }
    pos = (pos + 1) & mask;
  }

  table_slot[pos] = pos;  // slot id == table position
  lru[pos] = lru_counter;
  slot_of_run[run_prefix[i]] = pos;
}

// Each run is summed serially (sorted order => deterministic) and folded into its slot with
// one atomic per field per voxel per insert.
__global__ void accumulate_kernel(
  const unsigned long long* sorted_keys,
  const unsigned int* sorted_order,
  const unsigned int* run_prefix,
  size_t n,
  const float4* t_pts,
  const float* t_covs,
  const unsigned int* slot_of_run,
  float4* sum_pt,
  float* sum_cov,
  unsigned int* count) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || sorted_keys[i] == EMPTY || !(i == 0 || sorted_keys[i - 1] != sorted_keys[i])) {
    return;
  }

  // Find the run extent [i, end)
  size_t end = i + 1;
  while (end < n && sorted_keys[end] == sorted_keys[i]) {
    end++;
  }

  float4 sp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float sc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (size_t k = i; k < end; k++) {
    const unsigned int src = sorted_order[k];
    const float4 p = t_pts[src];
    sp.x += p.x;
    sp.y += p.y;
    sp.z += p.z;
    sp.w += p.w;
#pragma unroll
    for (int c = 0; c < 9; c++) {
      sc[c] += t_covs[src * 9 + c];
    }
  }

  const unsigned int slot = slot_of_run[run_prefix[i]];
  atomicAdd(&sum_pt[slot].x, sp.x);
  atomicAdd(&sum_pt[slot].y, sp.y);
  atomicAdd(&sum_pt[slot].z, sp.z);
  atomicAdd(&sum_pt[slot].w, sp.w);
#pragma unroll
  for (int c = 0; c < 9; c++) {
    atomicAdd(&sum_cov[slot * 9 + c], sc[c]);
  }
  atomicAdd(&count[slot], static_cast<unsigned int>(end - i));
}

// Finalize: mean = sum_pt / count, cov = sum_cov / count (live slots only).
__global__ void finalize_kernel(
  const unsigned long long* table_key,
  const float4* sum_pt,
  const float* sum_cov,
  const unsigned int* count,
  size_t cap,
  float4* mean,
  float* cov) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= cap || table_key[pos] == EMPTY || count[pos] == 0) {
    return;
  }
  const float inv_n = 1.0f / static_cast<float>(count[pos]);
  const float4 sp = sum_pt[pos];
  mean[pos] = make_float4(sp.x * inv_n, sp.y * inv_n, sp.z * inv_n, 1.0f);
#pragma unroll
  for (int c = 0; c < 9; c++) {
    cov[pos * 9 + c] = sum_cov[pos * 9 + c] * inv_n;
  }
}

// LRU eviction: clear slots not touched for `horizon` inserts.
__global__ void evict_kernel(
  unsigned long long* table_key,
  const unsigned long long* lru,
  size_t cap,
  unsigned long long counter,
  unsigned long long horizon,
  unsigned int* live_count) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= cap || table_key[pos] == EMPTY) {
    return;
  }
  if (lru[pos] + horizon < counter) {
    table_key[pos] = EMPTY;
    atomicSub(live_count, 1u);
  }
}

__global__ void clear_key_kernel(unsigned long long* table_key, size_t cap) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < cap) {
    table_key[pos] = EMPTY;
  }
}

__global__ void zero_u32_kernel(unsigned int* v, size_t n) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < n) {
    v[pos] = 0;
  }
}

__global__ void zero_f4_kernel(float4* v, size_t n) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < n) {
    v[pos] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

__global__ void zero_f_kernel(float* v, size_t n) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < n) {
    v[pos] = 0.0f;
  }
}

// Rehash for growth: move every live slot (table entry + payload) into a doubled table.
__global__ void rehash_kernel(
  const unsigned long long* old_key,
  const float4* old_sum_pt,
  const float* old_sum_cov,
  const unsigned int* old_count,
  const unsigned long long* old_lru,
  size_t old_cap,
  unsigned long long* new_key,
  unsigned int* new_slot,
  float4* new_sum_pt,
  float* new_sum_cov,
  unsigned int* new_count,
  unsigned long long* new_lru,
  unsigned int new_mask) {
  const size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= old_cap || old_key[pos] == EMPTY) {
    return;
  }
  const unsigned long long key = old_key[pos];
  unsigned int npos = voxel_hash(key, new_mask);
  while (atomicCAS(new_key + npos, EMPTY, key) != EMPTY) {
    npos = (npos + 1) & new_mask;
  }
  new_slot[npos] = npos;
  new_sum_pt[npos] = old_sum_pt[pos];
  new_count[npos] = old_count[pos];
  new_lru[npos] = old_lru[pos];
#pragma unroll
  for (int c = 0; c < 9; c++) {
    new_sum_cov[npos * 9 + c] = old_sum_cov[pos * 9 + c];
  }
}

}  // namespace

VoxelHashMap::VoxelHashMap(double leaf_size, size_t lru_horizon, size_t lru_clear_cycle) : leaf_size_(leaf_size), inv_leaf_(1.0f / leaf_size) {
  lru_horizon_val = lru_horizon;
  lru_clear_cycle_val = lru_clear_cycle;

  size_t cap = 1 << 18;
  table_key.resize(cap);
  table_slot.resize(cap);
  sum_pt.resize(cap);
  sum_cov.resize(cap * 9);
  count.resize(cap);
  lru.resize(cap);
  mean.resize(cap);
  cov.resize(cap * 9);
  mask_ = static_cast<unsigned int>(cap - 1);

  clear_key_kernel<<<grid_size(cap), BLOCK>>>(table_key.raw(), cap);
  zero_u32_kernel<<<grid_size(cap), BLOCK>>>(count.raw(), cap);
  zero_f4_kernel<<<grid_size(cap), BLOCK>>>(sum_pt.raw(), cap);
  zero_f_kernel<<<grid_size(cap * 9), BLOCK>>>(sum_cov.raw(), cap * 9);
  SGC_CHECK(cudaGetLastError());

  live_count_.resize(1);
  unsigned int zero = 0;
  live_count_.upload(&zero, 1);
}

void VoxelHashMap::grow() {
  const size_t old_cap = table_key.size();
  const size_t new_cap = old_cap * 2;

  GpuBuffer<unsigned long long> new_key(new_cap);
  GpuBuffer<unsigned int> new_slot(new_cap);
  GpuBuffer<float4> new_sum_pt(new_cap);
  GpuBuffer<float> new_sum_cov(new_cap * 9);
  GpuBuffer<unsigned int> new_count(new_cap);
  GpuBuffer<unsigned long long> new_lru(new_cap);
  GpuBuffer<float4> new_mean(new_cap);
  GpuBuffer<float> new_cov(new_cap * 9);
  clear_key_kernel<<<grid_size(new_cap), BLOCK>>>(new_key.raw(), new_cap);
  SGC_CHECK(cudaGetLastError());
  rehash_kernel<<<grid_size(old_cap), BLOCK>>>(
    table_key.raw(), sum_pt.raw(), sum_cov.raw(), count.raw(), lru.raw(), old_cap, new_key.raw(), new_slot.raw(), new_sum_pt.raw(), new_sum_cov.raw(), new_count.raw(), new_lru.raw(),
    static_cast<unsigned int>(new_cap - 1));
  SGC_CHECK(cudaGetLastError());

  table_key = std::move(new_key);
  table_slot = std::move(new_slot);
  sum_pt = std::move(new_sum_pt);
  sum_cov = std::move(new_sum_cov);
  count = std::move(new_count);
  lru = std::move(new_lru);
  mean = std::move(new_mean);
  cov = std::move(new_cov);
  mask_ = static_cast<unsigned int>(new_cap - 1);
}

void VoxelHashMap::evict() {
  evict_kernel<<<grid_size(table_key.size()), BLOCK>>>(table_key.raw(), lru.raw(), table_key.size(), lru_counter_, lru_horizon_val, live_count_.raw());
  SGC_CHECK(cudaGetLastError());
  unsigned int live = 0;
  live_count_.download(&live, 1);
  live_slots_ = live;
}

void VoxelHashMap::insert(const GpuCloud& cloud, const Eigen::Isometry3d& T) {
  const size_t n = cloud.size();
  if (n == 0) {
    return;
  }

  if (t_pts_.size() != n) {
    t_pts_.resize(n);
    t_covs_.resize(n * 9);
    keys_.resize(n);
    order_.resize(n);
    sorted_keys_.resize(n);
    sorted_order_.resize(n);
    run_flags_.resize(n);
    run_prefix_.resize(n);
    num_runs_buf_.resize(1);
  }
  if (h_T_.size() != 4) {
    h_T_.resize(4);
  }

  // T as 4 float4 rows
  {
    std::vector<float4> hT(4);
    const Eigen::Matrix4f Tf = T.matrix().cast<float>();
    for (int r = 0; r < 4; r++) {
      hT[r] = make_float4(Tf(r, 0), Tf(r, 1), Tf(r, 2), Tf(r, 3));
    }
    h_T_.upload(hT.data(), 4);
  }

  transform_kernel<<<grid_size(n), BLOCK>>>(cloud.points.raw(), cloud.covs.raw(), n, reinterpret_cast<const float*>(h_T_.raw()), t_pts_.raw(), t_covs_.raw());
  SGC_CHECK(cudaGetLastError());
  compute_keys_kernel<<<grid_size(n), BLOCK>>>(t_pts_.raw(), n, inv_leaf_, keys_.raw());
  SGC_CHECK(cudaGetLastError());
  iota_kernel<<<grid_size(n), BLOCK>>>(order_.raw(), n);
  SGC_CHECK(cudaGetLastError());

  size_t sort_temp = 0, scan_temp = 0, reduce_temp = 0;
  SGC_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_temp, keys_.raw(), sorted_keys_.raw(), order_.raw(), sorted_order_.raw(), static_cast<int>(n)));
  SGC_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp, run_flags_.raw(), run_prefix_.raw(), static_cast<int>(n)));
  SGC_CHECK(cub::DeviceReduce::Sum(nullptr, reduce_temp, run_flags_.raw(), num_runs_buf_.raw(), static_cast<int>(n)));
  const size_t temp_needed = std::max({sort_temp, scan_temp, reduce_temp});
  if (cub_temp_.size() < temp_needed) {
    cub_temp_.resize(temp_needed);
    cub_temp_size_ = temp_needed;
  }

  SGC_CHECK(cub::DeviceRadixSort::SortPairs(cub_temp_.raw(), cub_temp_size_, keys_.raw(), sorted_keys_.raw(), order_.raw(), sorted_order_.raw(), static_cast<int>(n)));
  run_flags_kernel<<<grid_size(n), BLOCK>>>(sorted_keys_.raw(), n, run_flags_.raw());
  SGC_CHECK(cudaGetLastError());
  SGC_CHECK(cub::DeviceScan::ExclusiveSum(cub_temp_.raw(), cub_temp_size_, run_flags_.raw(), run_prefix_.raw(), static_cast<int>(n)));
  SGC_CHECK(cub::DeviceReduce::Sum(cub_temp_.raw(), cub_temp_size_, run_flags_.raw(), num_runs_buf_.raw(), static_cast<int>(n)));

  unsigned int num_runs = 0;
  num_runs_buf_.download(&num_runs, 1);
  if (num_runs == 0) {
    return;
  }
  slot_of_run_.resize(num_runs);

  // Grow before claiming if the load factor is high
  unsigned int live = 0;
  live_count_.download(&live, 1);
  if (live + num_runs > table_key.size() * 6 / 10) {
    grow();
  }

  claim_slots_kernel<<<grid_size(n), BLOCK>>>(
    sorted_keys_.raw(), n, mask_, table_key.raw(), table_slot.raw(), slot_of_run_.raw(), run_prefix_.raw(), lru.raw(), live_count_.raw(), lru_counter_);
  SGC_CHECK(cudaGetLastError());

  accumulate_kernel<<<grid_size(n), BLOCK>>>(
    sorted_keys_.raw(), sorted_order_.raw(), run_prefix_.raw(), n, t_pts_.raw(), t_covs_.raw(), slot_of_run_.raw(), sum_pt.raw(), sum_cov.raw(), count.raw());
  SGC_CHECK(cudaGetLastError());

  finalize_kernel<<<grid_size(table_key.size()), BLOCK>>>(table_key.raw(), sum_pt.raw(), sum_cov.raw(), count.raw(), table_key.size(), mean.raw(), cov.raw());
  SGC_CHECK(cudaGetLastError());

  lru_counter_++;
  inserts_since_clear_++;

  live_count_.download(&live, 1);
  live_slots_ = live;

  if (inserts_since_clear_ >= lru_clear_cycle_val) {
    inserts_since_clear_ = 0;
    evict();
  }
}

}  // namespace sgc
