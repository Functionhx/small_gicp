// SPDX-License-Identifier: MIT
#include <sgc/voxel/hash_index.hpp>

#include <sgc/core/check.hpp>

namespace sgc {

namespace {

constexpr int BLOCK = 256;
constexpr unsigned long long EMPTY = 0xFFFFFFFFFFFFFFFFull;

// Insert (key -> point index) with atomicCAS linear probing. Keys are unique.
__global__ void hash_insert_kernel(const unsigned long long* keys, int n, ulonglong2* table, unsigned int mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const unsigned long long key = keys[i];
  unsigned int pos = voxel_hash(key, mask);
  while (true) {
    const unsigned long long prev = atomicCAS(reinterpret_cast<unsigned long long*>(table + pos), EMPTY, key);
    if (prev == EMPTY) {
      table[pos].y = static_cast<unsigned long long>(i);
      return;
    }
    if (prev == key) {
      return;  // duplicate (cannot happen for unique keys, defensive)
    }
    pos = (pos + 1) & mask;
  }
}

__global__ void hash_clear_kernel(ulonglong2* table, size_t cap) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cap) {
    table[i].x = EMPTY;
    table[i].y = 0;
  }
}

}  // namespace

void VoxelHashIndex::build(const unsigned long long* d_keys, size_t n) {
  size_t cap = 16;
  while (cap < n * 2) {
    cap <<= 1;
  }
  if (table_.size() != cap) {
    table_.resize(cap);
  }
  mask_ = static_cast<unsigned int>(cap - 1);

  hash_clear_kernel<<<static_cast<int>((cap + BLOCK - 1) / BLOCK), BLOCK>>>(table_.raw(), cap);
  SGC_CHECK(cudaGetLastError());

  const int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
  hash_insert_kernel<<<grid, BLOCK>>>(d_keys, static_cast<int>(n), table_.raw(), mask_);
  SGC_CHECK(cudaGetLastError());
}

}  // namespace sgc
