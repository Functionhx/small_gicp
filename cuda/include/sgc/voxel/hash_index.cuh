// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>

namespace sgc {

/// @brief Device-side view of a static voxel hash index (open addressing, linear probing).
///        Each slot is a single 16B (key, value) pair so a probe costs one dependent load.
struct HashIndexView {
  const ulonglong2* table = nullptr;  ///< Slot (key, value); key == ~0ull marks an empty slot
  unsigned int mask = 0;              ///< capacity - 1 (capacity is a power of two; 0 = not built)

  __device__ __forceinline__ bool ready() const { return mask != 0; }
};

/// @brief splitmix64 finalizer.
__device__ __forceinline__ unsigned int voxel_hash(unsigned long long key, unsigned int mask) {
  unsigned long long z = key + 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z = z ^ (z >> 31);
  return static_cast<unsigned int>(z) & mask;
}

/// @brief O(1) expected voxel lookup. Returns the point index or -1 when the voxel is absent.
__device__ __forceinline__ int hash_lookup(const HashIndexView& index, unsigned long long key) {
  unsigned int pos = voxel_hash(key, index.mask);
  while (true) {
    const ulonglong2 slot = index.table[pos];
    if (slot.x == key) {
      return static_cast<int>(slot.y);
    }
    if (slot.x == 0xFFFFFFFFFFFFFFFFull) {
      return -1;
    }
    pos = (pos + 1) & index.mask;
  }
}

}  // namespace sgc
