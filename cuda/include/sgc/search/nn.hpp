// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

#ifdef __CUDACC__
#define SGC_HD __host__ __device__ __forceinline__
#else
#define SGC_HD inline
#endif

/// @brief Correspondence search strategy.
enum class NNStrategy {
  Voxel3,   ///< Probe the 3x3x3 voxel neighborhood
  Voxel5,   ///< Probe the 5x5x5 voxel neighborhood
  ExactBF,  ///< Exact brute-force scan of all target points
};

/// @brief Find the index of `key` in the sorted unique key array (lower_bound). Returns -1 when absent.
SGC_HD int find_voxel(const unsigned long long* keys, int n, unsigned long long key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (keys[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < n && keys[lo] == key) {
    return lo;
  }
  return -1;
}

/// @brief Number of probe offsets for each strategy (27 / 125 / 0 for brute force).
constexpr int num_probe_offsets(NNStrategy s) {
  return s == NNStrategy::Voxel3 ? 27 : (s == NNStrategy::Voxel5 ? 125 : 0);
}

}  // namespace sgc
