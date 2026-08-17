// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/search/nn.hpp>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

/// @brief Device-side single-query nearest neighbor over a voxel-bucket indexed cloud.
///        Voxel3/Voxel5 probe the 3x3x3 / 5x5x5 voxel neighborhood (approximate, but the true
///        NN is almost always inside it for roughly aligned clouds); ExactBF scans all points.
/// @param keys      Sorted unique voxel keys
/// @param num_keys  Number of keys
/// @param pts       Points (bucket centroids, same order as keys)
/// @param q         Query point
/// @param inv_leaf  1 / voxel size
/// @param out_idx   [out] Found point index or -1
/// @param out_d2    [out] Squared distance to the found point
template <NNStrategy Strategy>
__device__ __forceinline__ void nn_query(
  const unsigned long long* keys,
  int num_keys,
  const float4* pts,
  float4 q,
  float inv_leaf,
  int* out_idx,
  float* out_d2) {
  if (Strategy == NNStrategy::ExactBF) {
    int best = -1;
    float best_d2 = 3.4e38f;
    for (int j = 0; j < num_keys; j++) {
      const float4 p = pts[j];
      const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
      const float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = j;
      }
    }
    *out_idx = best;
    *out_d2 = best_d2;
    return;
  }

  const unsigned long long qkey = voxel_key(q, inv_leaf);
  if (qkey == INVALID_KEY) {
    *out_idx = -1;
    *out_d2 = 0.0f;
    return;
  }
  const VoxelCoord c = key_coord(qkey);

  int best = -1;
  float best_d2 = 3.4e38f;
  const int r = Strategy == NNStrategy::Voxel3 ? 1 : 2;
  for (int ox = -r; ox <= r; ox++) {
    for (int oy = -r; oy <= r; oy++) {
      for (int oz = -r; oz <= r; oz++) {
        const VoxelCoord coord{c.x + ox, c.y + oy, c.z + oz};
        const int j = find_voxel(keys, num_keys, coord_key(coord));
        if (j < 0) {
          continue;
        }
        const float4 p = pts[j];
        const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_d2) {
          best_d2 = d2;
          best = j;
        }
      }
    }
  }

  *out_idx = best;
  *out_d2 = best_d2;
}

}  // namespace sgc
