// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/search/nn.hpp>
#include <sgc/voxel/hash_index.cuh>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

/// @brief Device-side single-query nearest neighbor over a voxel-bucket indexed cloud.
///        Voxel3/Voxel5 probe the 3x3x3 / 5x5x5 voxel neighborhood via the O(1) hash index.
///        When the best hit is farther than `expand_d2` (or nothing is found), the probe
///        expands once to a 9x9x9 neighborhood so that correspondences within the typical
///        max_correspondence_distance (= 4 voxels) are still found, matching the kd-tree
///        behavior on initially misaligned frames. ExactBF scans all points.
/// @param hidx       Voxel hash index view
/// @param keys       Sorted unique voxel keys (fallback when the hash is not built)
/// @param num_keys   Number of keys
/// @param pts        Points (bucket centroids, same order as keys)
/// @param q          Query point
/// @param inv_leaf   1 / voxel size
/// @param expand_d2  Expansion threshold (squared); queries whose best is worse than this
///                   probe the 9^3 neighborhood. Use 0.0f to disable.
/// @param out_idx    [out] Found point index or -1
/// @param out_d2     [out] Squared distance to the found point
template <NNStrategy Strategy>
__device__ __forceinline__ void nn_query(
  const HashIndexView& hidx,
  const unsigned long long* keys,
  int num_keys,
  const float4* pts,
  float4 q,
  float inv_leaf,
  float expand_d2,
  float search_cap_d2,
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
        const unsigned long long key = coord_key(coord);
        int j;
        if (hidx.ready()) {
          j = hash_lookup(hidx, key);
        } else {
          j = find_voxel(keys, num_keys, key);
        }
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

  // Adaptive expansion for initially misaligned frames: the cheap window found nothing
  // promising, so probe the window that covers the correspondence rejection radius. Voxels
  // whose closest possible point is farther than the current best are pruned, which reduces
  // the effective scan to the sphere shell around the query.
  if (expand_d2 > 0.0f && (best < 0 || best_d2 > expand_d2)) {
    const int R = 4;
    // Query position inside its own voxel, in [0, 1)
    const float fx = q.x * inv_leaf - floorf(q.x * inv_leaf);
    const float fy = q.y * inv_leaf - floorf(q.y * inv_leaf);
    const float fz = q.z * inv_leaf - floorf(q.z * inv_leaf);
    for (int ox = -R; ox <= R; ox++) {
      const float gx = ox == 0 ? 0.0f : (abs(ox) - 1 + (ox > 0 ? 1.0f - fx : fx));
      for (int oy = -R; oy <= R; oy++) {
        const float gy = oy == 0 ? 0.0f : (abs(oy) - 1 + (oy > 0 ? 1.0f - fy : fy));
        for (int oz = -R; oz <= R; oz++) {
          if (max(max(abs(ox), abs(oy)), abs(oz)) <= r) {
            continue;  // already probed
          }
          const float gz = oz == 0 ? 0.0f : (abs(oz) - 1 + (oz > 0 ? 1.0f - fz : fz));
          // Voxel-unit distance to the closest point of this voxel; prune when it cannot beat
          // the current best or can never come under the correspondence rejection radius.
          const float prune_ref = best >= 0 ? best_d2 : search_cap_d2;
          if ((gx * gx + gy * gy + gz * gz) / (inv_leaf * inv_leaf) >= prune_ref) {
            continue;
          }
          const VoxelCoord coord{c.x + ox, c.y + oy, c.z + oz};
          const unsigned long long key = coord_key(coord);
          int j;
          if (hidx.ready()) {
            j = hash_lookup(hidx, key);
          } else {
            j = find_voxel(keys, num_keys, key);
          }
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
  }

  *out_idx = best;
  *out_d2 = best_d2;
}

}  // namespace sgc
