// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/reg/gicp.hpp>
#include <sgc/voxel/voxel_hash_map.hpp>

namespace sgc {

/// @brief GPU VGICP registration engine (scan-to-model over an incremental Gaussian voxel map).
///        Mirrors the upstream benchmark engine: the correspondence for each transformed source
///        point is the Gaussian of the voxel it falls into (center-voxel single probe,
///        `set_search_offsets(1)` semantics), rejected when the point-to-voxel-mean distance
///        exceeds max_dist_sq. LM constants are identical to GicpGpu.
struct VgicpGpu {
  int max_iterations = 20;
  int max_inner_iterations = 10;
  double init_lambda = 1e-3;
  double lambda_factor = 10.0;
  double max_dist_sq = 1.0;
  double translation_eps = 1e-3;
  double rotation_eps = 0.1 * M_PI / 180.0;

  /// @brief Align source against the voxel map.
  /// @param map    Incremental Gaussian voxel map (world frame)
  /// @param source Source cloud (points + covs)
  /// @param init_T Initial guess (T_world_source)
  GicpResult align(const VoxelHashMap& map, const GpuCloud& source, const Eigen::Isometry3d& init_T);

protected:
  bool converged(const Eigen::Matrix<double, 6, 1>& delta) const {
    return delta.head<3>().norm() <= rotation_eps && delta.tail<3>().norm() <= translation_eps;
  }

  // Scratch buffers
  GpuBuffer<float> d_T_;
  GpuBuffer<double> partials_;
  GpuBuffer<unsigned int> inlier_count_;
  size_t num_warps_ = 0;
  CorrCache cache_;
};

}  // namespace sgc
