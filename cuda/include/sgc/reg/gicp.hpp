// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Geometry>

#include <sgc/points/gpu_cloud.hpp>
#include <sgc/reg/reduction.hpp>
#include <sgc/search/nn.hpp>

namespace sgc {

/// @brief Registration result, mirroring small_gicp::RegistrationResult.
struct GicpResult {
  GicpResult() = default;
  explicit GicpResult(const Eigen::Isometry3d& init_T) : T_target_source(init_T) {}

  Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
  bool converged = false;
  int iterations = 0;
  size_t num_inliers = 0;
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  double error = 0.0;
};

/// @brief Shared optimized scan-to-scan GPU registration engine.
///        Factor-specific kernels are selected once on the host so ICP, point-to-plane ICP,
///        and GICP do not carry per-point runtime branches.
struct ScanToScanGpu {
  explicit ScanToScanGpu(RegistrationFactor factor) : factor_(factor) {}

  NNStrategy nn = NNStrategy::Voxel3;
  int max_iterations = 20;
  int max_inner_iterations = 10;
  double init_lambda = 1e-3;
  double lambda_factor = 10.0;
  double max_dist_sq = 1.0;
  double translation_eps = 1e-3;
  double rotation_eps = 0.1 * M_PI / 180.0;

  /// @brief Align source to target.
  /// @param target    Target cloud (plus normals/covariances required by the selected factor)
  /// @param source    Source cloud (plus covariances required by GICP)
  /// @param init_T    Initial guess
  /// @param leaf_size Voxel size of the target bucket index
  GicpResult align(const GpuCloud& target, const GpuCloud& source, const Eigen::Isometry3d& init_T, float leaf_size);

  Linearizer linearizer;  // Reusable scratch

protected:
  bool converged(const Eigen::Matrix<double, 6, 1>& delta) const { return delta.head<3>().norm() <= rotation_eps && delta.tail<3>().norm() <= translation_eps; }

  RegistrationFactor factor_;
};

/// @brief GPU GICP registration engine (scan-to-scan).
///        LM loop constants and semantics are ported line-by-line from
///        small_gicp::LevenbergMarquardtOptimizer + TerminationCriteria.
struct GicpGpu : public ScanToScanGpu {
  GicpGpu() : ScanToScanGpu(RegistrationFactor::GICP) {}
};

}  // namespace sgc
