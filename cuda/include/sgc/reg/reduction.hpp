// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Geometry>

#include <sgc/core/buffer.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/search/nn.hpp>

namespace sgc {

/// @brief Cached correspondences (target index + fused mahalanobis 3x3) for error re-evaluation.
struct CorrCache {
  GpuBuffer<int> target_idx;        ///< Per source point: target point index or -1
  GpuBuffer<float> mahalanobis;     ///< Per source point: 9 floats (row-major 3x3)

  void resize(size_t n) {
    if (target_idx.size() != n) {
      target_idx.resize(n);
    }
    if (mahalanobis.size() != n * 9) {
      mahalanobis.resize(n * 9);
    }
  }
};

/// @brief Fused linearization + deterministic reduction over the GPU.
///        Holds device scratch buffers; one instance can be reused across iterations and frames.
class Linearizer {
public:
  /// @brief Linearize all source factors against the target and reduce to H (6x6), b (6), e (1).
  ///        Deterministic: fixed launch config, warp fp32 shuffle reduction, fp64 partials, ordered CPU final sum.
  /// @param target      Target cloud (points + keys + covs)
  /// @param source      Source cloud (points + covs)
  /// @param T           Linearization point (T_target_source)
  /// @param max_dist_sq Correspondence rejection threshold (squared)
  /// @param nn          NN strategy
  /// @param leaf_size   Voxel size of the target bucket index
  /// @param cache       [out] Correspondence cache for eval_error_cached
  /// @param h_out       [out] 43 doubles: H row-major 6x6, then b (6), then e
  /// @return            Number of inlier factors
  size_t linearize_and_reduce(
    const GpuCloud& target,
    const GpuCloud& source,
    const Eigen::Isometry3d& T,
    double max_dist_sq,
    NNStrategy nn,
    float leaf_size,
    CorrCache& cache,
    double* h_out);

  /// @brief Re-evaluate the error at a new T using cached correspondences (no new NN search).
  double eval_error_cached(const GpuCloud& target, const GpuCloud& source, const Eigen::Isometry3d& T, const CorrCache& cache);

private:
  void prepare(size_t num_source);

  GpuBuffer<float> d_T_;                   ///< 16 floats, row-major 4x4
  GpuBuffer<double> partials_;             ///< num_warps * 43
  GpuBuffer<unsigned int> inlier_count_;
  GpuBuffer<double> error_out_;
  size_t num_warps_ = 0;
  int block_ = 256;
};

}  // namespace sgc
