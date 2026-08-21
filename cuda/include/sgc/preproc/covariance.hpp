// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/points/gpu_cloud.hpp>
#include <sgc/search/nn.hpp>

namespace sgc {

/// @brief Estimate oriented per-point normals on the GPU.
///        Uses the same exact/adaptive kNN search as estimate_covariances and mirrors
///        small_gicp::estimate_normals (smallest eigenvector, oriented toward the origin).
void estimate_normals(GpuCloud& cloud, float leaf_size, int num_neighbors = 20, int max_shell = 12);

/// @brief Estimate per-point covariances on the GPU.
///        Mirrors small_gicp::estimate_covariances: exact kNN (k=20, includes self) via an adaptive
///        expanding-shell voxel search with a rigorous early-stop bound (equivalent to kd-tree kNN up
///        to ties), sample covariance, then eigenvalue replacement (1e-3, 1, 1) which reduces to
///        cov = I - 0.999 * n * n^T with n the smallest-eigenvalue eigenvector.
///        Points with fewer than 5 neighbors get cov = I.
/// @param cloud          [in/out] Downsampled cloud (points + keys); covs is filled
/// @param leaf_size      Downsampling resolution (voxel size of the bucket index)
/// @param num_neighbors  Number of neighbors (default 20, same as upstream)
void estimate_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors = 20, int max_shell = 12);

/// @brief Estimate normals and covariances in one neighbor-search/eigendecomposition pass.
///        This is the preferred preprocessing path when both attributes are needed.
void estimate_normals_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors = 20, int max_shell = 12);

}  // namespace sgc
