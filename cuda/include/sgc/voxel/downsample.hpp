// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>

#include <sgc/core/buffer.hpp>
#include <sgc/points/gpu_cloud.hpp>

namespace sgc {

/// @brief GPU voxelgrid downsampling.
///        Sorts points by voxel key (radix sort) and computes per-voxel centroids.
///        The output point order (sorted by voxel key) doubles as the spatial index
///        used by the NN search strategies. Deterministic: fixed launch config + fp64 reduction.
class Downsampler {
public:
  /// @brief Downsample cloud.points in place (replaced by sorted voxel centroids) and fill cloud.keys.
  /// @param cloud     [in/out] Cloud whose points buffer holds num_raw raw points
  /// @param num_raw   Number of raw input points in cloud.points
  /// @param leaf_size Downsampling resolution
  void run(GpuCloud& cloud, size_t num_raw, double leaf_size);

  /// @brief Number of voxels found by the last run.
  size_t num_voxels() const { return num_buckets_; }

private:
  void ensure_sizes(size_t n);

  GpuBuffer<unsigned long long> keys_in_, sorted_keys_, out_keys_;
  GpuBuffer<unsigned int> values_in_, sorted_values_, flags_, prefix_, starts_;
  GpuBuffer<float4> out_points_;
  GpuBuffer<unsigned char> cub_temp_;
  GpuBuffer<unsigned int> reduce_out_, invalid_start_;
  size_t cub_temp_size_ = 0;
  size_t num_buckets_ = 0;
};

}  // namespace sgc
