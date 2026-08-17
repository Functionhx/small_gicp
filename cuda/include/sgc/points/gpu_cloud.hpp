// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include <Eigen/Core>

#include <sgc/core/buffer.hpp>
#include <sgc/voxel/hash_index.hpp>

namespace sgc {

/// @brief Device point cloud with its voxel-bucket index (built by preprocessing).
struct GpuCloud {
  GpuBuffer<float4> points;               ///< Points (centroids after downsampling, sorted by voxel key)
  GpuBuffer<float> covs;                  ///< Per-point 3x3 covariances, row-major, 9 floats per point
  GpuBuffer<unsigned long long> keys;     ///< Unique sorted voxel keys (one per point, same order)
  VoxelHashIndex index;                   ///< O(1) voxel hash index over keys (built by Downsampler)

  /// @brief Number of indexed points (valid after downsampling).
  size_t size() const { return keys.size(); }

  /// @brief Create a raw (not yet preprocessed) cloud from host points.
  static GpuCloud from_host(const std::vector<Eigen::Vector4f>& host) {
    GpuCloud cloud;
    cloud.points.resize(host.size());
    std::vector<float4> tmp(host.size());
    for (size_t i = 0; i < host.size(); i++) {
      tmp[i] = make_float4(host[i].x(), host[i].y(), host[i].z(), host[i].w());
    }
    if (!tmp.empty()) {
      cloud.points.upload(tmp.data(), tmp.size());
    }
    return cloud;
  }

  /// @brief Upload an existing device-side buffer of points (takes a copy).
  void set_points(const float4* device_ptr, size_t n) {
    points.resize(n);
    if (n) {
      SGC_CHECK(cudaMemcpy(points.raw(), device_ptr, n * sizeof(float4), cudaMemcpyDeviceToDevice));
    }
  }

  std::vector<Eigen::Vector4f> download_points() const {
    std::vector<float4> tmp(points.size());
    std::vector<Eigen::Vector4f> out(points.size());
    if (!tmp.empty()) {
      points.download(tmp.data(), tmp.size());
    }
    for (size_t i = 0; i < out.size(); i++) {
      out[i] = Eigen::Vector4f(tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w);
    }
    return out;
  }

  std::vector<unsigned long long> download_keys() const {
    std::vector<unsigned long long> out(keys.size());
    if (!out.empty()) {
      keys.download(out.data(), out.size());
    }
    return out;
  }

  std::vector<float> download_covs() const {
    std::vector<float> out(covs.size());
    if (!out.empty()) {
      covs.download(out.data(), out.size());
    }
    return out;
  }
};

}  // namespace sgc
