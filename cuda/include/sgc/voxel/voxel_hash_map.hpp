// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Geometry>

#include <sgc/core/buffer.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

/// @brief Incremental Gaussian voxel map for VGICP (scan-to-model), mirroring
///        small_gicp::IncrementalVoxelMap<GaussianVoxel>.
///        Slots accumulate sum(T*p) and sum(T*Cs*T^T); finalize divides by count.
///        Determinism: points are sorted by voxel key before accumulation and each voxel's
///        run is summed serially by one thread, so the accumulation order is fixed.
///        LRU: slots not touched for `lru_horizon` inserts are evicted every `lru_clear_cycle` inserts.
class VoxelHashMap {
public:
  explicit VoxelHashMap(double leaf_size, size_t lru_horizon = 100, size_t lru_clear_cycle = 10);

  /// @brief Insert a preprocessed cloud transformed by T.
  void insert(const GpuCloud& cloud, const Eigen::Isometry3d& T);

  /// @brief Number of live voxels.
  size_t num_voxels() const { return live_slots_; }

  /// @brief Voxel size of the map.
  double leaf_size() const { return leaf_size_; }

  // Device buffers (slot-indexed; mean/cov finalized after each insert)
  GpuBuffer<unsigned long long> table_key;     ///< Hash table: voxel key or EMPTY
  GpuBuffer<unsigned int> table_slot;          ///< Hash table: slot index
  GpuBuffer<float4> sum_pt;                    ///< Slot: sum of transformed points
  GpuBuffer<float> sum_cov;                    ///< Slot: sum of transformed covariances (9 floats)
  GpuBuffer<unsigned int> count;               ///< Slot: number of accumulated points
  GpuBuffer<unsigned long long> lru;           ///< Slot: last insert counter value
  GpuBuffer<float4> mean;                      ///< Slot: finalized mean
  GpuBuffer<float> cov;                        ///< Slot: finalized covariance (9 floats)

  unsigned int mask() const { return mask_; }
  unsigned long long lru_counter() const { return lru_counter_; }
  size_t lru_horizon_val = 100;
  size_t lru_clear_cycle_val = 10;

private:
  void grow();
  void evict();

  double leaf_size_;
  float inv_leaf_;
  unsigned int mask_ = 0;
  size_t live_slots_ = 0;
  unsigned long long lru_counter_ = 0;
  size_t inserts_since_clear_ = 0;

  // Scratch reused across inserts
  GpuBuffer<float4> t_pts_;
  GpuBuffer<float> t_covs_;
  GpuBuffer<unsigned long long> keys_;
  GpuBuffer<unsigned int> order_;
  GpuBuffer<unsigned long long> sorted_keys_;
  GpuBuffer<unsigned int> sorted_order_;
  GpuBuffer<unsigned int> run_flags_;
  GpuBuffer<unsigned int> run_prefix_;
  GpuBuffer<unsigned int> num_runs_buf_;
  GpuBuffer<unsigned int> slot_of_run_;
  GpuBuffer<unsigned char> cub_temp_;
  size_t cub_temp_size_ = 0;
  GpuBuffer<float4> h_T_;
  GpuBuffer<unsigned int> live_count_;
};

}  // namespace sgc
