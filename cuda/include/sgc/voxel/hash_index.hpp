// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/core/buffer.hpp>
#include <sgc/voxel/hash_index.cuh>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

/// @brief Static voxel hash index over the unique sorted keys of a GpuCloud.
///        Built once after downsampling; used by covariance shells, NN strategies, and registration.
class VoxelHashIndex {
public:
  /// @brief Build the table from n unique sorted keys. Capacity = next pow2 >= 2n.
  void build(const unsigned long long* d_keys, size_t n);

  /// @brief Device view for kernel arguments.
  HashIndexView view() const { return HashIndexView{table_.raw(), mask_}; }

  size_t capacity() const { return table_.size(); }

private:
  GpuBuffer<ulonglong2> table_;  ///< (key, point index) slots
  unsigned int mask_ = 0;
};

}  // namespace sgc
