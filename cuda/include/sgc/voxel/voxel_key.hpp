// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace sgc {

/// @brief 21 bits per axis voxel coordinate packing, identical to small_gicp's voxelgrid_sampling.
static constexpr int COORD_BIT_SIZE = 21;
static constexpr unsigned int COORD_BIT_MASK = (1u << COORD_BIT_SIZE) - 1;
static constexpr int COORD_OFFSET = 1 << (COORD_BIT_SIZE - 1);
static constexpr unsigned long long INVALID_KEY = std::numeric_limits<unsigned long long>::max();

struct VoxelCoord {
  int x, y, z;
};

/// @brief Fast floor with the same semantics as small_gicp::fast_floor (truncation + correction).
__host__ __device__ __forceinline__ int ffloor(float v) {
#ifdef __CUDA_ARCH__
  const int t = __float2int_rz(v);
  return t - (v < static_cast<float>(t));
#else
  const int t = static_cast<int>(v);
  return t - (v < static_cast<float>(t));
#endif
}

/// @brief Compute the packed voxel key of a point. Returns INVALID_KEY when out of the 21bit range.
__host__ __device__ __forceinline__ unsigned long long voxel_key(float4 p, float inv_leaf) {
  const int cx = ffloor(p.x * inv_leaf) + COORD_OFFSET;
  const int cy = ffloor(p.y * inv_leaf) + COORD_OFFSET;
  const int cz = ffloor(p.z * inv_leaf) + COORD_OFFSET;
  if (cx < 0 || cx > static_cast<int>(COORD_BIT_MASK) || cy < 0 || cy > static_cast<int>(COORD_BIT_MASK) || cz < 0 || cz > static_cast<int>(COORD_BIT_MASK)) {
    return INVALID_KEY;
  }
  return (static_cast<unsigned long long>(cx)) |                                   //
         (static_cast<unsigned long long>(cy) << (COORD_BIT_SIZE * 1)) |            //
         (static_cast<unsigned long long>(cz) << (COORD_BIT_SIZE * 2));
}

/// @brief Unpack a voxel key into integer coordinates.
__host__ __device__ __forceinline__ VoxelCoord key_coord(unsigned long long key) {
  return VoxelCoord{
    static_cast<int>(key & COORD_BIT_MASK) - COORD_OFFSET,
    static_cast<int>((key >> (COORD_BIT_SIZE * 1)) & COORD_BIT_MASK) - COORD_OFFSET,
    static_cast<int>((key >> (COORD_BIT_SIZE * 2)) & COORD_BIT_MASK) - COORD_OFFSET};
}

/// @brief Pack integer coordinates into a voxel key (no range check).
__host__ __device__ __forceinline__ unsigned long long coord_key(VoxelCoord c) {
  const unsigned int cx = static_cast<unsigned int>(c.x + COORD_OFFSET) & COORD_BIT_MASK;
  const unsigned int cy = static_cast<unsigned int>(c.y + COORD_OFFSET) & COORD_BIT_MASK;
  const unsigned int cz = static_cast<unsigned int>(c.z + COORD_OFFSET) & COORD_BIT_MASK;
  return (static_cast<unsigned long long>(cx)) |                                   //
         (static_cast<unsigned long long>(cy) << (COORD_BIT_SIZE * 1)) |            //
         (static_cast<unsigned long long>(cz) << (COORD_BIT_SIZE * 2));
}

}  // namespace sgc
