// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/core/buffer.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/search/nn.hpp>

namespace sgc {

/// @brief Batch nearest neighbor search (host API, mainly for validation and debugging).
/// @param target    Voxel-bucket indexed target cloud
/// @param queries   Query points
/// @param out_idx   [out] Found point index per query (-1 when nothing found)
/// @param out_d2    [out] Squared distance per query
/// @param strategy  Search strategy
/// @param leaf_size Voxel size of the target index
void nn_search(const GpuCloud& target, const GpuBuffer<float4>& queries, GpuBuffer<int>& out_idx, GpuBuffer<float>& out_d2, NNStrategy strategy, float leaf_size);

}  // namespace sgc
