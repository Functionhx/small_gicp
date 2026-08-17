// SPDX-License-Identifier: MIT
#include <sgc/search/nn_search.hpp>

#include <sgc/core/check.hpp>
#include <sgc/search/nn_query.cuh>

namespace sgc {

namespace {

constexpr int BLOCK = 256;

template <NNStrategy Strategy>
__global__ void nn_kernel(
  HashIndexView hidx,
  const unsigned long long* keys,
  int num_keys,
  const float4* pts,
  const float4* queries,
  int num_queries,
  float inv_leaf,
  int* out_idx,
  float* out_d2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_queries) {
    return;
  }
  nn_query<Strategy>(hidx, keys, num_keys, pts, queries[i], inv_leaf, 4.0f / (inv_leaf * inv_leaf), 3.4e38f, out_idx + i, out_d2 + i);
}

}  // namespace

void nn_search(const GpuCloud& target, const GpuBuffer<float4>& queries, GpuBuffer<int>& out_idx, GpuBuffer<float>& out_d2, NNStrategy strategy, float leaf_size) {
  out_idx.resize(queries.size());
  out_d2.resize(queries.size());
  if (queries.size() == 0) {
    return;
  }

  const HashIndexView hidx = target.index.view();
  const int num_keys = static_cast<int>(target.size());
  const int num_queries = static_cast<int>(queries.size());
  const int grid = (num_queries + BLOCK - 1) / BLOCK;
  const float inv_leaf = 1.0f / leaf_size;

  switch (strategy) {
    case NNStrategy::Voxel3:
      nn_kernel<NNStrategy::Voxel3><<<grid, BLOCK>>>(hidx, target.keys.raw(), num_keys, target.points.raw(), queries.raw(), num_queries, inv_leaf, out_idx.raw(), out_d2.raw());
      break;
    case NNStrategy::Voxel5:
      nn_kernel<NNStrategy::Voxel5><<<grid, BLOCK>>>(hidx, target.keys.raw(), num_keys, target.points.raw(), queries.raw(), num_queries, inv_leaf, out_idx.raw(), out_d2.raw());
      break;
    case NNStrategy::ExactBF:
      nn_kernel<NNStrategy::ExactBF><<<grid, BLOCK>>>(hidx, target.keys.raw(), num_keys, target.points.raw(), queries.raw(), num_queries, inv_leaf, out_idx.raw(), out_d2.raw());
      break;
  }
  SGC_CHECK(cudaGetLastError());
}

}  // namespace sgc
