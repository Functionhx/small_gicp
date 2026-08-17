// SPDX-License-Identifier: MIT
#include <sgc/core/buffer.hpp>

namespace sgc {

__global__ void fill_kernel(float* data, size_t n, float value) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = value;
  }
}

void launch_fill_kernel(float* data, size_t n, float value) {
  if (n == 0) {
    return;
  }
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  fill_kernel<<<grid, block>>>(data, n, value);
  SGC_CHECK(cudaGetLastError());
}

}  // namespace sgc
