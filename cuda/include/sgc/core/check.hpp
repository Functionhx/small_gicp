// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace sgc {

/// @brief Throw std::runtime_error with file:line and the CUDA error string on failure.
inline void check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda error at ") + file + ":" + std::to_string(line) + " : " + cudaGetErrorString(err));
  }
}

}  // namespace sgc

#define SGC_CHECK(call) ::sgc::check((call), __FILE__, __LINE__)
