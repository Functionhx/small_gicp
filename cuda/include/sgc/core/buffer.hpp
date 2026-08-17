// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include <sgc/core/check.hpp>

namespace sgc {

/// @brief Fill a device buffer with a constant (test/smoke utility). Launchable from host code.
void launch_fill_kernel(float* data, size_t n, float value);

/// @brief RAII device memory buffer. Non-copyable, movable.
template <typename T>
class GpuBuffer {
public:
  GpuBuffer() = default;
  explicit GpuBuffer(size_t n) { resize(n); }
  ~GpuBuffer() { destroy(); }

  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  GpuBuffer(GpuBuffer&& other) noexcept : ptr_(other.ptr_), n_(other.n_) {
    other.ptr_ = nullptr;
    other.n_ = 0;
  }
  GpuBuffer& operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
      destroy();
      ptr_ = other.ptr_;
      n_ = other.n_;
      other.ptr_ = nullptr;
      other.n_ = 0;
    }
    return *this;
  }

  /// @brief Resize the buffer (deallocates and reallocates when the size changes).
  void resize(size_t n) {
    if (n == n_) {
      return;
    }
    destroy();
    if (n > 0) {
      SGC_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
    n_ = n;
  }

  T* raw() { return ptr_; }
  const T* raw() const { return ptr_; }
  size_t size() const { return n_; }

  void upload(const T* host, size_t n) { SGC_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice)); }
  void download(T* host, size_t n) const { SGC_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost)); }

private:
  void destroy() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    n_ = 0;
  }

  T* ptr_ = nullptr;
  size_t n_ = 0;
};

}  // namespace sgc
