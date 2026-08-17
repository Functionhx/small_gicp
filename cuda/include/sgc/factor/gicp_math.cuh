// SPDX-License-Identifier: MIT
#pragma once

namespace sgc {

/// @brief Row-major 3x3 float matrix device helper.
struct M3 {
  float m[9];

  __device__ __forceinline__ float at(int r, int c) const { return m[r * 3 + c]; }
  __device__ __forceinline__ float& at(int r, int c) { return m[r * 3 + c]; }
};

/// @brief 3-vector device helper.
struct V3 {
  float x, y, z;
};

__device__ __forceinline__ V3 col(const M3& a, int c) { return V3{a.m[c], a.m[3 + c], a.m[6 + c]}; }

__device__ __forceinline__ V3 operator*(const M3& a, const V3& v) {
  return V3{a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z, a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z, a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
}

__device__ __forceinline__ M3 operator+(const M3& a, const M3& b) {
  M3 r;
  #pragma unroll
  for (int i = 0; i < 9; i++) {
    r.m[i] = a.m[i] + b.m[i];
  }
  return r;
}

__device__ __forceinline__ float dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/// @brief Closed-form inverse of a 3x3 matrix via the adjugate.
__device__ __forceinline__ M3 inv3(const M3& a) {
  const float c00 = a.at(1, 1) * a.at(2, 2) - a.at(1, 2) * a.at(2, 1);
  const float c01 = a.at(1, 2) * a.at(2, 0) - a.at(1, 0) * a.at(2, 2);
  const float c02 = a.at(1, 0) * a.at(2, 1) - a.at(1, 1) * a.at(2, 0);
  const float det = a.at(0, 0) * c00 + a.at(0, 1) * c01 + a.at(0, 2) * c02;
  // Guard against singular / underflowed determinants (NaN/Inf poisoning the linearized system)
  const float inv_det = 1.0f / (fabsf(det) < 1e-30f ? (det < 0.0f ? -1e-30f : 1e-30f) : det);

  M3 r;
  r.at(0, 0) = c00 * inv_det;
  r.at(0, 1) = (a.at(0, 2) * a.at(2, 1) - a.at(0, 1) * a.at(2, 2)) * inv_det;
  r.at(0, 2) = (a.at(0, 1) * a.at(1, 2) - a.at(0, 2) * a.at(1, 1)) * inv_det;
  r.at(1, 0) = c01 * inv_det;
  r.at(1, 1) = (a.at(0, 0) * a.at(2, 2) - a.at(0, 2) * a.at(2, 0)) * inv_det;
  r.at(1, 2) = (a.at(0, 2) * a.at(1, 0) - a.at(0, 0) * a.at(1, 2)) * inv_det;
  r.at(2, 0) = c02 * inv_det;
  r.at(2, 1) = (a.at(0, 1) * a.at(2, 0) - a.at(0, 0) * a.at(2, 1)) * inv_det;
  r.at(2, 2) = (a.at(0, 0) * a.at(1, 1) - a.at(0, 1) * a.at(1, 0)) * inv_det;
  return r;
}

}  // namespace sgc
