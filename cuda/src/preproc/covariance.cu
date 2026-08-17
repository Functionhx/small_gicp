// SPDX-License-Identifier: MIT
#include <sgc/preproc/covariance.hpp>

#include <sgc/core/check.hpp>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_K = 32;

// Cyclic Jacobi rotations for a symmetric 3x3 matrix. a is row-major and becomes diagonal;
// v's columns converge to the eigenvectors (v initially identity).
__device__ void jacobi_eigen3x3(float* a, float* v) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      v[i * 3 + j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (int sweep = 0; sweep < 16; sweep++) {
    // Find the largest off-diagonal element
    int p = 0, q = 1;
    float largest = fabsf(a[1]);
    if (fabsf(a[2]) > largest) {
      p = 0;
      q = 2;
      largest = fabsf(a[2]);
    }
    if (fabsf(a[5]) > largest) {
      p = 1;
      q = 2;
      largest = fabsf(a[5]);
    }
    if (largest < 1e-12f) {
      break;
    }

    const float app = a[p * 3 + p];
    const float aqq = a[q * 3 + q];
    const float apq = a[p * 3 + q];

    const float theta = 0.5f * atan2f(2.0f * apq, aqq - app);
    const float c = cosf(theta);
    const float s = sinf(theta);

    // A <- R^T A R (rows then columns)
    for (int k = 0; k < 3; k++) {
      const float akp = a[k * 3 + p];
      const float akq = a[k * 3 + q];
      a[k * 3 + p] = c * akp - s * akq;
      a[k * 3 + q] = s * akp + c * akq;
    }
    for (int k = 0; k < 3; k++) {
      const float apk = a[p * 3 + k];
      const float aqk = a[q * 3 + k];
      a[p * 3 + k] = c * apk - s * aqk;
      a[q * 3 + k] = s * apk + c * aqk;
    }

    // V <- V R (columns are eigenvectors)
    for (int k = 0; k < 3; k++) {
      const float vkp = v[k * 3 + p];
      const float vkq = v[k * 3 + q];
      v[k * 3 + p] = c * vkp - s * vkq;
      v[k * 3 + q] = s * vkp + c * vkq;
    }
  }
}

// Compute the covariance with eigenvalue replacement from a neighbor list.
__device__ void compute_cov_from_neighbors(const float4* points, const int* idx, int n, float* out) {
  if (n < 5) {
    // Upstream sets the identity matrix for points with too few neighbors
    out[0] = 1.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
    out[3] = 0.0f;
    out[4] = 1.0f;
    out[5] = 0.0f;
    out[6] = 0.0f;
    out[7] = 0.0f;
    out[8] = 1.0f;
    return;
  }

  // Sample covariance C = (sum(pp^T) - mean * sum(p)^T) / n
  float sx = 0.0f, sy = 0.0f, sz = 0.0f;
  float sxx = 0.0f, sxy = 0.0f, sxz = 0.0f;
  float syy = 0.0f, syz = 0.0f, szz = 0.0f;
  for (int k = 0; k < n; k++) {
    const float4 p = points[idx[k]];
    sx += p.x;
    sy += p.y;
    sz += p.z;
    sxx += p.x * p.x;
    sxy += p.x * p.y;
    sxz += p.x * p.z;
    syy += p.y * p.y;
    syz += p.y * p.z;
    szz += p.z * p.z;
  }
  const float inv_n = 1.0f / n;
  const float mx = sx * inv_n, my = sy * inv_n, mz = sz * inv_n;

  float a[9] = {
    (sxx - mx * sx) * inv_n, (sxy - mx * sy) * inv_n, (sxz - mx * sz) * inv_n,  //
    (sxy - my * sx) * inv_n, (syy - my * sy) * inv_n, (syz - my * sz) * inv_n,  //
    (sxz - mz * sx) * inv_n, (syz - mz * sy) * inv_n, (szz - mz * sz) * inv_n};

  float v[9];
  jacobi_eigen3x3(a, v);

  // Smallest-eigenvalue column (post-Jacobi diagonal)
  int min_axis = 0;
  if (a[4] < a[0]) {
    min_axis = 1;
  }
  if (a[8] < a[min_axis * 3 + min_axis]) {
    min_axis = 2;
  }
  const float nx = v[min_axis];
  const float ny = v[3 + min_axis];
  const float nz = v[6 + min_axis];

  // cov = I - 0.999 * n * n^T  (eigenvalue replacement (1e-3, 1, 1))
  out[0] = 1.0f - 0.999f * nx * nx;
  out[1] = -0.999f * nx * ny;
  out[2] = -0.999f * nx * nz;
  out[3] = -0.999f * ny * nx;
  out[4] = 1.0f - 0.999f * ny * ny;
  out[5] = -0.999f * ny * nz;
  out[6] = -0.999f * nz * nx;
  out[7] = -0.999f * nz * ny;
  out[8] = 1.0f - 0.999f * nz * nz;
}

__device__ __forceinline__ void push_candidate(float* dist, int* idx, int& n, int num_neighbors, float d2, int j) {
  if (n < num_neighbors || d2 < dist[n - 1]) {
    int k = n < num_neighbors ? n++ : num_neighbors - 1;
    while (k > 0 && dist[k - 1] > d2) {
      dist[k] = dist[k - 1];
      idx[k] = idx[k - 1];
      k--;
    }
    dist[k] = d2;
    idx[k] = j;
  }
}

// Pass 1: adaptive expanding-shell exact kNN over voxel buckets with a rigorous early-stop bound.
__global__ void covariance_shell_kernel(
  const float4* points,
  const unsigned long long* keys,
  int num_points,
  float inv_leaf,
  int num_neighbors,
  float* covs,
  unsigned int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points) {
    return;
  }

  float dist[MAX_K];
  int idx[MAX_K];
  int n = 0;

  const float4 pi = points[i];
  const VoxelCoord ci = key_coord(keys[i]);

  // Position of the point inside its own voxel, in [0, 1)
  const float fx = pi.x * inv_leaf - floorf(pi.x * inv_leaf);
  const float fy = pi.y * inv_leaf - floorf(pi.y * inv_leaf);
  const float fz = pi.z * inv_leaf - floorf(pi.z * inv_leaf);
  const float frac_min = fminf(fminf(fminf(fx, 1.0f - fx), fminf(fy, 1.0f - fy)), fminf(fz, 1.0f - fz));

  constexpr int MAX_SHELL = 8;
  bool terminated = false;
  for (int s = 0; s <= MAX_SHELL && n < num_neighbors; s++) {
    // Probe only shell s (voxels with max|offset| == s)
    for (int ox = -s; ox <= s; ox++) {
      for (int oy = -s; oy <= s; oy++) {
        for (int oz = -s; oz <= s; oz++) {
          if (std::max(std::max(std::abs(ox), std::abs(oy)), std::abs(oz)) < s) {
            continue;
          }
          const VoxelCoord coord{ci.x + ox, ci.y + oy, ci.z + oz};
          const int j = find_voxel(keys, num_points, coord_key(coord));
          if (j < 0) {
            continue;
          }
          const float4 pj = points[j];
          const float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
          push_candidate(dist, idx, n, num_neighbors, dx * dx + dy * dy + dz * dz, j);
        }
      }
    }

    // Early stop: the k-th best is closer than the nearest point outside the current cube
    if (n == num_neighbors) {
      const float bound = (frac_min + s) / inv_leaf;
      if (dist[n - 1] <= bound * bound) {
        terminated = true;
        break;
      }
    }
  }

  // High bit marks an unterminated (inexact) search that pass 2 must redo with brute force
  counts[i] = terminated ? static_cast<unsigned int>(n) : (static_cast<unsigned int>(n) | 0x80000000u);
  if (terminated) {
    compute_cov_from_neighbors(points, idx, n, covs + i * 9);
  }
}

// Pass 2: exact brute-force completion for sparse points that could not fill k neighbors within the shell budget.
__global__ void covariance_bruteforce_kernel(
  const float4* points,
  int num_points,
  int num_neighbors,
  float* covs,
  const unsigned int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points) {
    return;
  }
  const unsigned int state = counts[i];
  const bool terminated = (state & 0x80000000u) == 0;
  if (terminated && (state & 0x7fffffffu) >= static_cast<unsigned int>(num_neighbors)) {
    return;  // already exact
  }

  float dist[MAX_K];
  int idx[MAX_K];
  int n = 0;

  const float4 pi = points[i];
  for (int j = 0; j < num_points; j++) {
    const float4 pj = points[j];
    const float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
    push_candidate(dist, idx, n, num_neighbors, dx * dx + dy * dy + dz * dz, j);
  }

  compute_cov_from_neighbors(points, idx, n, covs + i * 9);
}

}  // namespace

void estimate_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors) {
  if (cloud.size() == 0) {
    return;
  }
  cloud.covs.resize(cloud.size() * 9);

  const int num_points = static_cast<int>(cloud.size());
  const int grid = (num_points + BLOCK - 1) / BLOCK;
  const float inv_leaf = 1.0f / leaf_size;

  GpuBuffer<unsigned int> counts(num_points);
  covariance_shell_kernel<<<grid, BLOCK>>>(cloud.points.raw(), cloud.keys.raw(), num_points, inv_leaf, num_neighbors, cloud.covs.raw(), counts.raw());
  SGC_CHECK(cudaGetLastError());
  covariance_bruteforce_kernel<<<grid, BLOCK>>>(cloud.points.raw(), num_points, num_neighbors, cloud.covs.raw(), counts.raw());
  SGC_CHECK(cudaGetLastError());
}

}  // namespace sgc
