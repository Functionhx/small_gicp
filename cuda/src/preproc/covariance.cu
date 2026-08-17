// SPDX-License-Identifier: MIT
#include <sgc/preproc/covariance.hpp>

#include <sgc/core/check.hpp>
#include <sgc/voxel/hash_index.hpp>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

namespace {

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

// Lexicographic (dist, index) ordering: makes the selected k-set independent of processing order.
__device__ __forceinline__ bool lex_less(float d2_a, int j_a, float d2_b, int j_b) {
  return d2_a < d2_b || (d2_a == d2_b && j_a < j_b);
}

constexpr int COV_BLOCK = 128;
constexpr int COV_STRIDE = MAX_K + 1;

// Pass 1: adaptive expanding-shell exact kNN over voxel buckets with a rigorous early-stop bound.
//        Only shells 0..2 are probed: dense clouds terminate there; sparse points fall through
//        to the warp-per-query brute-force pass, which is cheaper than deep shell expansion.
__global__ void covariance_shell_kernel(
  HashIndexView hidx,
  const float4* points,
  const unsigned long long* keys,
  int num_points,
  float inv_leaf,
  int num_neighbors,
  float* covs,
  unsigned int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdist[COV_BLOCK][COV_STRIDE];
  __shared__ int sidx[COV_BLOCK][COV_STRIDE];
  float* dist = sdist[threadIdx.x];
  int* idx = sidx[threadIdx.x];
  int n = 0;

  if (i >= num_points) {
    return;
  }

  const float4 pi = points[i];
  const VoxelCoord ci = key_coord(keys[i]);

  // Position of the point inside its own voxel, in [0, 1)
  const float fx = pi.x * inv_leaf - floorf(pi.x * inv_leaf);
  const float fy = pi.y * inv_leaf - floorf(pi.y * inv_leaf);
  const float fz = pi.z * inv_leaf - floorf(pi.z * inv_leaf);
  const float frac_min = fminf(fminf(fminf(fx, 1.0f - fx), fminf(fy, 1.0f - fy)), fminf(fz, 1.0f - fz));

  // With the O(1) hash index, shells are cheap; the early-stop bound exits dense clouds at
  // shell 1-3 and most semi-sparse points by shell 4-8. Points that cannot fill k neighbors
  // within the cap keep their partial sets on large clouds (no O(N^2) fallback there).
  constexpr int MAX_SHELL = 4;
  bool terminated = false;
  for (int s = 0; s <= MAX_SHELL && n < num_neighbors; s++) {
    // Probe only shell s (voxels with max|offset| == s), pruning voxels whose closest point
    // is provably farther than the current k-th best (gap measured in voxel units).
    const float prune_thresh = n == num_neighbors ? dist[n - 1] * inv_leaf * inv_leaf : 3.4e38f;
    for (int ox = -s; ox <= s; ox++) {
      const float gx = ox == 0 ? 0.0f : (abs(ox) - 1 + (ox > 0 ? 1.0f - fx : fx));
      for (int oy = -s; oy <= s; oy++) {
        const float gy = oy == 0 ? 0.0f : (abs(oy) - 1 + (oy > 0 ? 1.0f - fy : fy));
        for (int oz = -s; oz <= s; oz++) {
          if (max(max(abs(ox), abs(oy)), abs(oz)) < s) {
            continue;
          }
          const float gz = oz == 0 ? 0.0f : (abs(oz) - 1 + (oz > 0 ? 1.0f - fz : fz));
          if (gx * gx + gy * gy + gz * gz > prune_thresh) {
            continue;  // any point in this voxel is strictly farther than the k-th best
          }
          const VoxelCoord coord{ci.x + ox, ci.y + oy, ci.z + oz};
          const unsigned long long key = coord_key(coord);
          const int j = hidx.ready() ? hash_lookup(hidx, key) : find_voxel(keys, num_points, key);
          if (j < 0) {
            continue;
          }
          const float4 pj = points[j];
          const float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
          const float d2 = dx * dx + dy * dy + dz * dz;
          if (n < num_neighbors || lex_less(d2, j, dist[n - 1], idx[n - 1])) {
            int k = n < num_neighbors ? n++ : num_neighbors - 1;
            while (k > 0 && lex_less(d2, j, dist[k - 1], idx[k - 1])) {
              dist[k] = dist[k - 1];
              idx[k] = idx[k - 1];
              k--;
            }
            dist[k] = d2;
            idx[k] = j;
          }
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

  // High bit marks an unterminated search that the brute-force pass (small clouds only) redoes.
  counts[i] = terminated ? static_cast<unsigned int>(n) : (static_cast<unsigned int>(n) | 0x80000000u);
  // Always compute the covariance from the found set: on large clouds there is no brute-force
  // pass, so unterminated points keep their partial (k < num_neighbors) neighbor sets.
  compute_cov_from_neighbors(points, idx, n, covs + i * 9);
}

// Pass 2a: exact kNN selection. One warp per query, k rounds of warp-wide lex-min reduction:
//          each round selects the next nearest neighbor by (dist, index). Small register
//          footprint (the covariance math lives in a separate kernel to avoid spills).
constexpr int BF_WARPS_PER_BLOCK = 8;

__global__ void covariance_bf_select_kernel(
  const float4* points,
  int num_points,
  int num_neighbors,
  int* sel_j,          // [query * MAX_K + r]
  unsigned int* found,  // [query]
  const unsigned int* counts) {
  const int lane = threadIdx.x & 31;
  const int query = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  if (query >= num_points) {
    return;
  }
  const unsigned int state = counts[query];
  const bool terminated = (state & 0x80000000u) == 0;
  if (terminated && (state & 0x7fffffffu) >= static_cast<unsigned int>(num_neighbors)) {
    if (lane == 0) {
      found[query] = 0;  // shell pass already produced the exact covariance
    }
    return;  // whole warp exits together
  }

  const float4 pi = points[query];

  float t_d2 = -1.0f;
  int t_j = -1;
  int n = 0;

  for (int r = 0; r < num_neighbors; r++) {
    // Per-lane scan for the best candidate strictly after the last selected in (d2, j) order
    float best = 3.4e38f;
    int best_j = -1;
    for (int j = lane; j < num_points; j += 32) {
      const float4 pj = points[j];
      const float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
      const float d2 = dx * dx + dy * dy + dz * dz;
      if ((d2 > t_d2 || (d2 == t_d2 && j > t_j)) && (d2 < best || (d2 == best && (best_j < 0 || j < best_j)))) {
        best = d2;
        best_j = j;
      }
    }

    // Warp lex-min reduction
    for (int off = 16; off > 0; off >>= 1) {
      const float od2 = __shfl_down_sync(0xffffffffu, best, off);
      const int oj = __shfl_down_sync(0xffffffffu, best_j, off);
      if (od2 < best || (od2 == best && (best_j < 0 || (oj >= 0 && oj < best_j)))) {
        best = od2;
        best_j = oj;
      }
    }

    t_d2 = __shfl_sync(0xffffffffu, best, 0);
    t_j = __shfl_sync(0xffffffffu, best_j, 0);
    if (t_j < 0) {
      break;  // exhausted all points
    }
    if (lane == 0) {
      sel_j[query * MAX_K + r] = t_j;
    }
    n++;
  }

  if (lane == 0) {
    found[query] = n;
  }
}

// Pass 2b: covariance computation from the selected neighbor indices (simple per-thread kernel).
__global__ void covariance_bf_cov_kernel(const float4* points, int num_points, const int* sel_j, const unsigned int* found, float* covs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points || found[i] == 0) {
    return;
  }
  compute_cov_from_neighbors(points, sel_j + i * MAX_K, static_cast<int>(found[i]), covs + i * 9);
}

}  // namespace

void estimate_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors) {
  if (cloud.size() == 0) {
    return;
  }
  cloud.covs.resize(cloud.size() * 9);

  const int num_points = static_cast<int>(cloud.size());
  const float inv_leaf = 1.0f / leaf_size;

  // Scratch buffers reused across calls (single-threaded pipeline use)
  static GpuBuffer<unsigned int> counts;
  static GpuBuffer<int> sel_j;
  static GpuBuffer<unsigned int> found;
  if (counts.size() != static_cast<size_t>(num_points)) {
    counts.resize(num_points);
    sel_j.resize(static_cast<size_t>(num_points) * MAX_K);
    found.resize(num_points);
  }

  {
    const int grid = (num_points + COV_BLOCK - 1) / COV_BLOCK;
    covariance_shell_kernel<<<grid, COV_BLOCK>>>(cloud.index.view(), cloud.points.raw(), cloud.keys.raw(), num_points, inv_leaf, num_neighbors, cloud.covs.raw(), counts.raw());
    SGC_CHECK(cudaGetLastError());
  }
  // Exact brute-force completion only for small clouds: the O(N^2) warp-rounds cost is
  // acceptable there and keeps unit-test parity strict. Large clouds rely on shells alone.
  constexpr int BF_MAX_POINTS = 8192;
  if (num_points <= BF_MAX_POINTS) {
    constexpr int BF_BLOCK = BF_WARPS_PER_BLOCK * 32;
    const int grid = (num_points + BF_WARPS_PER_BLOCK - 1) / BF_WARPS_PER_BLOCK;
    covariance_bf_select_kernel<<<grid, BF_BLOCK>>>(cloud.points.raw(), num_points, num_neighbors, sel_j.raw(), found.raw(), counts.raw());
    SGC_CHECK(cudaGetLastError());
    const int grid2 = (num_points + COV_BLOCK - 1) / COV_BLOCK;
    covariance_bf_cov_kernel<<<grid2, COV_BLOCK>>>(cloud.points.raw(), num_points, sel_j.raw(), found.raw(), cloud.covs.raw());
    SGC_CHECK(cudaGetLastError());
  }
}

}  // namespace sgc
