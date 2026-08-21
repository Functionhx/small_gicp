// SPDX-License-Identifier: MIT
#include <sgc/preproc/covariance.hpp>

#include <stdexcept>

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

// Closed-form smallest eigenvector for a symmetric 3x3 matrix. This replaces the
// transcendental-heavy Jacobi loop on the common path; nearly degenerate matrices fall
// back to Jacobi to preserve robustness.
__device__ bool smallest_eigenvector_closed_form(const float* a, float* nx, float* ny, float* nz) {
  const float a00 = a[0], a01 = a[1], a02 = a[2];
  const float a11 = a[4], a12 = a[5], a22 = a[8];
  const float offdiag_sq = a01 * a01 + a02 * a02 + a12 * a12;

  if (offdiag_sq < 1e-20f) {
    int axis = 0;
    if (a11 < a00) {
      axis = 1;
    }
    if (a22 < (axis == 0 ? a00 : a11)) {
      axis = 2;
    }
    *nx = axis == 0 ? 1.0f : 0.0f;
    *ny = axis == 1 ? 1.0f : 0.0f;
    *nz = axis == 2 ? 1.0f : 0.0f;
    return true;
  }

  const float q = (a00 + a11 + a22) / 3.0f;
  const float b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
  const float p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + 2.0f * offdiag_sq) / 6.0f);
  if (!(p > 1e-12f)) {
    return false;
  }

  const float inv_p = 1.0f / p;
  const float c00 = b00 * inv_p, c01 = a01 * inv_p, c02 = a02 * inv_p;
  const float c11 = b11 * inv_p, c12 = a12 * inv_p, c22 = b22 * inv_p;
  const float det_b = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02);
  const float r = fminf(1.0f, fmaxf(-1.0f, 0.5f * det_b));
  const float phi = acosf(r) / 3.0f;
  constexpr float TWO_PI_OVER_THREE = 2.0943951023931954923f;
  const float lambda = q + 2.0f * p * cosf(phi + TWO_PI_OVER_THREE);
  const float lambda_max = q + 2.0f * p * cosf(phi);
  const float lambda_mid = 3.0f * q - lambda - lambda_max;
  if (lambda_mid - lambda <= 1e-4f * p) {
    return false;  // The smallest eigenspace is poorly conditioned; use Jacobi.
  }

  const float m00 = a00 - lambda, m11 = a11 - lambda, m22 = a22 - lambda;
  // Cross products of row pairs of A-lambda*I. The longest is the most stable null vector.
  const float c0x = a01 * a12 - a02 * m11;
  const float c0y = a02 * a01 - m00 * a12;
  const float c0z = m00 * m11 - a01 * a01;
  const float c1x = a01 * m22 - a02 * a12;
  const float c1y = a02 * a02 - m00 * m22;
  const float c1z = m00 * a12 - a01 * a02;
  const float c2x = m11 * m22 - a12 * a12;
  const float c2y = a12 * a02 - a01 * m22;
  const float c2z = a01 * a12 - m11 * a02;
  const float n0 = c0x * c0x + c0y * c0y + c0z * c0z;
  const float n1 = c1x * c1x + c1y * c1y + c1z * c1z;
  const float n2 = c2x * c2x + c2y * c2y + c2z * c2z;

  float vx = c0x, vy = c0y, vz = c0z, norm_sq = n0;
  if (n1 > norm_sq) {
    vx = c1x;
    vy = c1y;
    vz = c1z;
    norm_sq = n1;
  }
  if (n2 > norm_sq) {
    vx = c2x;
    vy = c2y;
    vz = c2z;
    norm_sq = n2;
  }
  if (!(norm_sq > 1e-20f)) {
    return false;
  }

  const float inv_norm = rsqrtf(norm_sq);
  vx *= inv_norm;
  vy *= inv_norm;
  vz *= inv_norm;
  const float rx = a00 * vx + a01 * vy + a02 * vz - lambda * vx;
  const float ry = a01 * vx + a11 * vy + a12 * vz - lambda * vy;
  const float rz = a02 * vx + a12 * vy + a22 * vz - lambda * vz;
  const float matrix_norm_sq = a00 * a00 + a11 * a11 + a22 * a22 + 2.0f * offdiag_sq;
  if (rx * rx + ry * ry + rz * rz > 1e-8f * fmaxf(matrix_norm_sq, 1e-20f)) {
    return false;
  }
  *nx = vx;
  *ny = vy;
  *nz = vz;
  return true;
}

// Compute normals and/or covariance with eigenvalue replacement from a neighbor list.
// Null outputs are skipped so ICP variants pay only for the features they require.
__device__ void compute_features_from_neighbors(const float4* points, const int* idx, int n, const float4 query_point, float* out_cov, float4* out_normal) {
  if (n < 5) {
    // Upstream uses normal=0 and covariance=I for insufficient neighborhoods.
    if (out_normal != nullptr) {
      *out_normal = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (out_cov != nullptr) {
      out_cov[0] = 1.0f;
      out_cov[1] = 0.0f;
      out_cov[2] = 0.0f;
      out_cov[3] = 0.0f;
      out_cov[4] = 1.0f;
      out_cov[5] = 0.0f;
      out_cov[6] = 0.0f;
      out_cov[7] = 0.0f;
      out_cov[8] = 1.0f;
    }
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
    (sxx - mx * sx) * inv_n,
    (sxy - mx * sy) * inv_n,
    (sxz - mx * sz) * inv_n,  //
    (sxy - my * sx) * inv_n,
    (syy - my * sy) * inv_n,
    (syz - my * sz) * inv_n,  //
    (sxz - mz * sx) * inv_n,
    (syz - mz * sy) * inv_n,
    (szz - mz * sz) * inv_n};

  float nx, ny, nz;
  if (!smallest_eigenvector_closed_form(a, &nx, &ny, &nz)) {
    float v[9];
    jacobi_eigen3x3(a, v);
    int min_axis = 0;
    if (a[4] < a[0]) {
      min_axis = 1;
    }
    if (a[8] < a[min_axis * 3 + min_axis]) {
      min_axis = 2;
    }
    nx = v[min_axis];
    ny = v[3 + min_axis];
    nz = v[6 + min_axis];
  }

  if (out_normal != nullptr) {
    // Mirror small_gicp::NormalSetter: orient the normal toward the sensor origin.
    if (query_point.x * nx + query_point.y * ny + query_point.z * nz > 0.0f) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    *out_normal = make_float4(nx, ny, nz, 0.0f);
  }

  if (out_cov != nullptr) {
    // cov = I - 0.999 * n * n^T  (eigenvalue replacement (1e-3, 1, 1))
    out_cov[0] = 1.0f - 0.999f * nx * nx;
    out_cov[1] = -0.999f * nx * ny;
    out_cov[2] = -0.999f * nx * nz;
    out_cov[3] = -0.999f * ny * nx;
    out_cov[4] = 1.0f - 0.999f * ny * ny;
    out_cov[5] = -0.999f * ny * nz;
    out_cov[6] = -0.999f * nz * nx;
    out_cov[7] = -0.999f * nz * ny;
    out_cov[8] = 1.0f - 0.999f * nz * nz;
  }
}

// Lexicographic (dist, index) ordering: makes the selected k-set independent of processing order.
__device__ __forceinline__ bool lex_less(float d2_a, int j_a, float d2_b, int j_b) {
  return d2_a < d2_b || (d2_a == d2_b && j_a < j_b);
}

constexpr int COV_BLOCK = 128;

// Pass 1 (warp-cooperative): exact kNN over voxel buckets with a rigorous early-stop bound.
//        One warp per point: lanes stride over each shell's voxels (latency hidden), hits are
//        merged into a shared-memory top-k by lane 0 with lexicographic (dist, index) ordering.
//        Exact for every point whose k-th neighbor lies within MAX_SHELL voxels.
constexpr int COV_WARPS = 4;
constexpr int COV_WSTRIDE = MAX_K + 1;

__global__ void covariance_shell_kernel(
  HashIndexView hidx,
  const float4* points,
  const unsigned long long* keys,
  int num_points,
  float inv_leaf,
  int num_neighbors,
  int max_shell,
  float* covs,
  float4* normals,
  unsigned int* counts) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int query = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  __shared__ float s_d2[COV_WARPS][COV_WSTRIDE];
  __shared__ int s_j[COV_WARPS][COV_WSTRIDE];
  __shared__ int s_n[COV_WARPS];
  __shared__ int s_stop[COV_WARPS];

  float* d2 = s_d2[warp];
  int* jj = s_j[warp];
  if (lane == 0) {
    s_n[warp] = 0;
    s_stop[warp] = 0;
  }
  __syncwarp();

  if (query >= num_points) {
    return;
  }

  const float4 pi = points[query];
  const VoxelCoord ci = key_coord(keys[query]);

  // Position of the point inside its own voxel, in [0, 1)
  const float fx = pi.x * inv_leaf - floorf(pi.x * inv_leaf);
  const float fy = pi.y * inv_leaf - floorf(pi.y * inv_leaf);
  const float fz = pi.z * inv_leaf - floorf(pi.z * inv_leaf);
  const float frac_min = fminf(fminf(fminf(fx, 1.0f - fx), fminf(fy, 1.0f - fy)), fminf(fz, 1.0f - fz));

  // Warp-cooperative shells stay cheap far beyond the per-thread version
  for (int s = 0; s <= max_shell && s_stop[warp] == 0; s++) {
    const int n3 = (2 * s + 1) * (2 * s + 1) * (2 * s + 1);

    // Round-based scan: each round every lane probes one voxel, then all 32 candidates
    // are merged into the shared top-k one at a time (no candidate is dropped).
    for (int base = 0; base < n3; base += 32) {
      const int t = base + lane;
      float cand_d2 = 3.4e38f;
      int cand_j = -1;
      if (t < n3) {
        const int ox = t / ((2 * s + 1) * (2 * s + 1)) - s;
        const int rem = t % ((2 * s + 1) * (2 * s + 1));
        const int oy = rem / (2 * s + 1) - s;
        const int oz = rem % (2 * s + 1) - s;
        if (max(max(abs(ox), abs(oy)), abs(oz)) == s) {
          const VoxelCoord coord{ci.x + ox, ci.y + oy, ci.z + oz};
          const unsigned long long key = coord_key(coord);
          const int j = hidx.ready() ? hash_lookup(hidx, key) : find_voxel(keys, num_points, key);
          if (j >= 0) {
            const float4 pj = points[j];
            const float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
            cand_d2 = dx * dx + dy * dy + dz * dz;
            cand_j = j;
          }
        }
      }

      unsigned int mask = __ballot_sync(0xffffffffu, cand_j >= 0);
      while (mask != 0) {
        const int l = __ffs(mask) - 1;
        mask &= mask - 1;
        const float cd2 = __shfl_sync(0xffffffffu, cand_d2, l);
        const int cj = __shfl_sync(0xffffffffu, cand_j, l);
        if (lane == 0) {
          const int n = s_n[warp];
          if (n < num_neighbors || lex_less(cd2, cj, d2[n - 1], jj[n - 1])) {
            int k = n < num_neighbors ? s_n[warp]++ : num_neighbors - 1;
            while (k > 0 && lex_less(cd2, cj, d2[k - 1], jj[k - 1])) {
              d2[k] = d2[k - 1];
              jj[k] = jj[k - 1];
              k--;
            }
            d2[k] = cd2;
            jj[k] = cj;
          }
        }
        __syncwarp();
      }
    }

    // Early stop: the k-th best is closer than the nearest point outside the current cube
    if (s_n[warp] >= num_neighbors) {
      const float bound = (frac_min + s) / inv_leaf;
      if (lane == 0 && d2[num_neighbors - 1] <= bound * bound) {
        s_stop[warp] = 1;
      }
      __syncwarp();
    }
  }

  if (lane == 0) {
    const int n = s_n[warp];
    // Only a search that fired the early-stop bound is provably exact; anything else
    // (including a full set at the shell cap) falls through to the brute-force pass.
    counts[query] = s_stop[warp] != 0 ? static_cast<unsigned int>(n) : (static_cast<unsigned int>(n) | 0x80000000u);
    compute_features_from_neighbors(points, jj, n, pi, covs != nullptr ? covs + query * 9 : nullptr, normals != nullptr ? normals + query : nullptr);
  }
}

// Pass 2a: exact kNN selection. One warp per query, k rounds of warp-wide lex-min reduction:
//          each round selects the next nearest neighbor by (dist, index). Small register
//          footprint (the covariance math lives in a separate kernel to avoid spills).
constexpr int BF_WARPS_PER_BLOCK = 8;

__global__ void covariance_bf_select_kernel(
  const float4* points,
  int num_points,
  int num_neighbors,
  int* sel_j,           // [query * MAX_K + r]
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
__global__ void covariance_bf_cov_kernel(const float4* points, int num_points, const int* sel_j, const unsigned int* found, float* covs, float4* normals) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points || found[i] == 0) {
    return;
  }
  compute_features_from_neighbors(
    points,
    sel_j + i * MAX_K,
    static_cast<int>(found[i]),
    points[i],
    covs != nullptr ? covs + i * 9 : nullptr,
    normals != nullptr ? normals + i : nullptr);
}

}  // namespace

namespace {

void estimate_features(GpuCloud& cloud, float leaf_size, int num_neighbors, int max_shell, bool with_covariances, bool with_normals) {
  if (cloud.size() == 0) {
    return;
  }
  if (num_neighbors < 1 || num_neighbors > MAX_K) {
    throw std::invalid_argument("num_neighbors must be in [1, 32]");
  }
  if (max_shell < 1 || max_shell > 12) {
    throw std::invalid_argument("max_shell must be in [1, 12]");
  }
  if (with_covariances) {
    cloud.covs.resize(cloud.size() * 9);
  }
  if (with_normals) {
    cloud.normals.resize(cloud.size());
  }

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
    constexpr int WARP_BLOCK = COV_WARPS * 32;
    const int grid = (num_points + COV_WARPS - 1) / COV_WARPS;
    covariance_shell_kernel<<<grid, WARP_BLOCK>>>(
      cloud.index.view(),
      cloud.points.raw(),
      cloud.keys.raw(),
      num_points,
      inv_leaf,
      num_neighbors,
      max_shell,
      with_covariances ? cloud.covs.raw() : nullptr,
      with_normals ? cloud.normals.raw() : nullptr,
      counts.raw());
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
    covariance_bf_cov_kernel<<<grid2, COV_BLOCK>>>(
      cloud.points.raw(),
      num_points,
      sel_j.raw(),
      found.raw(),
      with_covariances ? cloud.covs.raw() : nullptr,
      with_normals ? cloud.normals.raw() : nullptr);
    SGC_CHECK(cudaGetLastError());
  }
}

}  // namespace

void estimate_normals(GpuCloud& cloud, float leaf_size, int num_neighbors, int max_shell) {
  estimate_features(cloud, leaf_size, num_neighbors, max_shell, false, true);
}

void estimate_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors, int max_shell) {
  estimate_features(cloud, leaf_size, num_neighbors, max_shell, true, false);
}

void estimate_normals_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors, int max_shell) {
  estimate_features(cloud, leaf_size, num_neighbors, max_shell, true, true);
}

}  // namespace sgc
