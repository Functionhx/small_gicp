// SPDX-License-Identifier: MIT
#include <sgc/reg/reduction.hpp>

#include <array>

#include <sgc/core/check.hpp>
#include <sgc/factor/gicp_math.cuh>
#include <sgc/search/nn_query.cuh>
#include <sgc/voxel/hash_index.hpp>

namespace sgc {

namespace {

constexpr int NUM_OUT = 43;  // H(36) + b(6) + e(1)

// Finalize warp partials on the device. One thread owns one output element and sums
// warp slots in their original order, preserving deterministic ordering while reducing
// D2H traffic from O(num_warps * 43) to exactly 43 doubles.
__global__ void reduce_partials_kernel(const double* partials, size_t num_warps, double* out) {
  const int k = threadIdx.x;
  if (k >= NUM_OUT) {
    return;
  }

  double sum = 0.0;
  for (size_t w = 0; w < num_warps; w++) {
    sum += partials[w * NUM_OUT + k];
  }
  out[k] = sum;
}

__global__ void reduce_error_kernel(const double* partials, size_t num_warps, double* out) {
  double sum = 0.0;
  for (size_t w = 0; w < num_warps; w++) {
    sum += partials[w * NUM_OUT + 42];
  }
  out[0] = sum;
}

// Warp-cooperative batch NN: one warp per source query. Phase 1 probes the 5^3 window with a
// warp reduction; phase 2 (only when the best is worse than the expansion threshold) probes
// the pruned 9^3 sphere. Deterministic: warp-min over (dist, index) pairs.
__global__ void nn_batch_kernel(
  HashIndexView hidx,
  const unsigned long long* keys,
  int num_keys,
  const float4* pts,
  const float4* queries,
  int num_queries,
  const float* T,  // row-major 4x4
  float inv_leaf,
  float max_dist_sq,
  int* out_idx,
  float* out_d2) {
  const int lane = threadIdx.x & 31;
  const int query = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (query >= num_queries) {
    return;
  }

  // Search at the transformed position q = T * p (same as the per-thread path)
  const float4 ps = queries[query];
  const float4* T4 = reinterpret_cast<const float4*>(T);
  const float4 r0 = T4[0], r1 = T4[1], r2 = T4[2];
  const float4 q =
    make_float4(r0.x * ps.x + r0.y * ps.y + r0.z * ps.z + r0.w, r1.x * ps.x + r1.y * ps.y + r1.z * ps.z + r1.w, r2.x * ps.x + r2.y * ps.y + r2.z * ps.z + r2.w, 1.0f);
  const unsigned long long qkey = voxel_key(q, inv_leaf);
  if (qkey == INVALID_KEY) {
    if (lane == 0) {
      out_idx[query] = -1;
      out_d2[query] = 0.0f;
    }
    return;
  }
  const VoxelCoord c = key_coord(qkey);

  float best_d2 = 3.4e38f;
  int best = -1;

  // Phase 1: 5x5x5 window, lanes stride
  {
    const int r = 2;
    const int n3 = 125;
    float my_d2 = 3.4e38f;
    int my_j = -1;
    for (int t = lane; t < n3; t += 32) {
      const int ox = t / 25 - r;
      const int rem = t % 25;
      const int oy = rem / 5 - r;
      const int oz = rem % 5 - r;
      const VoxelCoord coord{c.x + ox, c.y + oy, c.z + oz};
      const int j = hidx.ready() ? hash_lookup(hidx, coord_key(coord)) : find_voxel(keys, num_keys, coord_key(coord));
      if (j >= 0) {
        const float4 p = pts[j];
        const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < my_d2) {
          my_d2 = d2;
          my_j = j;
        }
      }
    }
    // Warp min reduction over (d2, j)
    for (int off = 16; off > 0; off >>= 1) {
      const float od2 = __shfl_down_sync(0xffffffffu, my_d2, off);
      const int oj = __shfl_down_sync(0xffffffffu, my_j, off);
      if (od2 < my_d2) {
        my_d2 = od2;
        my_j = oj;
      }
    }
    best_d2 = __shfl_sync(0xffffffffu, my_d2, 0);
    best = __shfl_sync(0xffffffffu, my_j, 0);
  }

  // Phase 2: pruned 9^3 sphere when the window was not promising. Pruning against the
  // (stale, larger) phase-1 best is conservative: skipped voxels cannot beat the final best.
  const float expand_d2 = 4.0f / (inv_leaf * inv_leaf);
  if (best < 0 || best_d2 > expand_d2) {
    const float fx = q.x * inv_leaf - floorf(q.x * inv_leaf);
    const float fy = q.y * inv_leaf - floorf(q.y * inv_leaf);
    const float fz = q.z * inv_leaf - floorf(q.z * inv_leaf);
    const float prune_ref = best >= 0 ? best_d2 : max_dist_sq;
    const float leaf = 1.0f / inv_leaf;

    float my_d2 = 3.4e38f;
    int my_j = -1;
    const int R = 4;
    const int W = 2 * R + 1;
    const int n3 = W * W * W;
    for (int t = lane; t < n3; t += 32) {
      const int ox = t / (W * W) - R;
      const int rem = t % (W * W);
      const int oy = rem / W - R;
      const int oz = rem % W - R;
      if (max(max(abs(ox), abs(oy)), abs(oz)) <= 2) {
        continue;  // phase 1 already covered the 5^3 window
      }
      const float gx = ox == 0 ? 0.0f : (abs(ox) - 1 + (ox > 0 ? 1.0f - fx : fx));
      const float gy = oy == 0 ? 0.0f : (abs(oy) - 1 + (oy > 0 ? 1.0f - fy : fy));
      const float gz = oz == 0 ? 0.0f : (abs(oz) - 1 + (oz > 0 ? 1.0f - fz : fz));
      if ((gx * gx + gy * gy + gz * gz) * (leaf * leaf) >= prune_ref) {
        continue;
      }
      const VoxelCoord coord{c.x + ox, c.y + oy, c.z + oz};
      const int j = hidx.ready() ? hash_lookup(hidx, coord_key(coord)) : find_voxel(keys, num_keys, coord_key(coord));
      if (j >= 0) {
        const float4 p = pts[j];
        const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < my_d2) {
          my_d2 = d2;
          my_j = j;
        }
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float od2 = __shfl_down_sync(0xffffffffu, my_d2, off);
      const int oj = __shfl_down_sync(0xffffffffu, my_j, off);
      if (od2 < my_d2) {
        my_d2 = od2;
        my_j = oj;
      }
    }
    const float p2_d2 = __shfl_sync(0xffffffffu, my_d2, 0);
    const int p2_j = __shfl_sync(0xffffffffu, my_j, 0);
    if (p2_j >= 0 && p2_d2 < best_d2) {
      best_d2 = p2_d2;
      best = p2_j;
    }
  }

  if (lane == 0) {
    out_idx[query] = best;
    out_d2[query] = best_d2;
  }
}

template <RegistrationFactor Factor>
__device__ __forceinline__ M3 factor_weight(const float4* target_normals, const float* target_covs, const float* source_covs, int target_index, int source_index, const M3& R) {
  M3 M{};
  if constexpr (Factor == RegistrationFactor::ICP) {
    M.m[0] = 1.0f;
    M.m[4] = 1.0f;
    M.m[8] = 1.0f;
  } else if constexpr (Factor == RegistrationFactor::PointToPlaneICP) {
    const float4 n = target_normals[target_index];
    // Match small_gicp::PointToPlaneICPFactor exactly:
    // H = J^T diag(n)^2 J, b = J^T diag(n)^2 r.
    M.m[0] = n.x * n.x;
    M.m[4] = n.y * n.y;
    M.m[8] = n.z * n.z;
  } else {
    // GICP: M = (Ct + R * Cs * R^T)^-1.
    M3 Ct, Cs;
#pragma unroll
    for (int k = 0; k < 9; k++) {
      Ct.m[k] = target_covs[target_index * 9 + k];
      Cs.m[k] = source_covs[source_index * 9 + k];
    }
    M3 RCsRt;
#pragma unroll
    for (int r = 0; r < 3; r++) {
#pragma unroll
      for (int c = 0; c < 3; c++) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 3; k++) {
          const float Rik = R.at(r, k);
#pragma unroll
          for (int l = 0; l < 3; l++) {
            s += Rik * Cs.at(k, l) * R.at(c, l);
          }
        }
        RCsRt.at(r, c) = s;
      }
    }
    M = inv3(Ct + RCsRt);
  }
  return M;
}

// Fused per-point registration linearization: transform -> NN -> factor weight -> H, b, e,
// then a deterministic warp shuffle reduction (fp32) with fp64 partials at lane 0.
template <RegistrationFactor Factor, NNStrategy Strategy, bool ErrorOnly>
__global__ void linearize_kernel(
  HashIndexView hidx,
  const int* pre_j,
  const float* pre_d2,
  const float4* target_pts,
  const float4* target_normals,
  const unsigned long long* target_keys,
  const float* target_covs,
  const float4* source_pts,
  const float* source_covs,
  int num_source,
  int num_target,
  const float* T,  // row-major 4x4
  float max_dist_sq,
  float inv_leaf,
  double* partials,  // num_warps * 43
  unsigned int* inlier_count,
  int* corr_target_idx,
  float* corr_mahalanobis) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;

  float acc[NUM_OUT];
#pragma unroll
  for (int k = 0; k < NUM_OUT; k++) {
    acc[k] = 0.0f;
  }
  bool inlier = false;

  if (i < num_source) {
    const float4 ps = source_pts[i];
    const float4* T4 = reinterpret_cast<const float4*>(T);
    const float4 r0 = T4[0], r1 = T4[1], r2 = T4[2];
    // q = T * p_src
    const V3 q{
      r0.x * ps.x + r0.y * ps.y + r0.z * ps.z + r0.w,  //
      r1.x * ps.x + r1.y * ps.y + r1.z * ps.z + r1.w,  //
      r2.x * ps.x + r2.y * ps.y + r2.z * ps.z + r2.w};
    const M3 R{r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z};

    int j = -1;

    if (ErrorOnly) {
      // Error re-evaluation with cached correspondences (no NN search)
      j = corr_target_idx[i];
      if (j >= 0) {
        M3 M{};
        if constexpr (Factor == RegistrationFactor::GICP) {
#pragma unroll
          for (int k = 0; k < 9; k++) {
            M.m[k] = corr_mahalanobis[i * 9 + k];
          }
        } else {
          M = factor_weight<Factor>(target_normals, nullptr, nullptr, j, i, R);
        }
        const float4 pt = target_pts[j];
        const V3 res{pt.x - q.x, pt.y - q.y, pt.z - q.z};
        acc[42] = 0.5f * dot(res, M * res);
      }
    } else {
      float d2 = 0.0f;
      if (pre_j != nullptr) {
        j = pre_j[i];
        d2 = pre_d2[i];
      } else {
        nn_query<Strategy>(hidx, target_keys, num_target, target_pts, make_float4(q.x, q.y, q.z, 1.0f), inv_leaf, 4.0f / (inv_leaf * inv_leaf), max_dist_sq, &j, &d2);
      }

      if (j >= 0 && d2 <= max_dist_sq) {
        const M3 M = factor_weight<Factor>(target_normals, target_covs, source_covs, j, i, R);

        const float4 pt = target_pts[j];
        const V3 res{pt.x - q.x, pt.y - q.y, pt.z - q.z};

        // J = [R * skew(p_src) | -R]  (3x6, column vectors)
        const V3 p{ps.x, ps.y, ps.z};
        const V3 J0{r0.y * p.z - r0.z * p.y, r1.y * p.z - r1.z * p.y, r2.y * p.z - r2.z * p.y};
        const V3 J1{r0.z * p.x - r0.x * p.z, r1.z * p.x - r1.x * p.z, r2.z * p.x - r2.x * p.z};
        const V3 J2{r0.x * p.y - r0.y * p.x, r1.x * p.y - r1.y * p.x, r2.x * p.y - r2.y * p.x};
        const V3 J3{-r0.x, -r1.x, -r2.x};
        const V3 J4{-r0.y, -r1.y, -r2.y};
        const V3 J5{-r0.z, -r1.z, -r2.z};

        const V3 Mr = M * res;
        const V3 MJ0 = M * J0, MJ1 = M * J1, MJ2 = M * J2, MJ3 = M * J3, MJ4 = M * J4, MJ5 = M * J5;

        // H upper triangle (row-major 6x6)
        acc[0] = dot(J0, MJ0);
        acc[1] = dot(J0, MJ1);
        acc[2] = dot(J0, MJ2);
        acc[3] = dot(J0, MJ3);
        acc[4] = dot(J0, MJ4);
        acc[5] = dot(J0, MJ5);
        acc[7] = dot(J1, MJ1);
        acc[8] = dot(J1, MJ2);
        acc[9] = dot(J1, MJ3);
        acc[10] = dot(J1, MJ4);
        acc[11] = dot(J1, MJ5);
        acc[14] = dot(J2, MJ2);
        acc[15] = dot(J2, MJ3);
        acc[16] = dot(J2, MJ4);
        acc[17] = dot(J2, MJ5);
        acc[21] = dot(J3, MJ3);
        acc[22] = dot(J3, MJ4);
        acc[23] = dot(J3, MJ5);
        acc[28] = dot(J4, MJ4);
        acc[29] = dot(J4, MJ5);
        acc[35] = dot(J5, MJ5);

        // b = J^T M r
        acc[36] = dot(J0, Mr);
        acc[37] = dot(J1, Mr);
        acc[38] = dot(J2, Mr);
        acc[39] = dot(J3, Mr);
        acc[40] = dot(J4, Mr);
        acc[41] = dot(J5, Mr);

        // e = 0.5 * r^T M r
        acc[42] = 0.5f * dot(res, Mr);

        corr_target_idx[i] = j;
        if constexpr (Factor == RegistrationFactor::GICP) {
#pragma unroll
          for (int k = 0; k < 9; k++) {
            corr_mahalanobis[i * 9 + k] = M.m[k];
          }
        }
        inlier = true;
      } else {
        corr_target_idx[i] = -1;
      }
    }
  }

  // Deterministic warp reduction: fp32 shuffle tree within warp, fp64 partial at lane 0.
#pragma unroll
  for (int k = 0; k < NUM_OUT; k++) {
    float v = acc[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      partials[warp_id * NUM_OUT + k] = static_cast<double>(v);
    }
  }

  if (ErrorOnly == false) {
    const unsigned int mask = __ballot_sync(0xffffffffu, inlier && i < num_source);
    if (lane == 0) {
      atomicAdd(inlier_count, static_cast<unsigned int>(__popc(mask)));
    }
  }
}

}  // namespace

void Linearizer::prepare(size_t num_source) {
  if (d_T_.size() != 16) {
    d_T_.resize(16);
  }
  // Every launched warp (including tail warps covering only out-of-range threads) writes its
  // partial slot; size the buffer by the launched warp count, not by ceil(n / 32).
  const size_t num_warps = ((num_source + block_ - 1) / block_) * (block_ / 32);
  if (num_warps_ != num_warps) {
    partials_.resize(num_warps * NUM_OUT);
    num_warps_ = num_warps;
  }
  if (inlier_count_.size() != 1) {
    inlier_count_.resize(1);
  }
  if (reduced_out_.size() != NUM_OUT) {
    reduced_out_.resize(NUM_OUT);
  }
  if (error_out_.size() != 1) {
    error_out_.resize(1);
  }
  if (nn_j_.size() != static_cast<size_t>(num_source)) {
    nn_j_.resize(num_source);
    nn_d2_.resize(num_source);
  }
}

namespace {

void upload_T(GpuBuffer<float>& d_T, const Eigen::Isometry3d& T) {
  std::array<float, 16> h_T;
  const Eigen::Matrix4f Tf = T.matrix().cast<float>();
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      h_T[r * 4 + c] = Tf(r, c);
    }
  }
  d_T.upload(h_T.data(), 16);
}

template <RegistrationFactor Factor, NNStrategy Strategy, bool ErrorOnly>
void launch_factor_kernel(
  int grid,
  int block,
  HashIndexView hidx,
  const int* pre_j,
  const float* pre_d2,
  const GpuCloud& target,
  const GpuCloud& source,
  int num_source,
  int num_target,
  const float* d_T,
  float max_dist_sq,
  float inv_leaf,
  double* partials,
  unsigned int* inlier_count,
  int* corr_target_idx,
  float* corr_mahalanobis) {
  linearize_kernel<Factor, Strategy, ErrorOnly><<<grid, block>>>(
    hidx,
    pre_j,
    pre_d2,
    target.points.raw(),
    target.normals.raw(),
    target.keys.raw(),
    target.covs.raw(),
    source.points.raw(),
    source.covs.raw(),
    num_source,
    num_target,
    d_T,
    max_dist_sq,
    inv_leaf,
    partials,
    inlier_count,
    corr_target_idx,
    corr_mahalanobis);
}

template <RegistrationFactor Factor, bool ErrorOnly>
void launch_factor(
  NNStrategy nn,
  int grid,
  int block,
  HashIndexView hidx,
  const int* pre_j,
  const float* pre_d2,
  const GpuCloud& target,
  const GpuCloud& source,
  int num_source,
  int num_target,
  const float* d_T,
  float max_dist_sq,
  float inv_leaf,
  double* partials,
  unsigned int* inlier_count,
  int* corr_target_idx,
  float* corr_mahalanobis) {
  switch (nn) {
    case NNStrategy::Voxel3:
      launch_factor_kernel<Factor, NNStrategy::Voxel3, ErrorOnly>(
        grid,
        block,
        hidx,
        pre_j,
        pre_d2,
        target,
        source,
        num_source,
        num_target,
        d_T,
        max_dist_sq,
        inv_leaf,
        partials,
        inlier_count,
        corr_target_idx,
        corr_mahalanobis);
      break;
    case NNStrategy::Voxel5:
      launch_factor_kernel<Factor, NNStrategy::Voxel5, ErrorOnly>(
        grid,
        block,
        hidx,
        pre_j,
        pre_d2,
        target,
        source,
        num_source,
        num_target,
        d_T,
        max_dist_sq,
        inv_leaf,
        partials,
        inlier_count,
        corr_target_idx,
        corr_mahalanobis);
      break;
    case NNStrategy::ExactBF:
      launch_factor_kernel<Factor, NNStrategy::ExactBF, ErrorOnly>(
        grid,
        block,
        hidx,
        pre_j,
        pre_d2,
        target,
        source,
        num_source,
        num_target,
        d_T,
        max_dist_sq,
        inv_leaf,
        partials,
        inlier_count,
        corr_target_idx,
        corr_mahalanobis);
      break;
  }
}

}  // namespace

size_t Linearizer::linearize_and_reduce(
  const GpuCloud& target,
  const GpuCloud& source,
  const Eigen::Isometry3d& T,
  double max_dist_sq,
  NNStrategy nn,
  float leaf_size,
  RegistrationFactor factor,
  CorrCache& cache,
  double* h_out) {
  const int num_source = static_cast<int>(source.size());
  const int num_target = static_cast<int>(target.size());
  if (num_source == 0 || num_target == 0) {
    for (int k = 0; k < NUM_OUT; k++) {
      h_out[k] = 0.0;
    }
    return 0;
  }

  prepare(num_source);
  cache.resize(num_source, factor == RegistrationFactor::GICP);
  upload_T(d_T_, T);

  unsigned int zero = 0;
  inlier_count_.upload(&zero, 1);

  const int grid = (num_source + block_ - 1) / block_;
  const float inv_leaf = 1.0f / leaf_size;
  const HashIndexView hidx = target.index.view();

  const int* pre_j = nullptr;
  const float* pre_d2 = nullptr;
  if (nn == NNStrategy::Voxel3 || nn == NNStrategy::Voxel5) {
    // Warp-cooperative NN pass: one warp per query with pruned sphere expansion
    constexpr int NN_BLOCK = 256;
    const int nn_grid = (num_source + 7) / 8;
    nn_batch_kernel<<<nn_grid, NN_BLOCK>>>(
      hidx,
      target.keys.raw(),
      num_target,
      target.points.raw(),
      source.points.raw(),
      num_source,
      d_T_.raw(),
      inv_leaf,
      max_dist_sq,
      nn_j_.raw(),
      nn_d2_.raw());
    SGC_CHECK(cudaGetLastError());
    pre_j = nn_j_.raw();
    pre_d2 = nn_d2_.raw();
  }

  switch (factor) {
    case RegistrationFactor::ICP:
      launch_factor<RegistrationFactor::ICP, false>(
        nn,
        grid,
        block_,
        hidx,
        pre_j,
        pre_d2,
        target,
        source,
        num_source,
        num_target,
        d_T_.raw(),
        max_dist_sq,
        inv_leaf,
        partials_.raw(),
        inlier_count_.raw(),
        cache.target_idx.raw(),
        nullptr);
      break;
    case RegistrationFactor::PointToPlaneICP:
      launch_factor<RegistrationFactor::PointToPlaneICP, false>(
        nn,
        grid,
        block_,
        hidx,
        pre_j,
        pre_d2,
        target,
        source,
        num_source,
        num_target,
        d_T_.raw(),
        max_dist_sq,
        inv_leaf,
        partials_.raw(),
        inlier_count_.raw(),
        cache.target_idx.raw(),
        nullptr);
      break;
    case RegistrationFactor::GICP:
      launch_factor<RegistrationFactor::GICP, false>(
        nn,
        grid,
        block_,
        hidx,
        pre_j,
        pre_d2,
        target,
        source,
        num_source,
        num_target,
        d_T_.raw(),
        max_dist_sq,
        inv_leaf,
        partials_.raw(),
        inlier_count_.raw(),
        cache.target_idx.raw(),
        cache.mahalanobis.raw());
      break;
  }
  SGC_CHECK(cudaGetLastError());

  if (factor == RegistrationFactor::ICP) {
    // On Jetson, lightweight ICP finishes faster without an extra final-reduction launch.
    // Retain and reuse the host staging vector so this path does not allocate per iteration.
    host_partials_.resize(num_warps_ * NUM_OUT);
    partials_.download(host_partials_.data(), host_partials_.size());
    for (int k = 0; k < NUM_OUT; k++) {
      h_out[k] = 0.0;
    }
    for (size_t w = 0; w < num_warps_; w++) {
      for (int k = 0; k < NUM_OUT; k++) {
        h_out[k] += host_partials_[w * NUM_OUT + k];
      }
    }
  } else {
    reduce_partials_kernel<<<1, 64>>>(partials_.raw(), num_warps_, reduced_out_.raw());
    SGC_CHECK(cudaGetLastError());
    reduced_out_.download(h_out, NUM_OUT);
  }
  // Symmetrize H from the upper triangle
  for (int r = 0; r < 6; r++) {
    for (int c = 0; c < r; c++) {
      h_out[r * 6 + c] = h_out[c * 6 + r];
    }
  }

  unsigned int inliers = 0;
  inlier_count_.download(&inliers, 1);
  return inliers;
}

double Linearizer::eval_error_cached(const GpuCloud& target, const GpuCloud& source, const Eigen::Isometry3d& T, RegistrationFactor factor, const CorrCache& cache) {
  const int num_source = static_cast<int>(source.size());
  if (num_source == 0) {
    return 0.0;
  }

  prepare(num_source);
  upload_T(d_T_, T);

  const int grid = (num_source + block_ - 1) / block_;
  const HashIndexView hidx = target.index.view();
  switch (factor) {
    case RegistrationFactor::ICP:
      launch_factor<RegistrationFactor::ICP, true>(
        NNStrategy::Voxel3,
        grid,
        block_,
        hidx,
        nullptr,
        nullptr,
        target,
        source,
        num_source,
        0,
        d_T_.raw(),
        0.0f,
        0.0f,
        partials_.raw(),
        inlier_count_.raw(),
        const_cast<int*>(cache.target_idx.raw()),
        nullptr);
      break;
    case RegistrationFactor::PointToPlaneICP:
      launch_factor<RegistrationFactor::PointToPlaneICP, true>(
        NNStrategy::Voxel3,
        grid,
        block_,
        hidx,
        nullptr,
        nullptr,
        target,
        source,
        num_source,
        0,
        d_T_.raw(),
        0.0f,
        0.0f,
        partials_.raw(),
        inlier_count_.raw(),
        const_cast<int*>(cache.target_idx.raw()),
        nullptr);
      break;
    case RegistrationFactor::GICP:
      launch_factor<RegistrationFactor::GICP, true>(
        NNStrategy::Voxel3,
        grid,
        block_,
        hidx,
        nullptr,
        nullptr,
        target,
        source,
        num_source,
        0,
        d_T_.raw(),
        0.0f,
        0.0f,
        partials_.raw(),
        inlier_count_.raw(),
        const_cast<int*>(cache.target_idx.raw()),
        const_cast<float*>(cache.mahalanobis.raw()));
      break;
  }
  SGC_CHECK(cudaGetLastError());

  double e = 0.0;
  if (factor == RegistrationFactor::ICP) {
    host_partials_.resize(num_warps_ * NUM_OUT);
    partials_.download(host_partials_.data(), host_partials_.size());
    for (size_t w = 0; w < num_warps_; w++) {
      e += host_partials_[w * NUM_OUT + 42];
    }
  } else {
    reduce_error_kernel<<<1, 1>>>(partials_.raw(), num_warps_, error_out_.raw());
    SGC_CHECK(cudaGetLastError());
    error_out_.download(&e, 1);
  }
  return e;
}

}  // namespace sgc
