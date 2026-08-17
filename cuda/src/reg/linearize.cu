// SPDX-License-Identifier: MIT
#include <sgc/reg/reduction.hpp>

#include <vector>

#include <sgc/core/check.hpp>
#include <sgc/factor/gicp_math.cuh>
#include <sgc/search/nn_query.cuh>
#include <sgc/voxel/hash_index.hpp>

namespace sgc {

namespace {

constexpr int BLOCK = 256;
constexpr int NUM_OUT = 43;  // H(36) + b(6) + e(1)

// Fused per-point GICP linearization: transform -> NN -> mahalanobis -> H, b, e per point,
// then a deterministic warp shuffle reduction (fp32) with fp64 partials at lane 0.
template <NNStrategy Strategy, bool ErrorOnly>
__global__ void linearize_kernel(
  HashIndexView hidx,
  const float4* target_pts,
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
    const V3 q{r0.x * ps.x + r0.y * ps.y + r0.z * ps.z + r0.w,  //
               r1.x * ps.x + r1.y * ps.y + r1.z * ps.z + r1.w,  //
               r2.x * ps.x + r2.y * ps.y + r2.z * ps.z + r2.w};

    int j = -1;

    if (ErrorOnly) {
      // Error re-evaluation with cached correspondences (no NN search)
      j = corr_target_idx[i];
      if (j >= 0) {
        M3 M;
#pragma unroll
        for (int k = 0; k < 9; k++) {
          M.m[k] = corr_mahalanobis[i * 9 + k];
        }
        const float4 pt = target_pts[j];
        const V3 res{pt.x - q.x, pt.y - q.y, pt.z - q.z};
        acc[42] = 0.5f * dot(res, M * res);
      }
    } else {
      float d2 = 0.0f;
      nn_query<Strategy>(hidx, target_keys, num_target, target_pts, make_float4(q.x, q.y, q.z, 1.0f), inv_leaf, &j, &d2);

      if (j >= 0 && d2 <= max_dist_sq) {
        // M = (Ct + R * Cs * R^T)^-1
        M3 Ct, Cs;
#pragma unroll
        for (int k = 0; k < 9; k++) {
          Ct.m[k] = target_covs[j * 9 + k];
          Cs.m[k] = source_covs[i * 9 + k];
        }
        const M3 R{r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z};
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

        const M3 M = inv3(Ct + RCsRt);

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
#pragma unroll
        for (int k = 0; k < 9; k++) {
          corr_mahalanobis[i * 9 + k] = M.m[k];
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
  const size_t num_warps = (num_source + 31) / 32;
  if (num_warps_ != num_warps) {
    partials_.resize(num_warps * NUM_OUT);
    num_warps_ = num_warps;
  }
  if (inlier_count_.size() != 1) {
    inlier_count_.resize(1);
  }
}

namespace {

void upload_T(GpuBuffer<float>& d_T, const Eigen::Isometry3d& T) {
  std::vector<float> h_T(16);
  const Eigen::Matrix4f Tf = T.matrix().cast<float>();
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      h_T[r * 4 + c] = Tf(r, c);
    }
  }
  d_T.upload(h_T.data(), 16);
}

}  // namespace

size_t Linearizer::linearize_and_reduce(
  const GpuCloud& target,
  const GpuCloud& source,
  const Eigen::Isometry3d& T,
  double max_dist_sq,
  NNStrategy nn,
  float leaf_size,
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
  cache.resize(num_source);
  upload_T(d_T_, T);

  unsigned int zero = 0;
  inlier_count_.upload(&zero, 1);

  const int grid = (num_source + block_ - 1) / block_;
  const float inv_leaf = 1.0f / leaf_size;
  const HashIndexView hidx = target.index.view();

  switch (nn) {
    case NNStrategy::Voxel3:
      linearize_kernel<NNStrategy::Voxel3, false><<<grid, block_>>>(
        hidx, target.points.raw(), target.keys.raw(), target.covs.raw(), source.points.raw(), source.covs.raw(), num_source, num_target, d_T_.raw(), max_dist_sq,
        inv_leaf, partials_.raw(), inlier_count_.raw(), cache.target_idx.raw(), cache.mahalanobis.raw());
      break;
    case NNStrategy::Voxel5:
      linearize_kernel<NNStrategy::Voxel5, false><<<grid, block_>>>(
        hidx, target.points.raw(), target.keys.raw(), target.covs.raw(), source.points.raw(), source.covs.raw(), num_source, num_target, d_T_.raw(), max_dist_sq,
        inv_leaf, partials_.raw(), inlier_count_.raw(), cache.target_idx.raw(), cache.mahalanobis.raw());
      break;
    case NNStrategy::ExactBF:
      linearize_kernel<NNStrategy::ExactBF, false><<<grid, block_>>>(
        hidx, target.points.raw(), target.keys.raw(), target.covs.raw(), source.points.raw(), source.covs.raw(), num_source, num_target, d_T_.raw(), max_dist_sq,
        inv_leaf, partials_.raw(), inlier_count_.raw(), cache.target_idx.raw(), cache.mahalanobis.raw());
      break;
  }
  SGC_CHECK(cudaGetLastError());

  std::vector<double> h_partials(num_warps_ * NUM_OUT);
  partials_.download(h_partials.data(), h_partials.size());

  for (int k = 0; k < NUM_OUT; k++) {
    h_out[k] = 0.0;
  }
  for (size_t w = 0; w < num_warps_; w++) {
    for (int k = 0; k < NUM_OUT; k++) {
      h_out[k] += h_partials[w * NUM_OUT + k];
    }
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

double Linearizer::eval_error_cached(const GpuCloud& target, const GpuCloud& source, const Eigen::Isometry3d& T, const CorrCache& cache) {
  const int num_source = static_cast<int>(source.size());
  if (num_source == 0) {
    return 0.0;
  }

  prepare(num_source);
  upload_T(d_T_, T);

  const int grid = (num_source + block_ - 1) / block_;
  const HashIndexView hidx = target.index.view();
  linearize_kernel<NNStrategy::Voxel3, true><<<grid, block_>>>(
    hidx, target.points.raw(), target.keys.raw(), target.covs.raw(), source.points.raw(), source.covs.raw(), num_source, 0, d_T_.raw(), 0.0f, 0.0f, partials_.raw(),
    inlier_count_.raw(), const_cast<int*>(cache.target_idx.raw()), const_cast<float*>(cache.mahalanobis.raw()));
  SGC_CHECK(cudaGetLastError());

  std::vector<double> h_partials(num_warps_ * NUM_OUT);
  partials_.download(h_partials.data(), h_partials.size());

  double e = 0.0;
  for (size_t w = 0; w < num_warps_; w++) {
    e += h_partials[w * NUM_OUT + 42];
  }
  return e;
}

}  // namespace sgc
