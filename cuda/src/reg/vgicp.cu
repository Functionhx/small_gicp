// SPDX-License-Identifier: MIT
#include <sgc/reg/vgicp.hpp>

#include <array>

#include <sgc/core/check.hpp>
#include <sgc/factor/gicp_math.cuh>
#include <sgc/reg/lie.hpp>
#include <sgc/voxel/voxel_key.hpp>

namespace sgc {

namespace {

constexpr int BLOCK = 256;
constexpr int NUM_OUT = 43;
constexpr unsigned long long EMPTY = 0xFFFFFFFFFFFFFFFFull;

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

// Fused per-point VGICP linearization against the incremental voxel map:
// center-voxel probe -> mahalanobis -> H, b, e; deterministic warp reduction as in Linearizer.
template <bool ErrorOnly>
__global__ void vgicp_linearize_kernel(
  unsigned int table_mask,
  const unsigned long long* table_key,
  const unsigned int* table_slot,
  const float4* map_mean,
  const float* map_cov,
  float inv_leaf,
  const float4* source_pts,
  const float* source_covs,
  int num_source,
  const float* T,  // row-major 4x4
  float max_dist_sq,
  double* partials,
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
    const V3 q{
      r0.x * ps.x + r0.y * ps.y + r0.z * ps.z + r0.w,  //
      r1.x * ps.x + r1.y * ps.y + r1.z * ps.z + r1.w,  //
      r2.x * ps.x + r2.y * ps.y + r2.z * ps.z + r2.w};

    int slot = -1;

    if (ErrorOnly) {
      slot = corr_target_idx[i];
    } else {
      // Center-voxel single probe
      const unsigned long long key = voxel_key(make_float4(q.x, q.y, q.z, 1.0f), inv_leaf);
      if (key != EMPTY) {
        unsigned int pos = voxel_hash(key, table_mask);
        while (true) {
          const unsigned long long k = table_key[pos];
          if (k == key) {
            slot = static_cast<int>(table_slot[pos]);
            break;
          }
          if (k == EMPTY) {
            break;
          }
          pos = (pos + 1) & table_mask;
        }
      }
    }

    if (slot >= 0) {
      const float4 pt = map_mean[slot];
      const float dx = pt.x - q.x, dy = pt.y - q.y, dz = pt.z - q.z;
      const float d2 = dx * dx + dy * dy + dz * dz;

      if (ErrorOnly || d2 <= max_dist_sq) {
        M3 M;
        if (ErrorOnly) {
#pragma unroll
          for (int k = 0; k < 9; k++) {
            M.m[k] = corr_mahalanobis[i * 9 + k];
          }
        } else {
          M3 Ct, Cs;
#pragma unroll
          for (int k = 0; k < 9; k++) {
            Ct.m[k] = map_cov[slot * 9 + k];
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
          M = inv3(Ct + RCsRt);
#pragma unroll
          for (int k = 0; k < 9; k++) {
            corr_mahalanobis[i * 9 + k] = M.m[k];
          }
          corr_target_idx[i] = slot;
        }

        const V3 res{dx, dy, dz};
        const V3 p{ps.x, ps.y, ps.z};
        const V3 J0{r0.y * p.z - r0.z * p.y, r1.y * p.z - r1.z * p.y, r2.y * p.z - r2.z * p.y};
        const V3 J1{r0.z * p.x - r0.x * p.z, r1.z * p.x - r1.x * p.z, r2.z * p.x - r2.x * p.z};
        const V3 J2{r0.x * p.y - r0.y * p.x, r1.x * p.y - r1.y * p.x, r2.x * p.y - r2.y * p.x};
        const V3 J3{-r0.x, -r1.x, -r2.x};
        const V3 J4{-r0.y, -r1.y, -r2.y};
        const V3 J5{-r0.z, -r1.z, -r2.z};

        const V3 Mr = M * res;
        const V3 MJ0 = M * J0, MJ1 = M * J1, MJ2 = M * J2, MJ3 = M * J3, MJ4 = M * J4, MJ5 = M * J5;

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
        acc[36] = dot(J0, Mr);
        acc[37] = dot(J1, Mr);
        acc[38] = dot(J2, Mr);
        acc[39] = dot(J3, Mr);
        acc[40] = dot(J4, Mr);
        acc[41] = dot(J5, Mr);
        acc[42] = 0.5f * dot(res, Mr);

        inlier = true;
        if (ErrorOnly) {
          inlier = false;  // error pass does not recount
        }
      } else if (!ErrorOnly) {
        corr_target_idx[i] = -1;
      }
    } else if (!ErrorOnly) {
      corr_target_idx[i] = -1;
    }
  }

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

}  // namespace

GicpResult VgicpGpu::align(const VoxelHashMap& map, const GpuCloud& source, const Eigen::Isometry3d& init_T) {
  double lambda = init_lambda;
  GicpResult result(init_T);

  const int num_source = static_cast<int>(source.size());
  if (num_source == 0 || map.num_voxels() == 0) {
    return result;
  }

  if (d_T_.size() != 16) {
    d_T_.resize(16);
  }
  // The kernel launches ceil(n / BLOCK) blocks; every launched warp (including the tail
  // warps that only cover out-of-range threads) writes its partial slot. Size accordingly.
  const int grid = (num_source + BLOCK - 1) / BLOCK;
  const size_t num_warps = static_cast<size_t>(grid) * (BLOCK / 32);
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
  cache_.resize(num_source, true);
  // Initialize the correspondence cache on-device so any unwritten slot reads as no match.
  SGC_CHECK(cudaMemsetAsync(cache_.target_idx.raw(), 0xFF, static_cast<size_t>(num_source) * sizeof(int)));

  const float inv_leaf = 1.0f / static_cast<float>(map.leaf_size());
  std::array<double, NUM_OUT> out{};

  for (int i = 0; i < max_iterations && !result.converged; i++) {
    upload_T(d_T_, result.T_target_source);
    unsigned int zero = 0;
    inlier_count_.upload(&zero, 1);

    vgicp_linearize_kernel<false><<<grid, BLOCK>>>(
      map.mask(),
      map.table_key.raw(),
      map.table_slot.raw(),
      map.mean.raw(),
      map.cov.raw(),
      inv_leaf,
      source.points.raw(),
      source.covs.raw(),
      num_source,
      d_T_.raw(),
      max_dist_sq,
      partials_.raw(),
      inlier_count_.raw(),
      cache_.target_idx.raw(),
      cache_.mahalanobis.raw());
    SGC_CHECK(cudaGetLastError());

    reduce_partials_kernel<<<1, 64>>>(partials_.raw(), num_warps, reduced_out_.raw());
    SGC_CHECK(cudaGetLastError());
    reduced_out_.download(out.data(), NUM_OUT);
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < r; c++) {
        out[r * 6 + c] = out[c * 6 + r];
      }
    }
    unsigned int inliers = 0;
    inlier_count_.download(&inliers, 1);

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        H(r, c) = out[r * 6 + c];
      }
      b(r) = out[36 + r];
    }
    const double e = out[42];

    bool success = false;
    for (int j = 0; j < max_inner_iterations; j++) {
      const Eigen::Matrix<double, 6, 1> delta = (H + lambda * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-b);

      const Eigen::Isometry3d new_T = result.T_target_source * se3_exp(delta);
      upload_T(d_T_, new_T);
      vgicp_linearize_kernel<true><<<grid, BLOCK>>>(
        map.mask(),
        map.table_key.raw(),
        map.table_slot.raw(),
        map.mean.raw(),
        map.cov.raw(),
        inv_leaf,
        source.points.raw(),
        source.covs.raw(),
        num_source,
        d_T_.raw(),
        0.0f,
        partials_.raw(),
        inlier_count_.raw(),
        cache_.target_idx.raw(),
        cache_.mahalanobis.raw());
      SGC_CHECK(cudaGetLastError());

      reduce_error_kernel<<<1, 1>>>(partials_.raw(), num_warps, error_out_.raw());
      SGC_CHECK(cudaGetLastError());
      double new_e = 0.0;
      error_out_.download(&new_e, 1);

      if (new_e <= e) {
        result.converged = converged(delta);
        result.T_target_source = new_T;
        lambda /= lambda_factor;
        success = true;
        break;
      } else {
        lambda *= lambda_factor;
      }
    }

    result.iterations = i;
    result.H = H;
    result.b = b;
    result.error = e;
    result.num_inliers = inliers;

    if (!success) {
      break;
    }
  }

  return result;
}

}  // namespace sgc
