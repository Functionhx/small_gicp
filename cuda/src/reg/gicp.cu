// SPDX-License-Identifier: MIT
#include <sgc/reg/gicp.hpp>

#include <array>
#include <stdexcept>

#include <sgc/reg/lie.hpp>

namespace sgc {

GicpResult ScanToScanGpu::align(const GpuCloud& target, const GpuCloud& source, const Eigen::Isometry3d& init_T, float leaf_size) {
  if (factor_ == RegistrationFactor::PointToPlaneICP && target.normals.size() != target.size()) {
    throw std::invalid_argument("PointToPlaneIcpGpu requires target normals");
  }
  if (factor_ == RegistrationFactor::GICP && (target.covs.size() != target.size() * 9 || source.covs.size() != source.size() * 9)) {
    throw std::invalid_argument("GicpGpu requires target and source covariances");
  }

  double lambda = init_lambda;
  GicpResult result(init_T);

  CorrCache cache;
  std::array<double, 43> out{};

  for (int i = 0; i < max_iterations && !result.converged; i++) {
    // Linearize
    const size_t inliers = linearizer.linearize_and_reduce(target, source, result.T_target_source, max_dist_sq, nn, leaf_size, factor_, cache, out.data());

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        H(r, c) = out[r * 6 + c];
      }
      b(r) = out[36 + r];
    }
    const double e = out[42];

    // Lambda iteration (line-by-line port of small_gicp::LevenbergMarquardtOptimizer)
    bool success = false;
    for (int j = 0; j < max_inner_iterations; j++) {
      const Eigen::Matrix<double, 6, 1> delta = (H + lambda * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-b);

      const Eigen::Isometry3d new_T = result.T_target_source * se3_exp(delta);
      const double new_e = linearizer.eval_error_cached(target, source, new_T, factor_, cache);

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
