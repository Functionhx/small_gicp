// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/reg/reduction.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/rejector.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

// Upstream reference: sum of per-point H/b/e over the source cloud (double precision).
void reference_sum(
  const small_gicp::PointCloud& target,
  const small_gicp::PointCloud& source,
  const small_gicp::UnsafeKdTree<small_gicp::PointCloud>& tree,
  const Eigen::Isometry3d& T,
  Eigen::Matrix<double, 6, 6>& H,
  Eigen::Matrix<double, 6, 1>& b,
  double& e,
  size_t& inliers) {
  H.setZero();
  b.setZero();
  e = 0.0;
  inliers = 0;

  small_gicp::GICPFactor::Setting setting;
  small_gicp::DistanceRejector rejector;
  rejector.max_dist_sq = 1.0;

  for (size_t i = 0; i < source.size(); i++) {
    small_gicp::GICPFactor factor(setting);
    Eigen::Matrix<double, 6, 6> h;
    Eigen::Matrix<double, 6, 1> bb;
    double ee;
    if (factor.linearize(target, source, tree, T, i, rejector, &h, &bb, &ee)) {
      H += h;
      b += bb;
      e += ee;
      inliers++;
    }
  }
}

class LinearizeTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);

    // Synthetic pair: source = Delta * raw. The linearization point T is deliberately off the
    // optimum (extra perturbation) so that the gradient b is well away from zero and relative
    // error comparisons are meaningful.
    delta = Eigen::Translation<double, 3>(0.4, -0.3, 0.2) * Eigen::AngleAxisd(2.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ());
    const Eigen::Isometry3d perturb = Eigen::Translation<double, 3>(0.1, 0.05, -0.08) * Eigen::AngleAxisd(0.5 * M_PI / 180.0, Eigen::Vector3d(1, 1, 0).normalized());
    T = (delta * perturb).inverse();

    std::vector<Eigen::Vector4f> source_raw(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
      source_raw[i] = (delta * raw[i].cast<double>()).cast<float>();
    }

    // GPU clouds
    gpu_target = sgc::GpuCloud::from_host(raw);
    gpu_source = sgc::GpuCloud::from_host(source_raw);
    sgc::Downsampler downsampler;
    downsampler.run(gpu_target, raw.size(), 0.25);
    downsampler.run(gpu_source, source_raw.size(), 0.25);
    sgc::estimate_covariances(gpu_target, 0.25f, 20);
    sgc::estimate_covariances(gpu_source, 0.25f, 20);

    // Upstream reference clouds
    ref_target = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, 0.25);
    ref_source = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(source_raw, 0.25);
    {
      small_gicp::UnsafeKdTree<small_gicp::PointCloud> target_tree(*ref_target);
      small_gicp::estimate_covariances(*ref_target, target_tree, 20);
      small_gicp::UnsafeKdTree<small_gicp::PointCloud> source_tree(*ref_source);
      small_gicp::estimate_covariances(*ref_source, source_tree, 20);
    }
    tree = std::make_unique<small_gicp::UnsafeKdTree<small_gicp::PointCloud>>(*ref_target);

    reference_sum(*ref_target, *ref_source, *tree, T, ref_H, ref_b, ref_e, ref_inliers);
  }

  void check_parity(sgc::NNStrategy strategy, double h_tol, double b_tol, double e_tol, double inlier_tol) {
    sgc::Linearizer linearizer;
    sgc::CorrCache cache;
    std::vector<double> out(43);
    const size_t inliers = linearizer.linearize_and_reduce(gpu_target, gpu_source, T, 1.0, strategy, 0.25f, cache, out.data());

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        H(r, c) = out[r * 6 + c];
      }
      b(r) = out[36 + r];
    }
    const double e = out[42];

    EXPECT_LT((H - ref_H).norm() / ref_H.norm(), h_tol);
    EXPECT_LT((b - ref_b).norm() / ref_b.norm(), b_tol);
    EXPECT_LT(std::abs(e - ref_e) / std::abs(ref_e), e_tol);
    EXPECT_NEAR(static_cast<double>(inliers) / ref_inliers, 1.0, inlier_tol);

    // Self-consistency: the cached-correspondence error at the same T must equal the linearized e
    const double e_cached = linearizer.eval_error_cached(gpu_target, gpu_source, T, cache);
    EXPECT_LT(std::abs(e_cached - e) / std::abs(e), 1e-6);
  }

  std::vector<Eigen::Vector4f> raw;
  Eigen::Isometry3d delta{Eigen::Isometry3d::Identity()}, T{Eigen::Isometry3d::Identity()};
  sgc::GpuCloud gpu_target, gpu_source;
  small_gicp::PointCloud::Ptr ref_target, ref_source;
  std::unique_ptr<small_gicp::UnsafeKdTree<small_gicp::PointCloud>> tree;
  Eigen::Matrix<double, 6, 6> ref_H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> ref_b = Eigen::Matrix<double, 6, 1>::Zero();
  double ref_e = 0.0;
  size_t ref_inliers = 0;
};

TEST_F(LinearizeTest, HbEParityExactBF) {
  check_parity(sgc::NNStrategy::ExactBF, 1e-3, 1e-3, 1e-3, 1e-3);
}

TEST_F(LinearizeTest, HbEParityVoxel5) {
  check_parity(sgc::NNStrategy::Voxel5, 5e-3, 5e-3, 5e-3, 5e-3);
}

TEST_F(LinearizeTest, DeterministicAcrossRuns) {
  sgc::Linearizer linearizer;
  sgc::CorrCache cache;
  std::vector<double> first(43), again(43);
  linearizer.linearize_and_reduce(gpu_target, gpu_source, T, 1.0, sgc::NNStrategy::Voxel5, 0.25f, cache, first.data());
  for (int r = 0; r < 100; r++) {
    linearizer.linearize_and_reduce(gpu_target, gpu_source, T, 1.0, sgc::NNStrategy::Voxel5, 0.25f, cache, again.data());
    ASSERT_EQ(again, first);
  }
}

}  // namespace
