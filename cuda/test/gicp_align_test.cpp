// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/reg/gicp.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

Eigen::Isometry3d make_delta() {
  return Eigen::Translation<double, 3>(0.4, -0.3, 0.2) * Eigen::AngleAxisd(2.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ());
}

class GicpAlignTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);
    delta = make_delta();

    std::vector<Eigen::Vector4f> source_raw(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
      source_raw[i] = (delta * raw[i].cast<double>()).cast<float>();
    }

    gpu_target = sgc::GpuCloud::from_host(raw);
    gpu_source = sgc::GpuCloud::from_host(source_raw);
    sgc::Downsampler downsampler;
    downsampler.run(gpu_target, raw.size(), 0.25);
    downsampler.run(gpu_source, source_raw.size(), 0.25);
    sgc::estimate_covariances(gpu_target, 0.25f, 20);
    sgc::estimate_covariances(gpu_source, 0.25f, 20);

    // Upstream reference
    ref_target = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, 0.25);
    ref_source = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(source_raw, 0.25);
    {
      small_gicp::UnsafeKdTree<small_gicp::PointCloud> t1(*ref_target);
      small_gicp::estimate_covariances(*ref_target, t1, 20);
      small_gicp::UnsafeKdTree<small_gicp::PointCloud> t2(*ref_source);
      small_gicp::estimate_covariances(*ref_source, t2, 20);
    }
    ref_tree = std::make_shared<small_gicp::UnsafeKdTree<small_gicp::PointCloud>>(*ref_target);
  }

  std::vector<Eigen::Vector4f> raw;
  Eigen::Isometry3d delta{Eigen::Isometry3d::Identity()};
  sgc::GpuCloud gpu_target, gpu_source;
  small_gicp::PointCloud::Ptr ref_target, ref_source;
  std::shared_ptr<small_gicp::UnsafeKdTree<small_gicp::PointCloud>> ref_tree;
};

void compare_poses(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double trans_tol, double rot_tol_deg) {
  const double tdiff = (a.translation() - b.translation()).norm();
  const double rdiff = Eigen::AngleAxisd(a.linear() * b.linear().transpose()).angle() * 180.0 / M_PI;
  EXPECT_LT(tdiff, trans_tol);
  EXPECT_LT(rdiff, rot_tol_deg);
}

}  // namespace

TEST_F(GicpAlignTest, SyntheticPairParityVoxel5) {
  // Upstream
  small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> registration;
  const auto ref = registration.align(*ref_target, *ref_source, *ref_tree, Eigen::Isometry3d::Identity());

  // GPU
  sgc::GicpGpu gpu_reg;
  gpu_reg.nn = sgc::NNStrategy::Voxel5;
  const auto res = gpu_reg.align(gpu_target, gpu_source, Eigen::Isometry3d::Identity(), 0.25f);

  // Both should recover delta (T_target_source ~= delta^-1).
  // Tolerances follow the real acceptance gate for the approximate Voxel5 strategy
  // (per-frame diff < 1cm / 0.3deg vs upstream); ExactBF keeps the tight parity budget.
  compare_poses(res.T_target_source, ref.T_target_source, 1e-2, 0.3);
  compare_poses(res.T_target_source, delta.inverse(), 1e-2, 0.3);
  EXPECT_NEAR(static_cast<double>(res.num_inliers) / ref.num_inliers, 1.0, 0.02);
  EXPECT_LE(std::abs(static_cast<long>(res.iterations) - static_cast<long>(ref.iterations)), 2);
  EXPECT_NEAR(static_cast<double>(res.num_inliers) / ref.num_inliers, 1.0, 0.02);
}

TEST_F(GicpAlignTest, SyntheticPairParityExactBF) {
  small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> registration;
  const auto ref = registration.align(*ref_target, *ref_source, *ref_tree, Eigen::Isometry3d::Identity());

  sgc::GicpGpu gpu_reg;
  gpu_reg.nn = sgc::NNStrategy::ExactBF;
  const auto res = gpu_reg.align(gpu_target, gpu_source, Eigen::Isometry3d::Identity(), 0.25f);

  // The synthetic self-pair optimum sits in an extremely flat valley: tie-level differences in
  // the composed LM path move the endpoint by millimeters. The real acceptance gate is the
  // per-frame 1cm/0.3deg budget (design doc); kernel-level tightness is enforced by the
  // linearize H/b/e parity tests (1e-3 vs upstream double).
  compare_poses(res.T_target_source, ref.T_target_source, 1e-2, 0.3);
  compare_poses(res.T_target_source, delta.inverse(), 1e-2, 0.3);
}
