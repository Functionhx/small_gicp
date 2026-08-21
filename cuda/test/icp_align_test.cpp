// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/reg/icp.hpp>
#include <sgc/voxel/downsample.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

void compare_poses(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double trans_tol, double rot_tol_deg) {
  const double tdiff = (a.translation() - b.translation()).norm();
  const double rdiff = Eigen::AngleAxisd(a.linear() * b.linear().transpose()).angle() * 180.0 / M_PI;
  EXPECT_LT(tdiff, trans_tol);
  EXPECT_LT(rdiff, rot_tol_deg);
}

class IcpAlignTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);
    delta = Eigen::Translation<double, 3>(0.25, -0.18, 0.12) * Eigen::AngleAxisd(1.2 * M_PI / 180.0, Eigen::Vector3d::UnitZ());

    std::vector<Eigen::Vector4f> source_raw(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
      source_raw[i] = (delta * raw[i].cast<double>()).cast<float>();
    }

    gpu_target = sgc::GpuCloud::from_host(raw);
    gpu_source = sgc::GpuCloud::from_host(source_raw);
    sgc::Downsampler downsampler;
    downsampler.run(gpu_target, raw.size(), 0.25);
    downsampler.run(gpu_source, source_raw.size(), 0.25);

    ref_target = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, 0.25);
    ref_source = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(source_raw, 0.25);
    ref_tree = std::make_shared<small_gicp::UnsafeKdTree<small_gicp::PointCloud>>(*ref_target);
  }

  std::vector<Eigen::Vector4f> raw;
  Eigen::Isometry3d delta{Eigen::Isometry3d::Identity()};
  sgc::GpuCloud gpu_target, gpu_source;
  small_gicp::PointCloud::Ptr ref_target, ref_source;
  std::shared_ptr<small_gicp::UnsafeKdTree<small_gicp::PointCloud>> ref_tree;
};

}  // namespace

TEST_F(IcpAlignTest, PointToPointParityVoxel5) {
  small_gicp::Registration<small_gicp::ICPFactor, small_gicp::SerialReduction> cpu;
  const auto ref = cpu.align(*ref_target, *ref_source, *ref_tree, Eigen::Isometry3d::Identity());

  sgc::IcpGpu gpu;
  gpu.nn = sgc::NNStrategy::Voxel5;
  const auto result = gpu.align(gpu_target, gpu_source, Eigen::Isometry3d::Identity(), 0.25f);

  // Voxel5 follows a slightly different flat ICP descent path than the double-precision CPU
  // reference; ExactBF below keeps the tighter correspondence-path check.
  compare_poses(result.T_target_source, ref.T_target_source, 1.5e-2, 0.3);
  compare_poses(result.T_target_source, delta.inverse(), 1.5e-2, 0.3);
  EXPECT_NEAR(static_cast<double>(result.num_inliers) / ref.num_inliers, 1.0, 0.02);
}

TEST_F(IcpAlignTest, PointToPointParityExactBF) {
  small_gicp::Registration<small_gicp::ICPFactor, small_gicp::SerialReduction> cpu;
  const auto ref = cpu.align(*ref_target, *ref_source, *ref_tree, Eigen::Isometry3d::Identity());

  sgc::IcpGpu gpu;
  gpu.nn = sgc::NNStrategy::ExactBF;
  const auto result = gpu.align(gpu_target, gpu_source, Eigen::Isometry3d::Identity(), 0.25f);
  compare_poses(result.T_target_source, ref.T_target_source, 1e-2, 0.3);
}

TEST_F(IcpAlignTest, PointToPlaneParityVoxel5) {
  small_gicp::estimate_normals(*ref_target, *ref_tree, 20);
  sgc::estimate_normals(gpu_target, 0.25f, 20);

  small_gicp::Registration<small_gicp::PointToPlaneICPFactor, small_gicp::SerialReduction> cpu;
  const auto ref = cpu.align(*ref_target, *ref_source, *ref_tree, Eigen::Isometry3d::Identity());

  sgc::PointToPlaneIcpGpu gpu;
  gpu.nn = sgc::NNStrategy::Voxel5;
  const auto result = gpu.align(gpu_target, gpu_source, Eigen::Isometry3d::Identity(), 0.25f);

  compare_poses(result.T_target_source, ref.T_target_source, 1e-2, 0.3);
  compare_poses(result.T_target_source, delta.inverse(), 1e-2, 0.3);
  EXPECT_NEAR(static_cast<double>(result.num_inliers) / ref.num_inliers, 1.0, 0.02);
}

TEST_F(IcpAlignTest, PointToPlaneRequiresNormals) {
  sgc::PointToPlaneIcpGpu gpu;
  EXPECT_THROW(gpu.align(gpu_target, gpu_source, Eigen::Isometry3d::Identity(), 0.25f), std::invalid_argument);
}
