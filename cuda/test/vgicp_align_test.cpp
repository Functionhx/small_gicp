// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/voxel/voxel_hash_map.hpp>
#include <sgc/reg/vgicp.hpp>

#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

class VgicpAlignTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);

    delta = Eigen::Translation<double, 3>(0.3, -0.2, 0.15) * Eigen::AngleAxisd(1.5 * M_PI / 180.0, Eigen::Vector3d::UnitZ());

    std::vector<Eigen::Vector4f> source_raw(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
      source_raw[i] = (delta * raw[i].cast<double>()).cast<float>();
    }

    const double voxel_resolution = 1.0;

    // GPU
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

    // Upstream map built from the target (identity), matching the benchmark engine's first frame
    uref_map = std::make_shared<small_gicp::GaussianVoxelMap>(voxel_resolution);
    uref_map->insert(*ref_target);

    gmap = std::make_unique<sgc::VoxelHashMap>(voxel_resolution);
    gmap->insert(gpu_target, Eigen::Isometry3d::Identity());
    ASSERT_EQ(gmap->num_voxels(), uref_map->size());

  }

  std::vector<Eigen::Vector4f> raw;
  Eigen::Isometry3d delta{Eigen::Isometry3d::Identity()};
  sgc::GpuCloud gpu_target, gpu_source;
  small_gicp::PointCloud::Ptr ref_target, ref_source;
  std::shared_ptr<small_gicp::GaussianVoxelMap> uref_map;
  std::unique_ptr<sgc::VoxelHashMap> gmap;
};

}  // namespace

TEST_F(VgicpAlignTest, SyntheticPairParity) {
  // Upstream: align(map, source, map, init)
  small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> registration;
  const auto ref = registration.align(*uref_map, *ref_source, *uref_map, Eigen::Isometry3d::Identity());

  // GPU
  sgc::VgicpGpu gpu_reg;
  const auto res = gpu_reg.align(*gmap, gpu_source, Eigen::Isometry3d::Identity());

  const double tdiff = (res.T_target_source.translation() - ref.T_target_source.translation()).norm();
  const double rdiff = Eigen::AngleAxisd(res.T_target_source.linear() * ref.T_target_source.linear().transpose()).angle() * 180.0 / M_PI;
  EXPECT_LT(tdiff, 1e-2);
  EXPECT_LT(rdiff, 0.3);
}
