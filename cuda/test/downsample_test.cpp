// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <algorithm>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/voxel_key.hpp>
#include <sgc/voxel/downsample.hpp>

#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>

TEST(VoxelKey, PackUnpackRoundtrip) {
  const sgc::VoxelCoord c{123, -456, 789};
  const auto key = sgc::coord_key(c);
  const auto back = sgc::key_coord(key);
  EXPECT_EQ(back.x, c.x);
  EXPECT_EQ(back.y, c.y);
  EXPECT_EQ(back.z, c.z);
}

TEST(VoxelKey, FastFloorSemantics) {
  EXPECT_EQ(sgc::ffloor(1.5f), 1);
  EXPECT_EQ(sgc::ffloor(-1.5f), -2);
  EXPECT_EQ(sgc::ffloor(2.0f), 2);
  EXPECT_EQ(sgc::ffloor(-2.0f), -2);
  EXPECT_EQ(sgc::ffloor(0.9999f), 0);
}

TEST(Downsample, ParityWithUpstream) {
  const auto raw = sgc::io::read_ply("data/target.ply");
  ASSERT_GT(raw.size(), 10000u);
  const double leaf = 0.25;

  auto ref = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, leaf);
  ASSERT_GT(ref->size(), 1000u);

  sgc::GpuCloud cloud = sgc::GpuCloud::from_host(raw);
  sgc::Downsampler downsampler;
  downsampler.run(cloud, raw.size(), leaf);

  auto out = cloud.download_points();
  auto keys = cloud.download_keys();

  ASSERT_EQ(out.size(), ref->size());
  ASSERT_EQ(keys.size(), out.size());

  for (size_t i = 1; i < keys.size(); i++) {
    ASSERT_LT(keys[i - 1], keys[i]);
  }

  for (size_t i = 0; i < out.size(); i += 57) {
    EXPECT_NEAR(out[i].x(), ref->points[i].x(), 1e-4);
    EXPECT_NEAR(out[i].y(), ref->points[i].y(), 1e-4);
    EXPECT_NEAR(out[i].z(), ref->points[i].z(), 1e-4);
  }
}
