// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include <sgc/io/kitti.hpp>
#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>

TEST(IO, PlyRoundtrip) {
  const auto pts = sgc::io::read_ply("data/target.ply");
  ASSERT_GT(pts.size(), 10000u);

  sgc::GpuCloud cloud = sgc::GpuCloud::from_host(pts);
  EXPECT_EQ(cloud.size(), 0u);  // keys are empty before preprocessing

  const auto back = cloud.download_points();
  ASSERT_EQ(back.size(), pts.size());
  for (size_t i = 0; i < pts.size(); i += 997) {
    EXPECT_NEAR(back[i].x(), pts[i].x(), 1e-6f);
    EXPECT_NEAR(back[i].y(), pts[i].y(), 1e-6f);
    EXPECT_NEAR(back[i].z(), pts[i].z(), 1e-6f);
    EXPECT_NEAR(back[i].w(), 1.0f, 1e-6f);
  }
}

TEST(IO, KittilBinReader) {
  const std::string tmp = "/tmp/sgc_test.bin";
  {
    std::vector<float> v{1.f, 2.f, 3.f, 1.f, 4.f, 5.f, 6.f, 1.f};
    FILE* f = fopen(tmp.c_str(), "wb");
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
  }

  const auto pts = sgc::io::read_kitti_bin(tmp);
  ASSERT_EQ(pts.size(), 2u);
  EXPECT_EQ(pts[0].x(), 1.f);
  EXPECT_EQ(pts[1].y(), 5.f);
  EXPECT_EQ(pts[1].w(), 1.f);

  remove(tmp.c_str());
}
