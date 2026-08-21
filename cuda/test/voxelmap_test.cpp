// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <map>
#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/voxel/voxel_hash_map.hpp>

#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

// Dump the GPU map into a host (key -> (mean, cov, count)) map.
struct GpuMapDump {
  std::map<unsigned long long, Eigen::Vector4f> mean;
  std::map<unsigned long long, Eigen::Matrix3f> cov;
  std::map<unsigned long long, unsigned int> count;
};

GpuMapDump dump_gpu_map(const sgc::VoxelHashMap& map) {
  const size_t cap = map.table_key.size();
  std::vector<unsigned long long> h_keys(cap);
  std::vector<float4> h_mean(cap);
  std::vector<float> h_cov(cap * 9);
  std::vector<unsigned int> h_count(cap);
  map.table_key.download(h_keys.data(), cap);
  map.mean.download(h_mean.data(), cap);
  map.cov.download(h_cov.data(), cap * 9);
  map.count.download(h_count.data(), cap);

  GpuMapDump out;
  for (size_t i = 0; i < cap; i++) {
    if (h_keys[i] == 0xFFFFFFFFFFFFFFFFull || h_count[i] == 0) {
      continue;
    }
    out.mean[h_keys[i]] = Eigen::Vector4f(h_mean[i].x, h_mean[i].y, h_mean[i].z, 1.0f);
    Eigen::Matrix3f c;
    for (int r = 0; r < 3; r++) {
      for (int c2 = 0; c2 < 3; c2++) {
        c(r, c2) = h_cov[i * 9 + r * 3 + c2];
      }
    }
    out.cov[h_keys[i]] = c;
    out.count[h_keys[i]] = h_count[i];
  }
  return out;
}

// Dump the upstream map keyed by the same 21bit packing (valid at the same leaf size and range).
std::map<unsigned long long, std::pair<Eigen::Vector4d, Eigen::Matrix3d>> dump_upstream_map(const small_gicp::GaussianVoxelMap& map) {
  std::map<unsigned long long, std::pair<Eigen::Vector4d, Eigen::Matrix3d>> out;
  constexpr unsigned int mask21 = (1u << 21) - 1;
  for (const auto& voxel : map.flat_voxels) {
    const auto& coord = voxel->first.coord;
    const unsigned long long key = static_cast<unsigned long long>(coord.x() + (1 << 20)) |                            //
                                   (static_cast<unsigned long long>(coord.y() + (1 << 20)) << 21) |                     //
                                   (static_cast<unsigned long long>(coord.z() + (1 << 20)) << 42);
    out[key] = {voxel->second.mean, voxel->second.cov.topLeftCorner<3, 3>()};
  }
  return out;
}

class VoxelMapTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);

    gpu = sgc::GpuCloud::from_host(raw);
    sgc::Downsampler downsampler;
    downsampler.run(gpu, raw.size(), 0.5);
    sgc::estimate_covariances(gpu, 0.5f, 20);

    ref = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, 0.5);
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(*ref);
    small_gicp::estimate_covariances(*ref, tree, 20);
  }

  std::vector<Eigen::Vector4f> raw;
  sgc::GpuCloud gpu;
  small_gicp::PointCloud::Ptr ref;
};

TEST_F(VoxelMapTest, ParityWithUpstreamGaussianVoxelMap) {
  const Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
  const Eigen::Isometry3d T2 = Eigen::Translation<double, 3>(0.3, -0.2, 0.1) * Eigen::AngleAxisd(1.0 * M_PI / 180.0, Eigen::Vector3d::UnitY());

  // GPU map
  sgc::VoxelHashMap gmap(0.5);
  gmap.insert(gpu, T1);


  // Upstream map
  auto uref = std::make_shared<small_gicp::PointCloud>(*ref);
  small_gicp::GaussianVoxelMap umap(0.5);
  umap.insert(*uref);

  auto g = dump_gpu_map(gmap);
  auto u = dump_upstream_map(umap);
  ASSERT_EQ(g.count.size(), u.size());
  EXPECT_EQ(gmap.num_voxels(), u.size());

  size_t checked = 0;
  double max_mean_diff = 0.0, max_cov_rel = 0.0;
  for (const auto& [key, cnt] : g.count) {
    ASSERT_NE(u.count(key), 0u);
    const auto& [mean_d, cov_d] = u.at(key);
    const Eigen::Vector4f mean_f = g.mean.at(key);
    max_mean_diff = std::max(max_mean_diff, (mean_d - mean_f.cast<double>()).head<3>().norm());
    const Eigen::Matrix3f cov_f = g.cov.at(key);
    max_cov_rel = std::max(max_cov_rel, (cov_d - cov_f.cast<double>()).norm() / cov_d.norm());
    checked++;
  }
  EXPECT_EQ(checked, u.size());
  EXPECT_LT(max_mean_diff, 1e-3);
  EXPECT_LT(max_cov_rel, 2e-2);  // fp32 atomic sums over ~26 points/voxel vs double

  // Second insert with a non-identity transform
  gmap.insert(gpu, T2);
  umap.insert(*uref, T2);

  g = dump_gpu_map(gmap);
  u = dump_upstream_map(umap);
  EXPECT_EQ(g.count.size(), u.size());
  max_mean_diff = 0.0;
  for (const auto& [key, cnt] : g.count) {
    const auto& [mean_d, cov_d] = u.at(key);
    max_mean_diff = std::max(max_mean_diff, (mean_d - g.mean.at(key).cast<double>()).head<3>().norm());
  }
  EXPECT_LT(max_mean_diff, 2e-3);
}

TEST_F(VoxelMapTest, DeterministicAcrossRuns) {
  sgc::VoxelHashMap a(0.5), b(0.5);
  const Eigen::Isometry3d T = Eigen::Translation<double, 3>(0.1, 0.2, -0.1) * Eigen::AngleAxisd(0.5 * M_PI / 180.0, Eigen::Vector3d::UnitX());
  a.insert(gpu, T);
  a.insert(gpu, Eigen::Isometry3d::Identity());
  b.insert(gpu, T);
  b.insert(gpu, Eigen::Isometry3d::Identity());

  const auto da = dump_gpu_map(a);
  const auto db = dump_gpu_map(b);
  ASSERT_EQ(da.mean.size(), db.mean.size());
  for (const auto& [key, m] : da.mean) {
    const auto& mb = db.mean.at(key);
    EXPECT_EQ(m.x(), mb.x());
    EXPECT_EQ(m.y(), mb.y());
    EXPECT_EQ(m.z(), mb.z());
  }
}

TEST_F(VoxelMapTest, ClearRetainsStorageAndResetsState) {
  sgc::VoxelHashMap map(0.5);
  map.insert(gpu, Eigen::Isometry3d::Identity());
  ASSERT_GT(map.num_voxels(), 0u);

  auto* const table_allocation = map.table_key.raw();
  const size_t table_capacity = map.table_key.capacity();
  map.clear();
  EXPECT_EQ(map.num_voxels(), 0u);
  EXPECT_EQ(map.table_key.raw(), table_allocation);
  EXPECT_EQ(map.table_key.capacity(), table_capacity);

  map.insert(gpu, Eigen::Isometry3d::Identity());
  const auto recycled = dump_gpu_map(map);

  sgc::VoxelHashMap fresh(0.5);
  fresh.insert(gpu, Eigen::Isometry3d::Identity());
  const auto reference = dump_gpu_map(fresh);
  ASSERT_EQ(recycled.count, reference.count);
  ASSERT_EQ(recycled.mean.size(), reference.mean.size());
  for (const auto& [key, value] : recycled.mean) {
    const auto& expected = reference.mean.at(key);
    EXPECT_EQ(value.x(), expected.x());
    EXPECT_EQ(value.y(), expected.y());
    EXPECT_EQ(value.z(), expected.z());
  }
}

}  // namespace
