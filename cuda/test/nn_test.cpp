// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/search/nn_search.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>

namespace {

class NNTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);

    target = sgc::GpuCloud::from_host(raw);
    sgc::Downsampler downsampler;
    downsampler.run(target, raw.size(), 0.25);
    ASSERT_GT(target.size(), 1000u);

    // Upstream kd-tree reference
    ref = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, 0.25);
    tree = std::make_unique<small_gicp::UnsafeKdTree<small_gicp::PointCloud>>(*ref);

    queries = target.download_points();
  }

  // Runs a GPU strategy and returns the mismatch rate among inlier-range queries (kd dist^2 <= 1.0)
  double run_and_check(sgc::NNStrategy strategy, double& exact_same_index_rate) {
    const size_t n = queries.size();
    std::vector<sgc::GpuBuffer<float4>> q(1);
    q[0].resize(n);
    {
      std::vector<float4> tmp(n);
      for (size_t i = 0; i < n; i++) {
        tmp[i] = make_float4(queries[i].x(), queries[i].y(), queries[i].z(), 1.0f);
      }
      q[0].upload(tmp.data(), n);
    }
    sgc::GpuBuffer<int> out_idx(n);
    sgc::GpuBuffer<float> out_d2(n);
    sgc::nn_search(target, q[0], out_idx, out_d2, strategy, 0.25f);

    std::vector<int> h_idx(n);
    std::vector<float> h_d2(n);
    out_idx.download(h_idx.data(), n);
    out_d2.download(h_d2.data(), n);

    size_t inlier = 0, mismatch = 0, same_index = 0;
    for (size_t i = 0; i < n; i++) {
      size_t ki;
      double kd;
      tree->nearest_neighbor_search(queries[i].cast<double>(), &ki, &kd);
      if (kd > 1.0) {
        continue;  // would be rejected by DistanceRejector anyway
      }
      inlier++;
      if (h_idx[i] >= 0) {
        const double rel = std::abs(static_cast<double>(h_d2[i]) - kd) / kd;
        // Match = same found point index OR essentially equal distance
        const bool match = (static_cast<size_t>(h_idx[i]) == ki) || rel <= 1e-3;
        if (static_cast<size_t>(h_idx[i]) == ki) {
          same_index++;
        }
        if (!match) {
          mismatch++;
        }
      } else {
        mismatch++;  // GPU found nothing while kd-tree found an inlier
      }
    }
    exact_same_index_rate = static_cast<double>(same_index) / inlier;
    return static_cast<double>(mismatch) / inlier;
  }

  std::vector<Eigen::Vector4f> raw;
  std::vector<Eigen::Vector4f> queries;
  sgc::GpuCloud target;
  small_gicp::PointCloud::Ptr ref;
  std::unique_ptr<small_gicp::UnsafeKdTree<small_gicp::PointCloud>> tree;
};

TEST_F(NNTest, ExactBFMatchesKdTree) {
  double same_index_rate = 0.0;
  const double mismatch = run_and_check(sgc::NNStrategy::ExactBF, same_index_rate);
  EXPECT_LT(mismatch, 1e-4);
  EXPECT_GT(same_index_rate, 0.999);
}

TEST_F(NNTest, Voxel3Agreement) {
  double same_index_rate = 0.0;
  const double mismatch = run_and_check(sgc::NNStrategy::Voxel3, same_index_rate);
  EXPECT_LT(mismatch, 0.01);
}

TEST_F(NNTest, Voxel5Agreement) {
  double same_index_rate = 0.0;
  const double mismatch = run_and_check(sgc::NNStrategy::Voxel5, same_index_rate);
  EXPECT_LT(mismatch, 0.002);
}

}  // namespace
