// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

class CovarianceTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    ASSERT_GT(raw.size(), 10000u);

    gpu = sgc::GpuCloud::from_host(raw);
    sgc::Downsampler downsampler;
    downsampler.run(gpu, raw.size(), 0.25);
    ASSERT_GT(gpu.size(), 1000u);

    ref = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(raw, 0.25);
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(*ref);
    small_gicp::estimate_covariances(*ref, tree, 20);
  }

  // Returns the median relative error and fills all relative errors (sorted).
  double compare(std::vector<double>& rels) {
    const auto covs = gpu.download_covs();
    EXPECT_EQ(covs.size(), gpu.size() * 9);
    rels.clear();
    rels.reserve(ref->size());
    for (size_t i = 0; i < ref->size(); i++) {
      Eigen::Matrix3f gpu_cov;
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          gpu_cov(r, c) = covs[i * 9 + r * 3 + c];
        }
      }
      const Eigen::Matrix3f ref_cov = ref->covs[i].cast<float>().topLeftCorner<3, 3>();
      rels.push_back((gpu_cov - ref_cov).norm() / ref_cov.norm());
    }
    std::sort(rels.begin(), rels.end());
    return rels[rels.size() / 2];
  }

  std::vector<Eigen::Vector4f> raw;
  sgc::GpuCloud gpu;
  small_gicp::PointCloud::Ptr ref;
};

TEST_F(CovarianceTest, ParityWithUpstream) {
  sgc::estimate_covariances(gpu, 0.25f, 20);
  std::vector<double> rels;
  const double median = compare(rels);
  EXPECT_LT(median, 1e-4);
  EXPECT_LT(rels[rels.size() * 99 / 100], 1e-3);   // p99
  // A handful of points hit exact distance ties at the k-th boundary, swapping one neighbor.
  // Identical neighbor sets are proven to reproduce upstream covariances; see WORKLOG.
  EXPECT_LT(rels.back(), 5e-2);                    // max (isolated tie outliers only)
  EXPECT_LT(rels[rels.size() * 999 / 1000], 5e-3); // p99.9
}

TEST_F(CovarianceTest, DeterministicAcrossRuns) {
  sgc::estimate_covariances(gpu, 0.25f, 20);
  const auto first = gpu.download_covs();
  for (int r = 0; r < 3; r++) {
    sgc::estimate_covariances(gpu, 0.25f, 20);
    const auto again = gpu.download_covs();
    EXPECT_EQ(again, first);
  }
}

}  // namespace
