// SPDX-License-Identifier: MIT
// odometry_gpu: end-to-end odometry comparison harness (sgc GPU vs upstream CPU baseline).
//
// Usage:
//   odometry_gpu <kitti_velodyne_dir> [flags]     KITTI .bin sequence
//   odometry_gpu --synth <ply_path> [flags]       synthetic drifting sequence from one scan
// Flags: --exec full-gpu|hybrid|cpu --engine gicp|vgicp --nn voxel3|voxel5|exact-bf
//        --threads N --num_neighbors N --downsampling_resolution R --voxel_resolution R
//        --max_correspondence_distance R --max_frames N --report out.json --traj out.txt
//
// Timing semantics match upstream benchmark_odom.hpp: each frame's wall time includes
// downsampling + preprocessing + registration (I/O is outside the timed section).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sgc/bench/policy.hpp>
#include <sgc/io/kitti.hpp>
#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/preproc/covariance.hpp>
#include <sgc/reg/gicp.hpp>
#include <sgc/reg/vgicp.hpp>
#include <sgc/search/nn.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/voxel/voxel_hash_map.hpp>

#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace {

using Frame = std::vector<Eigen::Vector4f>;
using Clock = std::chrono::high_resolution_clock;

double msec_since(Clock::time_point t0) { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count() / 1e6; }

std::vector<double> summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const auto q = [&](double pct) { return v[std::min(v.size() - 1, static_cast<size_t>(v.size() * pct))]; };
  return {v.empty() ? 0.0 : v[0], q(0.5), q(0.95), v.empty() ? 0.0 : v.back()};
}

// ---------------------------------------------------------------- upstream CPU engines

struct CpuGicpOdometry {
  small_gicp::PointCloud::Ptr target;
  std::shared_ptr<small_gicp::UnsafeKdTree<small_gicp::PointCloud>> target_tree;

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    auto down = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(frame, p.downsampling_resolution);
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(*down);
    small_gicp::estimate_covariances(*down, tree, p.num_neighbors);

    if (target == nullptr) {
      target = down;
      target_tree = std::make_shared<small_gicp::UnsafeKdTree<small_gicp::PointCloud>>(*down);
      T_world = Eigen::Isometry3d::Identity();
      return T_world;
    }

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> reg;
    reg.rejector.max_dist_sq = p.max_correspondence_distance * p.max_correspondence_distance;
    const auto result = reg.align(*target, *down, *target_tree, Eigen::Isometry3d::Identity());

    T_world = T_world * result.T_target_source;
    target = down;
    target_tree = std::make_shared<small_gicp::UnsafeKdTree<small_gicp::PointCloud>>(*down);
    return T_world;
  }

  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();
};

struct CpuVgicpOdometry {
  std::shared_ptr<small_gicp::GaussianVoxelMap> voxelmap;

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    auto down = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(frame, p.downsampling_resolution);
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(*down);
    small_gicp::estimate_covariances(*down, tree, p.num_neighbors);

    if (voxelmap == nullptr) {
      voxelmap = std::make_shared<small_gicp::GaussianVoxelMap>(p.voxel_resolution);
      voxelmap->insert(*down);
      return Eigen::Isometry3d::Identity();
    }

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::SerialReduction> reg;
    const auto result = reg.align(*voxelmap, *down, *voxelmap, T_world);
    T_world = result.T_target_source;
    voxelmap->insert(*down, T_world);
    return T_world;
  }

  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();
};

// ---------------------------------------------------------------- upstream CPU engines (OpenMP)

struct CpuOmpGicpOdometry {
  small_gicp::PointCloud::Ptr target;
  std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> target_tree;
  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    auto down = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(frame, p.downsampling_resolution);
    auto tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(down, small_gicp::KdTreeBuilderOMP(p.num_threads));
    small_gicp::estimate_covariances_omp(*down, *tree, p.num_neighbors, p.num_threads);

    if (target == nullptr) {
      target = down;
      target_tree = tree;
      return T_world;
    }

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
    reg.rejector.max_dist_sq = p.max_correspondence_distance * p.max_correspondence_distance;
    reg.reduction.num_threads = p.num_threads;
    const auto result = reg.align(*target, *down, *target_tree, Eigen::Isometry3d::Identity());

    T_world = T_world * result.T_target_source;
    target = down;
    target_tree = tree;
    return T_world;
  }
};

struct CpuOmpVgicpOdometry {
  std::shared_ptr<small_gicp::GaussianVoxelMap> voxelmap;
  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    auto down = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(frame, p.downsampling_resolution);
    small_gicp::estimate_covariances_omp(*down, p.num_neighbors, p.num_threads);

    if (voxelmap == nullptr) {
      voxelmap = std::make_shared<small_gicp::GaussianVoxelMap>(p.voxel_resolution);
      voxelmap->insert(*down);
      return Eigen::Isometry3d::Identity();
    }

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> reg;
    reg.reduction.num_threads = p.num_threads;
    const auto result = reg.align(*voxelmap, *down, *voxelmap, T_world);
    T_world = result.T_target_source;
    voxelmap->insert(*down, T_world);
    return T_world;
  }
};

// ---------------------------------------------------------------- GPU engines

sgc::NNStrategy nn_of(const std::string& s) {
  if (s == "voxel3") {
    return sgc::NNStrategy::Voxel3;
  }
  if (s == "exact-bf") {
    return sgc::NNStrategy::ExactBF;
  }
  return sgc::NNStrategy::Voxel5;
}

struct GpuGicpOdometry {
  sgc::GpuCloud target;
  sgc::GicpGpu reg;
  bool has_target = false;
  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    sgc::GpuCloud cloud = sgc::GpuCloud::from_host(frame);
    sgc::Downsampler downsampler;
    downsampler.run(cloud, frame.size(), p.downsampling_resolution);
    sgc::estimate_covariances(cloud, p.downsampling_resolution, p.num_neighbors);

    if (!has_target) {
      target = std::move(cloud);
      has_target = true;
      return T_world;
    }

    reg.nn = nn_of(p.nn);
    reg.max_dist_sq = p.max_correspondence_distance * p.max_correspondence_distance;
    const auto result = reg.align(target, cloud, Eigen::Isometry3d::Identity(), p.downsampling_resolution);
    T_world = T_world * result.T_target_source;
    target = std::move(cloud);
    return T_world;
  }
};

struct GpuVgicpOdometry {
  std::unique_ptr<sgc::VoxelHashMap> voxelmap;
  sgc::VgicpGpu reg;
  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    sgc::GpuCloud cloud = sgc::GpuCloud::from_host(frame);
    sgc::Downsampler downsampler;
    downsampler.run(cloud, frame.size(), p.downsampling_resolution);
    sgc::estimate_covariances(cloud, p.downsampling_resolution, p.num_neighbors);

    if (voxelmap == nullptr) {
      voxelmap = std::make_unique<sgc::VoxelHashMap>(p.voxel_resolution);
      voxelmap->insert(cloud, Eigen::Isometry3d::Identity());
      return T_world;
    }

    reg.max_dist_sq = p.max_correspondence_distance * p.max_correspondence_distance;
    const auto result = reg.align(*voxelmap, cloud, T_world);
    T_world = result.T_target_source;
    voxelmap->insert(cloud, T_world);
    return T_world;
  }
};

// Hybrid: CPU preprocessing, GPU registration (data uploaded once, then GPU-resident)
struct HybridGicpOdometry {
  sgc::GpuCloud target;
  sgc::GicpGpu reg;
  bool has_target = false;
  Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d estimate(const Frame& frame, const sgc::bench::Policy& p) {
    auto down = small_gicp::voxelgrid_sampling<std::vector<Eigen::Vector4f>, small_gicp::PointCloud>(frame, p.downsampling_resolution);
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(*down);
    small_gicp::estimate_covariances(*down, tree, p.num_neighbors);

    std::vector<Eigen::Vector4f> pts_f(down->points.size());
    for (size_t k = 0; k < down->points.size(); k++) {
      pts_f[k] = down->points[k].cast<float>();
    }
    sgc::GpuCloud cloud = sgc::GpuCloud::from_host(pts_f);
    // Hybrid semantics: CPU downsampling + kd-tree/covariance preprocessing on CPU,
    // GPU bucketing + covariance estimation + registration. (The CPU-estimated covariances
    // would need a permutation to follow the GPU bucket sort, so the GPU re-estimates them.)
    sgc::Downsampler downsampler;  // idempotent re-bucketing at the same resolution
    downsampler.run(cloud, pts_f.size(), p.downsampling_resolution);
    sgc::estimate_covariances(cloud, p.downsampling_resolution, p.num_neighbors);

    if (!has_target) {
      target = std::move(cloud);
      has_target = true;
      return T_world;
    }

    reg.nn = nn_of(p.nn);
    reg.max_dist_sq = p.max_correspondence_distance * p.max_correspondence_distance;
    const auto result = reg.align(target, cloud, Eigen::Isometry3d::Identity(), p.downsampling_resolution);
    T_world = T_world * result.T_target_source;
    target = std::move(cloud);
    return T_world;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: odometry_gpu <kitti_dir | --synth ply> [flags]\n");
    return 1;
  }

  const bool synth = std::strcmp(argv[1], "--synth") == 0;
  if (synth && argc < 3) {
    std::fprintf(stderr, "--synth requires a ply path\n");
    return 1;
  }
  const sgc::bench::Policy p = sgc::bench::Policy::parse(argc, argv, synth ? 3 : 2);

  // ---------------------------------------------------------------- load frames
  std::vector<Frame> frames;
  if (synth) {
    const Frame base = sgc::io::read_ply(argv[2]);
    const int n = p.max_frames > 0 ? p.max_frames : 50;
    for (int i = 0; i < n; i++) {
      // Smooth drifting: 5cm + 0.15deg per frame on varying axes
      const double t = i;
      const Eigen::Isometry3d d = Eigen::Translation<double, 3>(0.05 * t, 0.02 * std::sin(0.1 * t), -0.01 * t) *
                                  Eigen::AngleAxisd(0.15 * M_PI / 180.0 * t, Eigen::Vector3d(std::sin(0.05 * t), std::cos(0.05 * t), 0.3).normalized());
      Frame f(base.size());
      for (size_t k = 0; k < base.size(); k++) {
        f[k] = (d * base[k].cast<double>()).cast<float>();
      }
      frames.push_back(std::move(f));
    }
  } else {
    const std::string dir = argv[1];
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) {
      std::fprintf(stderr, "failed to open %s\n", dir.c_str());
      return 1;
    }
    while (dirent* e = readdir(d)) {
      const std::string name = e->d_name;
      if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
        files.push_back(dir + "/" + name);
      }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    if (p.max_frames > 0 && files.size() > static_cast<size_t>(p.max_frames)) {
      files.resize(p.max_frames);
    }
    for (const auto& f : files) {
      frames.push_back(sgc::io::read_kitti_bin(f));
    }
  }
  std::fprintf(stderr, "frames=%zu policy=%s\n", frames.size(), p.tag().c_str());

  // ---------------------------------------------------------------- run engine
  std::vector<Eigen::Isometry3d> traj;
  std::vector<double> frame_ms;

  CpuGicpOdometry cpu_gicp;
  CpuVgicpOdometry cpu_vgicp;
  CpuOmpGicpOdometry cpu_omp_gicp;
  CpuOmpVgicpOdometry cpu_omp_vgicp;
  GpuGicpOdometry gpu_gicp;
  GpuVgicpOdometry gpu_vgicp;
  HybridGicpOdometry hybrid_gicp;

  for (auto& frame : frames) {
    // Warm up the GPU on the first frame only (context / module load, excluded from steady stats)
    const auto t0 = Clock::now();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    if (p.exec == "cpu") {
      T = p.engine == "vgicp" ? cpu_vgicp.estimate(frame, p) : cpu_gicp.estimate(frame, p);
    } else if (p.exec == "cpu-omp") {
      T = p.engine == "vgicp" ? cpu_omp_vgicp.estimate(frame, p) : cpu_omp_gicp.estimate(frame, p);
    } else if (p.exec == "hybrid") {
      T = hybrid_gicp.estimate(frame, p);
    } else {
      T = p.engine == "vgicp" ? gpu_vgicp.estimate(frame, p) : gpu_gicp.estimate(frame, p);
    }
    cudaDeviceSynchronize();
    frame_ms.push_back(msec_since(t0));
    traj.push_back(T);
  }

  // ---------------------------------------------------------------- report
  const auto stats = summarize(frame_ms);
  const double mean = std::accumulate(frame_ms.begin(), frame_ms.end(), 0.0) / std::max<size_t>(1, frame_ms.size());
  std::fprintf(stderr,
               "result policy=%s frames=%zu min=%.2f p50=%.2f p95=%.2f max=%.2f mean=%.2f [msec/frame]  throughput=%.1f [fps]\n",  //
               p.tag().c_str(), traj.size(), stats[0], stats[1], stats[2], stats[3], mean, 1000.0 / std::max(1e-9, mean));

  if (!p.traj.empty()) {
    FILE* f = std::fopen(p.traj.c_str(), "w");
    if (f) {
      for (const auto& T : traj) {
        const Eigen::Matrix<double, 3, 4> m = T.matrix().block<3, 4>(0, 0);
        std::fprintf(f, "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3),
                     m(2, 0), m(2, 1), m(2, 2), m(2, 3));
      }
      std::fclose(f);
    }
  }

  if (!p.report.empty()) {
    FILE* f = std::fopen(p.report.c_str(), "w");
    if (f) {
      size_t free_mem = 0, total_mem = 0;
      cudaMemGetInfo(&free_mem, &total_mem);
      std::fprintf(f,
                   "{\"policy\":\"%s\",\"frames\":%zu,\"min_ms\":%.3f,\"p50_ms\":%.3f,\"p95_ms\":%.3f,\"max_ms\":%.3f,\"mean_ms\":%.3f,\"fps\":%.2f,\"gpu_mem_used_mb\":%.1f}\n",
                   p.tag().c_str(), traj.size(), stats[0], stats[1], stats[2], stats[3], mean, 1000.0 / std::max(1e-9, mean), (total_mem - free_mem) / 1048576.0);
      std::fclose(f);
    }
  }

  return 0;
}
