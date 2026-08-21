// SPDX-License-Identifier: MIT
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

namespace sgc::bench {

/// @brief Runtime policy shared by the benchmark harness and tests.
struct Policy {
  std::string exec = "full-gpu";  // full-gpu | hybrid | cpu | cpu-omp
  std::string engine = "gicp";    // icp | plane_icp | gicp | vgicp | vgicp_s2s
  std::string nn = "voxel5";      // voxel3 | voxel5 | exact-bf (GICP engine only)
  int num_threads = 1;            // CPU engine threads
  int num_neighbors = 20;
  int covariance_max_shell = 12;  // 12=quality baseline, 8=Jetson balanced
  double downsampling_resolution = 0.25;
  double voxel_resolution = 1.0;  // VGICP map voxel size
  double max_correspondence_distance = 1.0;
  int max_frames = 0;  // 0 = all
  std::string report;  // json output path (optional)
  std::string traj;    // KITTI-format trajectory output path (optional)

  static Policy parse(int argc, char** argv, int first_flag) {
    Policy p;
    for (int i = first_flag; i < argc; i++) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
      if (a == "--exec") {
        p.exec = next();
      } else if (a == "--engine") {
        p.engine = next();
      } else if (a == "--nn") {
        p.nn = next();
      } else if (a == "--threads") {
        p.num_threads = std::stoi(next());
      } else if (a == "--num_neighbors") {
        p.num_neighbors = std::stoi(next());
      } else if (a == "--cov_max_shell") {
        p.covariance_max_shell = std::stoi(next());
      } else if (a == "--downsampling_resolution") {
        p.downsampling_resolution = std::stod(next());
      } else if (a == "--voxel_resolution") {
        p.voxel_resolution = std::stod(next());
      } else if (a == "--max_correspondence_distance") {
        p.max_correspondence_distance = std::stod(next());
      } else if (a == "--max_frames") {
        p.max_frames = std::stoi(next());
      } else if (a == "--report") {
        p.report = next();
      } else if (a == "--traj") {
        p.traj = next();
      }
    }
    return p;
  }

  std::string tag() const { return exec + "/" + engine + "/" + nn; }
};

}  // namespace sgc::bench
