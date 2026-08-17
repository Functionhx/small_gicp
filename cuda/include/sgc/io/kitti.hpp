// SPDX-License-Identifier: MIT
#pragma once

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sgc::io {

/// @brief Read a KITTI velodyne .bin scan (4 floats per point: x, y, z, intensity).
/// @param filename  Path to the .bin file
/// @return          Points as (x, y, z, 1)
inline std::vector<Eigen::Vector4f> read_kitti_bin(const std::string& filename) {
  FILE* f = fopen(filename.c_str(), "rb");
  if (f == nullptr) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return {};
  }

  fseek(f, 0, SEEK_END);
  const long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);

  const size_t num_points = bytes / 16;
  std::vector<float> buffer(num_points * 4);
  const size_t read = fread(buffer.data(), 4, buffer.size(), f);
  fclose(f);

  std::vector<Eigen::Vector4f> points(read / 4);
  for (size_t i = 0; i < points.size(); i++) {
    points[i] = Eigen::Vector4f(buffer[i * 4], buffer[i * 4 + 1], buffer[i * 4 + 2], 1.0f);
  }
  return points;
}

}  // namespace sgc::io
