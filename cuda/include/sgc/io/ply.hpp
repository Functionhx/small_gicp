// SPDX-License-Identifier: MIT
#pragma once

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sgc::io {

/// @brief Read points from a simple binary PLY file with float properties (x, y, z, ...).
/// @note  Ported from small_gicp's benchmark reader; only for simple PLY IO.
/// @param filename  Filename
/// @return          Points as (x, y, z, 1)
inline std::vector<Eigen::Vector4f> read_ply(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return {};
  }

  std::vector<std::string> properties;
  std::vector<Eigen::Vector4f> points;

  std::string line;
  while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
    if (line == "end_header") {
      break;
    }

    if (line.find("element") == 0) {
      std::stringstream sst(line);
      std::string token, vertex, num_points;
      sst >> token >> vertex >> num_points;
      if (token != "element" || vertex != "vertex") {
        std::cerr << "error: invalid ply format (line=" << line << ")" << std::endl;
        return {};
      }
      points.resize(std::stol(num_points));
    } else if (line.find("property") == 0) {
      std::stringstream sst(line);
      std::string token, type, name;
      sst >> token >> type >> name;
      if (type != "float") {
        std::cerr << "error: only float properties are supported!! (line=" << line << ")" << std::endl;
        return {};
      }
      properties.emplace_back(name);
    }
  }

  if (properties.size() < 3) {
    std::cerr << "error: invalid ply properties" << std::endl;
    return {};
  }

  const size_t stride = properties.size();
  std::vector<float> buffer(stride * points.size());
  ifs.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * buffer.size());

  for (size_t i = 0; i < points.size(); i++) {
    points[i] = Eigen::Vector4f(buffer[i * stride + 0], buffer[i * stride + 1], buffer[i * stride + 2], 1.0f);
  }
  return points;
}

}  // namespace sgc::io
