// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace sgc {

/// @brief Create a skew symmetric matrix.
inline Eigen::Matrix3d skew(const Eigen::Vector3d& x) {
  Eigen::Matrix3d s = Eigen::Matrix3d::Zero();
  s(0, 1) = -x[2];
  s(0, 2) = x[1];
  s(1, 0) = x[2];
  s(1, 2) = -x[0];
  s(2, 0) = -x[1];
  s(2, 1) = x[0];
  return s;
}

/*
 * SO3 expmap ported from Sophus via small_gicp (MIT license).
 * Copyright 2011-2017 Hauke Strasdat, 2012-2017 Steven Lovegrove.
 */
inline Eigen::Quaterniond so3_exp(const Eigen::Vector3d& omega) {
  const double theta_sq = omega.dot(omega);

  double imag_factor;
  double real_factor;
  if (theta_sq < 1e-10) {
    const double theta_quad = theta_sq * theta_sq;
    imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
    real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
  } else {
    const double theta = std::sqrt(theta_sq);
    const double half_theta = 0.5 * theta;
    imag_factor = std::sin(half_theta) / theta;
    real_factor = std::cos(half_theta);
  }

  return Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
}

/// @brief SE3 expmap (rotation-first). Identical to small_gicp::se3_exp.
inline Eigen::Isometry3d se3_exp(const Eigen::Matrix<double, 6, 1>& a) {
  const Eigen::Vector3d omega = a.head<3>();

  const double theta_sq = omega.dot(omega);
  const double theta = std::sqrt(theta_sq);

  Eigen::Isometry3d se3 = Eigen::Isometry3d::Identity();
  se3.linear() = so3_exp(omega).toRotationMatrix();

  if (theta < 1e-10) {
    se3.translation() = se3.linear() * a.tail<3>();
  } else {
    const Eigen::Matrix3d Omega = skew(omega);
    const Eigen::Matrix3d V = Eigen::Matrix3d::Identity() + (1.0 - std::cos(theta)) / theta_sq * Omega + (theta - std::sin(theta)) / (theta_sq * theta) * Omega * Omega;
    se3.translation() = V * a.tail<3>();
  }

  return se3;
}

}  // namespace sgc
