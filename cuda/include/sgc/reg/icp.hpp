// SPDX-License-Identifier: MIT
#pragma once

#include <sgc/reg/gicp.hpp>

namespace sgc {

/// @brief GPU point-to-point ICP registration engine (scan-to-scan).
///        Requires only downsampled/indexed points; no normal or covariance preprocessing.
struct IcpGpu : public ScanToScanGpu {
  IcpGpu() : ScanToScanGpu(RegistrationFactor::ICP) {}
};

/// @brief GPU point-to-plane ICP registration engine (scan-to-scan).
///        The target cloud must be preprocessed with estimate_normals or
///        estimate_normals_covariances.
struct PointToPlaneIcpGpu : public ScanToScanGpu {
  PointToPlaneIcpGpu() : ScanToScanGpu(RegistrationFactor::PointToPlaneICP) {}
};

}  // namespace sgc
