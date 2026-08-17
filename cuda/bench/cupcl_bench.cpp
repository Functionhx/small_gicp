// cuPCL cudaICP odometry benchmark on KITTI-00 (same loop policy as odometry_gpu).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "cuda_runtime.h"
#include "cudaICP.h"
#include "cudaFilter.h"

static std::vector<Eigen::Vector4f> read_bin(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return {};
  fseek(f, 0, SEEK_END); long n = ftell(f) / 16; fseek(f, 0, SEEK_SET);
  std::vector<float> buf(n * 4);
  fread(buf.data(), 4, buf.size(), f); fclose(f);
  std::vector<Eigen::Vector4f> pts(n);
  for (long i = 0; i < n; i++) pts[i] = Eigen::Vector4f(buf[i*4], buf[i*4+1], buf[i*4+2], 1.0f);
  return pts;
}

// CPU voxel downsample identical to upstream semantics (for the "downsampled input" variant)
static std::vector<Eigen::Vector4f> downsample(const std::vector<Eigen::Vector4f>& raw, double leaf) {
  const double inv = 1.0 / leaf;
  constexpr unsigned long long INVALID = ~0ull;
  std::vector<std::pair<unsigned long long, size_t>> keyed(raw.size());
  for (size_t i = 0; i < raw.size(); i++) {
    int cx = (int)raw[i].x()*inv - (raw[i].x() < (int)raw[i].x()*inv); // floor
    int cy = (int)raw[i].y()*inv - (raw[i].y() < (int)raw[i].y()*inv);
    int cz = (int)raw[i].z()*inv - (raw[i].z() < (int)raw[i].z()*inv);
    cx += 1<<20; cy += 1<<20; cz += 1<<20;
    if ((unsigned)cx > (1u<<21)-1 || (unsigned)cy > (1u<<21)-1 || (unsigned)cz > (1u<<21)-1) { keyed[i] = {INVALID, i}; continue; }
    keyed[i] = {(unsigned long long)cx | ((unsigned long long)cy<<21) | ((unsigned long long)cz<<42), i};
  }
  std::sort(keyed.begin(), keyed.end());
  std::vector<Eigen::Vector4f> out;
  Eigen::Vector4d sum = Eigen::Vector4d::Zero(); bool first = true; unsigned long long prev = 0;
  for (auto& [k, i] : keyed) {
    if (k == INVALID) continue;
    if (!first && k != prev) { out.push_back((sum / sum.w()).cast<float>()); sum.setZero(); }
    first = false; prev = k; sum += raw[i].cast<double>();
  }
  if (!first) out.push_back((sum / sum.w()).cast<float>());
  return out;
}

int main(int argc, char** argv) {
  const char* dir = argv[1];
  const int mode = argc > 2 ? (!strcmp(argv[2], "down") ? 1 : (!strcmp(argv[2], "gpu") ? 2 : 0)) : 0;  // 0=raw 1=cpu-down 2=gpu-down
  const int max_frames = argc > 3 ? atoi(argv[3]) : 100;

  std::vector<std::string> files;
  DIR* d = opendir(dir);
  while (dirent* e = readdir(d)) { std::string s = e->d_name; if (s.size() > 4 && s.substr(s.size()-4) == ".bin") files.push_back(std::string(dir) + "/" + s); }
  closedir(d);
  std::sort(files.begin(), files.end());
  if ((int)files.size() > max_frames) files.resize(max_frames);

  cudaStream_t stream; cudaStreamCreate(&stream);
  cudaFilter* gpu_filter = nullptr;
  if (mode == 2) {
    gpu_filter = new cudaFilter(stream);
    FilterParam_t fp;
    memset(&fp, 0, sizeof(fp));
    fp.type = VOXELGRID; fp.dim = 1; fp.upFilterLimits = 0; fp.downFilterLimits = 0; fp.limitsNegative = false;
    fp.voxelX = fp.voxelY = fp.voxelZ = 0.25f;
    gpu_filter->set(fp);
  }
  size_t maxN = 200000;
  std::vector<std::vector<Eigen::Vector4f>> frames;
  for (auto& f : files) { auto r = read_bin(f.c_str()); frames.push_back(std::move(r)); }

  cudaICP icp(maxN, maxN, stream);
  Eigen::Isometry3f T_world = Eigen::Isometry3f::Identity();
  std::vector<double> ms;
  float *dP, *dQ, *dSrcP, *dSrcQ; cudaMalloc(&dP, maxN*16); cudaMalloc(&dQ, maxN*16); cudaMalloc(&dSrcP, maxN*16); cudaMalloc(&dSrcQ, maxN*16);

  FILE* traj = fopen(argc > 4 ? argv[4] : "/tmp/cupcl_traj.txt", "w");
  // Raw frames are kept; downsampling happens INSIDE the timed loop (fair end-to-end)
  std::vector<std::vector<Eigen::Vector4f>> raw_frames = frames;
  for (size_t i = 0; i < frames.size(); i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (mode == 1) frames[i] = downsample(raw_frames[i], 0.25);  // CPU preprocessing, timed
    if (i > 0) {
      float M[16];
      if (mode == 2) {
        // Their best pipeline: raw H2D once, GPU voxel downsample, ICP on device buffers
        unsigned int nP = 0, nQ = 0;
        cudaMemcpy(dP, raw_frames[i].data(), raw_frames[i].size()*16, cudaMemcpyHostToDevice);
        cudaMemcpy(dQ, raw_frames[i-1].data(), raw_frames[i-1].size()*16, cudaMemcpyHostToDevice);
        gpu_filter->filter(dSrcP, &nP, dP, (unsigned)raw_frames[i].size());
        gpu_filter->filter(dSrcQ, &nQ, dQ, (unsigned)raw_frames[i-1].size());
        icp.icp(dSrcP, (int)nP, dSrcQ, (int)nQ, 1e-4f, 50, 1e-6, 0.5f, M, stream);
        cudaStreamSynchronize(stream);
        Eigen::Matrix4f Tm = Eigen::Map<Eigen::Matrix4f>(M);
        T_world = Eigen::Isometry3f(T_world.matrix() * Tm);
      } else {
      cudaMemcpy(dQ, frames[i-1].data(), frames[i-1].size()*16, cudaMemcpyHostToDevice);
      cudaMemcpy(dP, frames[i].data(), frames[i].size()*16, cudaMemcpyHostToDevice);
      // demo params: relative_mse=1e-4, MaxIter=50, threshold=1e-6, distance_threshold=0.5
      icp.icp(dP, (int)frames[i].size(), dQ, (int)frames[i-1].size(), 1e-4f, 50, 1e-6, 0.5f, M, stream);
      cudaStreamSynchronize(stream);
      Eigen::Matrix4f Tm = Eigen::Map<Eigen::Matrix4f>(M);
      T_world = Eigen::Isometry3f(T_world.matrix() * Tm);
      }
    }
    cudaStreamSynchronize(stream);
    ms.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-t0).count()/1e6);
    Eigen::Matrix<double,3,4> m = T_world.matrix().block<3,4>(0,0).cast<double>();
    fprintf(traj, "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n",
      m(0,0),m(0,1),m(0,2),m(0,3),m(1,0),m(1,1),m(1,2),m(1,3),m(2,0),m(2,1),m(2,2),m(2,3));
  }
  fclose(traj);
  std::sort(ms.begin(), ms.end());
  const char* modestr = mode==2?"cuFilter GPU downsample":(mode==1?"CPU downsample":"raw");
  printf("cuPCL-ICP (%s): frames=%zu p50=%.2fms mean=%.2fms p95=%.2f\n", modestr,
    ms.size(), ms[ms.size()/2], std::accumulate(ms.begin(),ms.end(),0.0)/ms.size(), ms[(size_t)(ms.size()*0.95)]);
  return 0;
}
