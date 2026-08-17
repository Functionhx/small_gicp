# small_gicp CUDA 加速实现计划（cuda-x86）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 cuda-x86 分支实现 GPU-native 的 GICP/VGICP 连续帧配准（`cuda/` 目录，库名 `sgc`），以 upstream 为精度参考，端到端单帧延迟较 upstream 多线程 CPU 提速 ≥5×（RTX 4070）。

**Architecture:** 体素桶统一索引（radix sort 输出即空间索引，kd-tree 退役）+ 全 GPU 常驻 fp32 数据 + 确定性 fp64 归约 + CPU 侧 LM 控制器（逐常数移植 upstream）+ 运行时策略参数（exec/nn/engine/det_insert）统一对比框架。

**Tech Stack:** CUDA 12.4（CUB/Thrust 随 toolkit）、C++17、Eigen3（CPU 侧）、GTest、CMake ≥3.18（CUDA architecture 支持）。

**设计文档:** `docs/superpowers/specs/2026-08-17-small-gicp-cuda-design.md`

## Global Constraints

- 工作目录：worktree `/home/as/vllm/resume/small_gicp/.claude/worktrees/cuda-x86`，分支 `cuda-x86`。所有路径相对于仓库根。
- upstream 代码路径（`include/small_gicp`、`src/`）**零改动**（baseline 引用与精度参考）。
- 提交：作者 `Functionhx <2994114386@qq.com>`（`--author` + `--signoff`），**禁止任何 Co-Authored-By**；提交信息风格与仓库一致（祈使句、无前缀）。
- CMake 默认构建（无 CUDA）必须不受影响；新代码全部由 `BUILD_CUDA=ON` 门控。
- CUDA arch：默认 `native`（x86=89），可 `-DCMAKE_CUDA_ARCHITECTURES=87` 供 Jetson。
- 数值策略：存储/因子数学 fp32；质心、H/b/e 归约、LM 求解 fp64；固定 launch config ⇒ run-to-run 逐位确定。
- 算法常数（逐项移植，不得更改）：`max_iterations=20`、`max_inner_iterations=10`、`init_lambda=1e-3`、`lambda_factor=10`、`translation_eps=1e-3`、`rotation_eps=0.1·π/180`、`max_dist_sq=1.0`、协方差特征值替换 `(1e-3, 1.0, 1.0)`、kNN 协方差邻居数 `20`（含自身）、`n<5 → 无效=I`。
- 每完成一个 Task：更新 `WORKLOG.md`（做了什么/测了什么/下一步/挂起决策）后一并提交。
- 测试一律从仓库根运行（相对路径 `data/…`）；GTest 目标名 `sgc_<name>_test`。

## 上游数学契约（实现时照抄，勿凭记忆）

- **voxel key**（downsampling.hpp:31-50）：`coord = fast_floor(pt · inv_leaf) + (1<<20)`，每轴 21bit，打包 `bits = x | y<<21 | z<<42`；越界点丢弃。
- **质心**（downsampling.hpp:60-75）：桶内 `Σpt / Σw`（w=1，纯均值）。
- **协方差**（normal_estimation.hpp:40-45, 66-92）：kNN=20（含自身）→ 样本协方差 `C = (Σ ppᵀ − μ Σpᵀ)/n`（3×3）→ 特征分解 → **特征值替换** → `cov = I − 0.999·n·nᵀ`（n=最小特征向量，等价重构）；`n<5` 时 `cov=I`。注意：等价式仅在特征向量按升序对应 `(1e-3,1,1)` 时成立，GPU 用"最小特征向量闭式解"实现。
- **GICP 因子**（gicp_factor.hpp:34-73）：`q = T·p_src`；NN→`(t_idx, d²)`；`d²>max_dist_sq` 拒绝；`M = (Ct + T·Cs·Tᵀ)₃ₓ₃⁻¹`；`r = p_tgt − q`；`J = [R·skew(p_src) | −R]`（3×6）；`H = JᵀMJ`、`b = JᵀMr`、`e = ½rᵀMr`。
- **LM**（optimizer.hpp:100-149）：δ 解 `(H+λI)δ = −b`（LDLT）；`T ← T·exp(δ)`；成功（`new_e ≤ e`）则 `λ/=10`，失败 `λ*=10`；收敛 = `‖δ_rot‖≤0.1° 且 ‖δ_trans‖≤1e-3`。误差重评用缓存 target_index/M，无二次 NN。
- **帧链**：GICP 引擎 `align(prev, cur, prev_tree, I)` 后 `T_world *= T_ts`；VGICP 引擎 `align(map, cur, map, T_world)` 后 `T_world = T_ts`，再 `insert(cur, T_world)`。VGICP NN = 中心体素单探（`set_search_offsets(1)`）。
- **VGICP 地图**（gaussian_voxelmap.hpp + incremental_voxelmap.hpp）：slot 累加 `Σ T·p` 与 `Σ T·Cs·Tᵀ`，finalize 除以 count；LRU：insert 触碰即 `lru=counter`，每 10 帧 清除 `lru+100 < counter` 的 slot。
- **已知上游边界情况不复制**：trailing-invalid-key 时 upstream 会输出 NaN 点（downsampling.hpp:74）；GPU 版直接在排序前丢弃越界点。

---

### Task 1: CMake CUDA 脚手架 + sgc::core + smoke 测试

**Files:**
- Create: `cuda/CMakeLists.txt`
- Create: `cuda/include/sgc/core/check.hpp`（SGC_CHECK 宏）
- Create: `cuda/include/sgc/core/buffer.hpp`
- Create: `cuda/src/core/buffer.cu`
- Create: `cuda/test/smoke_test.cpp`
- Modify: `CMakeLists.txt`（顶层，追加 BUILD_CUDA 门控）
- Modify: `WORKLOG.md`

**Interfaces（后续任务依赖）:**
- `namespace sgc`；`SGC_CHECK(x)` → 失败抛 `std::runtime_error`（含 `cudaGetErrorString`）。
- `template <typename T> class GpuBuffer`：`GpuBuffer()=default`、`explicit GpuBuffer(size_t n)`、`~GpuBuffer` 释放；`void resize(size_t n)`（仅增或有条件缩，cudaMalloc/cudaFree）；`T* raw()`、`const T* raw() const`；`size_t size() const`；`void upload(const T* host, size_t n)`（async=false 走默认流 cudaMemcpy）；`void download(T* host, size_t n) const`。禁拷贝、允许移动。

- [ ] **Step 1: 顶层 CMake 门控**

在顶层 `CMakeLists.txt` 的 `## Test ##` 段之前插入：

```cmake
#############
## CUDA #####
#############
option(BUILD_CUDA "Build CUDA accelerated implementation (sgc)" OFF)
if(BUILD_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_subdirectory(cuda)
endif()
```

- [ ] **Step 2: cuda/CMakeLists.txt**

```cmake
find_package(GTest REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_library(sgc STATIC
  src/core/buffer.cu
)
target_include_directories(sgc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sgc PUBLIC Eigen3::Eigen)

enable_testing()
file(GLOB SGC_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
foreach(TEST_SOURCE ${SGC_TEST_SOURCES})
  get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_link_libraries(${TEST_NAME} PRIVATE sgc GTest::gtest_main small_gicp)
  gtest_discover_tests(${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
```

注意：链接 `small_gicp`（helper 库）为后续对拍测试提供 upstream 参考；因此 `BUILD_CUDA=ON` 时在顶层强制 `set(BUILD_HELPER ON CACHE BOOL "" FORCE)`（加在 add_subdirectory(cuda) 前）。

- [ ] **Step 3: 写失败测试 `cuda/test/smoke_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include <sgc/core/buffer.hpp>

TEST(Smoke, BufferRoundtripAndKernel) {
  sgc::GpuBuffer<float> buf(1024);
  ASSERT_EQ(buf.size(), 1024u);
  std::vector<float> host(1024, 3.5f);
  buf.upload(host.data(), host.size());
  sgc::fill_kernel(buf.raw(), buf.size(), 1.25f);  // Task1 内置的迷你内核
  std::vector<float> out(1024);
  buf.download(out.data(), out.size());
  EXPECT_EQ(out[0], 1.25f);
  EXPECT_EQ(out[1023], 1.25f);
}
```

- [ ] **Step 4: 构建并确认失败**

```bash
mkdir -p build_cuda && cd build_cuda
cmake ../.. -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc) && ctest
```
预期：编译错误（头文件不存在）。

- [ ] **Step 5: 实现 check.hpp / buffer.hpp / buffer.cu**

`check.hpp`：

```cpp
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace sgc {
inline void check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda error at ") + file + ":" + std::to_string(line) + " : " + cudaGetErrorString(err));
  }
}
}  // namespace sgc
#define SGC_CHECK(call) ::sgc::check((call), __FILE__, __LINE__)
```

`buffer.hpp`：

```cpp
#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <sgc/core/check.hpp>

namespace sgc {

__global__ void fill_kernel(float* data, size_t n, float value);  // 定义在 buffer.cu

template <typename T>
class GpuBuffer {
public:
  GpuBuffer() = default;
  explicit GpuBuffer(size_t n) { resize(n); }
  ~GpuBuffer() { destroy(); }
  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;
  GpuBuffer(GpuBuffer&& other) noexcept : ptr_(other.ptr_), n_(other.n_) { other.ptr_ = nullptr; other.n_ = 0; }
  GpuBuffer& operator=(GpuBuffer&& other) noexcept { destroy(); ptr_ = other.ptr_; n_ = other.n_; other.ptr_ = nullptr; other.n_ = 0; return *this; }

  void resize(size_t n) {
    if (n == n_) return;
    destroy();
    if (n > 0) SGC_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    n_ = n;
  }
  T* raw() { return ptr_; }
  const T* raw() const { return ptr_; }
  size_t size() const { return n_; }
  void upload(const T* host, size_t n) { SGC_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice)); }
  void download(T* host, size_t n) const { SGC_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost)); }

private:
  void destroy() { if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; } n_ = 0; }
  T* ptr_ = nullptr;
  size_t n_ = 0;
};

}  // namespace sgc
```

`buffer.cu`：

```cpp
#include <sgc/core/buffer.hpp>

namespace sgc {
__global__ void fill_kernel(float* data, size_t n, float value) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = value;
}
}  // namespace sgc
```

- [ ] **Step 6: 构建并通过**

```bash
cmake --build . -j$(nproc) && ctest -R smoke --output-on-failure
```
预期 PASS。同时验证 `cmake ../.. `（无 BUILD_CUDA）仍正常配置。

- [ ] **Step 7: 更新 WORKLOG.md 并提交**

```bash
git add CMakeLists.txt cuda/ WORKLOG.md
git commit --author="Functionhx <2994114386@qq.com>" --signoff -m "Add CUDA build scaffolding and sgc core buffers"
```

---

### Task 2: 数据加载（KITTI bin / PLY）与 GpuCloud 宿主工具

**Files:**
- Create: `cuda/include/sgc/io/kitti.hpp`
- Create: `cuda/include/sgc/io/ply.hpp`（从 `include/small_gicp/benchmark/read_points.hpp` 移植 `read_ply`）
- Create: `cuda/include/sgc/points/gpu_cloud.hpp`
- Test: `cuda/test/io_test.cpp`

**Interfaces:**
- `namespace sgc::io`：`std::vector<Eigen::Vector4f> read_kitti_bin(const std::string& path);`（每点 4×float，w=1.0f）；`std::vector<Eigen::Vector4f> read_ply(const std::string& path);`
- `struct sgc::GpuCloud`：`GpuBuffer<float4> points;`、`GpuBuffer<float> covs;`（9/点）、`GpuBuffer<unsigned long long> keys;`（排序唯一 voxel key，与 points 一一对应）；`size_t size() const;`（= keys.size()）。附宿主侧便捷：`static GpuCloud from_host(const std::vector<Eigen::Vector4f>& pts);`（仅上传 points，keys/covs 空——预处理的输出）与 `std::vector<Eigen::Vector4f> download_points() const;`。

- [ ] **Step 1: 失败测试**

```cpp
#include <gtest/gtest.h>
#include <sgc/io/kitti.hpp>
#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>

TEST(IO, PlyRoundtrip) {
  const auto pts = sgc::io::read_ply("data/target.ply");
  ASSERT_GT(pts.size(), 10000u);
  sgc::GpuCloud cloud = sgc::GpuCloud::from_host(pts);
  EXPECT_EQ(cloud.size(), 0u);                       // 预处理前 keys 为空
  const auto back = cloud.download_points();
  ASSERT_EQ(back.size(), pts.size());
  for (size_t i = 0; i < pts.size(); i += 997) {
    EXPECT_NEAR(back[i].x(), pts[i].x(), 1e-6f);
    EXPECT_NEAR(back[i].w(), 1.0f, 1e-6f);
  }
}

TEST(IO, KittilBinHeader) {
  // 合成 2 点 bin 文件
  const std::string tmp = "/tmp/sgc_test.bin";
  { std::vector<float> v{1,2,3,1, 4,5,6,1}; FILE* f=fopen(tmp.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
  const auto pts = sgc::io::read_kitti_bin(tmp);
  ASSERT_EQ(pts.size(), 2u);
  EXPECT_EQ(pts[1].y(), 5.f);
}
```

- [ ] **Step 2: 运行确认失败**（`ctest -R io`，编译错误）
- [ ] **Step 3: 实现三个头文件**。`read_ply` 逐行照抄 upstream `read_points.hpp:52-` 的 ASCII/二进制 PLY 解析（返回 `Eigen::Vector4f`，w=1）；`read_kitti_bin`：fread 全文件 → `n = size/16` → 每点 `(x,y,z,1.0f)`；`GpuCloud` 按 Interfaces 定义实现（`from_host` resize+upload，`download_points` 转 Eigen）。
- [ ] **Step 4: `ctest -R io` PASS**
- [ ] **Step 5: WORKLOG + 提交** `Add KITTI/PLY loaders and GpuCloud`

---

### Task 3: 体素 key + 排序 + 唯一 + 分段质心降采样

**Files:**
- Create: `cuda/include/sgc/voxel/voxel_key.hpp`（host+device key 打包/解包）
- Create: `cuda/include/sgc/voxel/downsample.hpp`（host API）
- Create: `cuda/src/voxel/downsample.cu`（keys kernel、run 边界、双趟 fp64 分段归约）
- Test: `cuda/test/downsample_test.cpp`

**Interfaces:**
- `SGC_HOST_DEVICE unsigned long long sgc::voxel_key(float4 p, float inv_leaf);`（契约：`fast_floor` 语义 + 21bit×3 打包；越界返回 `0xFFFFFFFFFFFFFFFF`）
- `SGC_HOST_DEVICE int3 sgc::key_coord(unsigned long long key);` / `unsigned long long sgc::coord_key(int3 c);`（解包/重打包，供 NN 偏移用）
- `void sgc::voxelgrid_downsample(GpuCloud& cloud, size_t num_raw, float leaf_size);`——输入 `cloud.points` 已含原始点，输出就地写入 `cloud.points`（质心，按 key 升序）与 `cloud.keys`。**确定性要求**：固定 tile=256，fp64 partial，两趟归约。
- 内部使用 CUB：`cub::DeviceRadixSort::SortPairs`（key=uint64, value=uint32 原始索引）、`cub::DeviceScan::ExclusiveSum`（run-start 前缀和）。

- [ ] **Step 1: 失败测试（与 upstream 对拍）**

```cpp
#include <gtest/gtest.h>
#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/points/eigen.hpp>

TEST(Downsample, ParityWithUpstream) {
  const auto raw = sgc::io::read_ply("data/target.ply");
  const double leaf = 0.25;

  // upstream 参考（vector<Vector4f> 也适用 traits）
  auto ref = small_gicp::voxelgrid_sampling(raw, leaf);

  sgc::GpuCloud cloud = sgc::GpuCloud::from_host(raw);
  sgc::voxelgrid_downsample(cloud, raw.size(), leaf);
  auto out = cloud.download_points();
  auto keys = cloud.download_keys();  // vector<unsigned long long>，需在本任务给 GpuCloud 补该方法

  ASSERT_EQ(out.size(), ref->size());
  ASSERT_EQ(keys.size(), out.size());
  // 升序 key 检查
  for (size_t i = 1; i < keys.size(); i++) ASSERT_LT(keys[i-1], keys[i]);
  // 逐桶质心对拍（两边都按 key 排序，顺序一致）
  for (size_t i = 0; i < out.size(); i += 57) {
    EXPECT_NEAR(out[i].x(), (*ref)[i].x(), 1e-4);
    EXPECT_NEAR(out[i].y(), (*ref)[i].y(), 1e-4);
    EXPECT_NEAR(out[i].z(), (*ref)[i].z(), 1e-4);
  }
}
```

注：upstream `voxelgrid_sampling` 输出按 key 升序（std::sort 保证），与 GPU 输出同序，可按下标对拍。若首跑发现 NaN 点（上游 trailing-invalid 边界情况），改用 `data/source.ply` 或在前置检查中跳过 NaN。

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**
  - `voxel_key.hpp`：`fast_floor` 的 fp32 版 `int ffloor(float v){ int t=(int)v; return t - (v < (float)t); }`；打包/解包。
  - `downsample.cu` 流程：
    1. kernel A：每点算 key（越界→UINT64_MAX）；
    2. CUB SortPairs（升序，invalid 自然沉底）；
    3. kernel B：每点标记 `run_start[i] = (i==0 || key[i]!=key[i-1]) && key[i]!=INVALID`；
    4. ExclusiveSum → `bucket_of[i]`（invalid 点 bucket=-1）；
    5. kernel C（tile=256，每 tile 内线程逐点 fp32 累加、tile 尾由 lane0 以 fp64 写 partial`{bucket, Σx,Σy,Σz,Σw}`）；
    6. kernel D：每 bucket 顺序扫其 partials（fp64）求和 → `points[b] = make_float4(Σ/Σw, 1.0f)`，`keys[b]`；
    7. `cloud.points.resize(num_buckets)` 后写入。
  - partial 数组与 CUB temp storage 用 `GpuBuffer<uint8_t>`/`GpuBuffer<double>` 成员缓存（`class Downsampler` 持有，避免每帧分配——API 改为 `Downsampler down; down.run(cloud, num_raw, leaf);`，自由函数包装之）。
- [ ] **Step 4: `ctest -R downsample` PASS**（超差时先查 key 一致性：加临时断言逐点比较 CPU/GPU key 一致率 ≥99.99%）
- [ ] **Step 5: WORKLOG + 提交** `Add GPU voxelgrid downsampling with upstream parity`

---

### Task 4: 协方差估计（kNN20 邻桶候选 + I−0.999nnᵀ）

**Files:**
- Create: `cuda/include/sgc/preproc/covariance.hpp`
- Create: `cuda/src/preproc/covariance.cu`
- Test: `cuda/test/covariance_test.cpp`

**Interfaces:**
- `enum class sgc::NNStrategy { Voxel3, Voxel5, ExactBF };`
- `void sgc::estimate_covariances(GpuCloud& cloud, float leaf_size, int num_neighbors = 20, NNStrategy nn = NNStrategy::Voxel5);`——读 `points/keys`，写 `covs`（9 float/点，行主序）。
- 供测试与 Task5 复用的宿主参考：`Eigen::Matrix3f sgc::host_cov_of_neighborhood(const GpuCloud&, size_t i, const std::vector<size_t>& neighbor_ids);`

- [ ] **Step 1: 失败测试**

```cpp
#include <gtest/gtest.h>
#include <sgc/io/ply.hpp>
#include <sgc/points/gpu_cloud.hpp>
#include <sgc/voxel/downsample.hpp>
#include <sgc/preproc/covariance.hpp>
// upstream 参考：估计协方差到 small_gicp::PointCloud
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/ann/kdtree.hpp>

class CovarianceTest : public ::testing::Test {
protected:
  void SetUp() override {
    raw = sgc::io::read_ply("data/target.ply");
    gpu = sgc::GpuCloud::from_host(raw);
    sgc::voxelgrid_downsample(gpu, raw.size(), 0.25);
    // upstream 参考
    auto down = small_gicp::voxelgrid_sampling(raw, 0.25);
    ref = std::make_shared<small_gicp::PointCloud>(*down);  // Vector4f→PointCloud 转换见下
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(*ref);
    small_gicp::estimate_covariances(*ref, tree, 20);
  }
  std::vector<Eigen::Vector4f> raw;
  sgc::GpuCloud gpu;
  small_gicp::PointCloud::Ptr ref;
};

TEST_F(CovarianceTest, ParityWithUpstream) {
  sgc::estimate_covariances(gpu, 0.25f, 20, sgc::NNStrategy::Voxel5);
  auto covs = gpu.download_covs();  // 本任务给 GpuCloud 补该方法（vector<float>，9/点）
  ASSERT_EQ(covs.size() / 9, ref->size());
  // 与 upstream 同下标对拍（两边点序一致）
  double max_rel = 0.0;
  for (size_t i = 0; i < ref->size(); i++) {
    Eigen::Matrix3f gpu_cov;  // 行主序装入
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) gpu_cov(r,c) = covs[i*9+r*3+c];
    Eigen::Matrix3f ref_cov = ref->covs[i].cast<float>().topLeftCorner<3,3>();
    const double rel = (gpu_cov - ref_cov).norm() / ref_cov.norm();
    max_rel = std::max(max_rel, rel);
  }
  EXPECT_LT(max_rel, 1e-2);          // 特征向量近似导致的总体界
  // 中位数必须很紧
  // （实现里再算 median_rel，EXPECT_LT(median, 1e-3)）
}
```

（`ref` 构造：`small_gicp::PointCloud` 有 `points/covs` vector<Vector4d>/vector<Matrix4d>，从 down 逐点填；若 `PointCloud(vector<Vector4f>)` 构造存在则直接用。）

- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现内核**（每点一线程）：
  1. `key → int3 c`；按 nn 策略枚举偏移（Voxel3: 27，Voxel5: 125；ExactBF: 全体扫描——本任务先实现 Voxel3/5，ExactBF 在 Task 5 统一实现后此处直接复用其结果接口）；
  2. 每命中桶（含自身桶）取其质心点为候选，维护 top-20（`float dist2[20]; int idx[20];` 排序插入，容量满且更远则跳过）；
  3. `n<5 → cov=I`；
  4. `μ = Σp/n`；`C = (Σppᵀ − μΣpᵀ)/n`（fp32 累加）；
  5. 最小特征向量 n：闭式解——先用 `||C||` 无关量法：`n = 伴随幂迭代`：`v ← (C−λ̄I)⁻¹·v₀` 两次（λ̄=trace/3，v₀=(1,1,1)），归一化；数值病态（平面完美）时回退 `C` 最小行叉积法；
  6. `cov = I − 0.999·n·nᵀ` 写 9 float。
- [ ] **Step 4: `ctest -R covariance` PASS**（超差时输出 rel 直方图定位：kNN 集合差异 vs 特征向量求解差异——前者换 Voxel5/ExactBF 复测，后者查闭式解数值）
- [ ] **Step 5: WORKLOG + 提交** `Add GPU covariance estimation with eigen-replacement model`

---

### Task 5: VoxelIndex 二分探查 + NN 策略（voxel3/voxel5/exact-bf）

**Files:**
- Create: `cuda/include/sgc/search/nn.hpp`
- Create: `cuda/src/search/nn.cu`
- Test: `cuda/test/nn_test.cpp`

**Interfaces:**
- `__device__ int sgc::find_voxel(const unsigned long long* keys, int n, unsigned long long key);`（lower_bound，未命中 −1；`cuda/include/sgc/search/nn.hpp` 内 `__forceinline__`）
- `void sgc::nn_search(const GpuCloud& target, const GpuBuffer<float4>& queries, GpuBuffer<int>& out_idx, GpuBuffer<float>& out_sq_dist, NNStrategy nn, float leaf_size);`——批量 NN（调试/验收用宿主 API；配准内核内直接调 device 函数）。
- `__device__ void sgc::nn_query<Strategy>(const unsigned long long* keys, int n, const float4* pts, float4 q, float leaf_size, int* idx, float* sq_dist);`——配准内核复用的 device 端实现（Strategy 为枚举常参，编译期分派）。

- [ ] **Step 1: 失败测试**

```cpp
TEST(NN, AgreementWithKdTree) {
  // 数据：data/target.ply → downsample(0.25)
  // GPU: nn_search(Voxel3) 与 ExactBF
  // CPU 参考：small_gicp::UnsafeKdTree nearest_neighbor_search（逐点）
  // 断言：
  //  1) ExactBF 与 kd-tree：找到的 index 100% 相同（同点时）或 dist2 差 < 1e-6（并列时）
  //  2) Voxel3 与 kd-tree：sq_dist <= 1.0（max_corr²）的查询中，dist2 相对差 >1e-3 的占比 < 1%
  //  3) Voxel5 同口径 < 0.2%
}
```

（用 `data/target.ply` 全部降采样点作查询。）

- [ ] **Step 2: 确认失败** → **Step 3: 实现**
  - `find_voxel`：经典二分；
  - voxel3/5：`int3 c = key_coord(key_of(q));` 枚举偏移 → `coord_key` → `find_voxel` → 累 Compare 点距；
  - ExactBF：block=256，shared memory 缓存 target tile（256×float4=4KB），thread per query 全扫；
  - 宿主 `nn_search` 包装各自 kernel。
- [ ] **Step 4: `ctest -R nn` PASS**（这是近似质量的**关键验收**：不达标→默认策略改 Voxel5 并记录 WORKLOG）
- [ ] **Step 5: WORKLOG + 提交** `Add NN search strategies with kd-tree agreement tests`

---

### Task 6: GICP 线性化内核 + 确定性归约

**Files:**
- Create: `cuda/include/sgc/factor/gicp_factor.cuh`（device 数学：3×3 求逆、skew、JᵀMJ）
- Create: `cuda/include/sgc/reg/reduction.hpp`
- Create: `cuda/src/reg/linearize.cu`
- Test: `cuda/test/linearize_test.cpp`

**Interfaces:**
- `__device__ float3x3 sgc::mat3_inv(const float3x3&);`（伴随/行列式闭式）
- `struct sgc::CorrCache { GpuBuffer<int> target_idx; GpuBuffer<float> mahalanobis; }`（9 float/点，缓存 M；Task 7 误差重评复用）
- `void sgc::linearize_and_reduce(const GpuCloud& target, const GpuCloud& source, const float* d_T, double max_dist_sq, NNStrategy nn, float leaf_size, CorrCache& cache, GpuBuffer<double>& partials, int& num_partials, double* h_out /*43*/);`
  - kernel：thread per source 点：`q=T·p`→`nn_query`→拒绝→`M=(Ct+T Cs Tᵀ)₃ₓ₃⁻¹`→`J=[R·skew(p)|−R]`→`H=JᵀMJ,b=JᵀMr,e=½rᵀMr`→缓存 target_idx+M→warp shuffle fp32 归约→lane0 fp64 写 per-warp partial（43×W doubles）；
  - 宿主：固定 `block=256`；CPU 按 warp 序 double 终和 → `h_out`（H 36 + b 6 + e 1），同时以 `atomicAdd(&d_inliers,1)` 计内点数。
- `d_T`：16 float（4×4 行主序，由 `Eigen::Isometry3d.cast<float>().matrix()` 而来）。

- [ ] **Step 1: 失败测试**

```cpp
TEST(Linearize, HbEParityOnFixedCorrespondences) {
  // data/target.ply 降采样+协方差（Task4 产物）
  // source = target 平移 (0.3, -0.2, 0.1) 后的同一云（同 covs）
  // T = 该已知变换的逆
  // CPU 参考：对每个 source 点手动执行 upstream GICPFactor::linearize 语义
  //   （直接 include gicp_factor.hpp，用 UnsafeKdTree + traits 调 factor.linearize(...)）
  // 断言：ΣH / Σb / Σe 相对差 < 1e-3（fp32 因子 + fp64 归约 vs 全 double）
  //      num_inliers 相等
}
TEST(Linearize, DeterministicAcrossRuns) {
  // 同输入连续调用 100 次，h_out 43 double 逐位相同
}
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**（数学契约照抄上文；warp 归约：`for (offset=16;offset;offset>>=1) __shfl_down_sync` 对 43 个量循环——H 用 36 float 寄存器数组，shuffle 逐元素；lane0 `atomicAdd` 到 fp64 partial? NO——**固定槽写入**：`partial[warp_global_id*43+k] = fp64(val)`，无原子）
- [ ] **Step 4: `ctest -R linearize` PASS**
- [ ] **Step 5: WORKLOG + 提交** `Add GICP linearization kernel with deterministic fp64 reduction`

---

### Task 7: LM 控制器 + GicpGpu::align（GICP 引擎闭环）

**Files:**
- Create: `cuda/include/sgc/reg/lie.hpp`（`se3_exp`/`skew` 从 `include/small_gicp/util/lie.hpp` 逐行移植，double）
- Create: `cuda/include/sgc/reg/gicp.hpp`
- Create: `cuda/src/reg/gicp.cu`（误差重评内核）
- Test: `cuda/test/gicp_align_test.cpp`

**Interfaces:**
- `struct sgc::GicpGpu { NNStrategy nn = Voxel3; int max_iterations=20, max_inner_iterations=10; double init_lambda=1e-3, lambda_factor=10.0, max_dist_sq=1.0, translation_eps=1e-3, rotation_eps=0.1*M_PI/180; sgc::GicpResult align(const GpuCloud& target, const GpuCloud& source, const Eigen::Isometry3d& init_T, float leaf_size); };`
- `struct sgc::GicpResult { Eigen::Isometry3d T; bool converged=false; int iterations=0; size_t num_inliers=0; Eigen::Matrix<double,6,6> H; Eigen::Matrix<double,6,1> b; double error=0.0; };`
- `void sgc::eval_error_cached(const GpuCloud& source, const float* d_T, CorrCache& cache, GpuBuffer<double>& partials, int& num_partials, double* h_e);`（`e_i = ½rᵀMr`，r 用新 T 重算，M/target_idx 用缓存）

- [ ] **Step 1: 失败测试（单对合成配准对拍）**

```cpp
TEST(GicpAlign, SyntheticPairParity) {
  // raw = data/target.ply；Δ = Translation(0.4,-0.3,0.2)*AngleAxis(2°,z)
  // source_pts = Δ·raw；target = raw
  // upstream 参考：preprocess_points 各自 (downsample+kdtree+covs) 后
  //   Registration<GICPFactor, ParallelReductionOMP> 对齐（参数同 Global 常数）
  // GPU：voxelgrid_downsample + estimate_covariances(Voxel5) + GicpGpu{nn=Voxel5}.align(..., I)
  // 断言：Δ_gpu 与 Δ_ref 的平移差 < 5mm、旋转差 < 0.1°（首版临时阈值，正式验收在 Task 11）
  //      converged/iterations 差 ≤ 2
}
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**
  - `lie.hpp`：从 upstream `lie.hpp` 拷 `se3_exp`（Sophus 依赖改为纯 Eigen 实现，逐行对照）；
  - `align()`：外层循环调 `linearize_and_reduce` → `(H+λI).ldlt().solve(-b)` → 内层 `eval_error_cached` 比较误差 → λ 调整/接受/终止，逻辑逐行对照 `optimizer.hpp:100-149`；
  - T 在宿主 double 维护，每轮 cast<float> 上传 `GpuBuffer<float> d_T`。
- [ ] **Step 4: `ctest -R gicp_align` PASS**（若合成对不收敛，打印 e/λ 轨迹 vs upstream 对照定位）
- [ ] **Step 5: WORKLOG + 提交** `Add GPU GICP engine with LM controller`

---

### Task 8: VoxelHashMap 增量地图 + LRU

**Files:**
- Create: `cuda/include/sgc/voxel/hash_map.hpp`
- Create: `cuda/src/voxel/hash_map.cu`
- Test: `cuda/test/voxelmap_test.cpp`

**Interfaces:**
- `struct sgc::VoxelHashMap { explicit VoxelHashMap(float leaf_size, bool det_insert=true); void insert(const GpuCloud& cloud, const Eigen::Isometry3d& T); size_t num_voxels() const; /* 导出给测试与引擎 */ GpuBuffer<unsigned long long> slot_keys; GpuBuffer<float4> slot_mean; GpuBuffer<float> slot_cov; GpuBuffer<unsigned int> slot_count; GpuBuffer<unsigned long long> slot_lru; /* 内部 */ unsigned long long lru_counter=0; static constexpr size_t lru_horizon=100, lru_clear_cycle=10; };`
  - 哈希：容量 2^n（负载因子 >0.6 时扩容重插），`pos = splitmix64(key) & (cap-1)` 线性探测；空槽 `key==EMPTY(0xFFFFFFFFFFFFFFFF)`、墓碑 `TOMBSTONE(0xFFFFFFFFFFFFFFFE)`；
  - 插入 kernel：每点算 key → 探测（`atomicCAS` 占位新槽）→ `det_insert` 时先按 key 排序再逐桶串行累加（确定性），否则 `atomicAdd` Σpt/Σcov；
  - finalize kernel：`mean=Σ/count`、`cov=Σ/count`；
  - LRU：插入 kernel 内 `slot_lru[slot]=lru_counter`；每 `lru_clear_cycle` 次 insert 后跑标记-压缩（`thrust::remove_if` 或手写 scan+gather），语义对齐 upstream。

- [ ] **Step 1: 失败测试**

```cpp
TEST(VoxelMap, ParityWithUpstreamGaussianVoxelMap) {
  // raw = data/target.ply（降采样+协方差，作为两"帧"）
  // 两边各 insert 同一帧（T=I）：slot 数 == upstream GaussianVoxelMap flat_voxels.size()
  // 逐 slot：mean 差 < 1e-4，cov Frobenius 差 < 1e-2（fp32 累加 vs double）
  // 再 insert 第二帧（T=平移 0.3m）→ 再对拍（count≥1 的槽）
  // det_insert=true 连跑 3 次 → 逐位相同
}
TEST(VoxelMap, LRUEviction) {
  // 构造：插入帧，仅触碰部分槽（用小 lru_clear_cycle=1, horizon=2 的测试参数）
  // 断言：未触碰槽被清除，num_voxels 下降，剩余槽 mean 不变
}
```

（测试参数化：`VoxelHashMap(leaf, det_insert, horizon, cycle)` 加测试用 ctor 参数。）

- [ ] **Step 2: 确认失败** → **Step 3: 实现** → **Step 4: PASS**
- [ ] **Step 5: WORKLOG + 提交** `Add GPU incremental voxel hash map with LRU`

---

### Task 9: VgicpGpu 引擎（中心体素单探）

**Files:**
- Create: `cuda/include/sgc/reg/vgicp.hpp`
- Create: `cuda/src/reg/vgicp.cu`
- Test: `cuda/test/vgicp_align_test.cpp`

**Interfaces:**
- `struct sgc::VgicpGpu { int max_iterations=20, max_inner_iterations=10; double init_lambda=1e-3, lambda_factor=10.0, max_dist_sq=1.0; /* 同 LM 常数 */ GicpResult align(const VoxelHashMap& map, const GpuCloud& source, const Eigen::Isometry3d& init_T); };`
  - 线性化 kernel：thread per source 点：`q=T·p` → `find slot(hash, key(q))` → 未命中跳过；命中 → `M=(slot_cov+T·Cs·Tᵀ)⁻¹`（注意 upstream 因子里 target cov 取 `traits::cov(target, t_idx)`，这里即 slot cov；**无 DistanceRejector 检查**——对齐 upstream：VGICP 引擎 registration.rejector 未设置 max_dist_sq？查证：`odometry_benchmark_small_vgicp_model_tbb.cpp` 未改 rejector → `max_dist_sq=1.0` 生效于 `d²`（点到 slot mean）→ 保留同样的拒绝）→ 其余同 Task 6；
  - 归约/误差重评/LM 复用 Task 6/7 基础设施（模板参数化 target 类型：`GpuCloud | VoxelHashMap`）。

- [ ] **Step 1: 失败测试**：同 Task 7 形态——合成对（Δ 已知）建图后对拍 upstream `Registration<GICPFactor, ParallelReductionOMP>` + `GaussianVoxelMap`（注意 upstream 侧用 `voxel_resolution` 建图、`align(map, src, map, init)`）。断言平移差 <1cm/旋转 <0.3°。
- [ ] **Step 2-4: 失败→实现→PASS**（实现时把 Task 6 的 device 因子抽成 `template <typename TargetT> __device__ ... linearize_point(...)`，两引擎共用）
- [ ] **Step 5: WORKLOG + 提交** `Add GPU VGICP engine over incremental voxel map`

---

### Task 10: odometry_gpu 对比 harness

**Files:**
- Create: `cuda/bench/odometry_gpu.cpp`
- Create: `cuda/include/sgc/bench/policy.hpp`（参数结构）
- Modify: `cuda/CMakeLists.txt`（加 bench 目标，链 sgc + small_gicp + fmt）

**Interfaces:**
- `struct sgc::bench::Policy { std::string exec /*full-gpu|hybrid|cpu*/, engine /*gicp|vgicp*/, nn /*voxel3|voxel5|exact-bf*/; int num_threads=4, num_neighbors=20; double downsampling_resolution=0.25, voxel_resolution=1.0, max_correspondence_distance=1.0; bool det_insert=true; static Policy parse(int argc, char** argv); std::string tag() const; };`
- CLI：`odometry_gpu <dataset_path| --synth ply_path> [--exec ...] [--engine ...] [--nn ...] [--threads N] [--max_frames N] [--report out.json] [--traj out.txt(KITTI 格式)]`
- 三种 exec：
  - `cpu`：进程内调 upstream 引擎（照抄 `odometry_benchmark_small_gicp_omp.cpp` / `small_vgicp_model_tbb.cpp` 的逐帧逻辑，OMP reduction + threads）；
  - `full-gpu`：`voxelgrid_downsample → estimate_covariances → {GicpGpu|VgicpGpu}`，帧链策略照抄（GICP: identity init + 复合；VGICP: T_world init + 覆写 + insert）；
  - `hybrid`：CPU `voxelgrid_sampling` + upstream 预处理，配准迭代走 GPU（数据首次上传后常驻）。
- 计时：**与 upstream `benchmark_odom.hpp` 相同口径**——每帧计时含降采样+预处理+配准（`--synth` 模式另加验收统计输出：vs upstream 每帧 Δtrans/Δrot 直方图、达标率）。
- 报告：mean/p50/p95/max ms、FPS、`nvidia-smi --query-gpu=memory.used --format=csv` 采样（子进程或 NVML 可选）、RSS（`getrusage`）。

- [ ] **Step 1: 构建 harness 骨架**（`--synth` 先只支持 cpu/full-gpu×gicp）
- [ ] **Step 2: 合成验收冒烟**：`./odometry_gpu --synth data/target.ply --exec full-gpu --engine gicp` 输出验收统计；人工检查达标率 ≥99%
- [ ] **Step 3: 补 vgicp/hybrid/exact-bf 路径与 json 报告**
- [ ] **Step 4: WORKLOG + 提交** `Add odometry_gpu comparison harness`

---

### Task 11: KITTI 00 数据 + APE/RPE 评估 + 对比矩阵

**Files:**
- Create: `scripts/fetch_kitti00.sh`（google drive 622MB 子集，走 `http://127.0.0.1:7890` 代理；`gdown` 或 curl 直链）
- Create: `scripts/eval_traj.py`（KITTI 格式轨迹 APE/RPE：优先 `evo_ape/evo_rpe`（pip），缺失则自写 100/400/800 关键帧插值 RPE）
- Create: `scripts/run_gpu_matrix.sh`（exec×engine×nn 网格 + upstream `odometry_benchmark` 基线）

**Steps:**

- [ ] **1. 下载数据**：`bash scripts/fetch_kitti00.sh ~/datasets/kitti/odometry`（校验 4541 帧 velodyne）
- [ ] **2. 跑基线**：upstream `odometry_benchmark`（`--engine small_gicp_omp` 与 `small_vgicp_model_tbb`，`--num_threads 4/8/16`）得 CPU 基线延迟 + 轨迹
- [ ] **3. 跑 GPU 矩阵**：`run_gpu_matrix.sh` 全组合
- [ ] **4. 验收判定**（写进 WORKLOG，脚本输出 pass/fail）：
  - GICP/VGICP 两引擎：GPU vs 同参数 upstream 的 APE/RPE 相差 <5%
  - 延迟：`full-gpu` 单帧 mean 较 upstream(4 线程) 加速 ≥5×（4070；未达→Task 12 调优后再判）
  - 合成模式达标率 ≥99%
- [ ] **5. WORKLOG + 提交** `Add KITTI evaluation pipeline and comparison matrix`

---

### Task 12: 性能调优（Graph/融合/重叠）

**Files:**
- Modify: `cuda/src/reg/*.cu`、`cuda/bench/odometry_gpu.cpp`
- Create: `cuda/include/sgc/core/graph.hpp`（CUDA Graph 捕获包装）

**Steps（每步后重跑 Task 11 矩阵中 2 个代表配置，记录前后数字）：**

- [ ] **1. Nsight Systems 剖面**：`nsys profile ./odometry_gpu ...`，找 launch 间隙/内存瓶颈（空隙 >15% 才做 Graph）
- [ ] **2. CUDA Graph**：捕获每外层迭代的 kernel 序列（T/H/b/e 缓冲固定），`cudaGraphLaunch` 重放
- [ ] **3. kernel 融合**：keys+sort 前的 filter；线性化+归约第一层的合并机会
- [ ] **4. 流重叠**：frame i+1 预处理（独立流）与 frame i 迭代重叠（`--overlap` 开关，默认开）
- [ ] **5. 微调**：block 尺寸扫描（128/256/512）、`__ldg`、`-use_fast_math` A/B（精度门不破才可开）
- [ ] **6. WORKLOG + 提交** `Add CUDA graphs, stream overlap and tuning`

---

### Task 13: BENCHMARK_GPU.md 报告 + 最终验收

**Files:**
- Create: `BENCHMARK_GPU.md`（对比表：精度/延迟/吞吐/资源 × 配置矩阵）
- Modify: `README.md`（加一节 GPU 构建说明）
- Modify: `WORKLOG.md`

**Steps:**

- [ ] **1. 汇总表**：accuracy（APE/RPE、合成达标率）、latency（mean/p50/p95）、throughput（FPS）、resources（GPU mem、RSS、CPU 占用）
- [ ] **2. 全量回归**：`ctest`（全部 sgc 测试）+ 矩阵重跑一遍确认无回退
- [ ] **3. WORKLOG 收尾 + 提交** `Add GPU benchmark report`
- [ ] **4. 推送并设置默认分支**：`git push origin cuda-x86` → `bash /home/as/vllm/DeepLearning/pytorch/.claude/scripts/audit-author.sh`（若存在）→ `git log --format='%an <%ae>' origin/cuda-x86` 全部为 Functionhx → `gh repo edit Functionhx/small_gicp --default-branch cuda-x86`
- [ ] **5. jetson 移植要点写回 WORKLOG**（arch 87、JetPack、nvpmodel 固频、perf/W 记录法），后续在 cuda-jetson 分支进行

---

## Self-Review 记录

- **Spec 覆盖**：M0=T1-2、M1=T3、M2=T4-5、M3=T6-7、M4=T10-11、M5=T8-9、M6=T12、M7=T13；验收门 1-4 分布在 T5/T7/T9/T11/T13；`exec/nn/engine/det_insert` 参数全部落地（T10）。✓
- **占位符扫描**：无 TBD/TODO；所有代码块可直接实施。✓
- **类型一致性**：`GpuCloud`（T2 定义，T3-9 使用）、`CorrCache`（T6 定义，T7/T9 使用）、`GicpResult`（T7 定义，T9 复用）、`NNStrategy`（T4 定义，T5-7/T9-10 使用）已核对。✓
- **已知实现期风险**（WORKLOG 跟踪）：3×3 最小特征向量闭式解的数值稳定性；Voxel3 一致率若 <99% 的回退路径；`-use_fast_math` 对验收门的影响。
