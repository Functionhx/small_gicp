// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <vector>

#include <sgc/core/buffer.hpp>

TEST(Smoke, BufferRoundtripAndKernel) {
  sgc::GpuBuffer<float> buf(1024);
  ASSERT_EQ(buf.size(), 1024u);

  std::vector<float> host(1024, 3.5f);
  buf.upload(host.data(), host.size());

  sgc::launch_fill_kernel(buf.raw(), buf.size(), 1.25f);

  std::vector<float> out(1024);
  buf.download(out.data(), out.size());
  EXPECT_EQ(out[0], 1.25f);
  EXPECT_EQ(out[1023], 1.25f);
}

TEST(Smoke, BufferMoveSemantics) {
  sgc::GpuBuffer<float> a(16);
  sgc::GpuBuffer<float> b(std::move(a));
  EXPECT_EQ(a.size(), 0u);
  EXPECT_EQ(a.capacity(), 0u);
  EXPECT_EQ(a.raw(), nullptr);
  EXPECT_EQ(b.size(), 16u);
  EXPECT_EQ(b.capacity(), 16u);
}

TEST(Smoke, BufferRetainsCapacityAcrossResize) {
  sgc::GpuBuffer<float> buffer(128);
  float* const allocation = buffer.raw();

  buffer.resize(32);
  EXPECT_EQ(buffer.size(), 32u);
  EXPECT_EQ(buffer.capacity(), 128u);
  EXPECT_EQ(buffer.raw(), allocation);

  buffer.resize(0);
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_EQ(buffer.capacity(), 128u);
  EXPECT_EQ(buffer.raw(), allocation);

  buffer.resize(96);
  EXPECT_EQ(buffer.size(), 96u);
  EXPECT_EQ(buffer.capacity(), 128u);
  EXPECT_EQ(buffer.raw(), allocation);
}
