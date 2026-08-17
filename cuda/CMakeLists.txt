find_package(GTest REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(sgc STATIC
  src/core/buffer.cu
  src/voxel/downsample.cu
  src/voxel/hash_index.cu
  src/voxel/voxel_hash_map.cu
  src/preproc/covariance.cu
  src/search/nn.cu
  src/reg/linearize.cu
  src/reg/gicp.cu
  src/reg/vgicp.cu
)
target_include_directories(sgc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(sgc PUBLIC Eigen3::Eigen CUDA::cudart)
target_compile_options(sgc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

#############
## Tests ####
#############
if(BUILD_TESTS OR BUILD_CUDA)
  file(GLOB SGC_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
  foreach(TEST_SOURCE ${SGC_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
      sgc
      GTest::gtest_main
      small_gicp
    )
    gtest_discover_tests(${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
endif()

##############
## Benches ###
##############
add_executable(odometry_gpu bench/odometry_gpu.cpp)
target_link_libraries(odometry_gpu PRIVATE sgc CUDA::cudart)
