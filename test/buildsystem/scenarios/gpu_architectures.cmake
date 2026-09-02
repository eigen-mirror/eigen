# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")

# Architecture selection must not require an installed toolkit or GPU.
function(check_architectures toolkit eigen_arch cmake_arch environment_arch expected)
  set(CUDAToolkit_VERSION "${toolkit}")
  set(EIGEN_CUDA_COMPUTE_ARCH "${eigen_arch}")
  set(CMAKE_CUDA_ARCHITECTURES "${cmake_arch}")
  set(ENV{CUDAARCHS} "${environment_arch}")
  ei_cuda_resolve_compute_arch()
  bs_assert_streq("${EIGEN_CUDA_COMPUTE_ARCH}" "${expected}" "CUDA architecture selection")
endfunction()

check_architectures(12.8 "" "" "" 60)
check_architectures(13.3 "" "" "" 75)
check_architectures(13.3 "" "89" "" 89)
check_architectures(13.3 "" "75;89-real" "" "75;89-real")
check_architectures(13.3 "" "native" "" native)
check_architectures(13.3 "" "OFF" "" OFF)
check_architectures(13.3 "" "" "89" 89)
check_architectures(13.3 "" "89" "75" 89)
check_architectures(13.3 "86" "89" "75" 86)
