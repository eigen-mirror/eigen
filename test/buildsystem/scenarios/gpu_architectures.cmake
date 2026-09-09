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

# Observe the architecture setting at language enablement, before compiler ABI detection.
macro(enable_language language)
  bs_assert_streq("${language}" "HIP" "enabled GPU language")
  bs_assert_streq("${CMAKE_HIP_ARCHITECTURES}" "${expected}" "HIP architectures before compiler detection")
endmacro()

function(check_hip_architectures eigen_arch cmake_arch expected)
  set(EIGEN_TEST_CUDA OFF)
  set(EIGEN_TEST_HIP ON)
  unset(EIGEN_HIP_ARCHITECTURES CACHE)
  if(NOT "${eigen_arch}" STREQUAL "")
    set(EIGEN_HIP_ARCHITECTURES "${eigen_arch}")
  endif()
  set(CMAKE_HIP_ARCHITECTURES "${cmake_arch}")
  ei_gpu_testing_enable()
endfunction()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
  check_hip_architectures("" ""
    "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151")
  check_hip_architectures("gfx906;gfx1100" "" "gfx906;gfx1100")
  check_hip_architectures("gfx906" "gfx942" "gfx942")
endif()
