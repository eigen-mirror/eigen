# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

function(check_cuda_flags name environment toolchain expected_detection expected_build expected_cache)
  set(build "${WORK_DIR}/${name}-${clang}-${custom}")
  set(toolchain_file "${build}/toolchain.cmake")
  file(WRITE "${toolchain_file}" "set(CMAKE_CUDA_FLAGS_INIT \"${toolchain}\")\n")
  bs_configure("CUDA flags: ${name}" "${BS_CONSUMER_DIR}/gpu_flags" "${build}"
               "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}"
               "-DCMAKE_TOOLCHAIN_FILE=${toolchain_file}"
               "-DEIGEN_TEST_CUDA_CLANG=${clang}"
               "-DEIGEN_CUDA_CXX_FLAGS=${custom}"
               "-DENVIRONMENT_FLAGS=${environment}"
               "-DEXPECTED_DETECTION=${expected_detection}"
               "-DEXPECTED_BUILD=${expected_build}"
               "-DEXPECTED_CACHE=${expected_cache}" ${ARGN})

  # The cache keeps the initial defaults; changed Eigen flags replace the previous extras on reconfigure.
  file(WRITE "${toolchain_file}" "set(CMAKE_CUDA_FLAGS_INIT \"-DCHANGED_INIT=1\")\n")
  bs_configure("CUDA flags: ${name}, reconfigure" "${BS_CONSUMER_DIR}/gpu_flags" "${build}"
               "-DEIGEN_CUDA_CXX_FLAGS=-DCHANGED_EIGEN=1"
               "-DENVIRONMENT_FLAGS=-DCHANGED_ENV=1"
               "-DEXPECTED_DETECTION=${expected_build}")
endfunction()

foreach(clang OFF ON)
  foreach(custom "" "-DEIGEN_FLAG=1")
    check_cuda_flags(empty "" "" "" "-DPLATFORM_FLAG=1" "-DPLATFORM_FLAG=1")
    check_cuda_flags(environment "-DENV_FLAG=1" "" "-DENV_FLAG=1"
                     "-DENV_FLAG=1 -DPLATFORM_FLAG=1" "-DENV_FLAG=1 -DPLATFORM_FLAG=1")
    check_cuda_flags(toolchain "" "-DINIT_FLAG=1" "-DINIT_FLAG=1"
                     "-DINIT_FLAG=1 -DPLATFORM_FLAG=1" "-DINIT_FLAG=1 -DPLATFORM_FLAG=1")
    check_cuda_flags(combined "-DENV_FLAG=1" "-DINIT_FLAG=1" "-DENV_FLAG=1 -DINIT_FLAG=1"
                     "-DENV_FLAG=1 -DINIT_FLAG=1 -DPLATFORM_FLAG=1"
                     "-DENV_FLAG=1 -DINIT_FLAG=1 -DPLATFORM_FLAG=1")
    check_cuda_flags(cache "-DENV_FLAG=1" "-DINIT_FLAG=1" "-DCACHE_FLAG=1" "-DCACHE_FLAG=1" "-DCACHE_FLAG=1"
                     "-DCMAKE_CUDA_FLAGS=-DCACHE_FLAG=1")
    check_cuda_flags(cache_empty "-DENV_FLAG=1" "-DINIT_FLAG=1" "" "" "" "-DCMAKE_CUDA_FLAGS=")
    check_cuda_flags(normal "-DENV_FLAG=1" "-DINIT_FLAG=1" "-DNORMAL_FLAG=1" "-DNORMAL_FLAG=1"
                     "-DENV_FLAG=1 -DINIT_FLAG=1 -DPLATFORM_FLAG=1" "-DNORMAL_FLAGS=-DNORMAL_FLAG=1")
    check_cuda_flags(normal_empty "-DENV_FLAG=1" "-DINIT_FLAG=1" "" ""
                     "-DENV_FLAG=1 -DINIT_FLAG=1 -DPLATFORM_FLAG=1" "-DNORMAL_FLAGS=")
  endforeach()
endforeach()
