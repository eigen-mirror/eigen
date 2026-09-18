# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# How the .cu tests are compiled. Everything the GPU test build decides lives here, so ei_add_test_internal only
# has to ask which of three modes is in force:
#
#   cuda-language   CMake's first-class CUDA language, the default. nvcc, or clang as the CUDA compiler.
#   cuda-as-cxx     the .cu files compiled as C++ by a compiler that drives CUDA itself. nvc++ has no CMake CUDA
#                   language support, and CMake refuses clang as the CUDA compiler on Windows (issue #20776).
#   hip             CMake's HIP language.
#
# ei_gpu_testing_enable() is called once from the top-level CMakeLists, which is where enable_language() has to
# happen: it affects the calling directory and below, and test/ and contrib/test/ are siblings.

# Resolve EIGEN_CUDA_COMPUTE_ARCH against the toolkit version. Empty selects the oldest architecture the toolkit
# still compiles for: sm_60, Eigen's documented floor (see Macros.h), or sm_75 from CUDA 13 on, which dropped
# offline compilation for Pascal and Volta. Honor CMake's architecture setting (including CUDAARCHS) when the
# Eigen-specific override is empty.
macro(ei_cuda_resolve_compute_arch)
  if("${EIGEN_CUDA_COMPUTE_ARCH}" STREQUAL "")
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
      set(EIGEN_CUDA_COMPUTE_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    elseif(NOT "$ENV{CUDAARCHS}" STREQUAL "")
      set(EIGEN_CUDA_COMPUTE_ARCH "$ENV{CUDAARCHS}")
    elseif(CUDAToolkit_VERSION VERSION_LESS 13.0)
      set(EIGEN_CUDA_COMPUTE_ARCH 60)
    else()
      set(EIGEN_CUDA_COMPUTE_ARCH 75)
    endif()
  endif()
  set_property(GLOBAL PROPERTY EIGEN_CUDA_COMPUTE_ARCH_RESOLVED "${EIGEN_CUDA_COMPUTE_ARCH}")
endmacro()

macro(ei_gpu_testing_enable)
  set(EIGEN_GPU_TEST_MODE "none")

  if(EIGEN_TEST_CUDA)
    # Honours CUDAToolkit_ROOT, which is how one of several installed toolkits is chosen. Without it, an explicit nvcc
    # selects its own toolkit, <root>/bin/nvcc. find_package derives that only once the CUDA language is enabled, which
    # needs the toolkit version first; a toolkit other than the compiler's links a second CUDA runtime.
    if(CMAKE_CUDA_COMPILER AND NOT EIGEN_TEST_CUDA_NVC AND NOT EIGEN_TEST_CUDA_CLANG AND
       NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDAToolkit_ROOT})
      get_filename_component(CUDAToolkit_ROOT "${CMAKE_CUDA_COMPILER}" PROGRAM)
      get_filename_component(CUDAToolkit_ROOT "${CUDAToolkit_ROOT}" DIRECTORY)
      get_filename_component(CUDAToolkit_ROOT "${CUDAToolkit_ROOT}" DIRECTORY)
    endif()
    find_package(CUDAToolkit 11.8 REQUIRED)
    ei_cuda_resolve_compute_arch()

    if(EIGEN_TEST_CUDA_NVC OR (EIGEN_TEST_CUDA_CLANG AND WIN32))
      set(EIGEN_GPU_TEST_MODE "cuda-as-cxx")
      # CMake's named and suffixed architectures cannot be used verbatim in these compiler flags.
      foreach(arch IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
        if(NOT arch MATCHES "^[0-9]+$")
          message(FATAL_ERROR "EIGEN_CUDA_COMPUTE_ARCH=${EIGEN_CUDA_COMPUTE_ARCH}: this compiler route takes numeric "
                              "architectures only, for example 75;89.")
        endif()
      endforeach()
      if(EIGEN_TEST_CUDA_NVC)
        string(APPEND CMAKE_CXX_FLAGS " -cuda")
        foreach(arch IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
          string(APPEND CMAKE_CXX_FLAGS " -gpu=cc${arch}")
        endforeach()
      else()
        string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDAToolkit_TARGET_DIR}")
        foreach(arch IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
          string(APPEND CMAKE_CXX_FLAGS " --cuda-gpu-arch=sm_${arch}")
        endforeach()
      endif()
      string(APPEND CMAKE_CXX_FLAGS " ${EIGEN_CUDA_CXX_FLAGS}")
    else()
      set(EIGEN_GPU_TEST_MODE "cuda-language")
      # CMP0104 is NEW from 3.18, so the architectures have to be known before the language is enabled.
      if(EIGEN_CUDA_COMPUTE_ARCH MATCHES "native|all|all-major" AND CMAKE_VERSION VERSION_LESS 3.24)
        message(FATAL_ERROR "EIGEN_CUDA_COMPUTE_ARCH=${EIGEN_CUDA_COMPUTE_ARCH} needs CMake 3.24 or later; "
                            "name the architectures instead, for example 75;89.")
      endif()
      set(CMAKE_CUDA_ARCHITECTURES "${EIGEN_CUDA_COMPUTE_ARCH}")
      if(NOT CMAKE_CUDA_COMPILER)
        if(EIGEN_TEST_CUDA_CLANG)
          set(CMAKE_CUDA_COMPILER "${CMAKE_CXX_COMPILER}")
        else()
          set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
        endif()
      endif()
      # nvcc drives a host compiler; keep it the one the rest of the build uses. MSVC picks its own.
      if(NOT EIGEN_TEST_CUDA_CLANG AND NOT MSVC AND NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
      endif()
      if(NOT CMAKE_CUDA_STANDARD AND CMAKE_CXX_STANDARD)
        set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
      endif()
      # ccache in front of the C++ compiler should front the CUDA compiler too, which is what the generated nvcc
      # wrapper script used to arrange for FindCUDA.
      if(NOT CMAKE_CUDA_COMPILER_LAUNCHER AND CMAKE_CXX_COMPILER_LAUNCHER)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CMAKE_CXX_COMPILER_LAUNCHER}")
      endif()
      # Compiler identification compiles with CMAKE_CUDA_FLAGS, so flags it depends on go in before the language is
      # enabled: clang looks in /usr/local/cuda unless told otherwise, which need not be the toolkit found above, and
      # refuses a version it does not support; custom flags may carry nvcc's -allow-unsupported-compiler.
      set(eigen_cuda_flags "${EIGEN_CUDA_CXX_FLAGS}")
      if(EIGEN_TEST_CUDA_CLANG)
        string(PREPEND eigen_cuda_flags "--cuda-path=${CUDAToolkit_TARGET_DIR} ")
      endif()
      set(eigen_cuda_flags_initialized FALSE)
      if(DEFINED CMAKE_CUDA_FLAGS)
        set(eigen_cuda_flags_initialized TRUE)
      else()
        set(CMAKE_CUDA_FLAGS "$ENV{CUDAFLAGS} ${CMAKE_CUDA_FLAGS_INIT}")
      endif()
      string(APPEND CMAKE_CUDA_FLAGS " ${eigen_cuda_flags}")
      enable_language(CUDA)
      if(NOT eigen_cuda_flags_initialized)
        # Reveal CMake's initialized cache, including platform defaults added after compiler identification.
        # Keep Eigen's additions out of that cache so reconfiguration neither duplicates nor retains old flags.
        unset(CMAKE_CUDA_FLAGS)
        string(APPEND CMAKE_CUDA_FLAGS " ${eigen_cuda_flags}")
      endif()
      unset(eigen_cuda_flags)
      unset(eigen_cuda_flags_initialized)
      if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr -Xcudafe \"--display_error_number\"")
      endif()
      message(STATUS "CUDA tests: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}, "
                     "architectures ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

  elseif(EIGEN_TEST_HIP)
    if(CMAKE_VERSION VERSION_LESS 3.21)
      message(FATAL_ERROR "EIGEN_TEST_HIP needs CMake 3.21 or later for the HIP language.")
    endif()
    set(EIGEN_GPU_TEST_MODE "hip")
    # ROCM_PATH selected the installation for the find_package(HIP) path this replaces. The HIP language finds
    # ROCm through CMAKE_HIP_COMPILER_ROCM_ROOT and the Clang shipped beside it, so seed both from ROCM_PATH,
    # leaving anything the user configured directly, and anything CMake's own hipconfig probe finds when
    # ROCM_PATH names no installation, untouched.
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
    if(IS_DIRECTORY "${ROCM_PATH}")
      if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}")
      endif()
      if(NOT DEFINED CMAKE_HIP_COMPILER AND EXISTS "${ROCM_PATH}/llvm/bin/clang++")
        set(CMAKE_HIP_COMPILER "${ROCM_PATH}/llvm/bin/clang++")
      endif()
    endif()
    # Compiler ABI detection needs architectures even on build-only hosts without an AMD GPU.
    if(NOT DEFINED EIGEN_HIP_ARCHITECTURES)
      set(EIGEN_HIP_ARCHITECTURES
          gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151
          CACHE STRING "HIP GPU architectures to build Eigen's HIP tests for.")
    endif()
    # A nonempty false value such as OFF is CMake's request for no architecture flags, not an absent setting.
    if("${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
      set(CMAKE_HIP_ARCHITECTURES "${EIGEN_HIP_ARCHITECTURES}")
    endif()
    enable_language(HIP)
    message(STATUS "HIP tests: ${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}, "
                   "architectures ${CMAKE_HIP_ARCHITECTURES}")
  endif()
endmacro()

# The GPU part of ei_add_test_internal: build one .cu test in whichever mode is in force.
function(ei_add_gpu_test_executable targetname filename)
  if(EIGEN_GPU_TEST_MODE STREQUAL "hip")
    set_source_files_properties(${filename} PROPERTIES LANGUAGE HIP)
    add_executable(${targetname} ${filename})
    target_compile_definitions(${targetname} PRIVATE EIGEN_USE_HIP)
  elseif(EIGEN_GPU_TEST_MODE STREQUAL "cuda-as-cxx")
    # The compiler drives CUDA itself, from flags set in ei_gpu_testing_enable.
    set_source_files_properties(${filename} PROPERTIES LANGUAGE CXX)
    add_executable(${targetname} ${filename})
    # nvc++ already links its shared runtime. Adding the static runtime duplicates kernel registration state.
    if(NOT EIGEN_TEST_CUDA_NVC)
      target_link_libraries(${targetname} CUDA::cudart_static)
    endif()
  else()
    add_executable(${targetname} ${filename})
    target_link_libraries(${targetname} CUDA::cudart_static)
  endif()
endfunction()
