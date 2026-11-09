# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

FILE(GLOB examples_SRCS "*.cpp")

set(EIGEN_SYCL ON)
list(APPEND CMAKE_EXE_LINKER_FLAGS -pthread)
set(CMAKE_CXX_STANDARD 17)

FOREACH(example_src ${examples_SRCS})
  GET_FILENAME_COMPONENT(example ${example_src} NAME_WE)
  ei_add_test_internal(${example} example_${example})
  ADD_DEPENDENCIES(contrib_examples example_${example})
ENDFOREACH(example_src)
set(EIGEN_SYCL OFF)
