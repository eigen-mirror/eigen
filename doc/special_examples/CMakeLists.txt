# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

add_executable(Tutorial_sparse_example Tutorial_sparse_example.cpp Tutorial_sparse_example_details.cpp)
target_link_libraries(Tutorial_sparse_example ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO} Eigen3::Eigen)
add_dependencies(all_examples Tutorial_sparse_example)

add_custom_command(
  TARGET Tutorial_sparse_example
  POST_BUILD
  COMMAND Tutorial_sparse_example
  ARGS ${CMAKE_CURRENT_BINARY_DIR}/Tutorial_sparse_example.pgm
)

add_executable(random_generators random_generators.cpp)
target_link_libraries(random_generators ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO} Eigen3::Eigen)
add_dependencies(all_examples random_generators)

add_custom_command(
  TARGET random_generators
  POST_BUILD
  COMMAND random_generators
  ARGS >${CMAKE_CURRENT_BINARY_DIR}/random_generators.out
)
