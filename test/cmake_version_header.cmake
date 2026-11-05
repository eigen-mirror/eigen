# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# CMake installs the Eigen/Version header that the configure step generates from
# cmake/Version.in in place of the source one, so the generated header must
# define every version macro exactly as the source header does. This also
# catches a source header whose EIGEN_VERSION_STRING disagrees with its fields,
# and a master header without a prerelease tag: only release commits on release
# branches clear EIGEN_PRERELEASE_VERSION (see RELEASING.md).

foreach(header IN ITEMS SOURCE_HEADER GENERATED_HEADER)
  if(NOT EXISTS "${${header}}")
    message(FATAL_ERROR "${header} \"${${header}}\" does not exist.")
  endif()
endforeach()
file(READ "${SOURCE_HEADER}" source)
file(READ "${GENERATED_HEADER}" generated)

foreach(macro IN ITEMS EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION
                       EIGEN_PATCH_VERSION EIGEN_PRERELEASE_VERSION
                       EIGEN_BUILD_VERSION EIGEN_VERSION_STRING)
  set(regex "#define[ \t]+${macro}[ \t]+([0-9]+|\"[^\"]*\")")
  string(REGEX MATCH "${regex}" source_match "${source}")
  set(source_value "${CMAKE_MATCH_1}")
  string(REGEX MATCH "${regex}" generated_match "${generated}")
  set(generated_value "${CMAKE_MATCH_1}")
  if(source_match STREQUAL "")
    message(FATAL_ERROR "${SOURCE_HEADER} does not define ${macro}.")
  endif()
  if(generated_match STREQUAL "")
    message(FATAL_ERROR "${GENERATED_HEADER} does not define ${macro}.")
  endif()
  if(NOT generated_value STREQUAL source_value)
    message(FATAL_ERROR "${macro} is ${generated_value} in the generated header "
                        "${GENERATED_HEADER} but ${source_value} in ${SOURCE_HEADER}.")
  endif()
  set(${macro} "${source_value}")
endforeach()

if(EIGEN_BUILD_VERSION STREQUAL "\"master\"" AND EIGEN_PRERELEASE_VERSION STREQUAL "\"\"")
  message(FATAL_ERROR "${SOURCE_HEADER} declares a release (EIGEN_PRERELEASE_VERSION \"\") "
                      "with EIGEN_BUILD_VERSION \"master\": master must keep a prerelease "
                      "tag such as \"dev\".")
endif()
