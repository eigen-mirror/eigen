###
#
# @copyright (c) 2009-2014 The University of Tennessee and The University
#                          of Tennessee Research Foundation.
#                          All rights reserved.
# @copyright (c) 2012-2014 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
###
#
# - Find PASTIX include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(PASTIX
#               [REQUIRED] # Fail with error if pastix is not found
#               [COMPONENTS <comp1> <comp2> ...] # dependencies
#              )
#
# Requires PaStiX 6.x and its pkg-config metadata, including SpM and ordering dependencies.
#
# This module finds headers and pastix library.
# Results are reported in variables:
#  PASTIX_FOUND            - True if headers and requested libraries were found
#  PASTIX_VERSION          - Detected PaStiX version
#  PASTIX_INCLUDE_DIRS     - pastix include directories
#  PASTIX_LIBRARIES        - pastix libraries
#  PASTIX_INCLUDE_DIRS_DEP - pastix + dependencies include directories
#  PASTIX_LIBRARIES_DEP    - pastix libraries + dependencies
#
# The user can give specific paths where to find the libraries adding cmake
# options at configure (ex: cmake path/to/project -DPASTIX_DIR=path/to/pastix):
#  PASTIX_DIR              - Where to find the base directory of pastix
# SPDX-License-Identifier: BSD-3-Clause

#=============================================================================
# Copyright 2012-2013 Inria
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013      Florent Pruvost
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file MORSE-Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of Morse, substitute the full
#  License text for the above reference.)


find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  set(_eigen_pastix_prefix_path "${CMAKE_PREFIX_PATH}")
  if(PASTIX_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${PASTIX_DIR}")
  endif()
  pkg_check_modules(PASTIX6 QUIET IMPORTED_TARGET GLOBAL "pastix>=6")
  set(CMAKE_PREFIX_PATH "${_eigen_pastix_prefix_path}")
  unset(_eigen_pastix_prefix_path)
endif()

set(PASTIX_VERSION "${PASTIX6_VERSION}")
set(PASTIX_INCLUDE_DIRS ${PASTIX6_INCLUDE_DIRS})
set(PASTIX_INCLUDE_DIRS_DEP ${PASTIX6_INCLUDE_DIRS})
set(PASTIX_LIBRARIES "")
set(PASTIX_LIBRARIES_DEP "")
if(PASTIX6_FOUND AND PASTIX6_VERSION VERSION_LESS 7)
  # FindPkgConfig's imported target uses only the shared-link metadata, even
  # when find_library resolves PaStiX to a static archive.
  foreach(_pastix_library IN LISTS PASTIX6_LINK_LIBRARIES)
    if(_pastix_library MATCHES "[/\\\\]libpastix\\.a$")
      set(_pastix_static_libraries "")
      foreach(_pastix_dependency IN LISTS PASTIX6_STATIC_LIBRARIES)
        if(_pastix_dependency STREQUAL "pastix")
          list(APPEND _pastix_static_libraries "${_pastix_library}")
          continue()
        endif()
        find_library(_pastix_dependency_path NAMES ${_pastix_dependency}
          HINTS ${PASTIX6_STATIC_LIBRARY_DIRS})
        if(_pastix_dependency_path)
          list(APPEND _pastix_static_libraries "${_pastix_dependency_path}")
        else()
          list(APPEND _pastix_static_libraries "${_pastix_dependency}")
        endif()
        unset(_pastix_dependency_path CACHE)
      endforeach()
      set_target_properties(PkgConfig::PASTIX6 PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_pastix_static_libraries}"
        INTERFACE_LINK_DIRECTORIES "${PASTIX6_STATIC_LIBRARY_DIRS}"
        INTERFACE_LINK_OPTIONS "${PASTIX6_STATIC_LDFLAGS_OTHER}"
        INTERFACE_INCLUDE_DIRECTORIES "${PASTIX6_STATIC_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${PASTIX6_STATIC_CFLAGS_OTHER}")
      unset(_pastix_static_libraries)
      unset(_pastix_dependency)
      break()
    endif()
  endforeach()
  unset(_pastix_library)
  set(PASTIX_LIBRARIES PkgConfig::PASTIX6)
  set(PASTIX_LIBRARIES_DEP PkgConfig::PASTIX6)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PASTIX
  REQUIRED_VARS PASTIX6_FOUND PASTIX_LIBRARIES
  VERSION_VAR PASTIX_VERSION)
