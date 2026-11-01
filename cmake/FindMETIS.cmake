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
# - Find METIS include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(METIS
#               [REQUIRED]             # Fail with error if metis is not found
#              )
#
# This module finds headers and metis library.
# Results are reported in variables:
#  METIS_FOUND           - True if headers and requested libraries were found
#  METIS_INCLUDE_DIRS    - metis include directories
#  METIS_LIBRARY_DIRS    - Link directories for metis libraries
#  METIS_LIBRARIES       - metis component libraries to be linked
#
# The user can give specific paths where to find the libraries adding cmake
# options at configure (ex: cmake path/to/project -DMETIS_DIR=path/to/metis):
#  METIS_DIR             - Where to find the base directory of metis
#  METIS_INCDIR          - Where to find the header files
#  METIS_LIBDIR          - Where to find the library files
# The module can also look for the following environment variables if paths
# are not given as cmake variable: METIS_DIR, METIS_INCDIR, METIS_LIBDIR
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

set(METIS_DIR "" CACHE PATH "Installation directory of METIS library")

# Preserve the historical environment hints in addition to CMake's normal search paths.
if(WIN32)
  set(_metis_include_env INCLUDE)
  set(_metis_library_env LIB)
else()
  set(_metis_include_env INCLUDE C_INCLUDE_PATH CPATH INCLUDE_PATH)
  if(APPLE)
    set(_metis_library_env DYLD_LIBRARY_PATH)
  else()
    set(_metis_library_env LD_LIBRARY_PATH)
  endif()
endif()
set(_metis_include_hints)
foreach(variable IN LISTS _metis_include_env)
  file(TO_CMAKE_PATH "$ENV{${variable}}" paths)
  list(APPEND _metis_include_hints ${paths})
endforeach()
set(_metis_library_hints)
foreach(variable IN LISTS _metis_library_env)
  file(TO_CMAKE_PATH "$ENV{${variable}}" paths)
  list(APPEND _metis_library_hints ${paths})
endforeach()
if(NOT WIN32)
  list(APPEND _metis_library_hints ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}
    ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES})
endif()

set(_metis_include_suffixes)
if(METIS_INCDIR)
  set(_metis_include_hints "${METIS_INCDIR}")
elseif(METIS_DIR)
  set(_metis_include_hints "${METIS_DIR}")
  set(_metis_include_suffixes include include/metis)
elseif(DEFINED ENV{METIS_INCDIR} AND NOT "$ENV{METIS_INCDIR}" STREQUAL "")
  set(_metis_include_hints "$ENV{METIS_INCDIR}")
elseif(DEFINED ENV{METIS_DIR} AND NOT "$ENV{METIS_DIR}" STREQUAL "")
  set(_metis_include_hints "$ENV{METIS_DIR}" "$ENV{METIS_DIR}/include" "$ENV{METIS_DIR}/include/metis")
endif()
if(NOT METIS_INCDIR AND NOT METIS_DIR)
  list(APPEND _metis_include_hints ${CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES}
    ${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

set(_metis_library_suffixes)
if(METIS_LIBDIR)
  set(_metis_library_hints "${METIS_LIBDIR}")
elseif(METIS_DIR)
  set(_metis_library_hints "${METIS_DIR}")
  set(_metis_library_suffixes lib lib32 lib64)
elseif(DEFINED ENV{METIS_LIBDIR} AND NOT "$ENV{METIS_LIBDIR}" STREQUAL "")
  set(_metis_library_hints "$ENV{METIS_LIBDIR}")
elseif(DEFINED ENV{METIS_DIR} AND NOT "$ENV{METIS_DIR}" STREQUAL "")
  set(_metis_library_hints "$ENV{METIS_DIR}" "$ENV{METIS_DIR}/lib")
endif()

# Re-search on each call so changed installation hints also work with a warm cache.
set(METIS_metis.h_DIRS "METIS_metis.h_DIRS-NOTFOUND")
set(METIS_metis_LIBRARY "METIS_metis_LIBRARY-NOTFOUND")
find_path(METIS_metis.h_DIRS NAMES metis.h HINTS ${_metis_include_hints}
  PATH_SUFFIXES ${_metis_include_suffixes})
find_library(METIS_metis_LIBRARY NAMES metis HINTS ${_metis_library_hints}
  PATH_SUFFIXES ${_metis_library_suffixes})
set(METIS_INCLUDE_DIRS "${METIS_metis.h_DIRS}")
set(METIS_LIBRARIES "${METIS_metis_LIBRARY}")
set(METIS_LIBRARY_DIRS "METIS_LIBRARY_DIRS-NOTFOUND")

unset(METIS_WORKS CACHE)
unset(METIS_WORKS)
if(METIS_LIBRARIES)
  get_filename_component(METIS_LIBRARY_DIRS "${METIS_metis_LIBRARY}" DIRECTORY)
  string(REGEX REPLACE "/lib(32|64)?$" "" _metis_prefix "${METIS_LIBRARY_DIRS}")
  set(METIS_DIR_FOUND "${_metis_prefix}" CACHE PATH "Installation directory of METIS library" FORCE)

  include(CMakePushCheckState)
  include(CheckFunctionExists)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES "${METIS_LIBRARIES}")
  find_library(M_LIBRARY NAMES m)
  if(M_LIBRARY)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${M_LIBRARY}")
  endif()
  check_function_exists(METIS_NodeND METIS_WORKS)
  cmake_pop_check_state()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS DEFAULT_MSG METIS_LIBRARIES METIS_WORKS METIS_INCLUDE_DIRS)
mark_as_advanced(METIS_DIR METIS_DIR_FOUND METIS_metis.h_DIRS METIS_metis_LIBRARY METIS_WORKS M_LIBRARY)
