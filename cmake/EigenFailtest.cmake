# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Script-mode driver for the tests that ei_add_failtest() and
# ei_add_failtest_fixture() register (cmake/EigenTesting.cmake).
#
# build: deletes every failtest executable, then builds them all, so that an
#   executable exists afterwards exactly when its target built in this run.  It
#   always succeeds; the verdicts belong to the check tests.
# check: <name>_ok passes when its executable exists.  <name>_ko passes when its
#   executable is missing and the _ok twin's exists: a toolchain or build-system
#   failure breaks both variants, and must not pass as the asserted compile error.

cmake_minimum_required(VERSION 3.17)

set(config_args "")
if(NOT EIGEN_FAILTEST_CONFIG STREQUAL "")
  set(config_args --config "${EIGEN_FAILTEST_CONFIG}")
endif()

if(EIGEN_FAILTEST_ACTION STREQUAL "build")
  include("${EIGEN_FAILTEST_LIST}")
  file(REMOVE ${EIGEN_FAILTEST_FILES})

  set(keep_going "")
  set(parallel "")
  if(EIGEN_FAILTEST_GENERATOR MATCHES "Ninja")
    set(keep_going -k 0)
  elseif(EIGEN_FAILTEST_GENERATOR MATCHES "^(Unix|MinGW|MSYS) Makefiles$")
    set(keep_going -k)
    # Unlike Ninja, make is serial without -j.
    if(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
      cmake_host_system_information(RESULT jobs QUERY NUMBER_OF_LOGICAL_CORES)
      set(parallel --parallel ${jobs})
    endif()
  endif()

  if(keep_going)
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${EIGEN_FAILTEST_BINARY_DIR}" ${config_args}
                            --target buildfailtests ${parallel} -- ${keep_going})
  else()
    # Without a known keep-going flag, the first _ko failure would stop the
    # batch, so build one target at a time.
    foreach(target IN LISTS EIGEN_FAILTEST_TARGETS)
      execute_process(COMMAND "${CMAKE_COMMAND}" --build "${EIGEN_FAILTEST_BINARY_DIR}" ${config_args}
                              --target ${target})
    endforeach()
  endif()

elseif(EIGEN_FAILTEST_ACTION STREQUAL "check")
  if(DEFINED EIGEN_FAILTEST_KO)
    if(EXISTS "${EIGEN_FAILTEST_KO_FILE}")
      message(FATAL_ERROR "${EIGEN_FAILTEST_KO} built, but EIGEN_SHOULD_FAIL_TO_BUILD must break its compile.")
    endif()
    if(NOT EXISTS "${EIGEN_FAILTEST_OK_FILE}")
      message(FATAL_ERROR "${EIGEN_FAILTEST_OK} did not build either, so the failure of "
                          "${EIGEN_FAILTEST_KO} proves nothing; see the ${EIGEN_FAILTEST_OK} test.")
    endif()
  elseif(NOT EXISTS "${EIGEN_FAILTEST_OK_FILE}")
    # Rebuild the target alone to put its errors in this test's output.  The
    # lock keeps these builds from running concurrently in one binary directory.
    file(LOCK "${EIGEN_FAILTEST_LOCK}" GUARD PROCESS)
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${EIGEN_FAILTEST_BINARY_DIR}" ${config_args}
                            --target ${EIGEN_FAILTEST_OK})
    message(FATAL_ERROR "${EIGEN_FAILTEST_OK} did not build in the buildfailtests fixture.")
  endif()

else()
  message(FATAL_ERROR "Unknown EIGEN_FAILTEST_ACTION '${EIGEN_FAILTEST_ACTION}'.")
endif()
