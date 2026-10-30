# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

bs_configure("LAPACK summary without Python" "${BS_CONSUMER_DIR}/lapack_summary"
             "${WORK_DIR}/build" "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}")
# Selecting only the summary must generate its inputs, including under -j.
bs_run(WHAT "summary selects registered LAPACK tests without Python"
       COMMAND "${CMAKE_COMMAND}" -E chdir "${WORK_DIR}/build"
               "${CMAKE_CTEST_COMMAND}" -C "${BUILD_CONFIG}" --parallel 2
               -R "^LAPACK_Test_Summary$" --output-on-failure --no-tests=error)

set(summary "${EIGEN_SOURCE_DIR}/lapack/testing_summary.cmake")
set(report "${WORK_DIR}/report.txt")
set(manifest "${WORK_DIR}/outputs.txt")
set(first "${WORK_DIR}/first output.out")
set(second "${WORK_DIR}/second.out")
file(WRITE "${first}" "Note; unrelated run information\nAll tests passed ( 100 tests run)\n")
file(WRITE "${second}" "All tests passed ( 20 tests run)\n")
file(WRITE "${WORK_DIR}/stale.out" "999 out of 999 tests failed\n")
file(WRITE "${manifest}" "${first}\n${second}\n")
bs_run(WHAT "successful LAPACK summary" OUTPUT_VARIABLE output
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")
if(NOT output MATCHES "TOTAL: 120 tests, 0 numerical errors, 0 other errors")
  bs_fail("incorrect successful summary: ${output}")
endif()
file(READ "${report}" contents)
bs_assert_streq("${contents}" "Note; unrelated run information\nAll tests passed ( 100 tests run)\nAll tests passed ( 20 tests run)\n"
                "combined LAPACK report")

bs_run(WHAT "registered LAPACK example counts" OUTPUT_VARIABLE output
       COMMAND "${CMAKE_COMMAND}"
               "-DOUTPUT_MANIFEST=${WORK_DIR}/build/lapack-build/lapack_test_outputs.txt"
               "-DREPORT_FILE=${report}" -P "${summary}")
if(NOT output MATCHES "TOTAL: 328 tests, 0 numerical errors, 0 other errors")
  bs_fail("incorrect registered summary: ${output}")
endif()

file(WRITE "${manifest}" "${first}\n")
file(READ "${WORK_DIR}/build/lapack-reference/TESTING/dgbak.in" balancing)
string(REPLACE "0.000D+00" "0.125D+00" finite "${balancing}")
string(REPLACE "example number having largest error = 0"
               "example number having largest error = 7" finite "${finite}")
file(WRITE "${first}" "${finite}")
bs_run(WHAT "finite balancing error maximum" OUTPUT_VARIABLE output
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")
if(NOT output MATCHES "13 tests, 0 numerical errors, 0 other errors")
  bs_fail("incorrect finite balancing summary: ${output}")
endif()

foreach(kind bal bak gbal gbak)
  file(READ "${WORK_DIR}/build/lapack-reference/TESTING/d${kind}.in" balancing)
  string(REPLACE "number of examples where info is not 0 = 0"
                 "number of examples where info is not 0 = 2" failing "${balancing}")
  file(WRITE "${first}" "${failing}")
  bs_run(WHAT "balancing ${kind} routine errors" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "13 tests, 0 numerical errors, 2 other errors")
    bs_fail("incorrect balancing ${kind} error count: ${output}")
  endif()
endforeach()

foreach(kind bal gbal)
  file(READ "${WORK_DIR}/build/lapack-reference/TESTING/d${kind}.in" balancing)
  string(REPLACE "example number where ILO or IHI wrong = 0"
                 "example number where ILO or IHI wrong = 7" failing "${balancing}")
  string(REPLACE "number of examples where info is not 0 = 0"
                 "number of examples where info is not 0 = 1" failing "${failing}")
  file(WRITE "${first}" "${failing}")
  bs_run(WHAT "balancing ${kind} ILO/IHI error" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "13 tests, 0 numerical errors, 1 other errors")
    bs_fail("incorrect balancing ${kind} ILO/IHI error count: ${output}")
  endif()
endforeach()

foreach(error Infinity -Infinity NaN "************")
  string(REPLACE "0.000D+00" "${error}" failing "${balancing}")
  file(WRITE "${first}" "${failing}")
  bs_run(WHAT "balancing error ${error}" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "13 tests, 1 numerical errors, 0 other errors")
    bs_fail("unrecognized balancing error ${error}: ${output}")
  endif()
endforeach()

file(WRITE "${manifest}" "${first}\n${second}\n")
set(balance " .. test output of SGEBAL ..\n value of largest test error = 0.197D+01\n example number where info is not zero = 0\n example number where ILO or IHI wrong = 0\n example number having largest error = 11\n number of examples where info is not 0 = 0\n total number of examples tested = 13\n")
file(WRITE "${first}" "${balance}")
bs_run(WHAT "successful balance examples with ordinary tests" OUTPUT_VARIABLE output
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")
if(NOT output MATCHES "TOTAL: 33 tests, 0 numerical errors, 0 other errors")
  bs_fail("incorrect balance summary: ${output}")
endif()

file(WRITE "${manifest}" "${first}\n")
foreach(case infinity nan overflow info bounds unrecorded_info)
  set(contents "${balance}")
  set(expected "13 tests, 1 numerical errors, 0 other errors")
  if(case STREQUAL "infinity")
    string(REPLACE "0.197D+01" "Infinity" contents "${contents}")
  elseif(case STREQUAL "nan")
    string(REPLACE "0.197D+01" "NaN" contents "${contents}")
    string(REPLACE "value of largest" "ratio of largest" contents "${contents}")
  elseif(case STREQUAL "overflow")
    string(REPLACE "0.197D+01" "************" contents "${contents}")
  elseif(case STREQUAL "info")
    string(REPLACE "info is not zero = 0" "DGGBAL info is not 0 = 7" contents "${contents}")
    string(REPLACE "info is not 0 = 0" "info is not 0 = 3" contents "${contents}")
    set(expected "13 tests, 0 numerical errors, 3 other errors")
  elseif(case STREQUAL "bounds")
    string(REPLACE "ILO or IHI wrong = 0" "ILO or IHI is wrong = 4" contents "${contents}")
    string(REPLACE "info is not 0 = 0" "info is not 0 = 1" contents "${contents}")
    set(expected "13 tests, 0 numerical errors, 1 other errors")
  else()
    string(REPLACE "info is not zero = 0" "DGGBAK(L) info is not 0 = 5" contents "${contents}")
    set(expected "13 tests, 0 numerical errors, 1 other errors")
  endif()
  file(WRITE "${first}" "${contents}")
  bs_run(WHAT "LAPACK balance summary: ${case}" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "${expected}")
    bs_fail("wrong balance failure for ${case}: ${output}")
  endif()
endforeach()

set(ec "${WORK_DIR}/dec.out")
file(WRITE "${ec}" " Error in DLALN2: RMAX = 0.123E+03\n Error in DLANV2: RMAX = 0.456E+03\n")
foreach(mode alone mixed)
  file(WRITE "${manifest}" "${ec}\n")
  if(mode STREQUAL "mixed")
    file(APPEND "${manifest}" "${second}\n")
  endif()
  bs_run(WHAT "LAPACK diagnostic-only xCHKEC output: ${mode}" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "dec.out: 0 tests, 0 numerical errors, 2 other errors"
      OR output MATCHES "No LAPACK numerical tests were analyzed")
    bs_fail("incorrect xCHKEC failure diagnostic: ${output}")
  endif()
endforeach()

foreach(precision S D C Z)
  foreach(routine GEEQU GBEQU POEQU PPEQU PBEQU)
    foreach(mode mixed alone)
      file(WRITE "${manifest}" "${first}\n")
      file(WRITE "${first}" " ${precision}${routine} failed test with value  0.100D+03 exceeding threshold  0.200D+02\n")
      set(tests_run 0)
      if(mode STREQUAL "mixed")
        file(APPEND "${first}" " All tests for ${precision}GE routines passed the threshold (    100 tests run)\n")
        set(tests_run 100)
      endif()
      bs_run(WHAT "LAPACK equilibration failure: ${precision}${routine}, ${mode}"
             EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
             COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                     "-DREPORT_FILE=${report}" -P "${summary}")
      if(NOT output MATCHES "TOTAL: ${tests_run} tests, 1 numerical errors, 0 other errors"
          OR output MATCHES "No LAPACK numerical tests were analyzed")
        bs_fail("incorrect equilibration failure diagnostic: ${output}")
      endif()
    endforeach()
  endforeach()
endforeach()

foreach(case numerical diagnostics xerbla error_exits info_summary error_code recorded combined_errors empty mixed_empty zero_examples missing)
  file(WRITE "${manifest}" "${first}\n")
  if(case STREQUAL "numerical")
    file(WRITE "${first}" "3 out of 20 tests failed ( 20 tests run)\n")
    set(expected "20 tests, 3 numerical errors, 0 other errors")
  elseif(case STREQUAL "diagnostics")
    file(WRITE "${first}" "100 tests run)\nillegal argument\nDGETRF returned INFO = 1\n")
    set(expected "100 tests, 0 numerical errors, 2 other errors")
  elseif(case STREQUAL "xerbla")
    file(WRITE "${first}" "100 tests run)\n*** XERBLA was called with SRNAME = DGETRS instead of DGETRF ***\n")
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "error_exits")
    file(WRITE "${first}" "100 tests run)\n*** DGE routines failed the tests of the error exits ***\n")
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "info_summary")
    file(WRITE "${first}" "*** DGETRF returned with INFO = 2 instead of 1\nAll tests for DGE routines passed the threshold ( 100 tests run)\n1 error messages recorded\n")
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "error_code" OR case STREQUAL "recorded" OR case STREQUAL "combined_errors")
    file(WRITE "${first}" "")
    if(NOT case STREQUAL "recorded")
      file(APPEND "${first}" " *** Error code from DGETRF = 1 for M = 2, N = 2, type 1\n")
    endif()
    file(APPEND "${first}" " All tests for DGE routines passed the threshold ( 100 tests run)\n")
    if(NOT case STREQUAL "error_code")
      file(APPEND "${first}" " 1 error messages recorded\n")
    endif()
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "zero_examples")
    string(REPLACE "total number of examples tested = 13"
                   "total number of examples tested = 0" empty "${balancing}")
    file(WRITE "${first}" "${empty}")
    set(expected "No LAPACK numerical tests were analyzed")
  elseif(case STREQUAL "empty" OR case STREQUAL "mixed_empty")
    file(WRITE "${first}" "")
    if(case STREQUAL "mixed_empty")
      file(WRITE "${manifest}" "${second}\n${first}\n")
    endif()
    set(expected "No LAPACK numerical tests were analyzed")
  else()
    file(REMOVE "${first}")
    set(expected "Missing LAPACK test output")
  endif()
  bs_run(WHAT "LAPACK summary: ${case}" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}" "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "${expected}")
    bs_fail("wrong failure for ${case}: ${output}")
  endif()
endforeach()

file(WRITE "${WORK_DIR}/build/lapack-reference/TESTING/runtest.cmake" [=[
get_filename_component(name "${INPUT}" NAME)
if(name STREQUAL "stest.in")
  message(FATAL_ERROR "simulated LAPACK driver failure")
endif()
configure_file("${INPUT}" "${OUTPUT}" COPYONLY)
]=])
set(registered_report "${WORK_DIR}/build/lapack-build/TESTING/testing_results.txt")
file(WRITE "${registered_report}" "previous summary\n")
bs_run(WHAT "failed LAPACK fixture skips summary" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
       COMMAND "${CMAKE_COMMAND}" -E chdir "${WORK_DIR}/build"
               "${CMAKE_CTEST_COMMAND}" -C "${BUILD_CONFIG}" --parallel 2
               -R "^LAPACK_Test_Summary$" --output-on-failure --no-tests=error)
if(NOT output MATCHES "LAPACK_Test_Summary[^\n]*Not Run")
  bs_fail("summary was not skipped after a failed driver: ${output}")
endif()
file(READ "${registered_report}" contents)
bs_assert_streq("${contents}" "previous summary\n" "skipped LAPACK summary preserves its report")
