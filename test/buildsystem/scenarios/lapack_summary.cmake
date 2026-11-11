# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

bs_configure("LAPACK summary without Python" "${BS_CONSUMER_DIR}/lapack_summary"
             "${WORK_DIR}/build" "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}"
             "-DEIGEN_LAPACK_TEST_ALLOWED_FAILURES=")
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
bs_run_expect_output(WHAT "successful LAPACK summary"
       MATCHES "TOTAL: 120 tests, 0 numerical errors, 0 other errors"
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")
file(READ "${report}" contents)
bs_assert_streq("${contents}" "Note; unrelated run information\nAll tests passed ( 100 tests run)\nAll tests passed ( 20 tests run)\n"
                "combined LAPACK report")

bs_run_expect_output(WHAT "registered LAPACK example counts"
       MATCHES "TOTAL: 332 tests, 0 numerical errors, 0 other errors"
       COMMAND "${CMAKE_COMMAND}"
               "-DOUTPUT_MANIFEST=${WORK_DIR}/build/lapack-build/lapack_test_outputs.txt"
               "-DREPORT_FILE=${report}" -P "${summary}")

file(WRITE "${manifest}" "${first}\n")
file(READ "${WORK_DIR}/build/lapack-reference/TESTING/dgbak.in" balancing)
string(REPLACE "0.000D+00" "0.125D+00" finite "${balancing}")
string(REPLACE "example number having largest error = 0"
               "example number having largest error = 7" finite "${finite}")
file(WRITE "${first}" "${finite}")
bs_run_expect_output(WHAT "finite balancing error maximum"
       MATCHES "13 tests, 0 numerical errors, 0 other errors"
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")

foreach(kind bal bak gbal gbak)
  file(READ "${WORK_DIR}/build/lapack-reference/TESTING/d${kind}.in" balancing)
  string(REPLACE "number of examples where info is not 0 = 0"
                 "number of examples where info is not 0 = 2" failing "${balancing}")
  file(WRITE "${first}" "${failing}")
  bs_run_expect_output(WHAT "balancing ${kind} routine errors"
         EXPECT_RESULT FAILURE MATCHES "13 tests, 0 numerical errors, 2 other errors"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
endforeach()

foreach(kind bal gbal)
  file(READ "${WORK_DIR}/build/lapack-reference/TESTING/d${kind}.in" balancing)
  string(REPLACE "example number where ILO or IHI wrong = 0"
                 "example number where ILO or IHI wrong = 7" failing "${balancing}")
  string(REPLACE "number of examples where info is not 0 = 0"
                 "number of examples where info is not 0 = 1" failing "${failing}")
  file(WRITE "${first}" "${failing}")
  bs_run_expect_output(WHAT "balancing ${kind} ILO/IHI error"
         EXPECT_RESULT FAILURE MATCHES "13 tests, 0 numerical errors, 1 other errors"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
endforeach()

foreach(error Infinity -Infinity NaN "************")
  string(REPLACE "0.000D+00" "${error}" failing "${balancing}")
  file(WRITE "${first}" "${failing}")
  bs_run_expect_output(WHAT "balancing error ${error}"
         EXPECT_RESULT FAILURE MATCHES "13 tests, 1 numerical errors, 0 other errors"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
endforeach()

file(WRITE "${manifest}" "${first}\n${second}\n")
set(balance " .. test output of SGEBAL ..\n value of largest test error = 0.197D+01\n example number where info is not zero = 0\n example number where ILO or IHI wrong = 0\n example number having largest error = 11\n number of examples where info is not 0 = 0\n total number of examples tested = 13\n")
file(WRITE "${first}" "${balance}")
bs_run_expect_output(WHAT "successful balance examples with ordinary tests"
       MATCHES "TOTAL: 33 tests, 0 numerical errors, 0 other errors"
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")

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
  bs_run_expect_output(WHAT "LAPACK balance summary: ${case}"
         EXPECT_RESULT FAILURE MATCHES "${expected}"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
endforeach()

set(ec "${WORK_DIR}/dec.out")
file(WRITE "${ec}" " Error in DLALN2: RMAX = 0.123E+03\n Error in DLANV2: RMAX = 0.456E+03\n")
foreach(mode alone mixed)
  file(WRITE "${manifest}" "${ec}\n")
  if(mode STREQUAL "mixed")
    file(APPEND "${manifest}" "${second}\n")
  endif()
  bs_run_expect_output(WHAT "LAPACK diagnostic-only xCHKEC output: ${mode}"
         EXPECT_RESULT FAILURE MATCHES "dec.out: 0 tests, 0 numerical errors, 2 other errors"
         EXCLUDES "No LAPACK numerical tests were analyzed"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                 "-DREPORT_FILE=${report}" -P "${summary}")
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
      bs_run_expect_output(WHAT "LAPACK equilibration failure: ${precision}${routine}, ${mode}"
             EXPECT_RESULT FAILURE
             MATCHES "TOTAL: ${tests_run} tests, 1 numerical errors, 0 other errors"
             EXCLUDES "No LAPACK numerical tests were analyzed"
             COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
                     "-DREPORT_FILE=${report}" -P "${summary}")
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
  bs_run_expect_output(WHAT "LAPACK summary: ${case}"
         EXPECT_RESULT FAILURE MATCHES "${expected}"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}" "-DREPORT_FILE=${report}" -P "${summary}")
endforeach()

# xCHKDMD summaries: three DGEDMD checks and three DGEDMDQ checks.
set(dmd "${WORK_DIR}/ddmd.out")
set(dmd_passed " >>>> Z - U*V test PASSED.\n >>>> A*U test PASSED. \n >>>> Rezidual computation test PASSED.\n >>>> DGEDMD :: ALL TESTS PASSED.\n >>>> DGEDMD and DGEDMDQ computed singular values test PASSED.\n >>>> F - Q*R test PASSED.\n >>>> Rezidual computation test PASSED.\n >>>>>>> DGEDMDQ :: ALL TESTS PASSED.\n")
file(WRITE "${manifest}" "${dmd}\n")
set(dmd_case " FAILED! Something was wrong with the run.\n")
foreach(case passed check singular_values run summarized stopped)
  set(contents "${dmd_passed}")
  set(result FAILURE)
  if(case STREQUAL "passed")
    set(result 0)
    set(expected "6 tests, 0 numerical errors, 0 other errors")
  elseif(case STREQUAL "check")
    string(REPLACE " >>>> A*U test PASSED. \n"
                   " A*U test FAILED            3  time(s)\n           3 FAILURES!\n" contents "${contents}")
    set(expected "6 tests, 3 numerical errors, 0 other errors")
  elseif(case STREQUAL "singular_values")
    string(REPLACE " >>>> DGEDMD and DGEDMDQ computed singular values test PASSED.\n"
                   " DGEDMD and DGEDMDQ discrepancies in the singular values unacceptable            2  times. Test FAILED.\n"
                   contents "${contents}")
    set(expected "6 tests, 2 numerical errors, 0 other errors")
  elseif(case STREQUAL "run")
    # A per-case failure line that the summary does not count.
    string(PREPEND contents "${dmd_case}")
    set(expected "6 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "summarized")
    # Per-case failure lines for the singular-value and F - Q*R checks, which the summary counts again.
    string(REPLACE " >>>> DGEDMD and DGEDMDQ computed singular values test PASSED.\n >>>> F - Q*R test PASSED.\n"
                   " DGEDMD and DGEDMDQ discrepancies in the singular values unacceptable            1  times. Test FAILED.\n F - Q*R test FAILED            1  time(s)\n"
                   contents "${contents}")
    string(PREPEND contents "${dmd_case}${dmd_case}")
    set(expected "6 tests, 2 numerical errors, 0 other errors")
  else()
    # The driver stopped before its summary.
    set(contents "${dmd_case}")
    set(expected "ddmd.out: 0 tests, 0 numerical errors, 1 other errors")
  endif()
  file(WRITE "${dmd}" "${contents}")
  bs_run_expect_output(WHAT "LAPACK DMD summary: ${case}"
         EXPECT_RESULT ${result} MATCHES "${expected}"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}" "-DREPORT_FILE=${report}" -P "${summary}")
endforeach()

# Allowed failures bound each output's counts; FAILURES_FILE records them in the same format.
set(allowed "${WORK_DIR}/allowed.txt")
set(failures "${WORK_DIR}/failures.txt")
set(fragile "${WORK_DIR}/ded.out")
file(WRITE "${fragile}" "DEV:    3 out of   20 tests failed to pass the threshold\n *** Error code from DGEES = 6\n")
file(WRITE "${manifest}" "${fragile}\n${second}\n")
foreach(case none equal more fewer malformed duplicate unmatched)
  set(result FAILURE)
  if(case STREQUAL "none")
    file(WRITE "${allowed}" "# Nothing is allowed.\n\n")
    set(expected "ded.out: 20 tests, 3 numerical errors, 1 other errors; more than the 0 numerical and 0 other allowed")
  elseif(case STREQUAL "equal")
    file(WRITE "${allowed}" "# Comment\nded.out 3 1  # trailing comment\n")
    set(result 0)
    set(expected "ded.out: 20 tests, 3 numerical errors, 1 other errors; allowed")
  elseif(case STREQUAL "more")
    file(WRITE "${allowed}" "ded.out 2 1\n")
    set(expected "more than the 2 numerical and 1 other allowed.*LAPACK test output reports failures: ded.out")
  elseif(case STREQUAL "fewer")
    file(WRITE "${allowed}" "ded.out 5 1\n")
    set(result 0)
    set(expected "fewer than the 5 numerical and 1 other allowed, lower its entry")
  elseif(case STREQUAL "malformed")
    file(WRITE "${allowed}" "ded.out 3\n")
    set(expected "Malformed LAPACK allowed-failure entry: ded.out 3")
  elseif(case STREQUAL "duplicate")
    file(WRITE "${allowed}" "ded.out 3 1\nded.out 3 1\n")
    set(expected "Duplicate LAPACK allowed-failure entry: ded.out")
  else()
    file(WRITE "${allowed}" "ded.out 3 1\nsed.out 1 0\n")
    set(expected "Allowed-failure entries name no LAPACK test output: sed.out")
  endif()
  file(REMOVE "${failures}")
  bs_run_expect_output(WHAT "LAPACK allowed failures: ${case}"
         EXPECT_RESULT ${result} MATCHES "${expected}"
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}" "-DREPORT_FILE=${report}"
                 "-DALLOWED_FAILURES=${allowed}" "-DFAILURES_FILE=${failures}" -P "${summary}")
  if(case STREQUAL "equal")
    file(READ "${failures}" contents)
    bs_assert_streq("${contents}" "ded.out 3 1\n" "recorded LAPACK failures")
  endif()
endforeach()
bs_run_expect_output(WHAT "missing LAPACK allowed-failure file"
       EXPECT_RESULT FAILURE MATCHES "Missing LAPACK allowed-failure file"
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}" "-DREPORT_FILE=${report}"
               "-DALLOWED_FAILURES=${WORK_DIR}/absent.txt" -P "${summary}")

file(WRITE "${WORK_DIR}/build/lapack-reference/TESTING/runtest.cmake" [=[
get_filename_component(name "${INPUT}" NAME)
if(name STREQUAL "stest.in")
  message(FATAL_ERROR "simulated LAPACK driver failure")
endif()
configure_file("${INPUT}" "${OUTPUT}" COPYONLY)
]=])
set(registered_report "${WORK_DIR}/build/lapack-build/TESTING/testing_results.txt")
file(WRITE "${registered_report}" "previous summary\n")
bs_run_expect_output(WHAT "failed LAPACK fixture skips summary"
       EXPECT_RESULT FAILURE MATCHES "LAPACK_Test_Summary[^\n]*Not Run"
       COMMAND "${CMAKE_COMMAND}" -E chdir "${WORK_DIR}/build"
               "${CMAKE_CTEST_COMMAND}" -C "${BUILD_CONFIG}" --parallel 2
               -R "^LAPACK_Test_Summary$" --output-on-failure --no-tests=error)
file(READ "${registered_report}" contents)
bs_assert_streq("${contents}" "previous summary\n" "skipped LAPACK summary preserves its report")
