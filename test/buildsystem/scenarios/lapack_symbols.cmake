# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# lapack/testing_symbols.cmake on stand-in libraries: the routine split it must
# accept, and one of each split it must reject.
bs_configure("LAPACK symbol stand-ins" "${BS_CONSUMER_DIR}/lapack_symbols" "${WORK_DIR}/build")
bs_build("LAPACK symbol stand-ins" "${WORK_DIR}/build")
include("${WORK_DIR}/build/libraries-${BUILD_CONFIG}.cmake")
if(NOT NM)
  message(STATUS "[${SCENARIO}] skipped: the toolchain has no nm, so lapack/CMakeLists.txt registers no symbol check")
  return()
endif()

set(def "${WORK_DIR}/eigen_lapack.def")
set(exports "LIBRARY eigen_lapack\nEXPORTS\n    dgesv_\n    dpotrf_\n    xerbla_\n")
function(check_symbols what expected_result expected def_contents reference)
  file(WRITE "${def}" "${def_contents}")
  bs_run(WHAT "LAPACK symbol check: ${what}" EXPECT_RESULT ${expected_result} OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DNM=${NM}" "-DDEF_FILE=${def}"
                 "-DEIGEN_LAPACK=${EIGEN_LAPACK}" "-DEIGEN_BLAS=${EIGEN_BLAS}" "-DREFERENCE=${reference}"
                 ${ARGN} -P "${EIGEN_SOURCE_DIR}/lapack/testing_symbols.cmake")
  if(NOT output MATCHES "${expected}")
    bs_fail("wrong LAPACK symbol check result for ${what}: ${output}")
  endif()
endfunction()

# Passing requires counting the weak xerbla_, and ignoring the reference library's
# undefined dgesv_, the C++ helper and the C helper without a trailing underscore.
check_symbols("clean split" 0 "3 routines from Eigen, 1 from reference LAPACK" "${exports}" "${REFERENCE}")
string(REPLACE "    dpotrf_\n" "" unexported "${exports}")
check_symbols("unexported routine" FAILURE "eigen_lapack defines dpotrf_, which eigen_lapack\\.def does not export"
              "${unexported}" "${REFERENCE}")
check_symbols("undefined export" FAILURE "eigen_lapack\\.def exports dfoo_, which no Eigen library defines"
              "${exports}    dfoo_\n" "${REFERENCE}")
check_symbols("routine defined twice" FAILURE "dpotrf_ is defined by both Eigen and the reference LAPACK library"
              "${exports}" "${REFERENCE_OVERLAP}")

if(DEFINED EIGEN_LAPACK_SHARED)
  check_symbols("shared exports" 0 "3 routines from Eigen" "${exports}" "${REFERENCE}"
                "-DEIGEN_LAPACK_SHARED=${EIGEN_LAPACK_SHARED}" "-DEIGEN_BLAS_SHARED=${EIGEN_BLAS_SHARED}")
  check_symbols("hidden shared export" FAILURE "The shared eigen_lapack library does not export dpotrf_"
                "${exports}" "${REFERENCE}"
                "-DEIGEN_LAPACK_SHARED=${EIGEN_LAPACK_SHARED_HIDDEN}" "-DEIGEN_BLAS_SHARED=${EIGEN_BLAS_SHARED}")
endif()

# Exercise both C ABI spellings on every host, including weak and undefined symbols.
set(NM "${FAKE_NM}")
set(EIGEN_LAPACK "${WORK_DIR}/lapack-symbols.txt")
set(EIGEN_BLAS "${WORK_DIR}/blas-symbols.txt")
set(REFERENCE "${WORK_DIR}/reference-symbols.txt")
foreach(prefix "" "_")
  file(WRITE "${EIGEN_LAPACK}" "0000 T ${prefix}dgesv_\n0000 T ${prefix}dpotrf_\n"
                              "0000 T ${prefix}_Z18fake_lapack_helperv\n0000 T ${prefix}eigen_lapack_helper\n")
  file(WRITE "${EIGEN_BLAS}" "0000 T ${prefix}daxpy_\n0000 W ${prefix}xerbla_\n")
  file(WRITE "${REFERENCE}" "     U ${prefix}dgesv_\n0000 T ${prefix}dgetrf_\n")
  check_symbols("ABI prefix '${prefix}'" 0 "3 routines from Eigen, 1 from reference LAPACK"
                "${exports}" "${REFERENCE}")
  file(APPEND "${REFERENCE}" "0000 T ${prefix}dpotrf_\n")
  check_symbols("ABI prefix '${prefix}' overlap" FAILURE
                "dpotrf_ is defined by both Eigen and the reference LAPACK library" "${exports}" "${REFERENCE}")
endforeach()
