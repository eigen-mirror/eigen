# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

project(EigenLapack CXX)

if(EIGEN_BUILD_LAPACK AND EIGEN_BUILD_BLAS)

option(EIGEN_ENABLE_LAPACK_TESTS "Enable the Lapack unit tests" OFF)

if(EIGEN_ENABLE_LAPACK_TESTS)
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
      if ("${CMAKE_Fortran_COMPILER_VERSION}" VERSION_GREATER_EQUAL 10.0)
        # Reference LAPACK sources contain argument type mismatches.
        # Allow them to compile anyway with newer GNU versions.
        set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
      endif()
    endif()
    set(EIGEN_Fortran_COMPILER_WORKS ON)
  else()
    set(EIGEN_Fortran_COMPILER_WORKS OFF)
  endif()

  if(NOT EIGEN_Fortran_COMPILER_WORKS)
    message(STATUS "A Fortran compiler was not found; LAPACK unit tests won't be enabled")
    set(EIGEN_ENABLE_LAPACK_TESTS OFF)
  endif()
endif()

add_custom_target(lapack)
include_directories(../blas)

set(EigenLapack_SRCS
  dsecnd_INT_CPU_TIME.cpp second_INT_CPU_TIME.cpp single.cpp double.cpp complex_single.cpp complex_double.cpp
  ilaver.cpp
)

set(EIGEN_LAPACK_DEF "eigen_lapack.def")

if(EIGEN_ENABLE_LAPACK_TESTS)

  include(FetchContent)
  FetchContent_Declare(eigen_lapack_reference
    URL https://codeload.github.com/Reference-LAPACK/lapack/tar.gz/refs/tags/v3.12.1
    URL_HASH SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
    TLS_VERIFY ON)
  FetchContent_GetProperties(eigen_lapack_reference)
  if(NOT eigen_lapack_reference_POPULATED)
    FetchContent_Populate(eigen_lapack_reference)
  endif()
  set(LAPACK_SOURCE_DIR "${eigen_lapack_reference_SOURCE_DIR}")
  set(LAPACK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    set(EigenLapack_funcfilenames
        ssyev.f   dsyev.f   csyev.f   zsyev.f
        spotrf.f  dpotrf.f  cpotrf.f  zpotrf.f
        spotrs.f  dpotrs.f  cpotrs.f  zpotrs.f
        sgetrf.f  dgetrf.f  cgetrf.f  zgetrf.f
        sgetrs.f  dgetrs.f  cgetrs.f  zgetrs.f
        slarft.f  dlarft.f  clarft.f  zlarft.f
        slarfb.f  dlarfb.f  clarfb.f  zlarfb.f
        slarfg.f  dlarfg.f  clarfg.f  zlarfg.f
        slarf.f   dlarf.f   clarf.f   zlarf.f
        sladiv.f  dladiv.f  cladiv.f  zladiv.f
        ilaslr.f  iladlr.f  ilaclr.f  ilazlr.f
        ilaslc.f  iladlc.f  ilaclc.f  ilazlc.f
        dlapy2.f  dlapy3.f  slapy2.f  slapy3.f
        clacgv.f  zlacgv.f
        slamch.f  dlamch.f
        ilaver.f)

    file(GLOB ReferenceLapack_SRCS0 "${LAPACK_SOURCE_DIR}/SRC/*.f"
         "${LAPACK_SOURCE_DIR}/SRC/*.F" "${LAPACK_SOURCE_DIR}/SRC/*.f90"
         "${LAPACK_SOURCE_DIR}/SRC/*.F90"
         "${LAPACK_SOURCE_DIR}/SRC/DEPRECATED/*.f")
    list(APPEND ReferenceLapack_SRCS0
      "${LAPACK_SOURCE_DIR}/INSTALL/sroundup_lwork.f"
      "${LAPACK_SOURCE_DIR}/INSTALL/droundup_lwork.f")
    foreach(filename1 IN LISTS ReferenceLapack_SRCS0)
      get_filename_component(filename "${filename1}" NAME)
      list(FIND EigenLapack_SRCS ${filename} id1)
      list(FIND EigenLapack_funcfilenames ${filename} id2)
      if((id1 EQUAL -1) AND (id2 EQUAL -1))
        list(APPEND ReferenceLapack_SRCS "${filename1}")
      endif()
    endforeach()
endif()

set(EIGEN_LAPACK_TARGETS "")

add_library(eigen_lapack_static STATIC ${EigenLapack_SRCS} ${ReferenceLapack_SRCS})
target_link_libraries(eigen_lapack_static  eigen_blas_static)
list(APPEND EIGEN_LAPACK_TARGETS eigen_lapack_static)

if (EIGEN_BUILD_SHARED_LIBS)
  add_library(eigen_lapack SHARED ${EigenLapack_SRCS} ${EIGEN_LAPACK_DEF})
  # Build LAPACK but link BLAS.
  target_compile_definitions(eigen_lapack PUBLIC "EIGEN_BLAS_LINK_DLL" "EIGEN_LAPACK_BUILD_DLL")
  target_link_libraries(eigen_lapack  eigen_blas)
  set_target_properties(eigen_lapack PROPERTIES CXX_VISIBILITY_PRESET hidden
    VERSION ${EIGEN_VERSION_NUMBER}
    SOVERSION ${EIGEN_MAJOR_VERSION})
  list(APPEND EIGEN_LAPACK_TARGETS eigen_lapack)
endif()

foreach(target IN LISTS EIGEN_LAPACK_TARGETS)
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${target} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  target_link_libraries(${target} Eigen3::Eigen)
  add_dependencies(lapack ${target})
  if(EIGEN_INSTALL)
    install(TARGETS ${target}
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
  endif()
endforeach()


if(EIGEN_ENABLE_LAPACK_TESTS)

  set(BUILD_SINGLE      true)
  set(BUILD_DOUBLE      true)
  set(BUILD_COMPLEX     true)
  set(BUILD_COMPLEX16   true)

  if(MSVC_VERSION)
#  string(REPLACE "/STACK:10000000" "/STACK:900000000000000000"
#    CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  string(REGEX REPLACE "(.*)/STACK:(.*) (.*)" "\\1/STACK:900000000000000000 \\3"
    CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  endif()
  file(MAKE_DIRECTORY "${LAPACK_BINARY_DIR}/TESTING")
  set(LAPACK_LIBRARIES eigen_lapack_static)
  set(BLAS_LIBRARIES eigen_blas_static)
  set(TMGLIB tmglib)
  set(LAPACK_VERSION 3.12.1)
  set(LAPACK_MAJOR_VERSION 3)
  # Upstream's test libraries are build-only dependencies.
  function(lapack_install_library target)
  endfunction()
  add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/MATGEN" testing/MATGEN)
  add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/LIN" testing/LIN)
  add_subdirectory("${LAPACK_SOURCE_DIR}/TESTING/EIG" testing/EIG)
  # Some LAPACK 3.12.1 checks are expected to fail: obsolete bitwise eigenvalue
  # comparisons, unnormalized RFP errors, a balance-test denominator missing ABS,
  # and a reference-CSTEIN orthogonality limitation. These are LAPACK-side issues;
  # keep the tests enabled and let the summary report their failures.
  set(EIGEN_LAPACK_TEST_NAMES "")
  set(EIGEN_LAPACK_TEST_OUTPUTS "")
  macro(add_lapack_test output input target)
    set(TEST_INPUT "${LAPACK_SOURCE_DIR}/TESTING/${input}")
    set(TEST_OUTPUT "${LAPACK_BINARY_DIR}/TESTING/${output}")
    string(REPLACE "." "_" input_name ${input})
    set(testName "${target}_${input_name}")
    if(EXISTS "${TEST_INPUT}")
      add_dependencies(buildtests ${target})
      list(APPEND EIGEN_LAPACK_TEST_NAMES "LAPACK-${testName}")
      list(APPEND EIGEN_LAPACK_TEST_OUTPUTS "${TEST_OUTPUT}")
      add_test(NAME LAPACK-${testName}
        COMMAND "${CMAKE_COMMAND}"
        -DTEST=$<TARGET_FILE:${target}>
        -DINPUT=${TEST_INPUT}
        -DOUTPUT=${TEST_OUTPUT}
        -DINTDIR=${CMAKE_CFG_INTDIR}
        -P "${LAPACK_SOURCE_DIR}/TESTING/runtest.cmake")
    endif()
  endmacro()

  if (BUILD_SINGLE)
  add_lapack_test(stest.out stest.in xlintsts)
  #
  # ======== SINGLE RFP LIN TESTS ========================
  add_lapack_test(stest_rfp.out stest_rfp.in xlintstrfs)
  #
  #
  # ======== SINGLE EIG TESTS ===========================
  #

  add_lapack_test(snep.out nep.in xeigtsts)


  add_lapack_test(ssep.out sep.in xeigtsts)


  add_lapack_test(ssvd.out svd.in xeigtsts)


  add_lapack_test(sec.out sec.in xeigtsts)


  add_lapack_test(sed.out sed.in xeigtsts)


  add_lapack_test(sgg.out sgg.in xeigtsts)


  add_lapack_test(sgd.out sgd.in xeigtsts)


  add_lapack_test(ssb.out ssb.in xeigtsts)


  add_lapack_test(ssg.out ssg.in xeigtsts)


  add_lapack_test(sbal.out sbal.in xeigtsts)


  add_lapack_test(sbak.out sbak.in xeigtsts)


  add_lapack_test(sgbal.out sgbal.in xeigtsts)


  add_lapack_test(sgbak.out sgbak.in xeigtsts)


  add_lapack_test(sbb.out sbb.in xeigtsts)


  add_lapack_test(sglm.out glm.in xeigtsts)


  add_lapack_test(sgqr.out gqr.in xeigtsts)


  add_lapack_test(sgsv.out gsv.in xeigtsts)


  add_lapack_test(scsd.out csd.in xeigtsts)


  add_lapack_test(slse.out lse.in xeigtsts)
  endif()

  if (BUILD_DOUBLE)
  #
  # ======== DOUBLE LIN TESTS ===========================
  add_lapack_test(dtest.out dtest.in xlintstd)
  #
  # ======== DOUBLE RFP LIN TESTS ========================
  add_lapack_test(dtest_rfp.out dtest_rfp.in xlintstrfd)
  #
  # ======== DOUBLE EIG TESTS ===========================

  add_lapack_test(dnep.out nep.in xeigtstd)


  add_lapack_test(dsep.out sep.in xeigtstd)


  add_lapack_test(dsvd.out svd.in xeigtstd)


  add_lapack_test(dec.out dec.in xeigtstd)


  add_lapack_test(ded.out ded.in xeigtstd)


  add_lapack_test(dgg.out dgg.in xeigtstd)


  add_lapack_test(dgd.out dgd.in xeigtstd)


  add_lapack_test(dsb.out dsb.in xeigtstd)


  add_lapack_test(dsg.out dsg.in xeigtstd)


  add_lapack_test(dbal.out dbal.in xeigtstd)


  add_lapack_test(dbak.out dbak.in xeigtstd)


  add_lapack_test(dgbal.out dgbal.in xeigtstd)


  add_lapack_test(dgbak.out dgbak.in xeigtstd)


  add_lapack_test(dbb.out dbb.in xeigtstd)


  add_lapack_test(dglm.out glm.in xeigtstd)


  add_lapack_test(dgqr.out gqr.in xeigtstd)


  add_lapack_test(dgsv.out gsv.in xeigtstd)


  add_lapack_test(dcsd.out csd.in xeigtstd)


  add_lapack_test(dlse.out lse.in xeigtstd)
  endif()

  if (BUILD_COMPLEX)
  add_lapack_test(ctest.out ctest.in xlintstc)
  #
  # ======== COMPLEX RFP LIN TESTS ========================
  add_lapack_test(ctest_rfp.out ctest_rfp.in xlintstrfc)
  #
  # ======== COMPLEX EIG TESTS ===========================

  add_lapack_test(cnep.out nep.in xeigtstc)


  add_lapack_test(csep.out sep.in xeigtstc)


  add_lapack_test(csvd.out svd.in xeigtstc)


  add_lapack_test(cec.out cec.in xeigtstc)


  add_lapack_test(ced.out ced.in xeigtstc)


  add_lapack_test(cgg.out cgg.in xeigtstc)


  add_lapack_test(cgd.out cgd.in xeigtstc)


  add_lapack_test(csb.out csb.in xeigtstc)


  add_lapack_test(csg.out csg.in xeigtstc)


  add_lapack_test(cbal.out cbal.in xeigtstc)


  add_lapack_test(cbak.out cbak.in xeigtstc)


  add_lapack_test(cgbal.out cgbal.in xeigtstc)


  add_lapack_test(cgbak.out cgbak.in xeigtstc)


  add_lapack_test(cbb.out cbb.in xeigtstc)


  add_lapack_test(cglm.out glm.in xeigtstc)


  add_lapack_test(cgqr.out gqr.in xeigtstc)


  add_lapack_test(cgsv.out gsv.in xeigtstc)


  add_lapack_test(ccsd.out csd.in xeigtstc)


  add_lapack_test(clse.out lse.in xeigtstc)
  endif()

  if (BUILD_COMPLEX16)
  #
  # ======== COMPLEX16 LIN TESTS ========================
  add_lapack_test(ztest.out ztest.in xlintstz)
  #
  # ======== COMPLEX16 RFP LIN TESTS ========================
  add_lapack_test(ztest_rfp.out ztest_rfp.in xlintstrfz)
  #
  # ======== COMPLEX16 EIG TESTS ===========================

  add_lapack_test(znep.out nep.in xeigtstz)


  add_lapack_test(zsep.out sep.in xeigtstz)


  add_lapack_test(zsvd.out svd.in xeigtstz)


  add_lapack_test(zec.out zec.in xeigtstz)


  add_lapack_test(zed.out zed.in xeigtstz)


  add_lapack_test(zgg.out zgg.in xeigtstz)


  add_lapack_test(zgd.out zgd.in xeigtstz)


  add_lapack_test(zsb.out zsb.in xeigtstz)


  add_lapack_test(zsg.out zsg.in xeigtstz)


  add_lapack_test(zbal.out zbal.in xeigtstz)


  add_lapack_test(zbak.out zbak.in xeigtstz)


  add_lapack_test(zgbal.out zgbal.in xeigtstz)


  add_lapack_test(zgbak.out zgbak.in xeigtstz)


  add_lapack_test(zbb.out zbb.in xeigtstz)


  add_lapack_test(zglm.out glm.in xeigtstz)


  add_lapack_test(zgqr.out gqr.in xeigtstz)


  add_lapack_test(zgsv.out gsv.in xeigtstz)


  add_lapack_test(zcsd.out csd.in xeigtstz)


  add_lapack_test(zlse.out lse.in xeigtstz)
  endif()


  if (BUILD_SIMPLE)
      if (BUILD_DOUBLE)
  #
  # ======== SINGLE-DOUBLE PROTO LIN TESTS ==============
          add_lapack_test(dstest.out dstest.in xlintstds)
      endif()
  endif()


  if (BUILD_COMPLEX)
      if (BUILD_COMPLEX16)
  #
  # ======== COMPLEX-COMPLEX16 LIN TESTS ========================
          add_lapack_test(zctest.out zctest.in xlintstzc)
      endif()
  endif()

  # ==============================================================================

  if(EIGEN_LAPACK_TEST_NAMES)
    string(REPLACE ";" "\n" lapack_outputs "${EIGEN_LAPACK_TEST_OUTPUTS}")
    file(WRITE "${LAPACK_BINARY_DIR}/lapack_test_outputs.txt" "${lapack_outputs}\n")
    add_test(
      NAME LAPACK_Test_Summary
      WORKING_DIRECTORY ${LAPACK_BINARY_DIR}
      COMMAND "${CMAKE_COMMAND}"
              "-DOUTPUT_MANIFEST=${LAPACK_BINARY_DIR}/lapack_test_outputs.txt"
              "-DREPORT_FILE=${LAPACK_BINARY_DIR}/TESTING/testing_results.txt"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/testing_summary.cmake"
    )
    set_tests_properties(${EIGEN_LAPACK_TEST_NAMES} PROPERTIES FIXTURES_SETUP lapack_outputs)
    set_tests_properties(LAPACK_Test_Summary PROPERTIES FIXTURES_REQUIRED lapack_outputs)
  endif()

endif()

elseif(EIGEN_BUILD_LAPACK AND NOT EIGEN_BUILD_BLAS)
 message(FATAL_ERROR "EIGEN_BUILD_LAPACK requires EIGEN_BUILD_BLAS")
endif() #EIGEN_BUILD_LAPACK
