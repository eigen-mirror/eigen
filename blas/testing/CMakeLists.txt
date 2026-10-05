# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
  set(EIGEN_Fortran_COMPILER_WORKS ON)
else()
  set(EIGEN_Fortran_COMPILER_WORKS OFF)
endif()

if(EIGEN_BUILD_BLAS_ILP64)
  if(TARGET eigen_blas_ilp64)
    set(eigen_blas_ilp64_lib eigen_blas_ilp64)
  else()
    set(eigen_blas_ilp64_lib eigen_blas_ilp64_static)
  endif()
endif()

if (EIGEN_Fortran_COMPILER_WORKS)

macro(ei_add_blas_test testname)

  set(targetname ${testname})

  set(filename ${testname}.f)
  add_executable(${targetname} ${filename})

  target_link_libraries(${targetname} eigen_blas)

  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${targetname} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()

  target_link_libraries(${targetname} ${EXTERNAL_LIBS})

  add_test(${testname} "${Eigen_SOURCE_DIR}/blas/testing/runblastest.sh" "${testname}" "${Eigen_SOURCE_DIR}/blas/testing/${testname}.dat")
  add_dependencies(buildtests ${targetname})
  
endmacro()

ei_add_blas_test(sblat1)
ei_add_blas_test(sblat2)
ei_add_blas_test(sblat3)

ei_add_blas_test(dblat1)
ei_add_blas_test(dblat2)
ei_add_blas_test(dblat3)

ei_add_blas_test(cblat1)
ei_add_blas_test(cblat2)
ei_add_blas_test(cblat3)

ei_add_blas_test(zblat1)
ei_add_blas_test(zblat2)
ei_add_blas_test(zblat3)

# add_custom_target(level1)
# add_dependencies(level1 sblat1)

# Workaround for gfortran -O3 bug.  The optimization
# introduces a bug in the calculation of the norm of the error
# in the complex case.
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(cblat3 PRIVATE "-O0")
  target_compile_options(zblat3 PRIVATE "-O0")
endif()

if(EIGEN_BUILD_BLAS_ILP64)
  include(CheckFortranCompilerFlag)
  set(EIGEN_FORTRAN_INT64_FLAG "")
  foreach(flag "-fdefault-integer-8" "-i8")
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" flag_var "${flag}")
    check_fortran_compiler_flag("${flag}" EIGEN_FORTRAN_HAS_${flag_var})
    if(EIGEN_FORTRAN_HAS_${flag_var})
      set(EIGEN_FORTRAN_INT64_FLAG "${flag}")
      break()
    endif()
  endforeach()

  if(EIGEN_FORTRAN_INT64_FLAG)
    set(ilp64_test_dir "${CMAKE_CURRENT_BINARY_DIR}/ilp64")
    file(MAKE_DIRECTORY "${ilp64_test_dir}")

    macro(ei_add_blas_ilp64_test testname)
      set(targetname "${testname}_64")
      set(filename "${testname}.f")
      add_executable(${targetname} ${filename})
      target_compile_options(${targetname} PRIVATE "${EIGEN_FORTRAN_INT64_FLAG}")
      set_target_properties(${targetname} PROPERTIES
        OUTPUT_NAME "${testname}"
        RUNTIME_OUTPUT_DIRECTORY "${ilp64_test_dir}")
      target_link_libraries(${targetname} ${eigen_blas_ilp64_lib})
      if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
        target_link_libraries(${targetname} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
      endif()
      target_link_libraries(${targetname} ${EXTERNAL_LIBS})
      add_test(NAME ${targetname}
        COMMAND "${Eigen_SOURCE_DIR}/blas/testing/runblastest.sh" "${testname}" "${Eigen_SOURCE_DIR}/blas/testing/${testname}.dat"
        WORKING_DIRECTORY "${ilp64_test_dir}")
      add_dependencies(buildtests ${targetname})
    endmacro()

    ei_add_blas_ilp64_test(sblat1)
    ei_add_blas_ilp64_test(sblat2)
    ei_add_blas_ilp64_test(sblat3)

    ei_add_blas_ilp64_test(dblat1)
    ei_add_blas_ilp64_test(dblat2)
    ei_add_blas_ilp64_test(dblat3)

    ei_add_blas_ilp64_test(cblat1)
    ei_add_blas_ilp64_test(cblat2)
    ei_add_blas_ilp64_test(cblat3)

    ei_add_blas_ilp64_test(zblat1)
    ei_add_blas_ilp64_test(zblat2)
    ei_add_blas_ilp64_test(zblat3)

    if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
      target_compile_options(cblat3_64 PRIVATE "-O0")
      target_compile_options(zblat3_64 PRIVATE "-O0")
    endif()
  endif()
endif()

endif()

if(EIGEN_BUILD_BLAS_ILP64)
  add_executable(blas_ilp64 blas_ilp64.cpp)
  target_include_directories(blas_ilp64 PRIVATE "${Eigen_SOURCE_DIR}" "${Eigen_SOURCE_DIR}/test")
  target_compile_definitions(blas_ilp64 PRIVATE "EIGEN_USE_BLAS" "EIGEN_64BIT_BLAS")
  target_link_libraries(blas_ilp64 ${eigen_blas_ilp64_lib})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(blas_ilp64 ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  add_test(NAME blas_ilp64 COMMAND blas_ilp64)
  add_dependencies(buildtests blas_ilp64)
endif()

