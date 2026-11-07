# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Checks the routine split that the LAPACK test drivers link against:
#  - every routine eigen_lapack.def exports is defined by an Eigen library,
#  - every Fortran-callable routine eigen_lapack defines is exported,
#  - the reference LAPACK library defines none of Eigen's routines, so the
#    drivers cannot silently test a reference copy instead of Eigen's,
#  - on ELF platforms, the shared eigen_lapack exports every listed routine.

cmake_minimum_required(VERSION 3.18)

foreach(variable NM DEF_FILE EIGEN_LAPACK EIGEN_BLAS REFERENCE)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "LAPACK symbol check needs -D${variable}")
  endif()
endforeach()
# Fortran-callable global symbols that <library> defines, without ABI decoration.
function(fortran_symbols out library)
  if(NOT EXISTS "${library}")
    message(FATAL_ERROR "Missing library: ${library}")
  endif()
  execute_process(COMMAND "${NM}" ${ARGN} "${library}"
                  OUTPUT_VARIABLE listing RESULT_VARIABLE status ERROR_VARIABLE errors)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${NM} failed on ${library}: ${errors}")
  endif()
  # Weak definitions count: Eigen's xerbla_ is weak so that a program can replace it.
  # Mach-O and 32-bit x86 PE add a leading underscore; Fortran identifiers start with a letter.
  string(REGEX MATCHALL "[ \t][TDRBWV][ \t]+_?[a-z][a-z0-9_]*_\r?\n" symbols "${listing}")
  list(TRANSFORM symbols REPLACE "^[ \t][TDRBWV][ \t]+_?([a-z][a-z0-9_]*_)[\r\n]*$" "\\1")
  list(REMOVE_DUPLICATES symbols)
  set(${out} "${symbols}" PARENT_SCOPE)
endfunction()

file(STRINGS "${DEF_FILE}" exported REGEX "^[ \t]*[a-z0-9_]+_[ \t]*$")
list(TRANSFORM exported STRIP)
if(NOT exported)
  message(FATAL_ERROR "No routines found in ${DEF_FILE}")
endif()

fortran_symbols(lapack_defined "${EIGEN_LAPACK}" -g)
fortran_symbols(blas_defined "${EIGEN_BLAS}" -g)
fortran_symbols(reference_defined "${REFERENCE}" -g)

set(errors "")
foreach(symbol IN LISTS exported)
  if(NOT symbol IN_LIST lapack_defined AND NOT symbol IN_LIST blas_defined)
    string(APPEND errors "eigen_lapack.def exports ${symbol}, which no Eigen library defines\n")
  endif()
endforeach()
foreach(symbol IN LISTS lapack_defined)
  if(NOT symbol IN_LIST exported)
    string(APPEND errors "eigen_lapack defines ${symbol}, which eigen_lapack.def does not export\n")
  endif()
endforeach()
foreach(symbol IN LISTS lapack_defined blas_defined)
  if(symbol IN_LIST reference_defined)
    string(APPEND errors "${symbol} is defined by both Eigen and the reference LAPACK library\n")
  endif()
endforeach()

if(DEFINED EIGEN_LAPACK_SHARED)
  fortran_symbols(lapack_exported "${EIGEN_LAPACK_SHARED}" -D --defined-only)
  fortran_symbols(blas_exported "${EIGEN_BLAS_SHARED}" -D --defined-only)
  foreach(symbol IN LISTS exported)
    if(NOT symbol IN_LIST lapack_exported AND NOT symbol IN_LIST blas_exported)
      string(APPEND errors "The shared eigen_lapack library does not export ${symbol}; "
                           "declare it with EIGEN_LAPACK_API in lapack.h\n")
    endif()
  endforeach()
endif()

list(LENGTH exported exported_count)
list(LENGTH reference_defined reference_count)
message(STATUS "${exported_count} routines from Eigen, ${reference_count} from reference LAPACK")
if(errors)
  message(FATAL_ERROR "${errors}")
endif()
