#!/bin/bash

set -x

# Enter build directory.
rootdir=`pwd`
cd ${EIGEN_CI_BUILDDIR}

target=""
if [[ ${EIGEN_CI_CTEST_REGEX} ]]; then
  target="-R ${EIGEN_CI_CTEST_REGEX}"
elif [[ ${EIGEN_CI_CTEST_LABEL} ]]; then
  target="-L ${EIGEN_CI_CTEST_LABEL}"
fi

# Repeat tests up to EIGEN_CI_CTEST_REPEAT times.
# Tests that pass during the repeated attempts will return a non-zero error code.

run_ctest="ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${NPROC}"
run_ctest+=" --output-on-failure --no-compress-output"
run_ctest+=" --build-no-clean -T test ${target}"

eval "${run_ctest}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
  eval "${run_ctest} --repeat until-pass:${EIGEN_CI_CTEST_REPEAT}"
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    exit_code=42
  fi
fi


# Return to root directory.
cd ${rootdir}

set +x

exit $exit_code
