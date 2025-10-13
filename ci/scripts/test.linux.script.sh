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

run_ctest_retry="ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${NPROC}"
run_ctest_retry+=" --output-on-failure --no-compress-output"
run_ctest_retry+=" --repeat until-pass:${EIGEN_CI_CTEST_REPEAT}"

eval "${run_ctest}"

"${ctest_cmd[@]}"
exit_code=$?
if (( exit_code != 0 )); then
  echo "Retrying tests up to ${EIGEN_CI_CTEST_REPEAT} times."
  eval "${run_ctest_retry}"
  exit_code=$?
  if (( exit_code == 0 )); then
    echo "Tests passed on retry."
    exit_code=42
  else
    echo "Tests failed after retry attempts."
  fi
fi


# Return to root directory.
cd ${rootdir}

set +x

exit $exit_code
