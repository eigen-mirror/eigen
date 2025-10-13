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

declare -a ctest_cmd=(
    ctest
    ${EIGEN_CI_CTEST_ARGS}
    --parallel "${NPROC}"
    --output-on-failure
    --no-compress-output
    --build-no-clean
    -T test
    "${target}"
)

"${ctest_cmd[@]}"
exit_code=$?
if (( exit_code != 0 )); then
  echo "Retrying tests up to ${EIGEN_CI_CTEST_REPEAT} times."
  ctest_cmd+=( --rerun-failed "--repeat" "until-pass:${EIGEN_CI_CTEST_REPEAT}" )
  "${ctest_cmd[@]}"
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
