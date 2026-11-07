#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -eu

test_name=${1:?usage: runblastest.sh test-name [input-file]}
log="$test_name.log"
summary="$test_name.summ"
rm -f "$summary" "$test_name.snap"
: > "$log"

fail() {
  echo "Test $test_name failed: $*" >&2
  cat "$log" >&2
  if [[ -f "$summary" ]]; then cat "$summary" >&2; fi
  exit 1
}

# Level 1 writes its results to stdout and needs no input file. Levels 2 and 3
# name their summary file in the input deck. All three can fail with exit code 0.
if [[ "$test_name" == [sdcz]blat1 ]]; then
  data=/dev/null
else
  data=${2:-}
  [[ -f "$data" ]] || fail "missing input file: $data"
fi

if ! "./$test_name" < "$data" > "$log" 2>&1; then
  fail "driver exited unsuccessfully"
fi

reports=("$log")
if [[ "$test_name" != [sdcz]blat1 ]]; then
  [[ -f "$summary" ]] || fail "missing summary file"
  reports+=("$summary")
fi
if grep -Eq '(^|[[:space:]])FAIL([[:space:]]|$)|FAILED|FATAL ERROR|SUSPECT|TESTS ABANDONED' "${reports[@]}"; then
  fail "driver reported a failure"
fi

if [[ "$test_name" == [sdcz]blat1 ]]; then
  cases=$(grep -c 'Test of subprogram number' "$log" || true)
  passed=$(grep -c -- '----- PASS -----' "$log" || true)
  [[ "$cases" -gt 0 && "$cases" -eq "$passed" ]] || fail "missing test results"
else
  grep -q 'PASSED THE COMPUTATIONAL TESTS' "$summary" || fail "missing computational results"
  grep -q 'END OF TESTS' "$summary" || fail "incomplete summary"
fi
echo "Test $test_name passed"
