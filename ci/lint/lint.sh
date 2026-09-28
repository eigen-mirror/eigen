#!/bin/sh
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Runs every merge-request lint check in one job; see checkformat:lint.
#
#   lint.sh <base-sha>
#
# Every check runs even after one fails.  Exit status: 1 if a blocking check
# failed, 3 if only advisory checks did (the job's allow_failure exit code),
# 0 otherwise.  Writes lint-report.xml, one JUnit test case per check, so the
# merge request widget names the failing check.

base=${1:?usage: lint.sh <base-sha>}
blocking_failed=0
advisory_failed=0
cases=""
summary=""
nfail=0
ncases=0

# check <name> <blocking|advisory> <command...>
check() {
  name=$1 kind=$2
  shift 2
  printf '\033[0Ksection_start:%s:%s[collapsed=true]\r\033[0K%s (%s)\n' "$(date +%s)" "$name" "$name" "$kind"
  start=$(date +%s)
  "$@"
  status=$?
  secs=$(($(date +%s) - start))
  printf '\033[0Ksection_end:%s:%s\r\033[0K' "$(date +%s)" "$name"
  ncases=$((ncases + 1))
  if [ "$status" -eq 0 ]; then
    summary="$summary$(printf '  %-10s %-8s pass (%ss)' "$name" "$kind" "$secs")
"
    cases="$cases<testcase classname=\"lint\" name=\"$name\" time=\"$secs\"/>"
  else
    summary="$summary$(printf '  %-10s %-8s FAIL, exit %s (%ss)' "$name" "$kind" "$status" "$secs")
"
    cases="$cases<testcase classname=\"lint\" name=\"$name\" time=\"$secs\"><failure message=\"$kind check exited $status; see the $name section of the job log\"/></testcase>"
    nfail=$((nfail + 1))
    if [ "$kind" = blocking ]; then blocking_failed=1; else advisory_failed=1; fi
  fi
}

scripts() {
  rc=0
  for t in scripts/test_affected_tests.py ci/scripts/test_test_cache.py \
           ci/scripts/test_prune_runner_cache.py scripts/test_check_style.py \
           scripts/test_clang_tidy_hook.py; do
    echo "== $t"
    python3 "$t" || rc=1
  done
  # The edit-time hooks run on developers' Python, e.g. Ubuntu 24.04's 3.12,
  # not this image's.
  echo "== vermin"
  vermin -t=3.12- --violations --no-tips scripts ci/scripts || rc=1
  return $rc
}

# The Python helpers that decide what the rest of the pipeline does fail
# closed, but a wrong answer is silent, so their tests block; so does REUSE.
check scripts blocking scripts
check reuse blocking reuse lint
check format advisory git clang-format --diff --commit "$base"
check spelling advisory codespell --config setup.cfg

printf '<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="lint" tests="%s" failures="%s">%s</testsuite>\n' \
  "$ncases" "$nfail" "$cases" > lint-report.xml

printf '\nLint summary:\n%s' "$summary"
[ "$blocking_failed" -eq 0 ] || exit 1
[ "$advisory_failed" -eq 0 ] || exit 3
exit 0
