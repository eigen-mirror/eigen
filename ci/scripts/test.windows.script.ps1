# Change to build directory.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
$rootdir = Get-Location

# The affected-tests tier (see scripts/affected_tests.py) passes its CTest
# filter as a file rather than a variable so the regex is not bounded by CI
# variable limits.  "ALL" means run everything the paired build produced,
# "NONE" means the merge request affects no test at all.
if (${EIGEN_CI_CTEST_REGEX_FILE}) {
  $regex_file = ${EIGEN_CI_CTEST_REGEX_FILE}
  if (-Not [System.IO.Path]::IsPathRooted($regex_file)) {
    $regex_file = Join-Path ${rootdir} $regex_file
  }
  # Fail loudly rather than falling through: a missing selection would
  # otherwise silently run the whole suite against a partial build.
  if (-Not (Test-Path $regex_file)) {
    Write-Error ("EIGEN_CI_CTEST_REGEX_FILE=${EIGEN_CI_CTEST_REGEX_FILE} does not exist. " +
                 "The select:tests artifact is missing; refusing to guess a test filter.")
    Exit 1
  }
  $selection = (Get-Content $regex_file -Raw).Trim()
  if ($selection -eq "NONE") {
    Write-Host "No tests are affected by this merge request; nothing to run."
    Exit 0
  } elseif ($selection -eq "ALL") {
    $EIGEN_CI_CTEST_REGEX = ""
  } else {
    $EIGEN_CI_CTEST_REGEX = $selection
  }
}

cd $EIGEN_CI_BUILDDIR

# CTestTestfile.cmake and DartConfiguration.tcl name the build job's checkout by
# absolute path.  The Windows host runs several runner registrations, each with
# its own checkout directory, so a test job that lands on another one would look
# for the test executables in a directory that does not exist -- or, when it
# does, run whatever build another job left there.  Point them at this
# checkout; -ne and -replace ignore case, as Windows paths do.
$configured_root = Select-String -Path CMakeCache.txt -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$' |
                   Select-Object -First 1 | ForEach-Object { $_.Matches[0].Groups[1].Value }
$job_root = $rootdir.Path -replace '\\', '/'
if ($configured_root -and $configured_root -ne $job_root) {
  Write-Host "Relocating the CTest files from $configured_root to $job_root"
  $pattern = [regex]::Escape($configured_root)
  $replacement = $job_root.Replace('$', '$$')
  Get-ChildItem -Recurse -File -Include CTestTestfile.cmake, DartConfiguration.tcl | ForEach-Object {
    $text = [System.IO.File]::ReadAllText($_.FullName)
    [System.IO.File]::WriteAllText($_.FullName, ($text -replace $pattern, $replacement))
  }
}

# Verify GPU availability for GPU test suites. Never allow non-GPU runners
# to silently skip tests and report false-positive success.
if ($EIGEN_CI_CTEST_LABEL -eq "gpu" -or $Env:CI_JOB_NAME -like "*:cuda*") {
  $has_gpu = $false
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi > $null 2>&1
    if ($LASTEXITCODE -eq 0) { $has_gpu = $true }
  }
  if (-not $has_gpu) {
    Write-Error "ERROR: GPU test suite requested, but no functional NVIDIA device or driver was detected by nvidia-smi."
    exit 1
  }
}

# Determine number of processors for parallel tests.
$NPROC=${Env:NUMBER_OF_PROCESSORS}

# Total attempts for flaky tests, matching test.linux.script.sh.  No job sets
# this, and an empty value made the retry below spell "until-pass:", which ctest
# rejects as a usage error -- so the retry never ran and its exit status masked
# the test failure that triggered it.
if (-Not ${EIGEN_CI_CTEST_REPEAT}) { $EIGEN_CI_CTEST_REPEAT = 3 }

# @(), not "": an "ALL" selection leaves no regex and no label, so the
# no-filter case is reachable and must contribute no argument to ctest.
$target = @()
if (${EIGEN_CI_CTEST_REGEX}) {
  $target = "-R","${EIGEN_CI_CTEST_REGEX}"
} elseif (${EIGEN_CI_CTEST_LABEL}) {
  $target = "-L","${EIGEN_CI_CTEST_LABEL}"
}

# Same knob as test.linux.script.sh, for tests this runner cannot execute.
$exclude = @()
if (${EIGEN_CI_CTEST_EXCLUDE}) {
  $exclude = "-E","${EIGEN_CI_CTEST_EXCLUDE}"
}

$ctest_cmd = { ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${NPROC} --output-on-failure --no-compress-output --build-noclean ${target} ${exclude} @args }

Write-Host "Running initial tests..."

& $ctest_cmd "-T" "test"
$exit_code = $LASTEXITCODE

if ($exit_code -eq 0) {
  Write-Host "Tests passed on the first attempt."
}
else {
  Write-Host "Initial tests failed with exit code $exit_code. Retrying up to $EIGEN_CI_CTEST_REPEAT times..."
  & $ctest_cmd "--rerun-failed" "--repeat" "until-pass:$EIGEN_CI_CTEST_REPEAT"
  $exit_code = $LASTEXITCODE

  if ($exit_code -eq 0) {
    Write-Host "Tests passed on retry."
    $exit_code = 42
  }
}

# Return to root directory.
cd ${rootdir}

Exit $exit_code
