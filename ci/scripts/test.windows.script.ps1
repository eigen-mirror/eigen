# Change to build directory.
$rootdir = Get-Location
cd $EIGEN_CI_BUILDDIR

# Determine number of processors for parallel tests.
$NPROC=${Env:NUMBER_OF_PROCESSORS}

# Set target based on regex or label.
$target = ""
if (${EIGEN_CI_CTEST_REGEX}) {
  $target = "-R","${EIGEN_CI_CTEST_REGEX}"
} elseif (${EIGEN_CI_CTEST_LABEL}) {
  $target = "-L","${EIGEN_CI_CTEST_LABEL}"
}

$run_ctest = "ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${NPROC}"
$run_ctest += " --output-on-failure --no-compress-output"
$run_ctest += " --build-no-clean -T test ${target}"

Invoke-Expression $run_ctest
$exit_code = $LASTEXITCODE
if(${$exit_code} != 0) {
  $run_ctest += " --repeat until-pass:${EIGEN_CI_CTEST_REPEAT}"
  Invoke-Expression $run_ctest
  $exit_code = $LASTEXITCODE
  if(${$exit_code} == 0) {
    $exit_code = 42
  }
}

# Return to root directory.
cd ${rootdir}

Exit exit_code
