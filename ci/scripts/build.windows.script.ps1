# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

$rootdir = Get-Location

# The affected-tests tier (see scripts/affected_tests.py) passes its selection
# as a file rather than a variable so the list is not bounded by CI variable
# limits.  The file holds "NONE" or one target per line; the full-suite form is
# "buildtests" plus the targets it does not aggregate, and so takes the same
# path as any other list.
#
# Read first, before vcvarsall, the ccache provisioning and the CMake configure:
# a diff that reaches no test at all selects "NONE", and such a job should cost
# a checkout rather than a full configure.  Declining to schedule it at all
# would have to go through `rules:`, which GitLab evaluates when the pipeline is
# created -- before select:tests has produced this file -- so exiting early is
# the cheapest answer available within one pipeline.
$requested = @()
if (${EIGEN_CI_BUILD_TARGET_FILE}) {
  $target_file = ${EIGEN_CI_BUILD_TARGET_FILE}
  if (-Not [System.IO.Path]::IsPathRooted($target_file)) {
    $target_file = Join-Path ${rootdir} $target_file
  }
  # Fail loudly rather than falling through to the default target: a missing
  # selection would otherwise silently build the entire test suite.
  if (-Not (Test-Path $target_file)) {
    Write-Error ("EIGEN_CI_BUILD_TARGET_FILE=${EIGEN_CI_BUILD_TARGET_FILE} does not exist. " +
                 "The select:tests artifact is missing; refusing to guess a build target.")
    Exit 1
  }
  $requested = @(Get-Content $target_file | ForEach-Object { $_.Trim() } |
                 Where-Object { $_ } | Sort-Object -Unique)
  if ($requested -contains "NONE") {
    Write-Host "No tests are affected by this merge request; nothing to build."
    Exit 0
  }
}

# Find Visual Studio installation directory. -products * includes Build
# Tools installs, which vswhere's default product filter skips; without it a
# Build Tools-only runner gets an empty path here, vcvarsall never runs, and
# the configure step fails with "No CMAKE_CXX_COMPILER could be found".
$VS_INSTALL_DIR = &"${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath

# Run VCVarsAll.bat initialization script and extract environment variables.
# http://allen-mack.blogspot.com/2008/03/replace-visual-studio-command-prompt.html
cmd.exe /c "`"${VS_INSTALL_DIR}\VC\Auxiliary\Build\vcvarsall.bat`" $EIGEN_CI_MSVC_ARCH -vcvars_ver=$EIGEN_CI_MSVC_VER & set" |
  foreach {
    if ($_ -match "^([^=]+)=(.*)$") {
      set-item -force -LiteralPath "ENV:\$($Matches[1])" -value "$($Matches[2])"
    }
  }

# Create and enter build directory.
if (-Not (Test-Path ${EIGEN_CI_BUILDDIR})) {
    mkdir $EIGEN_CI_BUILDDIR
}
cd $EIGEN_CI_BUILDDIR

# We need to split EIGEN_CI_ADDITIONAL_ARGS, otherwise they are interpreted
# as a single argument.  Split by space, unless double-quoted.
$split_args = [regex]::Split(${EIGEN_CI_ADDITIONAL_ARGS}, ' (?=(?:[^"]|"[^"]*")*$)' )

# Compiler cache launcher selection:
# Prefer sccache (Mozilla Shared Compilation Cache) with native Google Cloud Storage
# remote backend (gs://eigen-gitlab-ci-cache). If sccache or GCS authentication is
# unavailable, fall back to ccache with local disk / GitLab runner cache.
#
# Trust is established at every use, not at download time: runner-installed ccache
# must report at least the version required for MSVC support (>= 4.8), and restored
# or freshly extracted binaries must match the pinned SHA-256 — the GitLab cache is
# writable job output, so restoring it does not authenticate it. Re-verifying per use
# also ensures version bumps invalidate stale cached copies.
$sccache_exe = ""
$ccache_exe = ""
$compiler_launcher = ""
$launchers = @()

if ("${EIGEN_CI_CCACHE}" -eq "on") {
  . (Join-Path ${rootdir} "ci/scripts/install_compiler_cache.ps1")
  if ("${env:EIGEN_CI_SCCACHE}" -ne "off") {
    $sccache_exe = Install-Sccache
  }
  $ccache_exe = Install-Ccache

  # 1. Try starting sccache server if available
  if ($sccache_exe) {
    $env:SCCACHE_DIR = Join-Path $env:CI_PROJECT_DIR ".sccache"
    $env:SCCACHE_CACHE_SIZE = if ($env:EIGEN_CI_CCACHE_MAXSIZE) { $env:EIGEN_CI_CCACHE_MAXSIZE } else { "4G" }
    # Rewrite paths relative to rootdir for cross-runner / cross-directory cache hits.
    $env:SCCACHE_BASEDIRS = $rootdir
    # Isolate daemon port per runner slot to avoid port collisions and process cross-kill.
    $env:SCCACHE_SERVER_PORT = [string](4226 + ([int]($env:CI_JOB_ID ? $env:CI_JOB_ID : 0) % 10000))

    $gcs_token = if ($env:EIGEN_GCS_CACHE_TOKEN_RW) { $env:EIGEN_GCS_CACHE_TOKEN_RW } else { $env:EIGEN_GCS_CACHE_TOKEN_RO }
    $cred_server_job = $null
    if ($gcs_token) {
      $env:SCCACHE_GCS_BUCKET = if ($env:EIGEN_CI_SCCACHE_GCS_BUCKET) { $env:EIGEN_CI_SCCACHE_GCS_BUCKET } else { "eigen-gitlab-ci-cache" }
      $env:SCCACHE_MULTILEVEL_CHAIN = "disk,gcs"
      if ($env:EIGEN_GCS_CACHE_TOKEN_RW) {
        $env:SCCACHE_GCS_RW_MODE = "READ_WRITE"
      } else {
        $env:SCCACHE_GCS_RW_MODE = "READ_ONLY"
      }
      $cred_port = [string](8200 + ([int]($env:CI_JOB_ID ? $env:CI_JOB_ID : 0) % 1000))
      $cred_server_job = Start-Job -ScriptBlock {
        param($tok, $p)
        $listener = New-Object System.Net.HttpListener
        $listener.Prefixes.Add("http://127.0.0.1:$p/")
        $listener.Start()
        while ($listener.IsListening) {
          $ctx = $listener.GetContext()
          $resp = $ctx.Response
          $resp.ContentType = "application/json"
          $body = [System.Text.Encoding]::UTF8.GetBytes("{`"access_token`":`"$tok`",`"token_type`":`"Bearer`",`"expires_in`":3600}")
          $resp.ContentLength64 = $body.Length
          $resp.OutputStream.Write($body, 0, $body.Length)
          $resp.Close()
        }
      } -ArgumentList $gcs_token, $cred_port
      $env:SCCACHE_GCS_CREDENTIALS_URL = "http://127.0.0.1:$cred_port/token"

      # Wait up to 1 second for local credential server to be ready
      for ($i = 0; $i -lt 20; $i++) {
        try {
          $res = Invoke-WebRequest -Uri "http://127.0.0.1:$cred_port/token" -UseBasicParsing -TimeoutSec 1
          if ($res.StatusCode -eq 200) { break }
        } catch {}
        Start-Sleep -Milliseconds 50
      }
    }

    & $sccache_exe --start-server | Out-Null
    if ($LASTEXITCODE -eq 0) {
      $compiler_launcher = "sccache"
      & $sccache_exe --zero-stats | Out-Null
      $sccache_cmake = $sccache_exe -replace '\\', '/'
      $launchers = "-DCMAKE_C_COMPILER_LAUNCHER=${sccache_cmake}",
                   "-DCMAKE_CXX_COMPILER_LAUNCHER=${sccache_cmake}"
      # Incompatible cache formats: purge restored .ccache\ so the uploaded cache
      # archive only holds .sccache\ and stays strictly below the 5 GB runner cap.
      $old_ccache = Join-Path $rootdir ".ccache"
      if (Test-Path $old_ccache) { Remove-Item -Recurse -Force $old_ccache }
    } else {
      Write-Warning "sccache server failed to start (check GCS credentials/network); falling back to ccache."
      if ($cred_server_job) {
        Stop-Job $cred_server_job -ErrorAction SilentlyContinue
        Remove-Job $cred_server_job -ErrorAction SilentlyContinue
      }
    }
  }

  # 2. Fall back to ccache if sccache is unavailable or failed to start
  if ((-not $compiler_launcher) -and $ccache_exe) {
    $compiler_launcher = "ccache"
    $old_sccache = Join-Path $rootdir ".sccache"
    if (Test-Path $old_sccache) { Remove-Item -Recurse -Force $old_sccache }
    # EIGEN_CI_CCACHE_* provide the YAML fallback defaults.  A runner may explicitly
    # set standard CCACHE_* variables (e.g. to a persistent host directory) in
    # config.toml without being overridden by the YAML template.
    if (-not $env:CCACHE_DIR -and $env:EIGEN_CI_CCACHE_DIR) {
      $env:CCACHE_DIR = $env:EIGEN_CI_CCACHE_DIR
    }
    if (-not $env:CCACHE_MAXSIZE -and $env:EIGEN_CI_CCACHE_MAXSIZE) {
      $env:CCACHE_MAXSIZE = $env:EIGEN_CI_CCACHE_MAXSIZE
    }
    if (-not $env:CCACHE_BASEDIR -and $env:EIGEN_CI_CCACHE_BASEDIR) {
      $env:CCACHE_BASEDIR = $env:EIGEN_CI_CCACHE_BASEDIR
    }
    if (-not $env:CCACHE_COMPRESSLEVEL -and $env:EIGEN_CI_CCACHE_COMPRESSLEVEL) {
      $env:CCACHE_COMPRESSLEVEL = $env:EIGEN_CI_CCACHE_COMPRESSLEVEL
    }

    # Forward slashes: CMake treats the launcher as a path-valued cache entry.
    $ccache_cmake = $ccache_exe -replace '\\', '/'
    $launchers = "-DCMAKE_C_COMPILER_LAUNCHER=${ccache_cmake}",
                 "-DCMAKE_CXX_COMPILER_LAUNCHER=${ccache_cmake}"
    # Log stats per job via CCACHE_STATSLOG rather than global --zero-stats /
    # --show-stats so concurrent jobs sharing a host CCACHE_DIR do not reset or
    # mix each other's counters.
    $env:CCACHE_STATSLOG = Join-Path (Get-Location) "ccache-stats.log"
    if (Test-Path $env:CCACHE_STATSLOG) {
      Remove-Item $env:CCACHE_STATSLOG -Force
    }
  }
}

# Configure build.
cmake -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel `
      -DEIGEN_TEST_CUSTOM_CXX_FLAGS="${EIGEN_CI_TEST_CUSTOM_CXX_FLAGS}" `
      ${launchers} ${split_args} "${rootdir}"

# Targets that this configuration did not register (optional dependencies such
# as CHOLMOD or CUDA) are dropped from the selection read above: ninja aborts on
# an unknown target, and this is the first point that knows what CMake actually
# configured.
$selected_targets = @()
if (${EIGEN_CI_BUILD_TARGET_FILE}) {
  # Not redirected with 2>: the runner sets $ErrorActionPreference to Stop, and
  # redirecting a native command's stderr promotes each line it writes to a
  # terminating error, which would abort the job ahead of the report below.
  $configured = @{}
  try {
    foreach ($line in (ninja -t targets all)) {
      if ($line -match '^([A-Za-z_0-9]+): phony$') { $configured[$Matches[1]] = $true }
    }
  } catch {
    Write-Warning "Enumerating configured targets failed: $_"
  }
  # An empty query means ninja is unusable, not that nothing is configured.
  # Without this the intersection below would be empty and the job would
  # trivially "succeed" having built nothing.
  if ($configured.Count -eq 0) {
    Write-Error "Could not enumerate configured targets via 'ninja -t targets'."
    cd ${rootdir}
    Exit 1
  }
  $selected_targets = @($requested | Where-Object { $configured.ContainsKey($_) })
  $unconfigured = @($requested | Where-Object { -Not $configured.ContainsKey($_) })
  Write-Host ("Affected selection: {0} of {1} requested targets are configured here." -f
              $selected_targets.Count, $requested.Count)
  if ($unconfigured.Count -gt 0) {
    Write-Host ("Not configured in this build: " + ($unconfigured -join " "))
  }
  if ($selected_targets.Count -eq 0) {
    Write-Host "None of the affected tests exist in this configuration; nothing to build."
    cd ${rootdir}
    Exit 0
  }
}

# Built as an array, not a string: PowerShell hands a single string containing
# spaces to a native command as one argument, which cmake rejects as an unknown
# target name.  An affected selection is always a list, and the SME-style
# multi-target EIGEN_CI_BUILD_TARGET is one too.
$target = @()
if ($selected_targets.Count -gt 0) {
  $target = @("--target") + $selected_targets
} elseif (${EIGEN_CI_BUILD_TARGET}) {
  $target = @("--target") + @(${EIGEN_CI_BUILD_TARGET} -split '\s+' | Where-Object { $_ })
}

# Windows builds sometimes fail due heap errors. In that case, try
# building the rest, then try to build again with a single thread.
cmake --build . ${target} -- -k0 || cmake --build . ${target} -- -k0 -j1

$success = $LASTEXITCODE

# Hit/miss summary for judging what the cache pays for on this job. Runs on
# failures too: the cache is pushed even then (cache:when: always), so the
# stats still describe what the next attempt can reuse.
if ($compiler_launcher -eq "sccache" -and $sccache_exe) {
  & $sccache_exe --show-stats
  & $sccache_exe --stop-server | Out-Null
  if ($cred_server_job) {
    Stop-Job $cred_server_job -ErrorAction SilentlyContinue
    Remove-Job $cred_server_job -ErrorAction SilentlyContinue
  }
} elseif ($compiler_launcher -eq "ccache" -and $ccache_exe) {
  & $ccache_exe --show-log-stats
  if (Test-Path $env:CCACHE_STATSLOG) {
    Remove-Item $env:CCACHE_STATSLOG -Force
  }
}

# Return to root directory.
cd ${rootdir}

# Explicitly propagate exit code to indicate pass/failure of build command.
if($success -ne 0) { Exit $success }
