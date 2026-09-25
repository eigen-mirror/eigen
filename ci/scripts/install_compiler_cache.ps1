# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Provision and verify compiler cache tools (sccache, ccache).
# Pinned release binaries are verified against strict SHA-256 checksums
# on both download and cache restoration to protect against supply-chain tampering.

function Install-Sccache {
  param([string]$RootDirectory = ${rootdir})

  $sccache_exe = ""

  $system_sccache = Get-Command sccache -ErrorAction SilentlyContinue
  if ($system_sccache) {
    $sccache_exe = $system_sccache.Source
  } else {
    $sccache_version = "0.18.0"
    $sccache_zip_sha256 = "8965c74d5e8a225244f741e18ad2f3f504f48228dc1bac948fc22761a348363d"
    $sccache_exe_sha256 = "b85863f6c0579c57a74071b8904f9850a8ef03dac26bfc94dc30225155bd7449"
    $sccache_bindir = Join-Path $RootDirectory ".sccache-bin"
    $cached_sccache = Join-Path $sccache_bindir "sccache.exe"

    # 1. Check cached sccache.exe
    if (Test-Path $cached_sccache) {
      if ((Get-FileHash $cached_sccache -Algorithm SHA256).Hash -eq $sccache_exe_sha256) {
        $sccache_exe = $cached_sccache
      } else {
        Write-Warning "Discarding cached sccache.exe that failed SHA-256 verification."
        Remove-Item $cached_sccache -Force -ErrorAction SilentlyContinue
      }
    }

    # 2. Download and verify sccache.zip
    if (-not $sccache_exe) {
      $zip = Join-Path ([System.IO.Path]::GetTempPath()) "sccache-${sccache_version}.zip"
      $download_success = $false
      for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
          $ProgressPreference = "SilentlyContinue"
          Invoke-WebRequest "https://github.com/mozilla/sccache/releases/download/v${sccache_version}/sccache-v${sccache_version}-x86_64-pc-windows-msvc.zip" -OutFile $zip -TimeoutSec 30
          $download_success = $true
          break
        } catch {
          if ($attempt -lt 3) { Start-Sleep -Seconds 2 } else { Write-Warning "sccache download failed: $_" }
        }
      }
      if ($download_success) {
        if ((Get-FileHash $zip -Algorithm SHA256).Hash -eq $sccache_zip_sha256) {
          $unpack = Join-Path ([System.IO.Path]::GetTempPath()) "sccache-unpack"
          Expand-Archive $zip -DestinationPath $unpack -Force
          New-Item -ItemType Directory -Force -Path $sccache_bindir | Out-Null
          $src = Join-Path $unpack "sccache-v${sccache_version}-x86_64-pc-windows-msvc/sccache.exe"
          Copy-Item $src $cached_sccache -Force
          if ((Get-FileHash $cached_sccache -Algorithm SHA256).Hash -eq $sccache_exe_sha256) {
            $sccache_exe = $cached_sccache
          } else {
            Write-Warning "Extracted sccache.exe failed SHA-256 verification; discarding."
            Remove-Item $cached_sccache -Force -ErrorAction SilentlyContinue
          }
          Remove-Item $unpack -Recurse -Force -ErrorAction SilentlyContinue
        } else {
          Write-Warning "sccache download failed SHA-256 verification."
        }
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
      }
    }
  }

  return $sccache_exe
}

function Install-Ccache {
  param([string]$RootDirectory = ${rootdir})

  $ccache_version = "4.13.6"
  $ccache_zip_sha256 = "3d7cebb05850ad704e197b3f1d3f0f924ab6c9fdfc561578e146184fe9d89380"
  $ccache_exe_sha256 = "1f285645553e61463b000c6c440301724894b4ba0174dbef6b4818d54b54b68d"
  # Caching MSVC needs ccache >= 4.8.
  $ccache_min_version = [version]"4.8"
  $ccache_bindir = Join-Path $RootDirectory ".ccache-bin"
  $cached_exe = Join-Path $ccache_bindir "ccache.exe"

  # 1. Check system ccache
  $system_ccache = Get-Command ccache -ErrorAction SilentlyContinue
  if ($system_ccache) {
    $system_version = ""
    try {
      $system_version = (& $system_ccache.Source --version | Select-Object -First 1)
    } catch {}
    if (($system_version -match "ccache version (\d+(\.\d+)+)") -and
        ([version]$Matches[1] -ge $ccache_min_version)) {
      return $system_ccache.Source
    } else {
      Write-Warning ("Ignoring system ccache at $($system_ccache.Source) " +
                     "('${system_version}'; need >= ${ccache_min_version}).")
    }
  }

  # 2. Check cached ccache.exe
  if (Test-Path $cached_exe) {
    if ((Get-FileHash $cached_exe -Algorithm SHA256).Hash -eq $ccache_exe_sha256) {
      return $cached_exe
    } else {
      Write-Warning "Discarding cached ccache.exe that failed SHA-256 verification."
      Remove-Item $cached_exe -Force -ErrorAction SilentlyContinue
    }
  }

  # 3. Download and verify ccache.zip
  $zip = Join-Path ([System.IO.Path]::GetTempPath()) "ccache-${ccache_version}.zip"
  $download_success = $false
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
      $ProgressPreference = "SilentlyContinue"
      Invoke-WebRequest "https://github.com/ccache/ccache/releases/download/v${ccache_version}/ccache-${ccache_version}-windows-x86_64.zip" -OutFile $zip -TimeoutSec 30
      $download_success = $true
      break
    } catch {
      if ($attempt -lt 3) { Start-Sleep -Seconds 2 } else { Write-Warning "ccache download failed: $_" }
    }
  }
  if ($download_success) {
    if ((Get-FileHash $zip -Algorithm SHA256).Hash -eq $ccache_zip_sha256) {
      $unpack = Join-Path ([System.IO.Path]::GetTempPath()) "ccache-unpack"
      Expand-Archive $zip -DestinationPath $unpack -Force
      New-Item -ItemType Directory -Force -Path $ccache_bindir | Out-Null
      Copy-Item (Join-Path $unpack "ccache-${ccache_version}-windows-x86_64/ccache.exe") $cached_exe -Force
      if ((Get-FileHash $cached_exe -Algorithm SHA256).Hash -eq $ccache_exe_sha256) {
        Remove-Item $unpack -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
        return $cached_exe
      } else {
        Write-Warning "Extracted ccache.exe failed SHA-256 verification; building without ccache."
        Remove-Item $cached_exe -Force -ErrorAction SilentlyContinue
      }
      Remove-Item $unpack -Recurse -Force -ErrorAction SilentlyContinue
    } else {
      Write-Warning "ccache download failed SHA-256 verification; building without ccache."
    }
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
  }

  return ""
}
