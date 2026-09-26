# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

echo "Running ${CI_JOB_NAME}"

echo "Host machine configuration:"
Get-CimInstance Win32_OperatingSystem -ErrorAction Continue |
  Select-Object Caption, Version, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory | Format-List
Get-CimInstance Win32_Processor -ErrorAction Continue |
  Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List
Get-PSDrive -PSProvider FileSystem | Format-Table Name, Used, Free -AutoSize

# Print configuration variables.
Get-Variable EIGEN* | Where-Object { $_.Name -notlike "*CACHE_TOKEN*" } | Format-Table -Wrap
Get-Variable CMAKE* | Format-Table -Wrap

# Run a custom before-script command.
if ("${EIGEN_CI_BEFORE_SCRIPT}") { Invoke-Expression -Command "${EIGEN_CI_BEFORE_SCRIPT}" }
