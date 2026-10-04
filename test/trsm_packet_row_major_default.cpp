// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// EIGEN_SUFFIXES;1;2
// The packet kernel's views of the column-major right-hand side must keep their
// storage order when EIGEN_DEFAULT_TO_ROW_MAJOR flips the default.
#ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#endif
#include "trsm_packet.cpp"  // NOLINT(bugprone-suspicious-include): Compile the suite with a different configuration.
