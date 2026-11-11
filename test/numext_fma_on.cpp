// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#undef EIGEN_SCALAR_MADD_USE_FMA
#define EIGEN_SCALAR_MADD_USE_FMA 1
#include "numext.cpp"  // NOLINT(bugprone-suspicious-include): Compile the suite with a different configuration.

// ei_add_test reads only this file, so list numext.cpp's CALL_SUBTEST_<N> parts here.
// EIGEN_SUFFIXES;1;2;3;4;5;6;7;8;9
