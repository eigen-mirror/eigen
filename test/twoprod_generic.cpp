// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_VECTORIZE_GENERIC 1
#define EIGEN_GENERIC_VECTOR_SIZE_BYTES 16
#include "twoprod.cpp"  // NOLINT(bugprone-suspicious-include): the same suite on the generic vector backend.
