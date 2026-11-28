// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_PCG_RANDOM
#define EIGEN_AVOID_THREAD_LOCAL
#define EIGEN_TEST_PART_16 1
#include "rand.cpp"  // NOLINT(bugprone-suspicious-include): Verify the forced std::rand fallback.

static_assert(EIGEN_HAS_THREAD_LOCAL_RANDOM == 0, "EIGEN_AVOID_THREAD_LOCAL must disable the PCG backend");
