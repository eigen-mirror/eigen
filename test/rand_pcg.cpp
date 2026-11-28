// Build the rand suite against the opt-in PCG backend, preserving its test parts.
// EIGEN_SUFFIXES;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#define EIGEN_USE_PCG_RANDOM
#include "rand.cpp"  // NOLINT(bugprone-suspicious-include): Compile the suite with the PCG backend.

// rand.cpp compiles its PCG checks out when the backend is off, so a silent fallback to std::rand() would pass.
#ifndef EIGEN_AVOID_THREAD_LOCAL
static_assert(EIGEN_HAS_THREAD_LOCAL_RANDOM == 1, "EIGEN_USE_PCG_RANDOM must enable the PCG backend");
#endif
