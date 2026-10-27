// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

// A block of an expression without direct access has linear coefficient access only if it is a vector or an inner
// panel at compile time, not merely a vector at runtime.
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define BLOCK_ROWS Dynamic
#else
#define BLOCK_ROWS 1
#endif

using namespace Eigen;

float foo(const MatrixXf &m) { return (m + m).block<BLOCK_ROWS, 6>(1, 0, 1, 6).coeff(3); }

int main() {}
