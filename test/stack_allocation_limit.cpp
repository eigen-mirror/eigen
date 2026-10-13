// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A backend whose blocked kernels need a larger stack panel may raise Eigen's own default limit --
// see the SVE and RVV blocks in ConfigureVectorization.h -- but an explicit EIGEN_STACK_ALLOCATION_LIMIT
// is a caller policy on stack safety and must reach DenseStorage.h and Memory.h unchanged. Rewriting
// one upward turns OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG off for objects the caller wanted rejected and
// switches internal buffers from the heap back to alloca.
//
// 1 is below anything a backend could derive, so this never passes vacuously, and nothing here
// allocates a fixed-size Eigen object that the limit would then reject.
#define EIGEN_STACK_ALLOCATION_LIMIT 1
#include "main.h"

static_assert(EIGEN_STACK_ALLOCATION_LIMIT == 1, "an explicit EIGEN_STACK_ALLOCATION_LIMIT must survive");

// Memory.h reads the macro at each use rather than through a saved constant, so check the users too.
void check_stack_allocation_is_disabled() {
  VERIFY_IS_EQUAL(int(sizeof(double) * 4 <= EIGEN_STACK_ALLOCATION_LIMIT), 0);
  ei_declare_aligned_stack_constructed_variable(double, buffer, 4, 0);
  VERIFY(buffer != nullptr);
  for (Index i = 0; i < 4; ++i) buffer[i] = double(i);
  for (Index i = 0; i < 4; ++i) VERIFY_IS_EQUAL(buffer[i], double(i));
}

// isApprox evaluates a lazy operand before its scaled fallback, which squares overflowing at 1e200 force; that
// temporary must not be a fixed-size object the limit rejects.
void check_approx_comparison_temporaries() {
  using Vector = Matrix<double, 2, 1>;
  VERIFY(Vector::Constant(1e200).isApprox(Vector::Constant(1e200)));
  VERIFY(!Vector::Constant(1e200).isApprox(Vector::Constant(2e200)));
}

// Triangular product and solver kernels declare internal panel buffers (TriangularMatrixMatrix,
// GeneralMatrixMatrixTriangular, TriangularSolverMatrix); they must fall back to dynamic allocation when
// EIGEN_STACK_ALLOCATION_LIMIT is smaller than the panel size.
template <typename Scalar>
void check_triangular_product_scratch() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat a = Mat::Random(16, 16);
  Mat b = Mat::Random(16, 16);
  Mat c = Mat::Zero(16, 16);
  c.noalias() += a.template triangularView<Lower>() * b;
  c.noalias() += b * a.template triangularView<Upper>();
  c.template triangularView<Lower>() += a * a.adjoint();
  // Use a non-multiple-of-PacketSize RHS column count to also exercise TriangularSolverMatrix::solve_padded.
  Mat rhs = Mat::Random(16, 3);
  a.diagonal().array() += Scalar(4);
  Mat x = a.template triangularView<Upper>().solve(rhs);
  VERIFY_IS_APPROX(a.template triangularView<Upper>() * x, rhs);
  Matrix<Scalar, Dynamic, Dynamic, RowMajor> a_row = a;
  Mat x_row = a_row.template triangularView<Upper>().solve(rhs);
  VERIFY_IS_APPROX(a_row.template triangularView<Upper>() * x_row, rhs);
}

EIGEN_DECLARE_TEST(stack_allocation_limit) {
  CALL_SUBTEST(check_stack_allocation_is_disabled());
  CALL_SUBTEST(check_approx_comparison_temporaries());
  CALL_SUBTEST(check_triangular_product_scratch<float>());
  CALL_SUBTEST(check_triangular_product_scratch<double>());
  CALL_SUBTEST(check_triangular_product_scratch<std::complex<double>>());
}
