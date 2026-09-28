// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Dst, typename Src, typename Functor>
void check_block_coeffwise_dispatch() {
#if EIGEN_UNALIGNED_VECTORIZE
  using Scalar = typename Dst::Scalar;
  using Kernel = internal::generic_dense_assignment_kernel<internal::evaluator<Dst>, internal::evaluator<Src>, Functor>;
  using Traits = typename Kernel::AssignmentTraits;
  constexpr bool packetSupported = internal::packet_traits<Scalar>::Vectorizable &&
                                   internal::packet_traits<Scalar>::HasMul &&
                                   internal::functor_traits<Functor>::PacketAccess;
  STATIC_CHECK(!packetSupported || Traits::Traversal == SliceVectorizedTraversal);
  STATIC_CHECK(!packetSupported || Traits::Unrolling == NoUnrolling);
  using Loop = internal::dense_assignment_loop_impl<Kernel, SliceVectorizedTraversal, NoUnrolling>;
  STATIC_CHECK(!packetSupported || EIGEN_MAX_ALIGN_BYTES < Loop::RequestedAlignment || Loop::DstIsAligned);
#endif
}

template <typename Scalar, int Order>
void block_coeffwise() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Product = internal::remove_all_t<
      decltype(std::declval<const Mat&>().block(0, 0, 0, 0).cwiseProduct(std::declval<const Mat&>()))>;
  check_block_coeffwise_dispatch<Mat, Product, internal::assign_op<Scalar, Scalar>>();
  check_block_coeffwise_dispatch<Mat, Product, internal::add_assign_op<Scalar, Scalar>>();
  check_block_coeffwise_dispatch<Mat, Product, internal::sub_assign_op<Scalar, Scalar>>();
  constexpr Index packetSize = internal::packet_traits<Scalar>::size;
  for (Index inner = 0; inner <= (std::max)(Index(33), 2 * packetSize + 1); ++inner) {
    for (Index offset : {0, 1, 3}) {
      const Index rows = Order == ColMajor ? inner : 5;
      const Index cols = Order == ColMajor ? 5 : inner;
      const Mat src = Mat::Random(rows + 7, cols + 7);
      const Mat rhs = Mat::Random(rows, cols);
      Mat dst = Mat::Constant(rows + 7, cols + 7, Scalar(3));
      Mat expected = dst;
      auto block = dst.block(offset, offset, rows, cols);
      block.array() = src.block(offset, offset, rows, cols).array() * rhs.array();
      for (Index j = 0; j < cols; ++j)
        for (Index i = 0; i < rows; ++i) expected(i + offset, j + offset) = src(i + offset, j + offset) * rhs(i, j);
      VERIFY_IS_APPROX(dst, expected);
      block.array() += src.block(offset, offset, rows, cols).array() * rhs.array();
      expected.block(offset, offset, rows, cols) *= Scalar(2);
      VERIFY_IS_APPROX(dst, expected);
      Mat packed = src.block(offset, offset, rows, cols).cwiseProduct(rhs);
      VERIFY_IS_APPROX(packed, (expected.block(offset, offset, rows, cols) / Scalar(2)));
      const auto product = src.block(offset, offset, rows, cols).cwiseProduct(rhs);
      const Mat initial = Mat::Random(rows, cols);
      Mat compound = initial;
      Mat reference(rows, cols);
      compound += product;
      for (Index j = 0; j < cols; ++j)
        for (Index i = 0; i < rows; ++i) reference(i, j) = initial(i, j) + src(i + offset, j + offset) * rhs(i, j);
      VERIFY_IS_APPROX(compound, reference);
      compound = initial;
      compound -= product;
      for (Index j = 0; j < cols; ++j)
        for (Index i = 0; i < rows; ++i) reference(i, j) = initial(i, j) - src(i + offset, j + offset) * rhs(i, j);
      VERIFY_IS_APPROX(compound, reference);
    }
  }
}

EIGEN_DECLARE_TEST(block_coeffwise) {
  CALL_SUBTEST((block_coeffwise<float, ColMajor>()));
  CALL_SUBTEST((block_coeffwise<float, RowMajor>()));
  CALL_SUBTEST((block_coeffwise<double, ColMajor>()));
  CALL_SUBTEST((block_coeffwise<double, RowMajor>()));
  CALL_SUBTEST((block_coeffwise<std::complex<float>, ColMajor>()));
  CALL_SUBTEST((block_coeffwise<std::complex<float>, RowMajor>()));
}
