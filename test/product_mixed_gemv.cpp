// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "product_test_helpers.h"

template <typename Scalar, int Order>
struct MixedGemvValueMapper {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const MatrixType& matrix;
  Index outer_stride;
  Scalar operator()(Index i, Index j) const { return matrix(i, j); }
  Index stride() const { return outer_stride; }
  template <typename Packet, int Alignment>
  Packet load(Index i, Index j) const {
    Scalar values[internal::unpacket_traits<Packet>::size];
    for (Index k = 0; k < internal::unpacket_traits<Packet>::size; ++k)
      values[k] = matrix(i + (Order == ColMajor ? k : 0), j + (Order == RowMajor ? k : 0));
    return internal::ploadu<Packet>(values);
  }
};

template <typename Real, int Order, bool RealLeft>
void mixed_gemv_value_mapper() {
  using Complex = std::complex<Real>;
  using LhsScalar = std::conditional_t<RealLeft, Real, Complex>;
  using RhsScalar = std::conditional_t<RealLeft, Complex, Real>;
  using LhsMapper = MixedGemvValueMapper<LhsScalar, Order>;
  using RhsMapper = MixedGemvValueMapper<RhsScalar, Order == ColMajor ? RowMajor : ColMajor>;
  const typename LhsMapper::MatrixType lhs = LhsMapper::MatrixType::Random(9, 7);
  const typename RhsMapper::MatrixType rhs = RhsMapper::MatrixType::Random(7, 1);
  Matrix<Complex, 9, 1> result = Matrix<Complex, 9, 1>::Zero();
  using Alpha = std::conditional_t<Order == ColMajor, RhsScalar, Complex>;
  internal::general_matrix_vector_product<Index, LhsScalar, LhsMapper, Order, true, RhsScalar, RhsMapper, true>::run(
      9, 7, LhsMapper{lhs, lhs.outerStride()}, RhsMapper{rhs, 1}, result.data(), 1, Alpha(1));
  for (Index i = 0; i < 9; ++i) {
    std::complex<long double> reference(0);
    long double magnitude = 0;
    for (Index j = 0; j < 7; ++j) {
      const auto term =
          std::conj(std::complex<long double>(lhs(i, j))) * std::conj(std::complex<long double>(rhs(j, 0)));
      reference += term;
      magnitude += std::abs(term);
    }
    VERIFY(std::abs(std::complex<long double>(result(i)) - reference) <=
           128 * NumTraits<Real>::epsilon() * (1 + magnitude));
  }
}

template <typename Real, int Order, bool RealLeft>
void mixed_gemv_layout() {
  using Complex = std::complex<Real>;
  using LhsScalar = std::conditional_t<RealLeft, Real, Complex>;
  using RhsScalar = std::conditional_t<RealLeft, Complex, Real>;
  using Lhs = Matrix<LhsScalar, Dynamic, Dynamic, Order>;
  using Rhs = Matrix<RhsScalar, Dynamic, 1>;
  for (Index rows : {Index(0), Index(1), Index(3), Index(8), Index(17), Index(33)}) {
    for (Index cols : {Index(0), Index(1), Index(3), Index(7), Index(8), Index(17), Index(129)}) {
      const Lhs lhs = Lhs::Random(rows, cols);
      const Rhs rhs = Rhs::Random(cols);
      const double factor = 16 * double(cols + 1);
      VERIFY(gemv_matches_long_double(lhs, rhs, factor));
      // Conjugating the real operand is the identity, so only the complex one varies.
      EIGEN_IF_CONSTEXPR (RealLeft) {
        VERIFY(gemv_matches_long_double(lhs, rhs.conjugate(), factor));
      } else {
        VERIFY(gemv_matches_long_double(lhs.conjugate(), rhs, factor));
      }
      Rhs storage = Rhs::Zero(2 * cols + 1);
      Map<Rhs, Unaligned, InnerStride<2>> strided_rhs(storage.data(), cols);
      strided_rhs = rhs;
      VERIFY(gemv_matches_long_double(lhs, strided_rhs, factor));
      Lhs padded = Lhs::Zero(rows + 2, cols + 2);
      padded.block(1, 1, rows, cols) = lhs;
      VERIFY(gemv_matches_long_double(padded.block(1, 1, rows, cols), rhs, factor));
    }
  }
  Matrix<LhsScalar, 8, 8, Order> fixed = Matrix<LhsScalar, 8, 8, Order>::Random();
  const Matrix<RhsScalar, 8, 1> fixed_rhs = Matrix<RhsScalar, 8, 1>::Random();
  VERIFY(gemv_matches_long_double(fixed, fixed_rhs, 16 * 9));

  fixed.setConstant(LhsScalar(NumTraits<Real>::quiet_NaN()));
  Matrix<Complex, 8, 1> result = Matrix<Complex, 8, 1>::Constant(Complex(7));
  using LhsMapper = internal::const_blas_data_mapper<LhsScalar, Index, Order>;
  using RhsMapper = internal::const_blas_data_mapper<RhsScalar, Index, Order == ColMajor ? RowMajor : ColMajor>;
  using Alpha = std::conditional_t<Order == ColMajor, RhsScalar, Complex>;
  internal::general_matrix_vector_product<Index, LhsScalar, LhsMapper, Order, false, RhsScalar, RhsMapper, false>::run(
      8, 8, LhsMapper(fixed.data(), fixed.outerStride()), RhsMapper(fixed_rhs.data(), 1), result.data(), 1, Alpha(0));
  VERIFY((result.array() == Complex(7)).all());
  EIGEN_IF_CONSTEXPR (!RealLeft) {
    result.noalias() += Complex(0) * (fixed * fixed_rhs);
    VERIFY((result.array() == Complex(7)).all());
  }
#if !defined(EIGEN_VECTORIZE_ALTIVEC) && !defined(EIGEN_VECTORIZE_VSX)
  // Dedicated AltiVec GEMV specializations require BLAS-style mappers.
  mixed_gemv_value_mapper<Real, Order, RealLeft>();
#endif
}

template <typename Real>
void mixed_gemv() {
  using Complex = std::complex<Real>;
  STATIC_CHECK((bool(internal::gemv_traits<Complex, Real>::Vectorizable) ==
                bool(internal::packet_traits<Complex>::Vectorizable)));
  STATIC_CHECK((bool(internal::gemv_traits<Real, Complex>::Vectorizable) ==
                bool(internal::packet_traits<Complex>::Vectorizable)));
  mixed_gemv_layout<Real, ColMajor, false>();
  mixed_gemv_layout<Real, ColMajor, true>();
  mixed_gemv_layout<Real, RowMajor, false>();
  mixed_gemv_layout<Real, RowMajor, true>();
}

template <typename Real, int Order>
class MixedProxyMatrix;

namespace Eigen {
namespace internal {
template <typename Real, int Order>
struct traits<MixedProxyMatrix<Real, Order>> : traits<Matrix<Real, 2, 2, Order>> {
  static constexpr unsigned int Flags = traits<Matrix<Real, 2, 2, Order>>::Flags & ~LvalueBit;
};
}  // namespace internal
}  // namespace Eigen

template <typename Real, int Order>
class MixedProxyMatrix : public MatrixBase<MixedProxyMatrix<Real, Order>> {
 public:
  struct Proxy {
    const Real* value;
    operator Real() const { return *value; }
  };
  MixedProxyMatrix() {
    for (Index i = 0; i < 2; ++i)
      for (Index j = 0; j < 2; ++j) {
        const Index index = Order == ColMajor ? i + 2 * j : 2 * i + j;
        values[index] = Real(i + 2 * j + 1);
        proxies[index].value = values + index;
      }
  }
  Index rows() const { return 2; }
  Index cols() const { return 2; }
  const Real* data() const { return values; }
  Index innerStride() const { return 1; }
  Index outerStride() const { return 2; }
  Real values[4];
  Proxy proxies[4];
};

namespace Eigen {
namespace internal {
template <typename Real, int Order>
struct evaluator<MixedProxyMatrix<Real, Order>> : evaluator_base<MixedProxyMatrix<Real, Order>> {
  using Xpr = MixedProxyMatrix<Real, Order>;
  static constexpr int CoeffReadCost = 1;
  static constexpr unsigned int Flags = DirectAccessBit | PacketAccessBit | (Order == RowMajor ? RowMajorBit : 0);
  static constexpr int Alignment = 0;
  const Xpr& matrix;
  explicit evaluator(const Xpr& xpr) : matrix(xpr) {}
  const typename Xpr::Proxy& coeff(Index row, Index col) const {
    return matrix.proxies[Order == ColMajor ? row + 2 * col : 2 * row + col];
  }
  template <int LoadMode, typename Packet>
  Packet packet(Index row, Index col) const {
    return ploadu<Packet>(matrix.data() + (Order == ColMajor ? row + 2 * col : 2 * row + col));
  }
};
}  // namespace internal
}  // namespace Eigen

template <typename Real, int Order>
void mixed_proxy_packets() {
  using Complex = std::complex<Real>;
  const MixedProxyMatrix<Real, Order> matrix;
  STATIC_CHECK((!internal::product_real_packet_access<MixedProxyMatrix<Real, Order>>::value));
  Matrix<Complex, 2, 1> rhs;
  rhs << Complex(1, 2), Complex(3, 4);
  const Matrix<Complex, 2, 1> column = matrix.lazyProduct(rhs);
  const Matrix<Complex, 1, 2> row = rhs.transpose().lazyProduct(matrix);
  VERIFY_IS_EQUAL((matrix * rhs).eval(), column);
  VERIFY_IS_EQUAL((rhs.transpose() * matrix).eval(), row);
  VERIFY_IS_EQUAL(column(0), Complex(10, 14));
  VERIFY_IS_EQUAL(column(1), Complex(14, 20));
  VERIFY_IS_EQUAL(row(0), Complex(7, 10));
  VERIFY_IS_EQUAL(row(1), Complex(15, 22));
}

template <typename Product>
void mixed_infinite_segments(const Product&, internal::false_type) {}

template <typename Product>
void mixed_infinite_segments(const Product& product, internal::true_type) {
  using Scalar = typename Product::Scalar;
  using Packet = typename internal::packet_traits<Scalar>::type;
  constexpr Index packet_size = internal::unpacket_traits<Packet>::size;
  internal::evaluator<Product> evaluator(product);
  for (Index begin = 0; begin <= packet_size; ++begin) {
    for (Index count = 0; count <= packet_size - begin; ++count) {
      if (begin + count > product.size()) continue;
      Scalar values[packet_size];
      internal::pstoreu(values, evaluator.template packetSegment<Unaligned, Packet>(0, 0, begin, count));
      for (Index i = 0; i < packet_size; ++i) {
        if (i >= begin && i < begin + count) {
          VERIFY_IS_EQUAL(values[i].real(), NumTraits<typename NumTraits<Scalar>::Real>::infinity());
          VERIFY_IS_EQUAL(values[i].imag(), NumTraits<typename NumTraits<Scalar>::Real>::infinity());
        }
      }
    }
  }
}

template <typename Result, typename Product>
void check_infinite_product(const Product& product) {
  using Real = typename NumTraits<typename Result::Scalar>::Real;
  const Result result = product;
  for (Index i = 0; i < result.size(); ++i) {
    VERIFY_IS_EQUAL(result(i).real(), NumTraits<Real>::infinity());
    VERIFY_IS_EQUAL(result(i).imag(), NumTraits<Real>::infinity());
  }
}

template <typename Product>
void check_infinite_lazy_product(const Product& product) {
  using Scalar = typename Product::Scalar;
  check_infinite_product<Matrix<Scalar, Dynamic, Dynamic>>(product);
  mixed_infinite_segments(product, internal::bool_constant<internal::packet_traits<Scalar>::Vectorizable>());
}

template <typename Real>
EIGEN_DONT_INLINE void mixed_infinite_factor() {
#if !defined(__FAST_MATH__)
  using Complex = std::complex<Real>;
  const Real inf = NumTraits<Real>::infinity();
  for (Index size = 1; size <= 9; ++size) {
    for (Index depth : {Index(1), Index(3)}) {
      const Matrix<Complex, Dynamic, Dynamic, ColMajor> a =
          Matrix<Complex, Dynamic, Dynamic, ColMajor>::Constant(size, depth, Complex(1, 2));
      const Matrix<Complex, Dynamic, Dynamic, RowMajor> transposed = a.transpose();
      const Matrix<Real, Dynamic, 1> factors = Matrix<Real, Dynamic, 1>::Constant(depth, inf);
      const Matrix<Real, Dynamic, Dynamic, ColMajor> real =
          Matrix<Real, Dynamic, Dynamic, ColMajor>::Constant(size, depth, inf);
      const Matrix<Complex, Dynamic, 1> complex_factors = Matrix<Complex, Dynamic, 1>::Constant(depth, Complex(1, 2));
      check_infinite_lazy_product(a.lazyProduct(factors));
      check_infinite_lazy_product(factors.transpose().lazyProduct(transposed));
      check_infinite_lazy_product(real.lazyProduct(complex_factors));
      check_infinite_lazy_product(complex_factors.transpose().lazyProduct(real.transpose()));
      if (size == 3 && depth == 3) {
        const auto a3 = a.template topLeftCorner<3, 3>();
        const auto transposed3 = transposed.template topLeftCorner<3, 3>();
        const auto real3 = real.template topLeftCorner<3, 3>();
        const auto factors3 = factors.template head<3>();
        const auto complex_factors3 = complex_factors.template head<3>();
        check_infinite_product<Matrix<Complex, 3, 1>>(a3 * factors3);
        check_infinite_product<Matrix<Complex, 1, 3>>(factors3.transpose() * transposed3);
        check_infinite_product<Matrix<Complex, 3, 1>>(real3 * complex_factors3);
        check_infinite_product<Matrix<Complex, 1, 3>>(complex_factors3.transpose() * real3.transpose());
      }
    }
  }
#endif
}

template <typename Lhs, typename Rhs>
void check_mixed_packets(const Lhs& lhs, const Rhs& rhs) {
  using Real = typename NumTraits<typename Lhs::Scalar>::Real;
  using Complex = std::complex<Real>;
  using Product = decltype(lhs.lazyProduct(rhs));
  using Result = typename Product::PlainObject;
  Matrix<Complex, Dynamic, Dynamic> reference(lhs.rows(), rhs.cols());
  for (Index j = 0; j < rhs.cols(); ++j) {
    for (Index i = 0; i < lhs.rows(); ++i) {
      Real re = 0, im = 0;
      for (Index k = 0; k < lhs.cols(); ++k) {
        const Complex a(lhs.coeff(i, k)), b(rhs.coeff(k, j));
        re += a.real() * b.real();
        im += NumTraits<typename Lhs::Scalar>::IsComplex ? a.imag() * b.real() : a.real() * b.imag();
      }
      reference(i, j) = Complex(re, im);
    }
  }
  Result result = lhs.lazyProduct(rhs);
  VERIFY_IS_EQUAL(result, reference);
  result.setZero();
  result.noalias() += lhs * rhs;
  VERIFY_IS_EQUAL(result, reference);
  VERIFY_IS_EQUAL(lhs.lazyProduct(rhs).sum(), reference.sum());
  Result reversed = lhs.lazyProduct(rhs).reverse();
  VERIFY_IS_EQUAL(reversed, reference.reverse());
  Matrix<Complex, Dynamic, Dynamic, Result::IsRowMajor ? RowMajor : ColMajor> padded(lhs.rows() + 1, rhs.cols() + 1);
  padded.setConstant(Complex(19, 23));
  auto destination = padded.topLeftCorner(lhs.rows(), rhs.cols());
  destination.noalias() = lhs * rhs;
  VERIFY_IS_EQUAL(destination, reference);
  VERIFY((padded.bottomRows(1).array() == Complex(19, 23)).all());
  VERIFY((padded.rightCols(1).array() == Complex(19, 23)).all());
}

template <typename Real, int Size>
void mixed_fixed_packets() {
  using Complex = std::complex<Real>;
  using ColMatrix = Matrix<Complex, Size, Size, ColMajor>;
  using RowMatrix = Matrix<Complex, Size, Size, RowMajor>;
  ColMatrix a;
  Matrix<Real, Size, 1> x;
  for (Index j = 0; j < Size; ++j) {
    x(j) = Real(j + 1) / Real(4);
    for (Index i = 0; i < Size; ++i) a(i, j) = Complex(Real(i + j + 1) / Real(2), Real(i - j) / Real(4));
  }
  using ColProduct = decltype(a.lazyProduct(x));
  STATIC_CHECK((bool(internal::evaluator<ColProduct>::Flags & PacketAccessBit) ==
                bool(internal::evaluator<ColMatrix>::Flags & PacketAccessBit)));
  check_mixed_packets(a, x);
  check_mixed_packets(a.conjugate(), x);
  const RowMatrix b = a;
  const Matrix<Real, 1, Size> row = x.transpose();
  using RowProduct = decltype(row.lazyProduct(b));
  STATIC_CHECK((bool(internal::evaluator<RowProduct>::Flags & PacketAccessBit) ==
                bool(internal::evaluator<RowMatrix>::Flags & PacketAccessBit)));
  check_mixed_packets(row, b);
  check_mixed_packets(row, b.conjugate());

  using RealColMatrix = Matrix<Real, Size, Size, ColMajor>;
  const RealColMatrix real = a.real();
  const Matrix<Complex, Size, 1> vector = a.col(0);
  using Expanded = decltype(real.lazyProduct(vector));
  STATIC_CHECK((bool(internal::evaluator<Expanded>::Flags & PacketAccessBit) ==
                (bool(internal::packet_traits<Complex>::Vectorizable) &&
                 bool(internal::evaluator<RealColMatrix>::Flags & PacketAccessBit))));
  check_mixed_packets(real, vector);
  check_mixed_packets(real, vector.conjugate());
  using RealRowMatrix = Matrix<Real, Size, Size, RowMajor>;
  const RealRowMatrix real_row = real;
  const Matrix<Complex, 1, Size> complex_row = vector.transpose();
  using ExpandedRow = decltype(complex_row.lazyProduct(real_row));
  STATIC_CHECK((bool(internal::evaluator<ExpandedRow>::Flags & PacketAccessBit) ==
                (bool(internal::packet_traits<Complex>::Vectorizable) &&
                 bool(internal::evaluator<RealRowMatrix>::Flags & PacketAccessBit))));
  check_mixed_packets(complex_row, real_row);
  check_mixed_packets(complex_row.conjugate(), real_row);
  check_mixed_packets(a.unaryExpr([](Complex value) { return value; }), x);
}

template <typename Real>
void mixed_dynamic_packets() {
  using Complex = std::complex<Real>;
  for (Index depth : {Index(0), Index(1), Index(3), Index(8)}) {
    Matrix<Complex, Dynamic, Dynamic, ColMajor> a(7, depth);
    Matrix<Real, Dynamic, Dynamic> b(depth, 3);
    for (Index j = 0; j < depth; ++j)
      for (Index i = 0; i < a.rows(); ++i) a(i, j) = Complex(Real(i + 1) / 2, Real(j + 1) / 4);
    for (Index j = 0; j < b.cols(); ++j)
      for (Index i = 0; i < depth; ++i) b(i, j) = Real(i - j + 1) / 4;
    check_mixed_packets(a, b);
    const Matrix<Real, Dynamic, Dynamic, ColMajor> real = a.real();
    const Matrix<Complex, Dynamic, Dynamic> complex = b.template cast<Complex>();
    check_mixed_packets(real, complex);
    check_mixed_packets(complex.transpose(), real.transpose());
    Matrix<Complex, Dynamic, Dynamic, RowMajor> transposed = a.transpose();
    Matrix<Real, 1, Dynamic> row(depth);
    for (Index i = 0; i < depth; ++i) row(i) = Real(i + 1) / 4;
    check_mixed_packets(row, transposed);
  }
}

// Compares bit-for-bit up to the NaN payload: NaN matches NaN, and zeros must agree in sign.
template <typename Real>
void verify_identical(const Real& actual, const Real& expected) {
  if ((numext::isnan)(expected)) {
    VERIFY((numext::isnan)(actual));
  } else {
    VERIFY_IS_EQUAL(actual, expected);
    if (expected == Real(0)) VERIFY_IS_EQUAL((std::signbit)(actual), (std::signbit)(expected));
  }
}

template <typename Real>
void verify_identical(const std::complex<Real>& actual, const std::complex<Real>& expected) {
  verify_identical(actual.real(), expected.real());
  verify_identical(actual.imag(), expected.imag());
}

template <typename Real>
void mixed_special_packets() {
  using Complex = std::complex<Real>;
  const Real inf = NumTraits<Real>::infinity(), nan = NumTraits<Real>::quiet_NaN(), zero(0);
  Matrix<Complex, 5, 1> a;
  a << Complex(inf, 2), Complex(2, inf), Complex(nan, 3), Complex(-zero, zero), Complex(4, -zero);
  const Matrix<Real, 1, 1> factor = Matrix<Real, 1, 1>::Constant(2);
  const Matrix<Complex, Dynamic, Dynamic, ColMajor> dynamic = a;
  const Matrix<Real, Dynamic, Dynamic> dynamic_factor = factor;
  const Matrix<Complex, 5, 1> result = a.lazyProduct(factor);
  const Matrix<Complex, 5, 1> dynamic_result = dynamic.lazyProduct(dynamic_factor);
  for (Index i = 0; i < a.size(); ++i) {
    verify_identical(result(i), a(i) * Real(2));
    verify_identical(dynamic_result(i), a(i) * Real(2));
  }

  // MSVC's real*complex operator introduces cross terms; both Eigen paths must scale components directly.
  const Complex negative_zero(-zero, -zero);
  for (const Complex& value : {internal::fast_mult_op<Real, Complex>()(Real(2), negative_zero),
                               internal::conj_helper<Real, Complex, false, false>().pmul(Real(2), negative_zero),
                               internal::conj_helper<Real, Complex, true, false>().pmul(Real(2), negative_zero)})
    verify_identical(value, negative_zero);
  for (const Complex& value : {internal::conj_helper<Real, Complex, false, true>().pmul(Real(2), negative_zero),
                               internal::conj_helper<Real, Complex, true, true>().pmul(Real(2), negative_zero)})
    verify_identical(value, Complex(-zero, zero));
  for (const Complex& value : {internal::fast_mult_op<Real, Complex>()(Real(2), a(0)),
                               internal::conj_helper<Real, Complex, false, false>().pmul(Real(2), a(0))})
    verify_identical(value, Complex(inf, 4));

  const Matrix<Real, 5, 1> real = Matrix<Real, 5, 1>::Constant(2);
  const Matrix<Complex, 1, 1> complex_factor = Matrix<Complex, 1, 1>::Constant(negative_zero);
  const Matrix<Complex, 5, 1> expanded = real.lazyProduct(complex_factor);
  const Matrix<Real, Dynamic, Dynamic, ColMajor> dynamic_real = real;
  const Matrix<Complex, Dynamic, Dynamic> dynamic_complex_factor = complex_factor;
  const Matrix<Complex, Dynamic, Dynamic> dynamic_expanded = dynamic_real.lazyProduct(dynamic_complex_factor);
  for (Index i = 0; i < real.size(); ++i) {
    verify_identical(expanded(i), negative_zero);
    verify_identical(dynamic_expanded(i), negative_zero);
  }

  Matrix<Real, 5, 1> real_special;
  real_special << inf, nan, zero, -zero, Real(2);
  const Complex c(2, -3);
  const Matrix<Complex, 5, 1> special = real_special.lazyProduct(Matrix<Complex, 1, 1>::Constant(c));
  for (Index i = 0; i < real_special.size(); ++i)
    verify_identical(special(i), Complex(real_special(i) * c.real(), real_special(i) * c.imag()));
}

template <typename Real>
void mixed_strided_packets() {
  using Complex = std::complex<Real>;
  using ColMatrix = Matrix<Complex, Dynamic, Dynamic, ColMajor>;
  struct Storage {
    Real padding;
    Complex values[9];
  } storage;
  storage.padding = Real(41);
  Map<Matrix<Complex, 3, 3, ColMajor>, Unaligned> unaligned(storage.values);
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) unaligned(i, j) = Complex(Real(i + j) / 2, Real(i - j) / 4);
  const Matrix<Real, 3, 1> ones = Matrix<Real, 3, 1>::Ones();
  check_mixed_packets(unaligned, ones);
  check_mixed_packets(ones.transpose(), unaligned.transpose());
  Real real_storage[9];
  Map<Matrix<Real, 3, 3, ColMajor>, Unaligned> real_map(real_storage);
  real_map = unaligned.real();
  const Matrix<Complex, 3, 1> complex_ones = Matrix<Complex, 3, 1>::Constant(Complex(1, 2));
  check_mixed_packets(real_map, complex_ones);
  check_mixed_packets(complex_ones.transpose(), real_map.transpose());
  VERIFY_IS_EQUAL(storage.padding, Real(41));
  for (Index inner : {Index(1), Index(2)}) {
    Matrix<Complex, 64, 1> strided_storage;
    strided_storage.setConstant(Complex(31, 37));
    Map<ColMatrix, Unaligned, Stride<Dynamic, Dynamic>> a(strided_storage.data() + 1, 7, 3,
                                                          Stride<Dynamic, Dynamic>(20, inner));
    for (Index j = 0; j < a.cols(); ++j)
      for (Index i = 0; i < a.rows(); ++i) a(i, j) = Complex(Real(i + j) / 2, Real(i - j) / 4);
    Matrix<Real, 3, 1> x;
    x << Real(1) / 4, Real(1) / 2, Real(3) / 4;
    check_mixed_packets(a, x);
    check_mixed_packets(x.transpose(), a.transpose());
    Matrix<Real, 64, 1> real_backing;
    real_backing.setConstant(Real(43));
    Map<Matrix<Real, Dynamic, Dynamic, ColMajor>, Unaligned, Stride<Dynamic, Dynamic>> strided_real_map(
        real_backing.data() + 1, 7, 3, Stride<Dynamic, Dynamic>(20, inner));
    strided_real_map = a.real();
    using Fallback = decltype(strided_real_map.lazyProduct(complex_ones));
    STATIC_CHECK((!(internal::evaluator<Fallback>::Flags & PacketAccessBit)));
    check_mixed_packets(strided_real_map, complex_ones);
    check_mixed_packets(complex_ones.transpose(), strided_real_map.transpose());
    Map<Matrix<Real, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<Dynamic>> unit_map(
        real_backing.data() + 1, 7, 3, OuterStride<Dynamic>(20));
    check_mixed_packets(unit_map, complex_ones);
    check_mixed_packets(complex_ones.transpose(), unit_map.transpose());
  }
}

template <typename Real>
void mixed_packet_tests() {
  mixed_proxy_packets<Real, ColMajor>();
  mixed_proxy_packets<Real, RowMajor>();
  mixed_fixed_packets<Real, 2>();
  mixed_fixed_packets<Real, 3>();
  mixed_fixed_packets<Real, 7>();
  mixed_fixed_packets<Real, 8>();
  mixed_dynamic_packets<Real>();
  mixed_special_packets<Real>();
  mixed_strided_packets<Real>();
  mixed_infinite_factor<Real>();
}

EIGEN_DECLARE_TEST(product_mixed_gemv) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(mixed_gemv<float>());
    CALL_SUBTEST_1(mixed_packet_tests<float>());
    CALL_SUBTEST_2(mixed_gemv<double>());
    CALL_SUBTEST_2(mixed_packet_tests<double>());
  }
}
