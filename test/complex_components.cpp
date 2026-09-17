// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

// Component access must not assume that a user-defined complex scalar has std::complex's layout.
struct PaddedComplex : std::complex<double> {
  using std::complex<double>::complex;
  double padding = 123;
};

template <>
struct Eigen::NumTraits<PaddedComplex> : Eigen::NumTraits<std::complex<double>> {};

template <typename Derived>
void check_component(const DenseBase<Derived>& component) {
  using Real = typename Derived::Scalar;
  Matrix<Real, Dynamic, Dynamic, Derived::IsRowMajor ? RowMajor : ColMajor> copy = component;
  Real sum = 0, minimum = component(0, 0), maximum = minimum, abs_maximum = numext::abs(minimum);
  for (Index col = 0; col < component.cols(); ++col) {
    for (Index row = 0; row < component.rows(); ++row) {
      const Real value = component(row, col);
      VERIFY_IS_EQUAL(copy(row, col), value);
      sum += value;
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
      if (numext::abs(value) > abs_maximum) abs_maximum = numext::abs(value);
    }
  }
  VERIFY_IS_EQUAL(component.sum(), sum);
  VERIFY_IS_EQUAL(component.template minCoeff<PropagateNaN>(), minimum);
  VERIFY_IS_EQUAL(component.template maxCoeff<PropagateNaN>(), maximum);
  VERIFY_IS_EQUAL(component.derived().cwiseAbs().template maxCoeff<PropagateNaN>(), abs_maximum);
  VERIFY_IS_CWISE_EQUAL(component.derived().colwise().sum().eval(), copy.colwise().sum().eval());
  VERIFY_IS_CWISE_EQUAL(component.derived().rowwise().sum().eval(), copy.rowwise().sum().eval());
}

template <typename MatrixType>
void component_expressions(Index rows, Index cols) {
  using Complex = typename MatrixType::Scalar;
  using Real = typename NumTraits<Complex>::Real;
  MatrixType values(rows, cols);
  for (Index i = 0; i < values.size(); ++i) values.data()[i] = Complex(Real(i % 13 - 6), Real(i % 17 + 1));
  const MatrixType& const_values = values;
  check_component(values.real());
  check_component(values.imag());
  check_component(const_values.real());
  check_component(const_values.imag());
  check_component(values.transpose().real());
  check_component(values.transpose().imag());
  check_component(values.conjugate().real());
  check_component(values.conjugate().imag());
  check_component(values.block(0, 0, rows - 1, cols - 1).real());
  check_component(values.block(0, 0, rows - 1, cols - 1).imag());
  check_component(const_values.block(0, 0, rows - 1, cols - 1).real());
  check_component(const_values.block(0, 0, rows - 1, cols - 1).imag());
  const auto real = values.real();
  const auto imag = values.imag();
  check_component(real.block(0, 0, rows - 1, cols - 1));
  check_component(imag.block(0, 0, rows - 1, cols - 1));

  MatrixType original = values;
  values.real() = original.imag();
  values.imag() = original.real();
  for (Index i = 0; i < values.size(); ++i) {
    VERIFY_IS_EQUAL(values.data()[i], Complex(original.data()[i].imag(), original.data()[i].real()));
  }
  values.real().block(0, 0, rows - 1, cols - 1).setConstant(Real(3));
  values.imag().array() += Real(2);
  for (Index col = 0; col < cols; ++col) {
    for (Index row = 0; row < rows; ++row) {
      VERIFY_IS_EQUAL(values(row, col).imag(), original(row, col).real() + Real(2));
      VERIFY_IS_EQUAL(values(row, col).real(), row < rows - 1 && col < cols - 1 ? Real(3) : original(row, col).imag());
    }
  }
}

template <typename Real>
void component_reductions() {
  using Complex = std::complex<Real>;
  using Vector = Matrix<Complex, Dynamic, 1>;
  constexpr bool vectorizable = (std::is_same<Real, float>::value || std::is_same<Real, double>::value) &&
                                internal::packet_traits<Real>::Vectorizable;
  using View = decltype(std::declval<Vector&>().real());
  using ConstOp = decltype(std::declval<const Vector&>().imag());
  STATIC_CHECK(bool(internal::redux_evaluator<View>::Flags & PacketAccessBit) == vectorizable);
  STATIC_CHECK(bool(internal::evaluator<ConstOp>::Flags & PacketAccessBit) == vectorizable);
  STATIC_CHECK(!(internal::evaluator<View>::Flags & PacketAccessBit));
  using Strided = Map<Vector, Unaligned, InnerStride<2>>;
  using StridedView = decltype(std::declval<Strided&>().real());
  STATIC_CHECK(!(internal::redux_evaluator<StridedView>::Flags & PacketAccessBit));
  using DynamicStride = Map<Vector, Unaligned, InnerStride<Dynamic>>;
  using DynamicView = decltype(std::declval<const DynamicStride&>().imag());
  STATIC_CHECK(!(internal::evaluator<DynamicView>::Flags & PacketAccessBit));
  using SumView = decltype((std::declval<Vector>() + std::declval<Vector>()).real());
  STATIC_CHECK(!(internal::evaluator<SumView>::Flags & PacketAccessBit));

  const Real nan = NumTraits<Real>::quiet_NaN();
  const Real inf = NumTraits<Real>::infinity();
  constexpr Index packet_size = internal::packet_traits<Real>::size;
  for (Index size : {Index(1), packet_size, packet_size + 1, 2 * packet_size - 1, Index(65)}) {
    Vector buffer(2 * size + 1);
    buffer.setConstant(Complex(Real(7), Real(-3)));
    Map<Vector> values(buffer.data() + 1, size);
    const auto& const_values = values;
    check_component(values.real());
    check_component(values.imag());
    Strided strided(buffer.data() + 1, size);
    check_component(strided.real());
    check_component(strided.imag());
    DynamicStride dynamic(buffer.data() + 1, size, InnerStride<Dynamic>(2));
    check_component(dynamic.real());
    check_component(dynamic.imag());
    for (Index i = 0; i < size; ++i) {
      for (Real special : {nan, inf, -inf, Real(0), -Real(0), std::numeric_limits<Real>::denorm_min()}) {
        values.setConstant(Complex(Real(7), Real(-3)));
        values(i) = Complex(special, Real(-3));
        Matrix<Real, Dynamic, 1> real = const_values.real();
        if ((numext::isnan)(special)) {
          VERIFY((numext::isnan)(real(i)));
          VERIFY((numext::isnan)(values.real().template maxCoeff<PropagateNaN>()));
          VERIFY((numext::isnan)(const_values.real().template minCoeff<PropagateNaN>()));
          if (size > 1) VERIFY_IS_EQUAL(values.real().template maxCoeff<PropagateNumbers>(), Real(7));
        } else {
          VERIFY_IS_EQUAL(real(i), special);
          VERIFY_IS_EQUAL((std::signbit)(real(i)), (std::signbit)(special));
        }
        VERIFY_IS_EQUAL(values.imag().template maxCoeff<PropagateNaN>(), Real(-3));
        values(i) = Complex(Real(7), special);
        Matrix<Real, Dynamic, 1> imag = const_values.imag();
        if ((numext::isnan)(special)) {
          VERIFY((numext::isnan)(imag(i)));
          VERIFY((numext::isnan)(values.imag().template minCoeff<PropagateNaN>()));
          VERIFY((numext::isnan)(const_values.imag().template maxCoeff<PropagateNaN>()));
          if (size > 1) VERIFY_IS_EQUAL(values.imag().template minCoeff<PropagateNumbers>(), Real(-3));
        } else {
          VERIFY_IS_EQUAL(imag(i), special);
          VERIFY_IS_EQUAL((std::signbit)(imag(i)), (std::signbit)(special));
        }
        VERIFY_IS_EQUAL(values.real().template minCoeff<PropagateNaN>(), Real(7));
      }
    }
  }
}

void component_custom_scalar() {
  using Vector = Matrix<PaddedComplex, Dynamic, 1>;
  Vector values(17);
  for (Index i = 0; i < values.size(); ++i) values(i) = PaddedComplex(double(i), double(2 * i + 1));
  const Vector& const_values = values;
  using RealView = decltype(const_values.real());
  using ImagView = decltype(const_values.imag());
  STATIC_CHECK(!(internal::evaluator<RealView>::Flags & PacketAccessBit));
  STATIC_CHECK(!(internal::evaluator<ImagView>::Flags & PacketAccessBit));
  check_component(const_values.real());
  check_component(const_values.imag());
}

EIGEN_DECLARE_TEST(complex_components) {
  CALL_SUBTEST_1(component_reductions<float>());
  CALL_SUBTEST_2(component_reductions<double>());
  CALL_SUBTEST_3(component_reductions<long double>());
  CALL_SUBTEST_3(component_custom_scalar());
  for (Index rows : {Index(2), Index(7), Index(17)}) {
    for (Index cols : {Index(3), Index(9)}) {
      EIGEN_UNUSED_VARIABLE(rows);
      EIGEN_UNUSED_VARIABLE(cols);
      CALL_SUBTEST_1((component_expressions<Matrix<std::complex<float>, Dynamic, Dynamic, ColMajor>>(rows, cols)));
      CALL_SUBTEST_1((component_expressions<Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>>(rows, cols)));
      CALL_SUBTEST_2((component_expressions<Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor>>(rows, cols)));
      CALL_SUBTEST_2((component_expressions<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(rows, cols)));
    }
  }
  CALL_SUBTEST_1((component_expressions<Matrix<std::complex<float>, 3, 5>>(3, 5)));
  CALL_SUBTEST_2((component_expressions<Matrix<std::complex<double>, 2, 2>>(2, 2)));
}
