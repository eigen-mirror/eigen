// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_NONCOMMUTATIVE_SCALAR_H
#define EIGEN_TEST_NONCOMMUTATIVE_SCALAR_H

#include <array>
#include <Eigen/Core>

namespace noncommutative_scalar {

// Scalars of distinct types whose mixed products have a third type, so that a kernel taking its operands in the wrong
// order fails to compile: <0> * <1> -> <2>, <1> * <0> -> <3>, and <0> + <1> -> <2>. Values multiply and add as ints.
template <int Tag>
struct TaggedScalar {
  explicit TaggedScalar(int v = 0) : value(v) {}
  int value;
};

inline TaggedScalar<2> operator*(const TaggedScalar<0>& lhs, const TaggedScalar<1>& rhs) {
  return TaggedScalar<2>(lhs.value * rhs.value);
}
inline TaggedScalar<3> operator*(const TaggedScalar<1>& lhs, const TaggedScalar<0>& rhs) {
  return TaggedScalar<3>(lhs.value * rhs.value);
}
inline TaggedScalar<2> operator+(const TaggedScalar<0>& lhs, const TaggedScalar<1>& rhs) {
  return TaggedScalar<2>(lhs.value + rhs.value);
}

// 2x2 integer matrices stored row-major: the product is associative but not commutative.
struct NoncommutingBlock {
  explicit NoncommutingBlock(int diagonal = 0) : values{{diagonal, 0, 0, diagonal}} {}
  NoncommutingBlock& operator=(int diagonal) {
    values = {{diagonal, 0, 0, diagonal}};
    return *this;
  }
  NoncommutingBlock& operator+=(const NoncommutingBlock& rhs) {
    for (int i = 0; i < 4; ++i) values[i] += rhs.values[i];
    return *this;
  }
  std::array<int, 4> values;
};

inline NoncommutingBlock operator*(const NoncommutingBlock& lhs, const NoncommutingBlock& rhs) {
  NoncommutingBlock result;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k) result.values[2 * i + j] += lhs.values[2 * i + k] * rhs.values[2 * k + j];
  return result;
}

inline double abs(const NoncommutingBlock& scalar) {
  double result = 0;
  for (int value : scalar.values) result += Eigen::numext::abs(value);
  return result;
}

// Quaternions: multiplication is associative and conjugation reverses products, but ab != ba in general.
struct Quaternion {
  double r, i, j, k;
  Quaternion(double real = 0, double x = 0, double y = 0, double z = 0) : r(real), i(x), j(y), k(z) {}
  Quaternion operator+(const Quaternion& b) const { return Quaternion(r + b.r, i + b.i, j + b.j, k + b.k); }
  Quaternion operator-(const Quaternion& b) const { return Quaternion(r - b.r, i - b.i, j - b.j, k - b.k); }
  Quaternion operator-() const { return Quaternion(-r, -i, -j, -k); }
  Quaternion operator*(const Quaternion& b) const {
    return Quaternion(r * b.r - i * b.i - j * b.j - k * b.k, r * b.i + i * b.r + j * b.k - k * b.j,
                      r * b.j - i * b.k + j * b.r + k * b.i, r * b.k + i * b.j - j * b.i + k * b.r);
  }
  Quaternion& operator+=(const Quaternion& b) { return *this = *this + b; }
  Quaternion& operator-=(const Quaternion& b) { return *this = *this - b; }
  Quaternion& operator*=(const Quaternion& b) { return *this = *this * b; }
  bool operator==(const Quaternion& b) const { return r == b.r && i == b.i && j == b.j && k == b.k; }
  bool operator!=(const Quaternion& b) const { return !(*this == b); }
};

inline double abs(const Quaternion& a) { return Eigen::numext::sqrt(a.r * a.r + a.i * a.i + a.j * a.j + a.k * a.k); }
inline Quaternion conj(const Quaternion& a) { return Quaternion(a.r, -a.i, -a.j, -a.k); }
inline double real(const Quaternion& a) { return a.r; }
inline double imag(const Quaternion& a) { return a.i; }

}  // namespace noncommutative_scalar

namespace Eigen {
template <>
struct ScalarBinaryOpTraits<
    noncommutative_scalar::TaggedScalar<0>, noncommutative_scalar::TaggedScalar<1>,
    internal::scalar_product_op<noncommutative_scalar::TaggedScalar<0>, noncommutative_scalar::TaggedScalar<1>>> {
  using ReturnType = noncommutative_scalar::TaggedScalar<2>;
};

template <>
struct ScalarBinaryOpTraits<
    noncommutative_scalar::TaggedScalar<1>, noncommutative_scalar::TaggedScalar<0>,
    internal::scalar_product_op<noncommutative_scalar::TaggedScalar<1>, noncommutative_scalar::TaggedScalar<0>>> {
  using ReturnType = noncommutative_scalar::TaggedScalar<3>;
};

template <>
struct ScalarBinaryOpTraits<
    noncommutative_scalar::TaggedScalar<0>, noncommutative_scalar::TaggedScalar<1>,
    internal::scalar_sum_op<noncommutative_scalar::TaggedScalar<0>, noncommutative_scalar::TaggedScalar<1>>> {
  using ReturnType = noncommutative_scalar::TaggedScalar<2>;
};

template <>
struct NumTraits<noncommutative_scalar::NoncommutingBlock>
    : GenericNumTraits<noncommutative_scalar::NoncommutingBlock> {
  using Real = double;
  static constexpr bool IsSigned = true;
  static constexpr bool RequireInitialization = true;
};

template <>
struct NumTraits<noncommutative_scalar::Quaternion> : GenericNumTraits<noncommutative_scalar::Quaternion> {
  using Real = double;
  using Literal = double;
  static constexpr bool IsComplex = true;
  static constexpr bool RequireInitialization = true;
};
namespace numext {
// The generic numext::abs of a complex scalar is hypot(real, imag), which would ignore the j and k components.
template <>
inline double abs(const noncommutative_scalar::Quaternion& a) {
  return noncommutative_scalar::abs(a);
}
}  // namespace numext
}  // namespace Eigen

#endif
