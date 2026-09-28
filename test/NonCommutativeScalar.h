// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_NONCOMMUTATIVE_SCALAR_H
#define EIGEN_TEST_NONCOMMUTATIVE_SCALAR_H

#include <Eigen/Core>

namespace noncommutative_scalar {

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

inline Quaternion conj(const Quaternion& a) { return Quaternion(a.r, -a.i, -a.j, -a.k); }
inline double real(const Quaternion& a) { return a.r; }
inline double imag(const Quaternion& a) { return a.i; }

}  // namespace noncommutative_scalar

namespace Eigen {
template <>
struct NumTraits<noncommutative_scalar::Quaternion> : GenericNumTraits<noncommutative_scalar::Quaternion> {
  using Real = double;
  using Literal = double;
  static constexpr bool IsComplex = true;
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

#endif
