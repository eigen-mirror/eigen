// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

// The coeffRef() of a RealView reinterprets the complex scalar as two adjacent reals, which holds for std::complex
// only. A complex scalar with another layout has no coeffRef().
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
struct ReverseComplex {
  double imaginary, real_part;
  ReverseComplex(double r = 0, double i = 0) : imaginary(i), real_part(r) {}
};
double real(const ReverseComplex &x) { return x.real_part; }
double imag(const ReverseComplex &x) { return x.imaginary; }
namespace Eigen {
template <>
struct NumTraits<ReverseComplex> : NumTraits<double> {
  static constexpr bool IsComplex = true;
};
}  // namespace Eigen
#define SCALAR ReverseComplex
#else
#include <complex>
#define SCALAR std::complex<double>
#endif

using namespace Eigen;

double &foo(Matrix<SCALAR, Dynamic, 1> &m) { return m.realView().coeffRef(1, 0); }

int main() {}
