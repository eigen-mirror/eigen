// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

// The const coeffRef() of a RealView, like the non-const one, exists for std::complex only.
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

double foo(const Matrix<SCALAR, Dynamic, 1> &m) {
  const auto view = m.realView();
  return view.coeffRef(1, 0);
}

int main() {}
