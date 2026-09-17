// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <bool Imaginary, int NaNPropagation, typename Vector>
EIGEN_DONT_INLINE typename Vector::RealScalar component_abs_max(Vector& values) {
  if (Imaginary) return values.imag().cwiseAbs().template maxCoeff<NaNPropagation>();
  return values.real().cwiseAbs().template maxCoeff<NaNPropagation>();
}

template <typename Real, bool Imaginary, bool Const, int NaNPropagation>
static void BM_ComponentAbsMax(benchmark::State& state) {
  using Complex = std::complex<Real>;
  using Vector = Matrix<Complex, Dynamic, 1>;
  Vector values = Vector::Random(state.range(0));
  std::conditional_t<Const, const Vector, Vector>& input = values;
  Real expected = 0;
  for (Index i = 0; i < values.size(); ++i) {
    const Real value = numext::abs(Imaginary ? values(i).imag() : values(i).real());
    if (value > expected) expected = value;
  }
  if (component_abs_max<Imaginary, NaNPropagation>(input) != expected) {
    state.SkipWithError("Incorrect component maximum");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(component_abs_max<Imaginary, NaNPropagation>(input));
  }
  state.SetBytesProcessed(state.iterations() * values.size() * sizeof(Complex));
}

// Small packet boundaries, the issue's cache-resident workload, and larger working sets.
#define COMPONENT_SIZES ->Arg(3)->Arg(8)->Arg(9)->Arg(64)->Arg(4096)->Arg(65536)
#define REGISTER_COMPONENT(Real, Imaginary, Const, Policy) \
  BENCHMARK_TEMPLATE(BM_ComponentAbsMax, Real, Imaginary, Const, Policy) COMPONENT_SIZES
#define REGISTER_TYPE(Real)                              \
  REGISTER_COMPONENT(Real, false, false, PropagateFast); \
  REGISTER_COMPONENT(Real, false, false, PropagateNaN);  \
  REGISTER_COMPONENT(Real, false, true, PropagateNaN);   \
  REGISTER_COMPONENT(Real, true, false, PropagateNaN);   \
  REGISTER_COMPONENT(Real, true, true, PropagateNaN)
REGISTER_TYPE(float);
REGISTER_TYPE(double);
#undef REGISTER_TYPE
#undef REGISTER_COMPONENT
#undef COMPONENT_SIZES
