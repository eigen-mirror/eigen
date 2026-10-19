// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <contrib/Eigen/NumericalDiff>
#include <algorithm>
#include <cmath>

template <typename Scalar_, int Options>
struct CubicFunctor {
  using Scalar = Scalar_;
  using InputType = Eigen::Matrix<Scalar, 2, 1>;
  using ValueType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, 2, Options>;

  static constexpr int InputsAtCompileTime = 2;
  static constexpr int ValuesAtCompileTime = Eigen::Dynamic;

  Eigen::Index count;

  Eigen::Index values() const { return count; }

  int operator()(const InputType& x, ValueType& y) const {
    for (Eigen::Index i = 0; i < count; ++i) y[i] = Scalar(i + 1) * x[0] * x[0] * x[0] + x[1] * x[1] * x[1];
    return 0;
  }
};

template <typename Derivative>
static void measureJacobian(benchmark::State& state, Derivative& derivative, typename Derivative::Scalar tolerance) {
  using Functor = typename Derivative::Functor;
  using Scalar = typename Functor::Scalar;
  const Eigen::Index count = derivative.values();
  typename Functor::InputType x;
  x << Scalar(0.5), Scalar(-1);
  typename Functor::JacobianType jac(count, 2);
  const int evaluations = derivative.df(x, jac);
  long double relativeError = 0;
  for (Eigen::Index i = 0; i < count; ++i) {
    const Scalar expected = Scalar(i + 1) * Scalar(0.75);
    relativeError = (std::max)(relativeError, std::abs(static_cast<long double>(jac(i, 0)) / (0.75L * (i + 1)) - 1));
    relativeError = (std::max)(relativeError, std::abs(static_cast<long double>(jac(i, 1)) / 3 - 1));
    const Scalar bound = tolerance * Eigen::numext::maxi(expected, Scalar(3));
    if (!(Eigen::numext::abs(jac(i, 0) - expected) <= bound && Eigen::numext::abs(jac(i, 1) - Scalar(3)) <= bound)) {
      state.SkipWithError("Incorrect cubic Jacobian");
      return;
    }
  }
  state.counters["evaluations"] = evaluations;
  state.counters["max_relative_error"] = static_cast<double>(relativeError);

  for (auto _ : state) {
    benchmark::DoNotOptimize(derivative.df(x, jac));
    benchmark::DoNotOptimize(jac.data());
    benchmark::ClobberMemory();
  }
}

template <typename Scalar, int Options>
static void BM_Ridders(benchmark::State& state) {
  using Functor = CubicFunctor<Scalar, Options>;
  Eigen::Ridders<Functor> derivative(Functor{state.range(0)});
  typename Eigen::Ridders<Functor>::Parameters parameters;
  parameters.maxSteps = static_cast<int>(state.range(1));
  derivative.setParameters(parameters);
  measureJacobian(state, derivative, Scalar(512) * Eigen::NumTraits<Scalar>::epsilon());
}

template <typename Scalar, int Options>
static void BM_Central(benchmark::State& state) {
  using Functor = CubicFunctor<Scalar, Options>;
  Eigen::NumericalDiff<Functor, Eigen::Central> derivative(Functor{state.range(0)});
  measureJacobian(state, derivative, Scalar(16) * std::sqrt(Eigen::NumTraits<Scalar>::epsilon()));
}

BENCHMARK_TEMPLATE(BM_Ridders, float, Eigen::ColMajor)->ArgsProduct({{1, 16, 128, 1024}, {2, 10}});
BENCHMARK_TEMPLATE(BM_Ridders, double, Eigen::ColMajor)->ArgsProduct({{1, 16, 128, 1024}, {2, 10}});
BENCHMARK_TEMPLATE(BM_Ridders, double, Eigen::RowMajor)->ArgsProduct({{1, 16, 128, 1024}, {2, 10}});
BENCHMARK_TEMPLATE(BM_Central, float, Eigen::ColMajor)->Arg(1)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_Central, double, Eigen::ColMajor)->Arg(1)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_Central, double, Eigen::RowMajor)->Arg(1)->Arg(16)->Arg(128)->Arg(1024);
