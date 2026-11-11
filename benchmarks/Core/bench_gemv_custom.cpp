// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include "../../test/NonCommutativeScalar.h"

namespace unannotated_gemv {
struct Scalar {
  double value;
  Scalar(double x = 0) : value(x) {}
  Scalar operator+(Scalar b) const { return value + b.value; }
  Scalar operator-(Scalar b) const { return value - b.value; }
  Scalar operator-() const { return -value; }
  Scalar operator*(Scalar b) const { return value * b.value; }
  Scalar& operator+=(Scalar b) {
    value += b.value;
    return *this;
  }
  Scalar& operator-=(Scalar b) {
    value -= b.value;
    return *this;
  }
  Scalar& operator*=(Scalar b) {
    value *= b.value;
    return *this;
  }
  bool operator==(Scalar b) const { return value == b.value; }
  bool operator!=(Scalar b) const { return value != b.value; }
};
}  // namespace unannotated_gemv

namespace unannotated_gemv {
struct Jet4 {
  double value;
  double gradient[4];
  Jet4(double x = 0) : value(x), gradient{} {}
  Jet4 operator+(const Jet4& b) const {
    Jet4 r(value + b.value);
    for (int i = 0; i < 4; ++i) r.gradient[i] = gradient[i] + b.gradient[i];
    return r;
  }
  Jet4 operator-(const Jet4& b) const {
    Jet4 r(value - b.value);
    for (int i = 0; i < 4; ++i) r.gradient[i] = gradient[i] - b.gradient[i];
    return r;
  }
  Jet4 operator-() const { return Jet4(0) - *this; }
  Jet4 operator*(const Jet4& b) const {
    Jet4 r(value * b.value);
    for (int i = 0; i < 4; ++i) r.gradient[i] = value * b.gradient[i] + gradient[i] * b.value;
    return r;
  }
  Jet4& operator+=(const Jet4& b) { return *this = *this + b; }
  Jet4& operator-=(const Jet4& b) { return *this = *this - b; }
  Jet4& operator*=(const Jet4& b) { return *this = *this * b; }
  bool operator==(const Jet4& b) const {
    if (value != b.value) return false;
    for (int i = 0; i < 4; ++i)
      if (gradient[i] != b.gradient[i]) return false;
    return true;
  }
  bool operator!=(const Jet4& b) const { return !(*this == b); }
};

using Components = std::array<long double, 5>;
inline Components components(const Scalar& x) { return {{x.value, 0, 0, 0, 0}}; }
inline Components components(const Jet4& x) {
  return {{x.value, x.gradient[0], x.gradient[1], x.gradient[2], x.gradient[3]}};
}
inline Components components(const noncommutative_scalar::Quaternion& x) { return {{x.r, x.i, x.j, x.k, 0}}; }
template <typename T>
T sample(double x) {
  return T(x);
}
template <>
inline Jet4 sample<Jet4>(double x) {
  Jet4 r(x);
  for (int i = 0; i < 4; ++i) r.gradient[i] = (i % 2 == 0 ? 1 : -1) * (i + 1) * x;
  return r;
}
template <>
inline noncommutative_scalar::Quaternion sample<noncommutative_scalar::Quaternion>(double x) {
  return {x, x * 0.25, -x * 0.5, x * 0.75};
}
}  // namespace unannotated_gemv

namespace Eigen {
template <>
struct NumTraits<unannotated_gemv::Scalar> : GenericNumTraits<unannotated_gemv::Scalar> {
  static constexpr bool RequireInitialization = true;
};
template <>
struct NumTraits<unannotated_gemv::Jet4> : GenericNumTraits<unannotated_gemv::Jet4> {
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

template <int Order, int Kind = 0, bool Left = false>
static void BM_CustomGemv(benchmark::State& state) {
  using Scalar =
      std::conditional_t<Kind == 0, unannotated_gemv::Scalar,
                         std::conditional_t<Kind == 1, unannotated_gemv::Jet4, noncommutative_scalar::Quaternion>>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Order>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Result = Eigen::Matrix<Scalar, Left ? 1 : Eigen::Dynamic, Left ? Eigen::Dynamic : 1,
                               Left ? Eigen::RowMajor : Eigen::ColMajor>;
  const Eigen::Index n = state.range(0);
  const bool scaled = state.range(1) != 0;
  Matrix a(n, n);
  Vector x(n);
  Result result(Left ? 1 : n, Left ? n : 1);
  for (Eigen::Index j = 0; j < n; ++j) {
    x(j) = unannotated_gemv::sample<Scalar>((j % 7 + 1) * 0.001);
    for (Eigen::Index i = 0; i < n; ++i) a(i, j) = unannotated_gemv::sample<Scalar>((i % 11 + j % 13 + 1) * 0.001);
  }
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(x.data());
  for (auto _ : state) {
    if (Left) {
      if (scaled)
        result.noalias() = Scalar(0.75) * (x.transpose() * a);
      else
        result.noalias() = x.transpose() * a;
    } else {
      if (scaled)
        result.noalias() = Scalar(0.75) * (a * x);
      else
        result.noalias() = a * x;
    }
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  // Full-result correctness is covered by the product tests; sample boundary and middle outputs here.
  for (Eigen::Index i : {Eigen::Index(0), n / 2, n - 1}) {
    unannotated_gemv::Components expected{}, magnitude{};
    for (Eigen::Index j = 0; j < n; ++j) {
      const auto av = unannotated_gemv::components(Left ? x(j) : a(i, j));
      const auto bv = unannotated_gemv::components(Left ? a(j, i) : x(j));
      unannotated_gemv::Components term{};
      if (Kind == 2) {
        term = {{av[0] * bv[0] - av[1] * bv[1] - av[2] * bv[2] - av[3] * bv[3],
                 av[0] * bv[1] + av[1] * bv[0] + av[2] * bv[3] - av[3] * bv[2],
                 av[0] * bv[2] - av[1] * bv[3] + av[2] * bv[0] + av[3] * bv[1],
                 av[0] * bv[3] + av[1] * bv[2] - av[2] * bv[1] + av[3] * bv[0], 0}};
      } else {
        term[0] = av[0] * bv[0];
        if (Kind == 1)
          for (int k = 1; k < 5; ++k) term[k] = av[0] * bv[k] + av[k] * bv[0];
      }
      for (int k = 0; k < 5; ++k) {
        expected[k] += term[k];
        magnitude[k] += std::abs(term[k]);
      }
    }
    const auto actual = unannotated_gemv::components(result(i));
    for (int k = 0; k < 5; ++k) {
      const long double scale = scaled ? 0.75L : 1.0L;
      const long double bound = 64 * n * Eigen::NumTraits<double>::epsilon() * (1 + scale * magnitude[k]);
      if (!(std::abs(actual[k] - scale * expected[k]) <= bound)) {
        state.SkipWithError("Incorrect custom-scalar GEMV result");
        return;
      }
    }
  }
}

#define EIGEN_BENCH_CUSTOM_GEMV(Order, Kind, Left) \
  BENCHMARK_TEMPLATE(BM_CustomGemv, Order, Kind, Left)->ArgsProduct({{16, 17, 64, 256, 2048}, {0, 1}})
EIGEN_BENCH_CUSTOM_GEMV(Eigen::ColMajor, 0, false);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::RowMajor, 0, false);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::ColMajor, 0, true);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::RowMajor, 0, true);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::ColMajor, 1, false);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::RowMajor, 1, false);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::ColMajor, 1, true);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::RowMajor, 1, true);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::ColMajor, 2, false);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::RowMajor, 2, false);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::ColMajor, 2, true);
EIGEN_BENCH_CUSTOM_GEMV(Eigen::RowMajor, 2, true);
#undef EIGEN_BENCH_CUSTOM_GEMV
