// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#if defined(__linux__)
#include <sys/resource.h>
#endif

template <typename Scalar, bool RankUpdate>
static void BM_Packing(benchmark::State& state) {
  Eigen::setNbThreads(1);
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Real = typename Eigen::NumTraits<Scalar>::Real;
  const Eigen::Index n = state.range(0);
  const MatrixType a = MatrixType::Random(n, n);
  const MatrixType b = MatrixType::Random(n, n);
  MatrixType c = MatrixType::Zero(n, n);
  const auto product = [&]() {
    if (RankUpdate)
      c.template selfadjointView<Eigen::Lower>().rankUpdate(a);
    else
      c.noalias() += a * b;
  };
  product();
  // Validate sampled coefficients without a reference allocation changing malloc's history.
  for (Eigen::Index i : {Eigen::Index(0), n / 2, n - 1}) {
    for (Eigen::Index j : {Eigen::Index(0), n / 2, n - 1}) {
      long double expected = 0, scale = 0;
      if (!RankUpdate || i >= j) {
        for (Eigen::Index k = 0; k < n; ++k) {
          const long double term = static_cast<long double>(a(i, k)) * (RankUpdate ? a(j, k) : b(k, j));
          expected += term;
          scale += Eigen::numext::abs(term);
        }
      }
      const long double bound = 8 * n * Eigen::NumTraits<Real>::epsilon() * scale;
      if (!(Eigen::numext::abs(static_cast<long double>(c(i, j)) - expected) <= bound)) {
        state.SkipWithError("matrix product validation failed");
        return;
      }
    }
  }
  c.setZero();
#if defined(__linux__)
  rusage before, after;
  getrusage(RUSAGE_SELF, &before);
#endif
  for (auto _ : state) {
    product();
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
#if defined(__linux__)
  getrusage(RUSAGE_SELF, &after);
  state.counters["minor_faults_per_call"] = double(after.ru_minflt - before.ru_minflt) / state.iterations();
#endif
  const double flops = RankUpdate ? double(n) * (n + 1) * n : 2.0 * n * n * n;
  state.counters["GFLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// clang-format off
#define PACKING_SIZES ->Arg(128)->Arg(144)->Arg(160)->Arg(176)->Arg(184)->Arg(192)->Arg(200)->Arg(240) \
                      ->Arg(256)->Arg(384)->Arg(448)->Arg(512)->Arg(768)
BENCHMARK(BM_Packing<float, false>)->Name("GEMM_float") PACKING_SIZES;
BENCHMARK(BM_Packing<double, false>)->Name("GEMM_double") PACKING_SIZES;
BENCHMARK(BM_Packing<float, true>)->Name("SYRK_float") PACKING_SIZES;
BENCHMARK(BM_Packing<double, true>)->Name("SYRK_double") PACKING_SIZES;
#undef PACKING_SIZES
// clang-format on
