// Benchmarks for vectorized coefficient-wise math functions.
//
// Each function is benchmarked on ArrayXf/ArrayXd with inputs chosen to
// stay in the valid domain and avoid NaN/Inf.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <unsupported/Eigen/SpecialFunctions>

using namespace Eigen;

// Macro to define a benchmark for a unary array operation.
// NAME:     benchmark function suffix (e.g. Exp)
// EXPR:     expression applied to the array (e.g. a.exp())
// LO, HI:  input range [LO, HI] mapped from the default Random() range [-1,1]
#define BENCH_CWISE_UNARY(NAME, EXPR, LO, HI)                                                    \
  template <typename Scalar>                                                                     \
  static void BM_##NAME(benchmark::State& state) {                                               \
    const Index n = state.range(0);                                                              \
    using Arr = Array<Scalar, Dynamic, 1>;                                                       \
    /* Map Random [-1,1] to [LO, HI] */                                                          \
    Arr a = (Arr::Random(n) + Scalar(1)) * Scalar((double(HI) - double(LO)) / 2.0) + Scalar(LO); \
    Arr b(n);                                                                                    \
    for (auto _ : state) {                                                                       \
      b = EXPR;                                                                                  \
      benchmark::DoNotOptimize(b.data());                                                        \
    }                                                                                            \
    state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);                        \
  }

// Transcendental functions
BENCH_CWISE_UNARY(Exp, a.exp(), -10, 10)
BENCH_CWISE_UNARY(Log, a.log(), 0.01, 100)
BENCH_CWISE_UNARY(Log1p, a.log1p(), -0.5, 100)
BENCH_CWISE_UNARY(Sqrt, a.sqrt(), 0, 100)
BENCH_CWISE_UNARY(Rsqrt, a.rsqrt(), 0.01, 100)

// Trigonometric functions
BENCH_CWISE_UNARY(Sin, a.sin(), -3.14, 3.14)
BENCH_CWISE_UNARY(Cos, a.cos(), -3.14, 3.14)
BENCH_CWISE_UNARY(Tan, a.tan(), -1.5, 1.5)
BENCH_CWISE_UNARY(Asin, a.asin(), -0.99, 0.99)
BENCH_CWISE_UNARY(Atan, a.atan(), -10, 10)

// Hyperbolic / special
BENCH_CWISE_UNARY(Tanh, a.tanh(), -5, 5)
BENCH_CWISE_UNARY(Erf, Eigen::erf(a), -4, 4)

// Simple operations (should be very fast / memory-bound)
BENCH_CWISE_UNARY(Abs, a.abs(), -100, 100)
BENCH_CWISE_UNARY(Square, a.square(), -100, 100)
BENCH_CWISE_UNARY(Cube, a.cube(), -10, 10)
BENCH_CWISE_UNARY(Ceil, a.ceil(), -100, 100)
BENCH_CWISE_UNARY(Floor, a.floor(), -100, 100)
BENCH_CWISE_UNARY(Round, a.round(), -100, 100)

// Sigmoid: 1 / (1 + exp(-x)), common in ML.
BENCH_CWISE_UNARY(Sigmoid, Scalar(1) / (Scalar(1) + (-a).exp()), -10, 10)

// Power: array^scalar
template <typename Scalar>
static void BM_Pow(benchmark::State& state) {
  const Index n = state.range(0);
  using Arr = Array<Scalar, Dynamic, 1>;
  Arr a = (Arr::Random(n) + Scalar(1)) * Scalar(50);  // [0, 100]
  Arr b(n);
  for (auto _ : state) {
    b = a.pow(Scalar(2.5));
    benchmark::DoNotOptimize(b.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar) * 2);
}

static void CwiseSizes(::benchmark::Benchmark* b) {
  for (int n : {1024, 4096, 16384, 65536, 262144, 1048576}) b->Arg(n);
}

// --- Register float ---
BENCHMARK(BM_Exp<float>)->Apply(CwiseSizes)->Name("Exp_float");
BENCHMARK(BM_Log<float>)->Apply(CwiseSizes)->Name("Log_float");
BENCHMARK(BM_Log1p<float>)->Apply(CwiseSizes)->Name("Log1p_float");
BENCHMARK(BM_Sqrt<float>)->Apply(CwiseSizes)->Name("Sqrt_float");
BENCHMARK(BM_Rsqrt<float>)->Apply(CwiseSizes)->Name("Rsqrt_float");
BENCHMARK(BM_Sin<float>)->Apply(CwiseSizes)->Name("Sin_float");
BENCHMARK(BM_Cos<float>)->Apply(CwiseSizes)->Name("Cos_float");
BENCHMARK(BM_Tan<float>)->Apply(CwiseSizes)->Name("Tan_float");
BENCHMARK(BM_Asin<float>)->Apply(CwiseSizes)->Name("Asin_float");
BENCHMARK(BM_Atan<float>)->Apply(CwiseSizes)->Name("Atan_float");
BENCHMARK(BM_Tanh<float>)->Apply(CwiseSizes)->Name("Tanh_float");
BENCHMARK(BM_Erf<float>)->Apply(CwiseSizes)->Name("Erf_float");
BENCHMARK(BM_Abs<float>)->Apply(CwiseSizes)->Name("Abs_float");
BENCHMARK(BM_Square<float>)->Apply(CwiseSizes)->Name("Square_float");
BENCHMARK(BM_Cube<float>)->Apply(CwiseSizes)->Name("Cube_float");
BENCHMARK(BM_Ceil<float>)->Apply(CwiseSizes)->Name("Ceil_float");
BENCHMARK(BM_Floor<float>)->Apply(CwiseSizes)->Name("Floor_float");
BENCHMARK(BM_Round<float>)->Apply(CwiseSizes)->Name("Round_float");
BENCHMARK(BM_Sigmoid<float>)->Apply(CwiseSizes)->Name("Sigmoid_float");
BENCHMARK(BM_Pow<float>)->Apply(CwiseSizes)->Name("Pow_float");

// --- Register double ---
BENCHMARK(BM_Exp<double>)->Apply(CwiseSizes)->Name("Exp_double");
BENCHMARK(BM_Log<double>)->Apply(CwiseSizes)->Name("Log_double");
BENCHMARK(BM_Log1p<double>)->Apply(CwiseSizes)->Name("Log1p_double");
BENCHMARK(BM_Sqrt<double>)->Apply(CwiseSizes)->Name("Sqrt_double");
BENCHMARK(BM_Rsqrt<double>)->Apply(CwiseSizes)->Name("Rsqrt_double");
BENCHMARK(BM_Sin<double>)->Apply(CwiseSizes)->Name("Sin_double");
BENCHMARK(BM_Cos<double>)->Apply(CwiseSizes)->Name("Cos_double");
BENCHMARK(BM_Tan<double>)->Apply(CwiseSizes)->Name("Tan_double");
BENCHMARK(BM_Asin<double>)->Apply(CwiseSizes)->Name("Asin_double");
BENCHMARK(BM_Atan<double>)->Apply(CwiseSizes)->Name("Atan_double");
BENCHMARK(BM_Tanh<double>)->Apply(CwiseSizes)->Name("Tanh_double");
BENCHMARK(BM_Erf<double>)->Apply(CwiseSizes)->Name("Erf_double");
BENCHMARK(BM_Abs<double>)->Apply(CwiseSizes)->Name("Abs_double");
BENCHMARK(BM_Square<double>)->Apply(CwiseSizes)->Name("Square_double");
BENCHMARK(BM_Cube<double>)->Apply(CwiseSizes)->Name("Cube_double");
BENCHMARK(BM_Ceil<double>)->Apply(CwiseSizes)->Name("Ceil_double");
BENCHMARK(BM_Floor<double>)->Apply(CwiseSizes)->Name("Floor_double");
BENCHMARK(BM_Round<double>)->Apply(CwiseSizes)->Name("Round_double");
BENCHMARK(BM_Sigmoid<double>)->Apply(CwiseSizes)->Name("Sigmoid_double");
BENCHMARK(BM_Pow<double>)->Apply(CwiseSizes)->Name("Pow_double");
