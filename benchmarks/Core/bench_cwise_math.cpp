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

BENCH_CWISE_UNARY(Log2, a.log2(), 0.01, 100)
BENCH_CWISE_UNARY(Exp2, a.exp2(), -10, 10)
BENCH_CWISE_UNARY(Expm1, a.expm1(), -2, 2)
BENCH_CWISE_UNARY(Cbrt, a.cbrt(), -100, 100)

// Trigonometric functions
BENCH_CWISE_UNARY(Sin, a.sin(), -3.14, 3.14)
BENCH_CWISE_UNARY(Cos, a.cos(), -3.14, 3.14)
BENCH_CWISE_UNARY(Tan, a.tan(), -1.5, 1.5)
BENCH_CWISE_UNARY(Asin, a.asin(), -0.99, 0.99)
BENCH_CWISE_UNARY(Acos, a.acos(), -0.99, 0.99)
BENCH_CWISE_UNARY(Atan, a.atan(), -10, 10)

// Hyperbolic / special
BENCH_CWISE_UNARY(Tanh, a.tanh(), -5, 5)
BENCH_CWISE_UNARY(Atanh, a.atanh(), -0.99, 0.99)
BENCH_CWISE_UNARY(Erf, Eigen::erf(a), -4, 4)

// Simple operations (should be very fast / memory-bound)
BENCH_CWISE_UNARY(Abs, a.abs(), -100, 100)
BENCH_CWISE_UNARY(Square, a.square(), -100, 100)
BENCH_CWISE_UNARY(Cube, a.cube(), -10, 10)
BENCH_CWISE_UNARY(Ceil, a.ceil(), -100, 100)
BENCH_CWISE_UNARY(Floor, a.floor(), -100, 100)
BENCH_CWISE_UNARY(Round, a.round(), -100, 100)
BENCH_CWISE_UNARY(Rint, a.rint(), -100, 100)
BENCH_CWISE_UNARY(Trunc, a.trunc(), -100, 100)

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

// Macro for complex unary benchmarks. Random() already produces complex
// values with real & imag in [-1,1]; scale both parts to [LO, HI].
#define BENCH_CWISE_UNARY_COMPLEX(NAME, EXPR, LO, HI)                                \
  template <typename RealScalar>                                                     \
  static void BM_##NAME##_complex(benchmark::State& state) {                         \
    using Scalar = std::complex<RealScalar>;                                         \
    const Index n = state.range(0);                                                  \
    using Arr = Array<Scalar, Dynamic, 1>;                                           \
    Arr a = (Arr::Random(n) + Scalar(RealScalar(1), RealScalar(1))) *                \
                Scalar(RealScalar((double(HI) - double(LO)) / 2.0), RealScalar(0)) + \
            Scalar(RealScalar(LO), RealScalar(LO));                                  \
    Arr b(n);                                                                        \
    for (auto _ : state) {                                                           \
      b = EXPR;                                                                      \
      benchmark::DoNotOptimize(b.data());                                            \
    }                                                                                \
    state.SetBytesProcessed(state.iterations() * n * Index(sizeof(Scalar)) * 2);     \
  }

// Macro for complex binary benchmarks (e.g. multiply, divide).
#define BENCH_CWISE_BINARY_COMPLEX(NAME, EXPR, LO, HI)                               \
  template <typename RealScalar>                                                     \
  static void BM_##NAME##_complex(benchmark::State& state) {                         \
    using Scalar = std::complex<RealScalar>;                                         \
    const Index n = state.range(0);                                                  \
    using Arr = Array<Scalar, Dynamic, 1>;                                           \
    Arr a = (Arr::Random(n) + Scalar(RealScalar(1), RealScalar(1))) *                \
                Scalar(RealScalar((double(HI) - double(LO)) / 2.0), RealScalar(0)) + \
            Scalar(RealScalar(LO), RealScalar(LO));                                  \
    Arr b = (Arr::Random(n) + Scalar(RealScalar(1), RealScalar(1))) *                \
                Scalar(RealScalar((double(HI) - double(LO)) / 2.0), RealScalar(0)) + \
            Scalar(RealScalar(LO), RealScalar(LO));                                  \
    Arr c(n);                                                                        \
    for (auto _ : state) {                                                           \
      c = EXPR;                                                                      \
      benchmark::DoNotOptimize(c.data());                                            \
    }                                                                                \
    state.SetBytesProcessed(state.iterations() * n * Index(sizeof(Scalar)) * 3);     \
  }

// Complex unary (SIMD implementations in GenericPacketMathFunctions.h)
BENCH_CWISE_UNARY_COMPLEX(Exp, a.exp(), -5, 5)
BENCH_CWISE_UNARY_COMPLEX(Log, a.log(), 0.01, 100)
BENCH_CWISE_UNARY_COMPLEX(Sqrt, a.sqrt(), -100, 100)
BENCH_CWISE_UNARY_COMPLEX(Square, a.square(), -10, 10)

// Complex binary (pdiv_complex, pmul_complex)
BENCH_CWISE_BINARY_COMPLEX(Mul, a* b, -10, 10)
BENCH_CWISE_BINARY_COMPLEX(Div, a / b, -10, 10)

static void CwiseSizes(::benchmark::Benchmark* b) {
  for (int n : {1024, 4096, 16384, 65536, 262144, 1048576}) b->Arg(n);
}

// --- Register float ---
BENCHMARK(BM_Exp<float>)->Apply(CwiseSizes)->Name("Exp_float");
BENCHMARK(BM_Log<float>)->Apply(CwiseSizes)->Name("Log_float");
BENCHMARK(BM_Log1p<float>)->Apply(CwiseSizes)->Name("Log1p_float");
BENCHMARK(BM_Log2<float>)->Apply(CwiseSizes)->Name("Log2_float");
BENCHMARK(BM_Sqrt<float>)->Apply(CwiseSizes)->Name("Sqrt_float");
BENCHMARK(BM_Rsqrt<float>)->Apply(CwiseSizes)->Name("Rsqrt_float");
BENCHMARK(BM_Exp2<float>)->Apply(CwiseSizes)->Name("Exp2_float");
BENCHMARK(BM_Expm1<float>)->Apply(CwiseSizes)->Name("Expm1_float");
BENCHMARK(BM_Cbrt<float>)->Apply(CwiseSizes)->Name("Cbrt_float");
BENCHMARK(BM_Sin<float>)->Apply(CwiseSizes)->Name("Sin_float");
BENCHMARK(BM_Cos<float>)->Apply(CwiseSizes)->Name("Cos_float");
BENCHMARK(BM_Tan<float>)->Apply(CwiseSizes)->Name("Tan_float");
BENCHMARK(BM_Asin<float>)->Apply(CwiseSizes)->Name("Asin_float");
BENCHMARK(BM_Acos<float>)->Apply(CwiseSizes)->Name("Acos_float");
BENCHMARK(BM_Atan<float>)->Apply(CwiseSizes)->Name("Atan_float");
BENCHMARK(BM_Tanh<float>)->Apply(CwiseSizes)->Name("Tanh_float");
BENCHMARK(BM_Atanh<float>)->Apply(CwiseSizes)->Name("Atanh_float");
BENCHMARK(BM_Erf<float>)->Apply(CwiseSizes)->Name("Erf_float");
BENCHMARK(BM_Abs<float>)->Apply(CwiseSizes)->Name("Abs_float");
BENCHMARK(BM_Square<float>)->Apply(CwiseSizes)->Name("Square_float");
BENCHMARK(BM_Cube<float>)->Apply(CwiseSizes)->Name("Cube_float");
BENCHMARK(BM_Ceil<float>)->Apply(CwiseSizes)->Name("Ceil_float");
BENCHMARK(BM_Floor<float>)->Apply(CwiseSizes)->Name("Floor_float");
BENCHMARK(BM_Round<float>)->Apply(CwiseSizes)->Name("Round_float");
BENCHMARK(BM_Rint<float>)->Apply(CwiseSizes)->Name("Rint_float");
BENCHMARK(BM_Trunc<float>)->Apply(CwiseSizes)->Name("Trunc_float");
BENCHMARK(BM_Sigmoid<float>)->Apply(CwiseSizes)->Name("Sigmoid_float");
BENCHMARK(BM_Pow<float>)->Apply(CwiseSizes)->Name("Pow_float");

// --- Register double ---
BENCHMARK(BM_Exp<double>)->Apply(CwiseSizes)->Name("Exp_double");
BENCHMARK(BM_Log<double>)->Apply(CwiseSizes)->Name("Log_double");
BENCHMARK(BM_Log1p<double>)->Apply(CwiseSizes)->Name("Log1p_double");
BENCHMARK(BM_Log2<double>)->Apply(CwiseSizes)->Name("Log2_double");
BENCHMARK(BM_Sqrt<double>)->Apply(CwiseSizes)->Name("Sqrt_double");
BENCHMARK(BM_Rsqrt<double>)->Apply(CwiseSizes)->Name("Rsqrt_double");
BENCHMARK(BM_Exp2<double>)->Apply(CwiseSizes)->Name("Exp2_double");
BENCHMARK(BM_Expm1<double>)->Apply(CwiseSizes)->Name("Expm1_double");
BENCHMARK(BM_Cbrt<double>)->Apply(CwiseSizes)->Name("Cbrt_double");
BENCHMARK(BM_Sin<double>)->Apply(CwiseSizes)->Name("Sin_double");
BENCHMARK(BM_Cos<double>)->Apply(CwiseSizes)->Name("Cos_double");
BENCHMARK(BM_Tan<double>)->Apply(CwiseSizes)->Name("Tan_double");
BENCHMARK(BM_Asin<double>)->Apply(CwiseSizes)->Name("Asin_double");
BENCHMARK(BM_Acos<double>)->Apply(CwiseSizes)->Name("Acos_double");
BENCHMARK(BM_Atan<double>)->Apply(CwiseSizes)->Name("Atan_double");
BENCHMARK(BM_Tanh<double>)->Apply(CwiseSizes)->Name("Tanh_double");
BENCHMARK(BM_Atanh<double>)->Apply(CwiseSizes)->Name("Atanh_double");
BENCHMARK(BM_Erf<double>)->Apply(CwiseSizes)->Name("Erf_double");
BENCHMARK(BM_Abs<double>)->Apply(CwiseSizes)->Name("Abs_double");
BENCHMARK(BM_Square<double>)->Apply(CwiseSizes)->Name("Square_double");
BENCHMARK(BM_Cube<double>)->Apply(CwiseSizes)->Name("Cube_double");
BENCHMARK(BM_Ceil<double>)->Apply(CwiseSizes)->Name("Ceil_double");
BENCHMARK(BM_Floor<double>)->Apply(CwiseSizes)->Name("Floor_double");
BENCHMARK(BM_Round<double>)->Apply(CwiseSizes)->Name("Round_double");
BENCHMARK(BM_Rint<double>)->Apply(CwiseSizes)->Name("Rint_double");
BENCHMARK(BM_Trunc<double>)->Apply(CwiseSizes)->Name("Trunc_double");
BENCHMARK(BM_Sigmoid<double>)->Apply(CwiseSizes)->Name("Sigmoid_double");
BENCHMARK(BM_Pow<double>)->Apply(CwiseSizes)->Name("Pow_double");

// --- Register complex<float> ---
BENCHMARK(BM_Exp_complex<float>)->Apply(CwiseSizes)->Name("Exp_complexf");
BENCHMARK(BM_Log_complex<float>)->Apply(CwiseSizes)->Name("Log_complexf");
BENCHMARK(BM_Sqrt_complex<float>)->Apply(CwiseSizes)->Name("Sqrt_complexf");
BENCHMARK(BM_Square_complex<float>)->Apply(CwiseSizes)->Name("Square_complexf");
BENCHMARK(BM_Mul_complex<float>)->Apply(CwiseSizes)->Name("Mul_complexf");
BENCHMARK(BM_Div_complex<float>)->Apply(CwiseSizes)->Name("Div_complexf");

// --- Register complex<double> ---
BENCHMARK(BM_Exp_complex<double>)->Apply(CwiseSizes)->Name("Exp_complexd");
BENCHMARK(BM_Log_complex<double>)->Apply(CwiseSizes)->Name("Log_complexd");
BENCHMARK(BM_Sqrt_complex<double>)->Apply(CwiseSizes)->Name("Sqrt_complexd");
BENCHMARK(BM_Square_complex<double>)->Apply(CwiseSizes)->Name("Square_complexd");
BENCHMARK(BM_Mul_complex<double>)->Apply(CwiseSizes)->Name("Mul_complexd");
BENCHMARK(BM_Div_complex<double>)->Apply(CwiseSizes)->Name("Div_complexd");
