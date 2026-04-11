// Benchmarks for Eigen TensorImagePatch extraction.

#define EIGEN_USE_THREADS

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

using namespace Eigen;

typedef float Scalar;

// --- Basic image patch extraction with PADDING_VALID ---
static void BM_ImagePatch_Valid(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int W = state.range(2);
  const int kH = state.range(3);
  const int kW = state.range(4);

  Tensor<Scalar, 4> input(C, H, W, 1);
  input.setRandom();
  const int outH = H - kH + 1;
  const int outW = W - kW + 1;
  Tensor<Scalar, 5> result(C, kH, kW, outH * outW, 1);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kW, 1, 1, 1, 1, PADDING_VALID);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * outH * outW * kH * kW * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- Basic image patch extraction with PADDING_SAME ---
static void BM_ImagePatch_Same(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int W = state.range(2);
  const int kH = state.range(3);
  const int kW = state.range(4);

  Tensor<Scalar, 4> input(C, H, W, 1);
  input.setRandom();
  const int outH = H;
  const int outW = W;
  Tensor<Scalar, 5> result(C, kH, kW, outH * outW, 1);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kW, 1, 1, 1, 1, PADDING_SAME);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * H * W * kH * kW * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- Image patch with strides (simulates strided convolution) ---
static void BM_ImagePatch_Strided(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int kH = state.range(2);
  const int stride = state.range(3);

  Tensor<Scalar, 4> input(C, H, H, 1);
  input.setRandom();
  const int outH = (H + stride - 1) / stride;
  Tensor<Scalar, 5> result(C, kH, kH, outH * outH, 1);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kH, stride, stride, 1, 1, PADDING_SAME);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * outH * outH * kH * kH * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- Image patch with dilation (atrous/dilated convolution) ---
static void BM_ImagePatch_Dilated(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int kH = state.range(2);
  const int dilation = state.range(3);

  Tensor<Scalar, 4> input(C, H, H, 1);
  input.setRandom();
  const int outH = H;
  Tensor<Scalar, 5> result(C, kH, kH, outH * outH, 1);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kH, 1, 1, dilation, dilation, PADDING_SAME);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * H * H * kH * kH * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- Image patch with explicit padding ---
static void BM_ImagePatch_ExplicitPadding(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int W = state.range(2);
  const int kH = state.range(3);

  const int pad = kH / 2;

  Tensor<Scalar, 4> input(C, H, W, 1);
  input.setRandom();
  const int outH = H + 2 * pad - kH + 1;
  const int outW = W + 2 * pad - kH + 1;
  Tensor<Scalar, 5> result(C, kH, kH, outH * outW, 1);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kH, 1, 1, 1, 1, 1, 1, pad, pad, pad, pad, Scalar(0));
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * outH * outW * kH * kH * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- Batched image patch (multiple images) ---
static void BM_ImagePatch_Batched(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int kH = state.range(2);
  const int batch = state.range(3);

  Tensor<Scalar, 4> input(C, H, H, batch);
  input.setRandom();
  const int outH = H;
  Tensor<Scalar, 5> result(C, kH, kH, outH * outH, batch);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kH, 1, 1, 1, 1, PADDING_SAME);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(batch) * C * H * H * kH * kH * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- ImageNet-style configurations (realistic CNN layer sizes) ---
static void BM_ImagePatch_ImageNet(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int kH = state.range(2);
  const int stride = state.range(3);

  Tensor<Scalar, 4> input(C, H, H, 1);
  input.setRandom();
  const int outH = (H + stride - 1) / stride;
  Tensor<Scalar, 5> result(C, kH, kH, outH * outH, 1);

  for (auto _ : state) {
    result = input.extract_image_patches(kH, kH, stride, stride, 1, 1, PADDING_SAME);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * outH * outH * kH * kH * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
}

// --- ThreadPool variant ---
static void BM_ImagePatch_ThreadPool(benchmark::State& state) {
  const int C = state.range(0);
  const int H = state.range(1);
  const int kH = state.range(2);
  const int threads = state.range(3);

  Tensor<Scalar, 4> input(C, H, H, 1);
  input.setRandom();

  ThreadPool tp(threads);
  ThreadPoolDevice dev(&tp, threads);

  const int outH = H;
  Tensor<Scalar, 5> result(C, kH, kH, outH * outH, 1);

  for (auto _ : state) {
    result.device(dev) = input.extract_image_patches(kH, kH, 1, 1, 1, 1, PADDING_SAME);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  const double bytes = static_cast<double>(C) * H * H * kH * kH * sizeof(Scalar);
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["threads"] = threads;
}

// --- Size generators ---

static void PatchSizes(::benchmark::internal::Benchmark* b) {
  // channels, H, W, kH, kW
  for (int c : {3, 32, 64}) {
    for (int hw : {32, 64, 128}) {
      for (int k : {3, 5, 7}) {
        b->Args({c, hw, hw, k, k});
      }
    }
  }
}

static void StridedSizes(::benchmark::internal::Benchmark* b) {
  // channels, H, kH, stride
  for (int c : {3, 64}) {
    for (int hw : {56, 112, 224}) {
      for (int k : {3, 5}) {
        for (int s : {1, 2}) {
          b->Args({c, hw, k, s});
        }
      }
    }
  }
}

static void DilatedSizes(::benchmark::internal::Benchmark* b) {
  // channels, H, kH, dilation
  for (int c : {3, 64}) {
    for (int hw : {32, 64}) {
      for (int k : {3, 5}) {
        for (int d : {2, 4}) {
          b->Args({c, hw, k, d});
        }
      }
    }
  }
}

static void ExplicitPaddingSizes(::benchmark::internal::Benchmark* b) {
  // channels, H, W, kH
  for (int c : {3, 64}) {
    for (int hw : {32, 64, 128}) {
      for (int k : {3, 5}) {
        b->Args({c, hw, hw, k});
      }
    }
  }
}

static void BatchedSizes(::benchmark::internal::Benchmark* b) {
  // channels, H, kH, batch
  for (int c : {3, 64}) {
    for (int hw : {32, 56}) {
      for (int k : {3, 5}) {
        for (int batch : {4, 16, 32}) {
          b->Args({c, hw, k, batch});
        }
      }
    }
  }
}

static void ImageNetSizes(::benchmark::internal::Benchmark* b) {
  // Realistic CNN layer configurations: channels, spatial_size, kernel, stride
  // AlexNet conv1: 3x227x227, 11x11, stride 4
  b->Args({3, 227, 11, 4});
  // VGG-style: 64x224x224, 3x3, stride 1
  b->Args({64, 224, 3, 1});
  // VGG deeper: 128x112x112, 3x3, stride 1
  b->Args({128, 112, 3, 1});
  // VGG deeper: 256x56x56, 3x3, stride 1
  b->Args({256, 56, 3, 1});
  // ResNet: 64x56x56, 3x3, stride 1
  b->Args({64, 56, 3, 1});
  // ResNet downsample: 128x56x56, 3x3, stride 2
  b->Args({128, 56, 3, 2});
  // ResNet: 256x28x28, 3x3, stride 1
  b->Args({256, 28, 3, 1});
  // ResNet: 512x14x14, 3x3, stride 1
  b->Args({512, 14, 3, 1});
  // MobileNet depthwise: 32x112x112, 3x3, stride 1
  b->Args({32, 112, 3, 1});
  // Inception 1x1 (degenerate patch): 192x28x28, 1x1, stride 1
  b->Args({192, 28, 1, 1});
}

static void ThreadPoolSizes(::benchmark::internal::Benchmark* b) {
  // channels, H, kH, threads
  for (int c : {64, 128}) {
    for (int hw : {56, 112}) {
      for (int k : {3, 5}) {
        for (int threads : {2, 4, 8}) {
          b->Args({c, hw, k, threads});
        }
      }
    }
  }
}

BENCHMARK(BM_ImagePatch_Valid)->Apply(PatchSizes);
BENCHMARK(BM_ImagePatch_Same)->Apply(PatchSizes);
BENCHMARK(BM_ImagePatch_Strided)->Apply(StridedSizes);
BENCHMARK(BM_ImagePatch_Dilated)->Apply(DilatedSizes);
BENCHMARK(BM_ImagePatch_ExplicitPadding)->Apply(ExplicitPaddingSizes);
BENCHMARK(BM_ImagePatch_Batched)->Apply(BatchedSizes);
BENCHMARK(BM_ImagePatch_ImageNet)->Apply(ImageNetSizes);
BENCHMARK(BM_ImagePatch_ThreadPool)->Apply(ThreadPoolSizes);
