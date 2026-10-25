// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_GPU
#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#include "main.h"
#include <contrib/Eigen/Tensor>

template <int Layout, int NumDims, typename TensorIndex>
void verify_gpu_convolution(const Tensor<float, NumDims, Layout, TensorIndex>& input,
                            const Tensor<float, 3, Layout, TensorIndex>& kernel, const array<TensorIndex, 3>& axes,
                            const Tensor<float, NumDims, Layout, TensorIndex>& expected) {
  GpuStreamDevice stream;
  GpuDevice device(&stream);
  bool nonzero = false, varying = false;
  for (Index i = 0; i < expected.size(); ++i) {
    nonzero |= expected.data()[i] != 0;
    varying |= expected.data()[i] != expected.data()[0];
  }
  VERIFY(nonzero);
  VERIFY(varying);
  Tensor<float, NumDims, Layout, TensorIndex> result;
  result.resize(expected.dimensions());
  float* a = static_cast<float*>(device.allocate(input.size() * sizeof(float)));
  float* k = static_cast<float*>(device.allocate(kernel.size() * sizeof(float)));
  float* out = static_cast<float*>(device.allocate(result.size() * sizeof(float)));
  device.memcpyHostToDevice(a, input.data(), input.size() * sizeof(float));
  device.memcpyHostToDevice(k, kernel.data(), kernel.size() * sizeof(float));
  // NaN sentinels make missing output writes fail even when the reference is zero.
  device.memset(out, 0xff, result.size() * sizeof(float));
  TensorMap<Tensor<float, NumDims, Layout, TensorIndex>> gpu_input(a, input.dimensions());
  TensorMap<Tensor<float, 3, Layout, TensorIndex>> gpu_kernel(k, kernel.dimensions());
  TensorMap<Tensor<float, NumDims, Layout, TensorIndex>> gpu_result(out, result.dimensions());
  gpu_result.device(device) = gpu_input.convolve(gpu_kernel, axes);
  device.memcpyDeviceToHost(result.data(), out, result.size() * sizeof(float));
  device.synchronize();
  for (Index i = 0; i < result.size(); ++i) VERIFY_IS_EQUAL(result.data()[i], expected.data()[i]);
  device.deallocate(a);
  device.deallocate(k);
  device.deallocate(out);
}

template <int Layout>
void check_large_convolution(const array<Index, 3>& kernel_dims, const array<Index, 3>& axes) {
  array<Index, 5> input_dims{{2, 3, 2, 3, 2}};
  for (int i = 0; i < 3; ++i) input_dims[axes[i]] = kernel_dims[i] + i + 1;
  Tensor<float, 5, Layout> input(input_dims);
  Tensor<float, 3, Layout> kernel(kernel_dims);
  // Small integers keep every product and partial sum exactly representable in float.
  for (Index i = 0; i < input.size(); ++i) input.data()[i] = float((i * 17 + i / 7) % 19 - 9);
  for (Index i = 0; i < kernel.size(); ++i) kernel.data()[i] = float((i * 7 + i / 5) % 7 - 2);
  Tensor<float, 5, Layout> expected = input.convolve(kernel, axes);
  verify_gpu_convolution(input, kernel, axes, expected);
}

template <int Layout, typename TensorIndex = Index>
void check_convolution_grid_stride(Index output_size = 0) {
  GpuStreamDevice stream;
  GpuDevice device(&stream);
  const Index kernel_size = device.sharedMemPerBlock() / sizeof(float) + 1;
  const Index threads = numext::mini(256, device.maxGpuThreadsPerBlock());
  const Index grid_stride = Index(device.getNumGpuMultiProcessors()) * 32 * threads;
  if (output_size == 0) output_size = 2 * grid_stride + 17;
  const Index max_input =
      static_cast<Index>(numext::mini<int64_t>(NumTraits<Index>::highest(), NumTraits<TensorIndex>::highest()));
  const Index max_output = max_input - kernel_size + 1;
  // Some devices cannot fit an oversized convolution's input in this narrow index type.
  if (max_output < 2) return;
  output_size = numext::mini(output_size, max_output);
  const array<TensorIndex, 3> axes{{0, 1, 2}};
  Tensor<float, 3, Layout, TensorIndex> input(1, 1, TensorIndex(kernel_size + output_size - 1));
  Tensor<float, 3, Layout, TensorIndex> kernel(1, 1, TensorIndex(kernel_size));
  Tensor<float, 3, Layout, TensorIndex> expected(1, 1, TensorIndex(output_size));
  input.setConstant(1);
  input.data()[kernel_size + output_size / 2 - 1] = 2;
  kernel.setConstant(1);
  // An independent sliding sum avoids a CPU convolution of O(kernel_size * output_size).
  Index sum = kernel_size;
  for (Index i = 0; i < output_size; ++i) {
    expected.data()[i] = float(sum);
    if (i + 1 < output_size) sum += Index(input.data()[i + kernel_size]) - Index(input.data()[i]);
  }
  if (std::is_same<TensorIndex, Index>::value) VERIFY(output_size > grid_stride);
  verify_gpu_convolution(input, kernel, axes, expected);
}

EIGEN_DECLARE_TEST(tensor_convolution_large_gpu) {
  GpuStreamDevice stream;
  GpuDevice device(&stream);
  // Always exceed this device's shared-memory limit, including newer GPUs.
  const Index oversized = device.sharedMemPerBlock() / (sizeof(float) * 3 * 50) + 1;
  const Index capacity = device.sharedMemPerBlock() / sizeof(float);
  const array<Index, 3> shapes[] = {{{3, 4, 2}},           {{3, 50, 120}},         {{3, 50, 145}},
                                    {{3, 50, oversized}},  {{1, 1, capacity - 1}}, {{1, 1, capacity}},
                                    {{1, 1, capacity + 1}}};
  const array<Index, 3> axes[] = {{{1, 2, 3}}, {{0, 2, 4}}, {{4, 0, 2}}};
  for (const auto& shape : shapes) {
    for (const auto& axis : axes) {
      CALL_SUBTEST(check_large_convolution<ColMajor>(shape, axis));
      CALL_SUBTEST(check_large_convolution<RowMajor>(shape, axis));
    }
  }
  CALL_SUBTEST(check_convolution_grid_stride<ColMajor>());
  CALL_SUBTEST(check_convolution_grid_stride<RowMajor>());
  CALL_SUBTEST((check_convolution_grid_stride<ColMajor, int16_t>(18000)));
  CALL_SUBTEST((check_convolution_grid_stride<RowMajor, int16_t>(18000)));
}
