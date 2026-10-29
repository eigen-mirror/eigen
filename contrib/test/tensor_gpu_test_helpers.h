// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_GPU_TEST_HELPERS_H
#define EIGEN_TENSOR_GPU_TEST_HELPERS_H

#include <vector>

#include <contrib/Eigen/Tensor>

namespace tensor_test {

using Eigen::Index;
using Eigen::Tensor;

class GpuTensorTestFixture {
 public:
  GpuTensorTestFixture() : device_(&stream_) {}
  GpuTensorTestFixture(const GpuTensorTestFixture&) = delete;
  GpuTensorTestFixture& operator=(const GpuTensorTestFixture&) = delete;

  ~GpuTensorTestFixture() {
    // Maps and pending expressions borrow these buffers from the fixture.
    device_.synchronize();
    for (void* buffer : buffers_) device_.deallocate(buffer);
  }

  const Eigen::GpuDevice& device() const { return device_; }

  template <typename Scalar, int NumDims, int Alignment = Eigen::Aligned, typename... Dimensions>
  Eigen::TensorMap<Tensor<Scalar, NumDims>, Alignment> tensor(Dimensions... dimensions) {
    const Eigen::DSizes<Index, NumDims> shape{dimensions...};
    void* buffer = device_.allocate(shape.TotalSize() * sizeof(Scalar));
    buffers_.push_back(buffer);
    return Eigen::TensorMap<Tensor<Scalar, NumDims>, Alignment>(static_cast<Scalar*>(buffer), shape);
  }

  template <typename Scalar, int NumDims, int Alignment>
  Tensor<Scalar, NumDims> toHost(const Eigen::TensorMap<Tensor<Scalar, NumDims>, Alignment>& values) {
    Tensor<Scalar, NumDims> host(values.dimensions());
    device_.memcpyDeviceToHost(host.data(), values.data(), values.size() * sizeof(Scalar));
    device_.synchronize();
    return host;
  }

 private:
  Eigen::GpuStreamDevice stream_;
  Eigen::GpuDevice device_;
  std::vector<void*> buffers_;
};

}  // namespace tensor_test

#endif  // EIGEN_TENSOR_GPU_TEST_HELPERS_H
