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

  // Uninitialized device memory shaped like a Tensor<Scalar, NumDims, Layout, IndexType>.
  template <typename Scalar, int NumDims, int Alignment = Eigen::Aligned, int Layout = Eigen::ColMajor,
            typename IndexType = Eigen::DenseIndex>
  Eigen::TensorMap<Tensor<Scalar, NumDims, Layout, IndexType>, Alignment> tensor(
      const Eigen::DSizes<IndexType, NumDims>& shape) {
    void* buffer = device_.allocate(shape.TotalSize() * sizeof(Scalar));
    buffers_.push_back(buffer);
    return Eigen::TensorMap<Tensor<Scalar, NumDims, Layout, IndexType>, Alignment>(static_cast<Scalar*>(buffer), shape);
  }

  // The dimensions can be empty, for a 0-d tensor.
  template <typename Scalar, int NumDims, int Alignment = Eigen::Aligned, int Layout = Eigen::ColMajor,
            typename IndexType = Eigen::DenseIndex, typename... Dimensions>
  Eigen::TensorMap<Tensor<Scalar, NumDims, Layout, IndexType>, Alignment> tensor(Dimensions... dimensions) {
    return tensor<Scalar, NumDims, Alignment, Layout, IndexType>(
        Eigen::DSizes<IndexType, NumDims>{static_cast<IndexType>(dimensions)...});
  }

  // A copy of host on the device.
  template <typename Scalar, int NumDims, int Layout, typename IndexType>
  Eigen::TensorMap<Tensor<Scalar, NumDims, Layout, IndexType>, Eigen::Aligned> fromHost(
      const Tensor<Scalar, NumDims, Layout, IndexType>& host) {
    auto values = tensor<Scalar, NumDims, Eigen::Aligned, Layout, IndexType>(host.dimensions());
    device_.memcpyHostToDevice(values.data(), host.data(), host.size() * sizeof(Scalar));
    return values;
  }

  template <typename Scalar, int NumDims, int Alignment, int Layout, typename IndexType>
  Tensor<Scalar, NumDims, Layout, IndexType> toHost(
      const Eigen::TensorMap<Tensor<Scalar, NumDims, Layout, IndexType>, Alignment>& values) {
    Tensor<Scalar, NumDims, Layout, IndexType> host;
    host.resize(values.dimensions());
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
