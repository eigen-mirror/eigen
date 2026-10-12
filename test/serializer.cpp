// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#if defined(EIGEN_TEST_PART_2)
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif

#include "main.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>
#endif

template <typename T>
struct RandomImpl {
  static auto Create(Eigen::Index rows, Eigen::Index cols) { return T::Random(rows, cols); }
};

template <typename Scalar, int Options, typename DenseIndex>
struct RandomImpl<Eigen::SparseMatrix<Scalar, Options, DenseIndex>> {
  using T = Eigen::SparseMatrix<Scalar, Options, DenseIndex>;

  static auto Create(Eigen::Index rows, Eigen::Index cols) {
    Eigen::SparseMatrix<Scalar, Options, DenseIndex> M(rows, cols);
    M.setZero();
    double density = 0.1;

    // Reserve some space along each inner dim.
    int nnz = static_cast<int>(std::ceil(density * 1.5 * M.innerSize()));
    M.reserve(Eigen::VectorXi::Constant(M.outerSize(), nnz));

    for (int j = 0; j < M.outerSize(); j++) {
      for (int i = 0; i < M.innerSize(); i++) {
        bool zero = (Eigen::internal::random<double>(0, 1) > density);
        if (!zero) {
          M.insertByOuterInner(j, i) = internal::random<Scalar>();
        }
      }
    }

    // 50-50 whether to compress or not.
    if (Eigen::internal::random<double>(0, 1) >= 0.5) {
      M.makeCompressed();
    }

    return M;
  }
};

template <typename Scalar, int Options, typename DenseIndex>
struct RandomImpl<Eigen::SparseVector<Scalar, Options, DenseIndex>> {
  using T = Eigen::SparseVector<Scalar, Options, DenseIndex>;

  static auto Create(Eigen::Index rows, Eigen::Index cols) {
    Eigen::SparseVector<Scalar, Options, DenseIndex> M(rows, cols);
    M.setZero();
    double density = 0.1;

    // Reserve some space along each inner dim.
    int nnz = static_cast<int>(density * 1.5 * M.innerSize());
    M.reserve(nnz);

    for (int i = 0; i < M.innerSize(); i++) {
      bool zero = (Eigen::internal::random<double>(0, 1) > density);
      if (!zero) {
        M.insert(i) = internal::random<Scalar>();
      }
    }

    return M;
  }
};

struct MyPodType {
  double x;
  int y;
  float z;
};

// Plain-old-data serialization.
void test_pod_type() {
  MyPodType initial = {1.3, 17, 1.9f};
  MyPodType clone = {-1, -1, -1};

  Eigen::Serializer<MyPodType> serializer;

  // Determine required size.
  size_t buffer_size = serializer.size(initial);
  VERIFY_IS_EQUAL(buffer_size, sizeof(MyPodType));

  // Serialize.
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = serializer.serialize(begin, end, initial);
  VERIFY(dest != nullptr);
  VERIFY_IS_EQUAL(dest - begin, buffer_size);

  // Deserialize.
  const uint8_t* src = serializer.deserialize(begin, end, clone);
  VERIFY(src != nullptr);
  VERIFY_IS_EQUAL(src - begin, buffer_size);
  VERIFY_IS_EQUAL(clone.x, initial.x);
  VERIFY_IS_EQUAL(clone.y, initial.y);
  VERIFY_IS_EQUAL(clone.z, initial.z);

  // Serialize with bounds checking errors.
  dest = serializer.serialize(begin, end - 1, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(begin, begin, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(nullptr, nullptr, initial);
  VERIFY(dest == nullptr);

  // Deserialize with bounds checking errors.
  src = serializer.deserialize(begin, end - 1, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(begin, begin, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(nullptr, nullptr, clone);
  VERIFY(src == nullptr);
}

// Matrix, Vector, Array
template <typename T>
void test_eigen_type(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();

  const T initial = RandomImpl<T>::Create(rows, cols);

  // Serialize.
  Eigen::Serializer<T> serializer;
  size_t buffer_size = serializer.size(initial);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = serializer.serialize(begin, end, initial);
  VERIFY(dest != nullptr);
  VERIFY_IS_EQUAL(dest - begin, buffer_size);

  // Deserialize.
  T clone;
  const uint8_t* src = serializer.deserialize(begin, end, clone);
  VERIFY(src != nullptr);
  VERIFY_IS_EQUAL(src - begin, buffer_size);
  VERIFY_IS_CWISE_EQUAL(clone, initial);

  // Serialize with bounds checking errors.
  dest = serializer.serialize(begin, end - 1, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(begin, begin, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(nullptr, nullptr, initial);
  VERIFY(dest == nullptr);

  // Deserialize with bounds checking errors.
  src = serializer.deserialize(begin, end - 1, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(begin, begin, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(nullptr, nullptr, clone);
  VERIFY(src == nullptr);
}

// A dense header whose sizes are negative, wrong for the type, or larger than the buffer is a parsing error.
template <typename T>
void test_malformed_dense_header(Index rows, Index cols) {
  struct Header {
    Index rows;
    Index cols;
  };
  std::vector<uint8_t> buffer(sizeof(Header) + 16 * sizeof(typename T::Scalar));
  const Header header = {rows, cols};
  std::memcpy(buffer.data(), &header, sizeof(header));
  T value;
  VERIFY(Eigen::Serializer<T>().deserialize(buffer.data(), buffer.data() + buffer.size(), value) == nullptr);
}

void test_malformed_dense_headers() {
  test_malformed_dense_header<Matrix2d>(4, 4);
  test_malformed_dense_header<Matrix2d>(1, 4);
  test_malformed_dense_header<Matrix<double, Dynamic, Dynamic, 0, 2, 2>>(4, 1);
  test_malformed_dense_header<MatrixXd>(-1, -4);
  test_malformed_dense_header<MatrixXd>(5, 4);
  test_malformed_dense_header<MatrixXd>(Index(1) << (8 * sizeof(Index) - 2), 4);
  test_malformed_dense_header<VectorXd>(4, 4);
}

// Serializes `valid` into a buffer with room to spare, applies `corrupt` to the header and payload, and checks that
// deserializing the result is a parsing error that leaves no entries in the destination.
template <typename T, typename Corrupt>
void verify_rejected(const T& valid, Corrupt corrupt) {
  using Header = typename Serializer<T>::Header;
  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(valid) + 256);
  uint8_t* begin = buffer.data();
  uint8_t* end = begin + buffer.size();
  VERIFY(serializer.serialize(begin, end, valid) != nullptr);
  Header header;
  std::memcpy(&header, begin, sizeof(Header));
  corrupt(header, begin + sizeof(Header));
  std::memcpy(begin, &header, sizeof(Header));
  T value;
  VERIFY(serializer.deserialize(begin, end, value) == nullptr);
  VERIFY_IS_EQUAL(value.nonZeros(), 0);
}

template <typename StorageIndex>
void set_index(uint8_t* array, Index i, StorageIndex v) {
  std::memcpy(array + i * sizeof(StorageIndex), &v, sizeof(v));
}

// Every header field and stored index of a sparse matrix decides where deserialize() writes or what later reads
// address, so a buffer that does not describe a valid matrix must be rejected.
template <int Options>
void test_malformed_sparse_matrix() {
  using T = SparseMatrix<float, Options, int>;
  using Header = typename Serializer<T>::Header;
  constexpr bool kRowMajor = T::IsRowMajor;
  // 3 x 2 with entries (0, 0), (2, 0), (1, 1): compressed, the payload is outer[3], inner[3], values[3] in column-major
  // order, and outer[4], inner[3], values[3] in row-major order.
  T m(3, 2);
  m.insert(0, 0) = 1;
  m.insert(2, 0) = 2;
  m.insert(1, 1) = 3;
  m.makeCompressed();
  const Index outer_size = m.outerSize();
  const auto inner = [outer_size](uint8_t* payload) { return payload + (outer_size + 1) * sizeof(int); };

  verify_rejected(T(1, 1), [](Header& h, uint8_t*) { h.outer_size = 10; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.outer_size = kRowMajor ? h.cols : h.rows; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.rows = -1; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.cols = -1; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.inner_buffer_size = -1; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.inner_buffer_size = 4; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.inner_buffer_size = Index(1) << 30; });
  verify_rejected(m, [](Header& h, uint8_t*) {
    const unsigned char two = 2;
    std::memcpy(&h.compressed, &two, 1);
  });
  verify_rejected(m, [](Header&, uint8_t* p) { set_index(p, 0, 1); });
  verify_rejected(m, [](Header&, uint8_t* p) { set_index(p, 1, 4); });
  verify_rejected(m, [&](Header&, uint8_t* p) { set_index(p, outer_size, 2); });
  verify_rejected(m, [&](Header&, uint8_t* p) { set_index(inner(p), 1, kRowMajor ? 2 : 3); });
  verify_rejected(m, [&](Header&, uint8_t* p) { set_index(inner(p), 2, -1); });

  // Uncompressed, the payload starts with innerNonZeros[outer_size], and the unused entries are not checked.
  T u = m;
  u.reserve(Eigen::VectorXi::Constant(outer_size, 3));
  VERIFY(!u.isCompressed());
  const int capacity0 = u.outerIndexPtr()[1] - u.outerIndexPtr()[0];
  verify_rejected(u, [&](Header&, uint8_t* p) { set_index(p, 0, capacity0 + 1); });
  verify_rejected(u, [&](Header&, uint8_t* p) { set_index(p, 0, -1); });
  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(u));
  VERIFY(serializer.serialize(buffer.data(), buffer.data() + buffer.size(), u) != nullptr);
  uint8_t* payload = buffer.data() + sizeof(Header);
  const int* outer = u.outerIndexPtr();
  const int* inner_non_zeros = u.innerNonZeroPtr();
  for (Index j = 0; j < outer_size; ++j) {
    for (Index k = outer[j] + inner_non_zeros[j]; k < outer[j + 1]; ++k)
      set_index(payload + (2 * outer_size + 1) * sizeof(int), k, -7);
  }
  T clone;
  VERIFY(serializer.deserialize(buffer.data(), buffer.data() + buffer.size(), clone) != nullptr);
  VERIFY_IS_CWISE_EQUAL(MatrixXf(clone), MatrixXf(m));
}

void test_malformed_sparse_vector() {
  using T = SparseVector<float, 0, int>;
  using Header = Serializer<T>::Header;
  T v(4);
  v.insert(1) = 1;
  v.insert(3) = 2;
  verify_rejected(v, [](Header& h, uint8_t*) { h.size = -1; });
  verify_rejected(v, [](Header& h, uint8_t*) { h.num_non_zeros = -1; });
  verify_rejected(v, [](Header& h, uint8_t*) { h.num_non_zeros = Index(1) << 30; });
  verify_rejected(v, [](Header&, uint8_t* p) { set_index(p, 1, 4); });
  verify_rejected(v, [](Header&, uint8_t* p) { set_index(p, 0, -1); });
}

#if defined(EIGEN_TEST_PART_2) && defined(__unix__) && defined(MAP_NORESERVE) && SIZE_MAX > 0xFFFFFFFFu
// 65536 x 65536 chars fit in a 4 GiB buffer, but resize() forms rows * cols in the 32-bit Index, where it overflows.
// The mapping only reserves address space: the header must be rejected before the payload is read.
void test_dense_header_index_overflow() {
  STATIC_CHECK(sizeof(Index) == 4);
  using T = Matrix<char, Dynamic, Dynamic>;
  using Header = Serializer<T>::Header;
  const std::size_t bytes = sizeof(Header) + (std::size_t(1) << 32);
  void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  VERIFY(map != MAP_FAILED);
  const Header header = {65536, 65536};
  std::memcpy(map, &header, sizeof(header));
  const uint8_t* begin = static_cast<const uint8_t*>(map);
  T value;
  VERIFY(Serializer<T>().deserialize(begin, begin + bytes, value) == nullptr);
  VERIFY_IS_EQUAL(munmap(map, bytes), 0);
}
#endif

// Test a collection of dense types.
template <typename T1, typename T2, typename T3>
void test_dense_types(const T1& type1, const T2& type2, const T3& type3) {
  // Make random inputs.
  const T1 x1 = T1::Random(type1.rows(), type1.cols());
  const T2 x2 = T2::Random(type2.rows(), type2.cols());
  const T3 x3 = T3::Random(type3.rows(), type3.cols());

  // Allocate buffer and serialize.
  size_t buffer_size = Eigen::serialize_size(x1, x2, x3);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = Eigen::serialize(begin, end, x1, x2, x3);
  VERIFY(dest != nullptr);

  // Clone everything.
  T1 y1;
  T2 y2;
  T3 y3;
  const uint8_t* src = Eigen::deserialize(begin, end, y1, y2, y3);
  VERIFY(src != nullptr);

  // Verify they equal.
  VERIFY_IS_CWISE_EQUAL(y1, x1);
  VERIFY_IS_CWISE_EQUAL(y2, x2);
  VERIFY_IS_CWISE_EQUAL(y3, x3);

  // Serialize everything with bounds checking errors.
  dest = Eigen::serialize(begin, end - 1, y1, y2, y3);
  VERIFY(dest == nullptr);
  dest = Eigen::serialize(begin, begin, y1, y2, y3);
  VERIFY(dest == nullptr);
  dest = Eigen::serialize(nullptr, nullptr, y1, y2, y3);
  VERIFY(dest == nullptr);

  // Deserialize everything with bounds checking errors.
  src = Eigen::deserialize(begin, end - 1, y1, y2, y3);
  VERIFY(src == nullptr);
  src = Eigen::deserialize(begin, begin, y1, y2, y3);
  VERIFY(src == nullptr);
  src = Eigen::deserialize(nullptr, nullptr, y1, y2, y3);
  VERIFY(src == nullptr);
}

EIGEN_DECLARE_TEST(serializer) {
  CALL_SUBTEST_1(test_pod_type());
  CALL_SUBTEST_1(test_malformed_dense_headers());
  CALL_SUBTEST_1(test_malformed_sparse_matrix<ColMajor>());
  CALL_SUBTEST_1(test_malformed_sparse_matrix<RowMajor>());
  CALL_SUBTEST_1(test_malformed_sparse_vector());

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(test_eigen_type(Eigen::Array33f()));
    CALL_SUBTEST_1(test_eigen_type(Eigen::ArrayXd(10)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::Vector3f()));
    CALL_SUBTEST_1(test_eigen_type(Eigen::Matrix4d()));
    CALL_SUBTEST_1(test_eigen_type(Eigen::MatrixXd(15, 17)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseMatrix<float>(13, 12)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseMatrix<float>(0, 3)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseVector<float>(17)));

    CALL_SUBTEST_1(test_dense_types(Eigen::Array33f(), Eigen::ArrayXd(10), Eigen::MatrixXd(15, 17)));
  }

  CALL_SUBTEST_2(test_malformed_dense_headers());
#if defined(EIGEN_TEST_PART_2) && defined(__unix__) && defined(MAP_NORESERVE) && SIZE_MAX > 0xFFFFFFFFu
  CALL_SUBTEST_2(test_dense_header_index_overflow());
#endif
}
