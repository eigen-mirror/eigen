// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/SparseCore>
#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
#include "AnnoyingScalar.h"

template <typename Scalar, typename StorageIndex, typename Values>
Index apply_sparse_scatter(Scalar* dense, const StorageIndex* indices, const MatrixBase<Values>& values) {
  const Index processed = internal::sparse_scatter_sub_packets(dense, indices, values);
  VERIFY(processed >= 0 && processed <= values.size());
  if (values.size() < 32 || NumTraits<Scalar>::IsComplex || !(internal::evaluator<Values>::Flags & PacketAccessBit))
    VERIFY_IS_EQUAL(processed, Index(0));
  else {
    using Packet = typename internal::find_best_packet<Scalar, 4>::type;
    VERIFY_IS_EQUAL(processed, values.size() - values.size() % internal::unpacket_traits<Packet>::size);
  }
  for (Index i = processed; i < values.size(); ++i) dense[indices[i]] -= values[i];
  return processed;
}

template <typename Scalar, typename StorageIndex>
void sparse_scatter() {
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const Index capacity = 257;
  Vector initial = Vector::Random(capacity), actual(capacity), expected(capacity);
  Vector storage = Vector::Random(2 * capacity + 1);
  Matrix<StorageIndex, Dynamic, 1> indices(capacity);
  for (Index i = 0; i < capacity; ++i) indices[i] = StorageIndex((37 * i) % capacity);
  const Scalar scale = internal::random<Scalar>();
  // |inputs| <= sqrt(2): allow rounding in one product and one subtraction.
  const RealScalar tolerance = RealScalar(8) * NumTraits<RealScalar>::epsilon();

  for (Index size = 0; size <= 65; ++size) {
    const Map<const Vector> values(storage.data() + 1, size);
    actual = expected = initial;
    for (Index i = 0; i < size; ++i) expected[indices[i]] -= scale * numext::conj(values[i]);
    const Index expectedProcessed = apply_sparse_scatter(actual.data(), indices.data(), scale * values.conjugate());
    VERIFY((actual - expected).cwiseAbs().maxCoeff() <= tolerance);

    actual = initial;
    const Index processed =
        internal::sparse_scatter_sub_packets<true>(actual.data(), indices.data(), values.data(), size, scale);
    VERIFY_IS_EQUAL(processed, expectedProcessed);
    for (Index i = processed; i < size; ++i) actual[indices[i]] -= numext::conj(values[i]) * scale;
    VERIFY((actual - expected).cwiseAbs().maxCoeff() <= tolerance);

    const Map<const Vector, Unaligned, InnerStride<2>> strided(storage.data() + 1, size);
    actual = expected = initial;
    for (Index i = 0; i < size; ++i) expected[indices[i]] -= strided[i] * scale;
    apply_sparse_scatter(actual.data(), indices.data(), strided * scale);
    VERIFY((actual - expected).cwiseAbs().maxCoeff() <= tolerance);
  }
}

void sparse_scatter_custom_scalar() {
  Matrix<AnnoyingScalar, Dynamic, 1> values(33), actual(67);
  VectorXi indices(33);
  for (Index i = 0; i < actual.size(); ++i) actual[i] = AnnoyingScalar(3);
  for (Index i = 0; i < values.size(); ++i) {
    values[i] = AnnoyingScalar(int(i));
    indices[i] = int(2 * i);
  }
  const AnnoyingScalar scale(2);
  VERIFY_IS_EQUAL(
      internal::sparse_scatter_sub_packets<false>(actual.data(), indices.data(), values.data(), values.size(), scale),
      Index(0));
  apply_sparse_scatter(actual.data(), indices.data(), values * AnnoyingScalar(2));
  for (Index i = 0; i < actual.size(); ++i) {
    const AnnoyingScalar expected(i % 2 == 0 && i < 66 ? 3 - int(i) : 3);
    VERIFY_IS_EQUAL(actual[i], expected);
  }
}

template <typename Scalar>
void sparse_scatter_special_values() {
  using Vector = Matrix<Scalar, Dynamic, 1>;
  Vector values = Vector::Zero(65), actual = Vector::Zero(64), expected = actual;
  VectorXi indices(64);
  for (Index i = 0; i < indices.size(); ++i) indices[i] = int(63 - i);
  values[1] = NumTraits<Scalar>::quiet_NaN();
  values[2] = NumTraits<Scalar>::infinity();
  values[3] = -NumTraits<Scalar>::infinity();
  values[4] = -Scalar(0);
  for (Index i = 0; i < actual.size(); ++i) expected[indices[i]] -= Scalar(2) * values[i + 1];
  apply_sparse_scatter(actual.data(), indices.data(), Scalar(2) * values.tail(64));
  for (Index i = 0; i < actual.size(); ++i) {
    if ((numext::isnan)(expected[i])) {
      VERIFY((numext::isnan)(actual[i]));
    } else {
      VERIFY_IS_EQUAL(actual[i], expected[i]);
      VERIFY_IS_EQUAL(bool((numext::signbit)(actual[i])), bool((numext::signbit)(expected[i])));
    }
  }
}

EIGEN_DECLARE_TEST(sparse_scatter) {
  CALL_SUBTEST_1((sparse_scatter<float, int>()));
  CALL_SUBTEST_1((sparse_scatter<double, long long>()));
  CALL_SUBTEST_1(sparse_scatter_special_values<float>());
  CALL_SUBTEST_1(sparse_scatter_special_values<double>());
  CALL_SUBTEST_2((sparse_scatter<std::complex<float>, int>()));
  CALL_SUBTEST_2((sparse_scatter<std::complex<double>, long long>()));
  CALL_SUBTEST_3(sparse_scatter_custom_scalar());
}
