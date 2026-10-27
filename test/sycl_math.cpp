// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_SYCL
#define EIGEN_TEST_NO_LONGDOUBLE
#include "main.h"
#include <Eigen/Geometry>

template <typename Scalar>
void sycl_scalar_math(sycl::queue& queue) {
  constexpr int kOutputs = 38;
  std::vector<Scalar> result(4 * kOutputs);
  {
    sycl::buffer<Scalar, 1> buffer(result.data(), sycl::range<1>(result.size()));
    queue.submit([&](sycl::handler& handler) {
      auto output = buffer.template get_access<sycl::access::mode::write>(handler);
      handler.parallel_for(sycl::range<1>(4), [=](sycl::id<1> id) {
        const Scalar x = Scalar(id[0] + 1) / Scalar(4);
        const int offset = int(id[0]) * kOutputs;
        output[offset + 0] = internal::psin(x);
        output[offset + 1] = internal::pcos(x);
        output[offset + 2] = internal::ptan(x);
        output[offset + 3] = internal::pasin(x);
        output[offset + 4] = internal::pacos(x);
        output[offset + 5] = internal::patan(x);
        output[offset + 6] = internal::psinh(x);
        output[offset + 7] = internal::pcosh(x);
        output[offset + 8] = internal::ptanh(x);
        output[offset + 9] = internal::pasinh(x);
        output[offset + 10] = internal::pacosh(x + Scalar(1));
        output[offset + 11] = internal::patanh(x / Scalar(2));
        output[offset + 12] = numext::log(x);
        output[offset + 13] = numext::log2(x);
        output[offset + 14] = numext::log10(x);
        output[offset + 15] = numext::exp(x);
        output[offset + 16] = Array<Scalar, 1, 1>::Constant(x).pow(Scalar(1.5))(0);
        output[offset + 17] = numext::cbrt(x);
        output[offset + 18] = numext::sqrt(x);
        output[offset + 19] = numext::hypot(x, Scalar(2));
        output[offset + 20] = numext::atan2(x, Scalar(2));
        output[offset + 21] = numext::copysign(x, Scalar(-1));
        output[offset + 22] = numext::rint(x + Scalar(0.25));
        const Matrix<Scalar, 3, 1> vector(Scalar(1), x, x / Scalar(2));
        output[offset + 23] = vector.template lpNorm<3>();
        output[offset + 24] = vector.norm();
        const Quaternion<Scalar> rotation(AngleAxis<Scalar>(x, Matrix<Scalar, 3, 1>::UnitZ()));
        output[offset + 25] = rotation.angularDistance(Quaternion<Scalar>::Identity());
        output[offset + 26] =
            rotation.slerp(Scalar(0.5), Quaternion<Scalar>::Identity()).angularDistance(Quaternion<Scalar>::Identity());
        output[offset + 27] = Rotation2D<Scalar>(x).toRotationMatrix()(0, 1);
        JacobiRotation<Scalar> jacobi;
        jacobi.makeJacobi(Scalar(2), x, Scalar(3));
        output[offset + 28] = numext::abs2(jacobi.c()) + numext::abs2(jacobi.s());
        output[offset + 29] = Rotation2D<Scalar>(Rotation2D<Scalar>(x).toRotationMatrix()).angle();
        output[offset + 30] = numext::copysign(Scalar(0), Scalar(-1));
        output[offset + 31] = numext::atan2(Scalar(-0.0), Scalar(-1));
        output[offset + 32] = Scalar(float(numext::tanh(half(float(x)))));
        output[offset + 33] = Scalar(float(numext::tanh(bfloat16(float(x)))));
        output[offset + 34] = Scalar(float(numext::sqrt(half(float(x)))));
        output[offset + 35] = Scalar(float(numext::sqrt(bfloat16(float(x)))));
        output[offset + 36] = Scalar(float(numext::log2(half(float(x)))));
        output[offset + 37] = Scalar(float(numext::log2(bfloat16(float(x)))));
      });
    });
    queue.wait_and_throw();
  }
  for (int i = 0; i < 4; ++i) {
    const Scalar x = Scalar(i + 1) / Scalar(4);
    const Scalar expected[] = {std::sin(x),
                               std::cos(x),
                               std::tan(x),
                               std::asin(x),
                               std::acos(x),
                               std::atan(x),
                               std::sinh(x),
                               std::cosh(x),
                               std::tanh(x),
                               std::asinh(x),
                               std::acosh(x + Scalar(1)),
                               std::atanh(x / Scalar(2)),
                               std::log(x),
                               std::log2(x),
                               std::log10(x),
                               std::exp(x),
                               std::pow(x, Scalar(1.5)),
                               std::cbrt(x),
                               std::sqrt(x),
                               std::hypot(x, Scalar(2)),
                               std::atan2(x, Scalar(2)),
                               -x,
                               std::rint(x + Scalar(0.25)),
                               std::pow(Scalar(1) + x * x * x * Scalar(1.125), Scalar(1) / Scalar(3)),
                               std::sqrt(Scalar(1) + x * x * Scalar(1.25)),
                               x,
                               x / Scalar(2),
                               -std::sin(x),
                               Scalar(1),
                               x};
    // Each composed rotation/norm uses O(10) operations and a few elementary functions.
    for (int j = 0; j < 30; ++j) {
      const Scalar tolerance =
          Scalar(128) * NumTraits<Scalar>::epsilon() * (std::max)(Scalar(1), std::abs(expected[j]));
      VERIFY(std::abs(result[i * kOutputs + j] - expected[j]) <= tolerance);
    }
    VERIFY_IS_EQUAL(result[i * kOutputs + 30], Scalar(0));
    VERIFY((numext::signbit)(result[i * kOutputs + 30]));
    VERIFY(result[i * kOutputs + 31] < Scalar(0));
    VERIFY(std::abs(result[i * kOutputs + 31] + Scalar(EIGEN_PI)) <= Scalar(4) * NumTraits<Scalar>::epsilon());
    VERIFY_IS_EQUAL(result[i * kOutputs + 32], Scalar(float(half(std::tanh(float(x))))));
    VERIFY_IS_EQUAL(result[i * kOutputs + 33], Scalar(float(bfloat16(std::tanh(float(x))))));
    VERIFY_IS_EQUAL(result[i * kOutputs + 34], Scalar(float(half(std::sqrt(float(x))))));
    VERIFY_IS_EQUAL(result[i * kOutputs + 35], Scalar(float(bfloat16(std::sqrt(float(x))))));
    VERIFY_IS_EQUAL(result[i * kOutputs + 36], Scalar(float(half(std::log2(float(x))))));
    VERIFY_IS_EQUAL(result[i * kOutputs + 37], Scalar(float(bfloat16(std::log2(float(x))))));
  }
}

EIGEN_DECLARE_TEST(sycl_math) {
  sycl::queue queue;
  CALL_SUBTEST(sycl_scalar_math<float>(queue));
  if (queue.get_device().has(sycl::aspect::fp64)) CALL_SUBTEST(sycl_scalar_math<double>(queue));
}
