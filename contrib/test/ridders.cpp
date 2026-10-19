// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <contrib/Eigen/NumericalDiff>
#include <contrib/Eigen/LevenbergMarquardt>

template <typename Scalar_, int Inputs = Dynamic, int Values = Dynamic, int Options = ColMajor>
struct RiddersFunctor {
  using Scalar = Scalar_;
  using InputType = Matrix<Scalar, Inputs, 1>;
  using ValueType = Matrix<Scalar, Values, 1>;
  using JacobianType = Matrix<Scalar, Values, Inputs, Options>;

  static constexpr int InputsAtCompileTime = Inputs;
  static constexpr int ValuesAtCompileTime = Values;

  mutable int evaluations = 0;

  int values() const { return 4; }

  int operator()(const InputType& x, ValueType& y) const {
    ++evaluations;
    y[0] = numext::exp(x[0]);
    y[1] = numext::sin(Scalar(50) * x[0]);
    y[2] = x[0] * x[1];
    y[3] = numext::exp(x[1]);
    return 0;
  }
};

template <typename Scalar, int Inputs, int Values, int Options>
void test_ridders_jacobian() {
  using F = RiddersFunctor<Scalar, Inputs, Values, Options>;

  Ridders<F> diff;
  typename F::InputType x(2), error(2);
  typename F::JacobianType jac(4, 2), expected(4, 2), withoutError(4, 2);

  x << Scalar(1), Scalar(-2);
  const typename F::InputType original = x;

  expected.row(0) << Scalar(std::exp(1.0L)), Scalar(0);
  expected.row(1) << Scalar(50 * std::cos(50.0L)), Scalar(0);
  expected.row(2) << x[1], x[0];
  expected.row(3) << Scalar(0), Scalar(std::exp(-2.0L));

  const int count = diff.df(x, jac, error);
  VERIFY_IS_EQUAL(count, diff.functor().evaluations);
  VERIFY(count >= 8 && count <= 40 && count % 2 == 0);

  VERIFY(jac.allFinite());
  VERIFY(error.allFinite());
  VERIFY((error.array() >= Scalar(0)).all());

  // The initial stencil spans half a radian in the oscillatory component; extrapolation
  // must remove its truncation error to reach a roundoff-scale bound.
  VERIFY((jac - expected).template lpNorm<Infinity>() <=
         Scalar(256) * NumTraits<Scalar>::epsilon() * expected.template lpNorm<Infinity>());

  VERIFY_IS_EQUAL(x, original);
  VERIFY_IS_EQUAL(diff.df(x, withoutError), count);
  VERIFY_IS_EQUAL(jac, withoutError);
}

struct RiddersScalarFunctor {
  using Scalar = double;
  using InputType = Matrix<double, 1, 1>;
  using ValueType = InputType;
  using JacobianType = InputType;

  static constexpr int InputsAtCompileTime = 1;
  static constexpr int ValuesAtCompileTime = 1;

  enum Function { Cubic, LargeCubic, Quintic, SquareRoot, NoisyExp, Identity, Invalid, InvalidRefinement };
  Function function = Cubic;
  mutable int evaluations = 0;

  int values() const { return 1; }

  int operator()(const InputType& x, ValueType& y) const {
    ++evaluations;
    switch (function) {
      case Cubic:
        y[0] = x[0] * x[0] * x[0];
        break;

      case LargeCubic:
        y[0] = 2e307 * x[0] * x[0] * x[0];
        break;

      case Quintic:
        y[0] = x[0] * x[0] * x[0] * x[0] * x[0];
        break;

      case SquareRoot:
        y[0] = std::sqrt(x[0]);
        break;

      case NoisyExp:
        y[0] = double(float(std::exp(x[0])));
        break;

      case Identity:
        y[0] = x[0];
        break;

      case Invalid:
        y[0] = NumTraits<double>::quiet_NaN();
        break;

      case InvalidRefinement:
        y[0] = evaluations <= 2 ? x[0] : NumTraits<double>::quiet_NaN();
        break;
    }
    return 0;
  }
};

void test_ridders_parameters() {
  using DiffType = Ridders<RiddersScalarFunctor>;

  DiffType diff;
  DiffType::Parameters parameters;
  parameters.initialStep = 0.25;
  parameters.stepReduction = 4;
  parameters.maxSteps = 2;
  diff.setParameters(parameters);

  RiddersScalarFunctor::InputType x, error;
  RiddersScalarFunctor::JacobianType jac;
  x[0] = 2;
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
  VERIFY_IS_EQUAL(jac[0], 12.0);

  // The cubic's central differences are 12 + h^2, with h = 1/2 and 1/8.
  VERIFY_IS_EQUAL(error[0], 0.25);

  parameters.maxSteps = 10;
  diff.setParameters(parameters);
  VERIFY(diff.df(x, jac, error) < 20);
  VERIFY_IS_EQUAL(jac[0], 12.0);
  VERIFY_IS_EQUAL(error[0], 0.0);

  parameters.initialStep = 0;
  VERIFY_RAISES_ASSERT(diff.setParameters(parameters));

  parameters.initialStep = 0.01;
  parameters.stepReduction = 1;
  VERIFY_RAISES_ASSERT(diff.setParameters(parameters));

  parameters.stepReduction = 2;
  parameters.maxSteps = 1;
  VERIFY_RAISES_ASSERT(diff.setParameters(parameters));

  parameters.maxSteps = 10;
  parameters.initialStep = NumTraits<double>::infinity();
  VERIFY_RAISES_ASSERT(diff.setParameters(parameters));

  parameters.initialStep = 0.01;
  parameters.stepReduction = NumTraits<double>::infinity();
  VERIFY_RAISES_ASSERT(diff.setParameters(parameters));

  parameters = DiffType::Parameters();
  if (NumTraits<Index>::highest() / 2 < NumTraits<int>::highest()) {
    parameters.maxSteps = static_cast<int>(NumTraits<Index>::highest() / 2) + 1;
    VERIFY_RAISES_ASSERT(diff.setParameters(parameters));
    --parameters.maxSteps;
  } else {
    parameters.maxSteps = NumTraits<int>::highest();
  }
  diff.setParameters(parameters);
}

void test_ridders_repeated_stencil() {
  using DiffType = Ridders<RiddersScalarFunctor>;

  for (int multiplier : {1, 16}) {
    DiffType diff;
    DiffType::Parameters parameters;
    parameters.stepReduction = 1 + multiplier * NumTraits<double>::epsilon();
    diff.setParameters(parameters);

    RiddersScalarFunctor::InputType x, error;
    RiddersScalarFunctor::JacobianType jac;
    x[0] = 1;
    VERIFY_IS_EQUAL(diff.df(x, jac, error), 2);
    VERIFY_IS_EQUAL(diff.functor().evaluations, 2);
    VERIFY((numext::isfinite)(jac[0]));
    VERIFY(std::abs(jac[0] - 3.0001) <= 32 * NumTraits<double>::epsilon());
    VERIFY((numext::isinf)(error[0]) && error[0] > 0);
  }
}

void test_ridders_dimensions() {
  Ridders<RiddersFunctor<double>> diff;
  VectorXd x = VectorXd::Ones(2), error;
  MatrixXd jac(3, 2);
  VERIFY_RAISES_ASSERT(diff.df(x, jac));

  jac.resize(4, 1);
  VERIFY_RAISES_ASSERT(diff.df(x, jac, error));

  jac.resize(4, 0);
  VERIFY_RAISES_ASSERT(diff.df(x, jac));
  VERIFY_IS_EQUAL(diff.functor().evaluations, 0);
}

template <typename Scalar_>
struct RiddersAffineFunctor {
  using Scalar = Scalar_;
  using InputType = Matrix<Scalar, 1, 1>;
  using ValueType = Matrix<Scalar, Dynamic, 1>;
  using JacobianType = Matrix<Scalar, Dynamic, 1>;

  static constexpr int InputsAtCompileTime = 1;
  static constexpr int ValuesAtCompileTime = Dynamic;

  ValueType slopes, offsets;
  Scalar center = Scalar(100);
  Scalar inputScale = Scalar(1);
  mutable int evaluations = 0;
  int finiteEvaluations = 20;

  Index values() const { return slopes.size(); }

  int operator()(const InputType& x, ValueType& y) const {
    ++evaluations;
    y = slopes * ((x[0] - center) * inputScale) + offsets;
    if (evaluations > finiteEvaluations) y.setConstant(NumTraits<Scalar>::quiet_NaN());
    return 0;
  }
};

template <typename Scalar>
void test_ridders_central_overflow() {
  using Functor = RiddersAffineFunctor<Scalar>;
  using DiffType = Ridders<Functor>;

  const Scalar large = NumTraits<Scalar>::highest() * Scalar(0.75);
  const Scalar tiny = (std::numeric_limits<Scalar>::denorm_min)();

  for (Index count : {1, 7, 17}) {
    Functor functor;
    functor.slopes.resize(count);
    functor.offsets.setZero(count);

    for (Index i = 0; i < count; ++i) {
      functor.slopes[i] = i % 3 == 0 ? large : (i % 3 == 1 ? Scalar(-8) : (std::numeric_limits<Scalar>::min)());
      if (i % 6 == 3) functor.slopes[i] = -large;
    }

    DiffType diff(functor);
    typename Functor::InputType x, error;
    typename Functor::JacobianType jac(count, 1);
    x[0] = functor.center;
    VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
    VERIFY(jac.allFinite());
    for (Index i = 0; i < count; ++i) VERIFY_IS_EQUAL(jac[i], functor.slopes[i]);
    VERIFY_IS_EQUAL(error[0], Scalar(0));

    // A failed refinement must retain finite entries, including subnormals, from the initial stencil.
    diff.functor().evaluations = 0;
    diff.functor().finiteEvaluations = 2;
    for (Index i = 2; i < count; i += 3) diff.functor().slopes[i] = tiny;
    VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
    VERIFY_IS_EQUAL(jac, diff.functor().slopes);
    VERIFY((numext::isinf)(error[0]) && error[0] > Scalar(0));
  }

  Functor functor;
  functor.slopes.setZero(1);
  functor.offsets.setConstant(1, large);

  DiffType diff(functor);
  typename Functor::InputType x, error;
  typename Functor::JacobianType jac(1, 1);
  x[0] = Scalar(100);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
  VERIFY_IS_EQUAL(jac[0], Scalar(0));

  x[0] = Scalar(0);
  diff.functor().center = Scalar(0);

  typename DiffType::Parameters parameters = diff.parameters();
  parameters.initialStep = (std::numeric_limits<Scalar>::min)();
  diff.setParameters(parameters);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
  VERIFY_IS_EQUAL(jac[0], Scalar(0));

  // Division first cannot repair a genuinely overflowing derivative or nonfinite samples.
  diff.functor().slopes[0] = NumTraits<Scalar>::highest();
  diff.functor().offsets[0] = Scalar(0);
  diff.functor().inputScale = Scalar(4);
  parameters.initialStep = Scalar(0.25);
  diff.setParameters(parameters);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 2);
  VERIFY((numext::isnan)(jac[0]));
  VERIFY((numext::isinf)(error[0]) && error[0] > Scalar(0));

  diff.functor().slopes[0] = NumTraits<Scalar>::infinity();
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 2);
  VERIFY((numext::isnan)(jac[0]));
  VERIFY((numext::isinf)(error[0]) && error[0] > Scalar(0));

  parameters.initialStep = Scalar(2);
  diff.setParameters(parameters);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 2);
  VERIFY((numext::isnan)(jac[0]));
}

void test_ridders_domain_and_noise() {
  using DiffType = Ridders<RiddersScalarFunctor>;

  DiffType diff;
  RiddersScalarFunctor::InputType x, error;
  RiddersScalarFunctor::JacobianType jac;

  diff.functor().function = RiddersScalarFunctor::SquareRoot;
  x[0] = 0.005;

  DiffType::Parameters parameters;
  parameters.initialStep = 0.01 * x[0];
  diff.setParameters(parameters);
  VERIFY(diff.df(x, jac, error) <= 20);

  const double expected = double(0.5L / std::sqrt(static_cast<long double>(x[0])));
  VERIFY(std::abs(jac[0] - expected) <= 256 * NumTraits<double>::epsilon() * expected);
  VERIFY((numext::isfinite)(error[0]));

  diff.functor().function = RiddersScalarFunctor::NoisyExp;
  x[0] = 1;
  parameters.initialStep = 0.01;
  diff.setParameters(parameters);
  VERIFY(diff.df(x, jac, error) < 20);
  VERIFY(std::abs(jac[0] - std::exp(1.0)) <= 256 * double(NumTraits<float>::epsilon()) * std::exp(1.0));
  VERIFY((numext::isfinite)(error[0]));
}

void test_ridders_richardson_overflow() {
  Ridders<RiddersScalarFunctor> diff;
  diff.functor().function = RiddersScalarFunctor::LargeCubic;
  RiddersScalarFunctor::InputType x, error;
  RiddersScalarFunctor::JacobianType jac;
  x[0] = 1;

  const int evaluations = diff.df(x, jac, error);
  VERIFY_IS_EQUAL(evaluations, diff.functor().evaluations);
  VERIFY(evaluations > 4 && evaluations <= 20);

  VERIFY((numext::isfinite)(jac[0]));
  VERIFY((numext::isfinite)(error[0]));
  VERIFY(error[0] >= 0);

  // The cubic's truncation error cancels on extrapolation; divide before comparing near the range limit.
  VERIFY(std::abs(jac[0] / 6e307 - 1) <= 256 * NumTraits<double>::epsilon());
}

void test_ridders_stencil() {
  using DiffType = Ridders<RiddersScalarFunctor>;

  DiffType diff;
  diff.functor().function = RiddersScalarFunctor::Quintic;

  DiffType::Parameters parameters;
  parameters.initialStep = 0.25;
  parameters.maxSteps = 3;
  diff.setParameters(parameters);

  RiddersScalarFunctor::InputType x, error, value;
  RiddersScalarFunctor::JacobianType jac;
  x[0] = 1;
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 6);
  VERIFY_IS_EQUAL(jac[0], 5.0);

  // At depth three the extrapolation is the degree-five-exact stencil on these six nodes.
  const Stencil<1, double, 6> stencil({-0.25, 0.25, -0.125, 0.125, -0.0625, 0.0625});
  double expected = 0;
  for (Index i = 0; i < 6; ++i) {
    RiddersScalarFunctor::InputType point;
    point[0] = x[0] + stencil.point(i);
    diff(point, value);
    expected += stencil.weight(i) * value[0];
  }

  VERIFY(std::abs(jac[0] - expected) <= 32 * NumTraits<double>::epsilon() * std::abs(jac[0]));
}

// Tests construction from a functor without a default constructor and optional inputs() forwarding.
struct OwnedRiddersFunctor : RiddersScalarFunctor {
  explicit OwnedRiddersFunctor(Function selected) { function = selected; }

  int inputs() const { return 1; }
};

void test_ridders_composition() {
  using Derivative = Ridders<OwnedRiddersFunctor>;

  static_assert(!std::is_base_of<OwnedRiddersFunctor, Derivative>::value, "Ridders owns its functor by composition");

  OwnedRiddersFunctor functor(RiddersScalarFunctor::Cubic);
  Derivative::Parameters parameters;
  parameters.initialStep = 0.25;
  parameters.maxSteps = 2;
  Derivative derivative(functor, parameters);
  VERIFY_IS_EQUAL(derivative.inputs(), 1);
  VERIFY_IS_EQUAL(derivative.values(), 1);
  VERIFY_IS_EQUAL(derivative.parameters().maxSteps, 2);

  const int* inputs = &Derivative::InputsAtCompileTime;
  const int* values = &Derivative::ValuesAtCompileTime;
  VERIFY_IS_EQUAL(*inputs, 1);
  VERIFY_IS_EQUAL(*values, 1);

  OwnedRiddersFunctor::InputType x, y, error;
  OwnedRiddersFunctor::JacobianType jac;
  x[0] = 2;
  VERIFY_IS_EQUAL(derivative(x, y), 0);
  VERIFY_IS_EQUAL(y[0], 8.0);

  VERIFY_IS_EQUAL(derivative.df(x, jac, error), 4);
  VERIFY_IS_EQUAL(jac[0], 12.0);
  VERIFY_IS_EQUAL(functor.evaluations, 0);

  const Derivative& constDerivative = derivative;
  VERIFY_IS_EQUAL(constDerivative.functor().evaluations, 5);
}

struct LinearRiddersFunctor : RiddersFunctor<double, 7, 7> {
  int values() const { return 7; }

  int operator()(const InputType& x, ValueType& y) const {
    y = x * 8;
    return 0;
  }
};

void test_ridders_step_alignment() {
  Ridders<LinearRiddersFunctor> derivative;
  LinearRiddersFunctor::InputType x;
  x << 0.0, 0.082, 1.13, -2350.7, 7.7e8 + 0.3, -1.0 - std::ldexp(1.0, -27), -1.0;
  LinearRiddersFunctor::JacobianType jac;

  derivative.df(x, jac);
  VERIFY_IS_EQUAL(jac, LinearRiddersFunctor::JacobianType::Identity() * 8);
}

void test_ridders_fallback() {
  using DiffType = Ridders<RiddersScalarFunctor>;

  DiffType diff;
  RiddersScalarFunctor::InputType x, error;
  RiddersScalarFunctor::JacobianType jac;
  x[0] = 1;

  diff.functor().function = RiddersScalarFunctor::Invalid;
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 2);
  VERIFY((numext::isnan)(jac[0]));
  VERIFY((numext::isinf)(error[0]) && error[0] > 0);

  diff.functor().function = RiddersScalarFunctor::InvalidRefinement;
  diff.functor().evaluations = 0;
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
  VERIFY_IS_EQUAL(jac[0], 1.0);
  VERIFY((numext::isinf)(error[0]));

  diff.functor().function = RiddersScalarFunctor::Identity;

  DiffType::Parameters parameters;
  parameters.initialStep = NumTraits<double>::epsilon() / 16;
  diff.setParameters(parameters);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 0);
  VERIFY((numext::isnan)(jac[0]));
  VERIFY((numext::isinf)(error[0]));

  parameters.initialStep = NumTraits<double>::epsilon();
  diff.setParameters(parameters);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 4);
  VERIFY_IS_EQUAL(jac[0], 1.0);
  VERIFY_IS_EQUAL(error[0], 0.0);
}

struct RiddersLevenbergMarquardtFunctor : DenseFunctor<double> {
  RiddersLevenbergMarquardtFunctor() : DenseFunctor<double>(1, 1) {}

  int operator()(const InputType& x, ValueType& y) const {
    y[0] = x[0] * x[0] - 2;
    return 0;
  }
};

void test_ridders_levenberg_marquardt() {
  using Functor = RiddersLevenbergMarquardtFunctor;
  using Derivative = Ridders<Functor>;

  static_assert(std::is_same<typename Derivative::QRSolver, typename Functor::QRSolver>::value,
                "Ridders must preserve the functor's QRSolver type");

  Derivative derivative;
  LevenbergMarquardt<Derivative> solver(derivative);
  VectorXd x = VectorXd::Ones(1);

  const LevenbergMarquardtSpace::Status status = solver.minimize(x);
  VERIFY(status >= LevenbergMarquardtSpace::RelativeReductionTooSmall &&
         status <= LevenbergMarquardtSpace::CosinusTooSmall);
  VERIFY(std::abs(x[0] - std::sqrt(2.0)) <= 256 * NumTraits<double>::epsilon());
}

struct EmptyRiddersFunctor : RiddersFunctor<double> {
  int values() const { return 0; }
};

void test_ridders_empty() {
  Ridders<EmptyRiddersFunctor> diff;
  VectorXd x = VectorXd::Ones(2), error;
  MatrixXd jac(0, 2);
  VERIFY_IS_EQUAL(diff.df(x, jac, error), 0);
  VERIFY_IS_EQUAL(error, VectorXd::Zero(2));
  VERIFY_IS_EQUAL(diff.functor().evaluations, 0);

  Ridders<RiddersFunctor<double>> noInputs;
  x.resize(0);
  jac.resize(4, 0);
  VERIFY_IS_EQUAL(noInputs.df(x, jac, error), 0);
  VERIFY_IS_EQUAL(error.size(), 0);
  VERIFY_IS_EQUAL(noInputs.functor().evaluations, 0);
}

EIGEN_DECLARE_TEST(ridders) {
  CALL_SUBTEST((test_ridders_jacobian<float, 2, 4, ColMajor>()));
  CALL_SUBTEST((test_ridders_jacobian<double, 2, 4, RowMajor>()));
  CALL_SUBTEST((test_ridders_jacobian<double, Dynamic, Dynamic, ColMajor>()));
  CALL_SUBTEST((test_ridders_jacobian<long double, Dynamic, Dynamic, RowMajor>()));

  CALL_SUBTEST(test_ridders_parameters());
  CALL_SUBTEST(test_ridders_repeated_stencil());
  CALL_SUBTEST(test_ridders_dimensions());

  CALL_SUBTEST(test_ridders_central_overflow<float>());
  CALL_SUBTEST(test_ridders_central_overflow<double>());

  CALL_SUBTEST(test_ridders_domain_and_noise());
  CALL_SUBTEST(test_ridders_richardson_overflow());
  CALL_SUBTEST(test_ridders_stencil());

  CALL_SUBTEST(test_ridders_composition());
  CALL_SUBTEST(test_ridders_step_alignment());
  CALL_SUBTEST(test_ridders_fallback());

  CALL_SUBTEST(test_ridders_levenberg_marquardt());
  CALL_SUBTEST(test_ridders_empty());
}
