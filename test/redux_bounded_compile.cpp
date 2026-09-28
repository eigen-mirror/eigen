// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>
#include <functional>

// Scalar only: InnerProduct's packet-enabled unit-stride remap loses this product's bound.
#ifdef EIGEN_DONT_VECTORIZE
using BoundedVector4d = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 4, 1>;
using BoundedMatrix4d = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 4>;

Eigen::MatrixXd bounded_transform(const Eigen::MatrixXd& vertices, const BoundedMatrix4d& transform) {
  Eigen::MatrixXd transformed(vertices.rows(), vertices.cols());
  for (Eigen::Index col = 0; col < vertices.cols(); ++col) {
    BoundedVector4d homogeneous = BoundedVector4d::Ones(vertices.rows() + 1);
    homogeneous.head(vertices.rows()) = vertices.col(col);
    transformed.col(col) = (transform * homogeneous).head(vertices.rows());
  }
  return transformed;
}
#endif

using BoundedVector3d = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;

std::function<Eigen::VectorXd(Eigen::Index)> evaluate_position;

template <int Mode>
EIGEN_ALWAYS_INLINE double bounded_residual(const BoundedVector3d& target, Eigen::Index size) {
  const Eigen::VectorXd position = evaluate_position(size);
  EIGEN_IF_CONSTEXPR (Mode == 0) {
    return (position - target.head(size)).norm();
  } else EIGEN_IF_CONSTEXPR (Mode == 1) {
    return Eigen::numext::sqrt((position - target.head(size)).array().square().sum());
  } else {
    return (position.array() - target.head(size).array()).matrix().norm();
  }
}

template <int Mode>
Eigen::VectorXd bounded_residuals(const Eigen::MatrixXd& points, Eigen::Index size) {
  Eigen::VectorXd result(points.cols());
  for (Eigen::Index i = 0; i < points.cols(); ++i)
    result(i) = bounded_residual<Mode>(BoundedVector3d(points.col(i)), size);
  return result;
}

int main() {
  evaluate_position = [](Eigen::Index size) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(size); };
  for (Eigen::Index size = 1; size <= 3; ++size) {
    const Eigen::MatrixXd points = Eigen::MatrixXd::Ones(size, 2);
    for (const Eigen::VectorXd& result :
         {bounded_residuals<0>(points, size), bounded_residuals<1>(points, size), bounded_residuals<2>(points, size)}) {
      for (Eigen::Index i = 0; i < result.size(); ++i)
        if (Eigen::numext::abs(result(i) - Eigen::numext::sqrt(double(size))) > 1e-12) return 1;
    }
  }
}
