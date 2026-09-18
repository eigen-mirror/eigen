// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>

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

int main() {}
