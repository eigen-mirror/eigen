// SPDX-FileCopyrightText: 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// SPDX-FileCopyrightText: 2010 Hauke Heibel <hauke.heibel@gmail.com>
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STL_CONTAINER_TEST_HELPERS_H
#define EIGEN_STL_CONTAINER_TEST_HELPERS_H

#include <cstdint>
#include <iterator>
#include <vector>

#include <Eigen/Geometry>

template <typename Container>
typename Container::reference stl_container_at(Container& container, std::size_t index,
                                               std::random_access_iterator_tag) {
  return container[index];
}

template <typename Container>
typename Container::reference stl_container_at(Container& container, std::size_t index,
                                               std::bidirectional_iterator_tag) {
  auto iterator = container.begin();
  std::advance(iterator, index);
  return *iterator;
}

template <typename Container>
typename Container::reference stl_container_at(Container& container, std::size_t index) {
  using Category = typename std::iterator_traits<typename Container::iterator>::iterator_category;
  return stl_container_at(container, index, Category{});
}

template <typename Derived>
const Derived& stl_container_coeffs(const Eigen::MatrixBase<Derived>& value) {
  return value.derived();
}

template <typename Scalar, int Dim, int Mode, int Options>
const typename Eigen::Transform<Scalar, Dim, Mode, Options>::MatrixType& stl_container_coeffs(
    const Eigen::Transform<Scalar, Dim, Mode, Options>& value) {
  return value.matrix();
}

template <typename Derived>
auto stl_container_coeffs(const Eigen::QuaternionBase<Derived>& value) -> decltype(value.coeffs()) {
  return value.coeffs();
}

template <typename Container>
void verify_stl_contiguity(const Container&) {}

template <typename Value, typename Allocator>
void verify_stl_contiguity(const std::vector<Value, Allocator>& values) {
  VERIFY(reinterpret_cast<std::uintptr_t>(&values[22]) ==
         reinterpret_cast<std::uintptr_t>(&values[21]) + sizeof(Value));
}

template <typename Container, typename... Initializers>
void check_stl_container_front(Container& values, Container& other, const typename Container::value_type& x,
                               const typename Container::value_type& y, const Initializers&... initializers) {
  values.front() = x;
  other.front() = other.back();
  VERIFY_IS_APPROX(other.front(), other.back());
  values = other;
  auto iterator = values.begin();
  auto other_iterator = other.begin();
  for (int i = 0; i < 20; ++i, ++iterator, ++other_iterator) VERIFY_IS_APPROX(*iterator, *other_iterator);

  values.resize(21, initializers...);
  values.back() = x;
  VERIFY_IS_APPROX(values.back(), x);
  values.resize(22, y);
  VERIFY_IS_APPROX(values.back(), y);
  values.push_back(x);
  VERIFY_IS_APPROX(values.back(), x);
}

template <typename Container>
void check_stl_container_indexed(Container& values, Container& other, const typename Container::value_type& x,
                                 const typename Container::value_type& y) {
  stl_container_at(values, 5) = x;
  stl_container_at(other, 6) = stl_container_at(values, 5);
  VERIFY_IS_APPROX(stl_container_at(other, 6), stl_container_at(values, 5));
  values = other;
  for (int i = 0; i < 20; ++i) VERIFY_IS_APPROX(stl_container_at(other, i), stl_container_at(values, i));

  values.resize(21);
  stl_container_at(values, 20) = x;
  VERIFY_IS_APPROX(stl_container_at(values, 20), x);
  values.resize(22, y);
  VERIFY_IS_APPROX(stl_container_at(values, 21), y);
  values.push_back(x);
  VERIFY_IS_APPROX(stl_container_at(values, 22), x);
  verify_stl_contiguity(values);

  // The old loop compared an address in the unchanged source, so it always appended 300 entries.
  for (int i = 0; i < 300; ++i) values.push_back(stl_container_at(other, i % other.size()));
  for (std::size_t i = 23; i < values.size(); ++i) {
    VERIFY(stl_container_coeffs(stl_container_at(values, i)) ==
           stl_container_coeffs(stl_container_at(other, (i - 23) % other.size())));
  }
}

template <template <typename> class Container, typename MatrixType>
void check_stl_container_matrix(const MatrixType& m) {
  const Index rows = m.rows(), cols = m.cols();
  const MatrixType x = MatrixType::Random(rows, cols), y = MatrixType::Random(rows, cols),
                   zero = MatrixType::Zero(rows, cols);
  Container<MatrixType> v(10, zero), w(20, y);
  check_stl_container_front(v, w, x, y, zero);
  Container<MatrixType> v2(10, zero), w2(20, y);
  check_stl_container_indexed(v2, w2, x, y);
}

template <template <typename> class Container, typename TransformType>
void check_stl_container_transform(const TransformType&) {
  using MatrixType = typename TransformType::MatrixType;
  const TransformType x(MatrixType::Random()), y(MatrixType::Random()), ti = TransformType::Identity();
  Container<TransformType> v(10, ti), w(20, y);
  check_stl_container_front(v, w, x, y, ti);
  Container<TransformType> v2(10), w2(20, y);
  check_stl_container_indexed(v2, w2, x, y);
}

template <template <typename> class Container, typename QuaternionType>
void check_stl_container_quaternion(const QuaternionType&) {
  using Coefficients = typename QuaternionType::Coefficients;
  const QuaternionType x(Coefficients::Random()), y(Coefficients::Random()), qi = QuaternionType::Identity();
  Container<QuaternionType> v(10, qi), w(20, y);
  check_stl_container_front(v, w, x, y, qi);
  Container<QuaternionType> v2(10, qi), w2(20, y);
  check_stl_container_indexed(v2, w2, x, y);
}

#endif  // EIGEN_STL_CONTAINER_TEST_HELPERS_H
