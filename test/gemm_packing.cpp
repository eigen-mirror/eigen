// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Real cache blocking lets these products exceed the stack allocation limit.
#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "main.h"
#include <Eigen/Core>

#ifdef EIGEN_TEST_WRAP_MALLOC
static volatile bool count_packing_allocations = false;
static volatile Index packing_allocations = 0;
// Preserve the linker's --wrap symbols without reserved C++ identifiers.
extern "C" void* real_malloc(std::size_t) asm("__real_malloc");
extern "C" void* wrap_malloc(std::size_t) asm("__wrap_malloc");
extern "C" void* wrap_malloc(std::size_t bytes) {
  if (count_packing_allocations) packing_allocations = packing_allocations + 1;
  return real_malloc(bytes);
}
#endif

template <typename LhsScalar, typename RhsScalar>
void packing_storage() {
  const std::size_t sizeA = EIGEN_STACK_ALLOCATION_LIMIT / sizeof(LhsScalar) + 3;
  const std::size_t sizeB = EIGEN_STACK_ALLOCATION_LIMIT / sizeof(RhsScalar) + 7;
  internal::gemm_packing_space<LhsScalar, RhsScalar> packing(sizeA, sizeB, nullptr, nullptr);
  VERIFY(packing.blockA() != nullptr);
  VERIFY(packing.blockB() != nullptr);
  const auto first = reinterpret_cast<std::uintptr_t>(packing.blockA());
  const auto second = reinterpret_cast<std::uintptr_t>(packing.blockB());
  const std::size_t alignment = (std::max)(std::size_t(EIGEN_STACK_ALIGN_BYTES), std::size_t(alignof(RhsScalar)));
  VERIFY_IS_EQUAL(second - first, (sizeA * sizeof(LhsScalar) + alignment - 1) / alignment * alignment);
  VERIFY_IS_EQUAL(first % alignof(LhsScalar), std::uintptr_t(0));
  VERIFY_IS_EQUAL(second % alignment, std::uintptr_t(0));
  packing.blockA()[0] = LhsScalar(1);
  packing.blockA()[sizeA - 1] = LhsScalar(2);
  packing.blockB()[0] = RhsScalar(3);
  packing.blockB()[sizeB - 1] = RhsScalar(4);
  VERIFY_IS_EQUAL(packing.blockA()[0], LhsScalar(1));
  VERIFY_IS_EQUAL(packing.blockA()[sizeA - 1], LhsScalar(2));

  std::vector<LhsScalar> suppliedA(sizeA);
  std::vector<RhsScalar> suppliedB(sizeB);
  internal::gemm_packing_space<LhsScalar, RhsScalar> supplied(sizeA, sizeB, suppliedA.data(), suppliedB.data());
  VERIFY(supplied.blockA() == suppliedA.data());
  VERIFY(supplied.blockB() == suppliedB.data());
  internal::gemm_packing_space<LhsScalar, RhsScalar> partial(sizeA, sizeB, suppliedA.data(), nullptr);
  VERIFY(partial.blockA() == suppliedA.data());
  VERIFY(partial.blockB() == nullptr);
#if defined(EIGEN_ALLOCA) && !defined(EIGEN_NO_ALLOCA) && EIGEN_STACK_ALLOCATION_LIMIT > 0
  internal::gemm_packing_space<LhsScalar, RhsScalar> small(1, 1, nullptr, nullptr);
  VERIFY(small.blockA() == nullptr);
  VERIFY(small.blockB() == nullptr);
  internal::gemm_packing_space<LhsScalar, RhsScalar> mixed(1, sizeB, nullptr, nullptr);
  VERIFY(mixed.blockA() == nullptr);
  VERIFY(mixed.blockB() == nullptr);
#endif
}

// Stricter than the alignment scratch_malloc guarantees, so either panel of this type keeps its own allocation.
constexpr std::size_t kScratchAlignment = EIGEN_STACK_ALIGN_BYTES > 0 ? EIGEN_STACK_ALIGN_BYTES
                                                                      : alignof(std::max_align_t);
struct alignas(2 * kScratchAlignment) OverAlignedScalar {
  double value = 0;
};

template <typename LhsScalar, typename RhsScalar>
void packing_alignment_fallback() {
  const std::size_t sizeA = EIGEN_STACK_ALLOCATION_LIMIT / sizeof(LhsScalar) + 3;
  const std::size_t sizeB = EIGEN_STACK_ALLOCATION_LIMIT / sizeof(RhsScalar) + 7;
  internal::gemm_packing_space<LhsScalar, RhsScalar> packing(sizeA, sizeB, nullptr, nullptr);
  VERIFY(packing.blockA() == nullptr);
  VERIFY(packing.blockB() == nullptr);
}

struct PackingScalar {
  static Index live;
  static Index remaining;
  int value = 0;
  PackingScalar() {
#ifdef EIGEN_EXCEPTIONS
    if (remaining == 0) throw std::bad_alloc();
#endif
    --remaining;
    ++live;
  }
  ~PackingScalar() { --live; }
};
Index PackingScalar::live = 0;
Index PackingScalar::remaining = 0;

namespace Eigen {
template <>
struct NumTraits<PackingScalar> : GenericNumTraits<PackingScalar> {
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

#ifdef EIGEN_EXCEPTIONS
template <typename LhsScalar, typename RhsScalar>
bool packing_throws_bad_alloc(std::size_t sizeA, std::size_t sizeB) {
  try {
    internal::gemm_packing_space<LhsScalar, RhsScalar> packing(sizeA, sizeB, nullptr, nullptr);
  } catch (const std::bad_alloc&) {
    return true;
  }
  return false;
}
#endif

void packing_lifetimes() {
  const std::size_t size = EIGEN_STACK_ALLOCATION_LIMIT / sizeof(PackingScalar) + 3;
  PackingScalar::remaining = Index(2 * size);
  {
    internal::gemm_packing_space<PackingScalar, PackingScalar> packing(size, size, nullptr, nullptr);
    VERIFY_IS_EQUAL(PackingScalar::live, Index(2 * size));
  }
  VERIFY_IS_EQUAL(PackingScalar::live, Index(0));
#ifdef EIGEN_EXCEPTIONS
  // Throw while constructing A's first element, A's second, B's first, and B's second; size >= 3 even when
  // EIGEN_STACK_ALLOCATION_LIMIT is 0.
  for (Index count : {Index(0), Index(1), Index(size), Index(size + 1)}) {
    PackingScalar::remaining = count;
    VERIFY((packing_throws_bad_alloc<PackingScalar, PackingScalar>(size, size)));
    VERIFY_IS_EQUAL(PackingScalar::live, Index(0));
  }
  const std::size_t max_bytes = (std::numeric_limits<std::ptrdiff_t>::max)();
  for (std::size_t sizeA : {max_bytes, max_bytes / 2})
    VERIFY((packing_throws_bad_alloc<char, char>(sizeA, max_bytes / 2 + 2)));
#endif
}

// Runs product(); where malloc is wrapped, returns whether it allocated at most once.
template <typename Product>
bool allocates_at_most_once(Product product) {
#ifdef EIGEN_TEST_WRAP_MALLOC
  packing_allocations = 0;
  count_packing_allocations = true;
#endif
  product();
#ifdef EIGEN_TEST_WRAP_MALLOC
  count_packing_allocations = false;
  return packing_allocations <= 1;
#else
  return true;
#endif
}

template <typename Scalar, int Order>
void packing_products() {
  Eigen::setNbThreads(1);
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Real = typename NumTraits<Scalar>::Real;
  for (Index n : {Index(145), Index(201)}) {
    const MatrixType a = MatrixType::Random(n, n + 3);
    const MatrixType b = MatrixType::Random(n + 3, n + 1);
    MatrixType result = MatrixType::Zero(n, n + 1);
    MatrixType reference = a.lazyProduct(b);
    VERIFY(allocates_at_most_once([&] { result.noalias() += a * b; }));
    const Real gemm_bound = Real(8) * Real(n + 3) * NumTraits<Real>::epsilon() * a.norm() * b.norm();
    VERIFY((result - reference).norm() <= gemm_bound);

    result.setZero(n, n);
    reference = a.lazyProduct(a.adjoint());
    const Real syrk_bound = Real(8) * Real(n + 3) * NumTraits<Real>::epsilon() * a.squaredNorm();
    VERIFY(allocates_at_most_once([&] { result.template selfadjointView<Lower>().rankUpdate(a); }));
    reference.template triangularView<StrictlyUpper>().setZero();
    VERIFY((result - reference).norm() <= syrk_bound);
    result.setZero();
    result.template selfadjointView<Upper>().rankUpdate(a.conjugate());
    reference = a.conjugate().lazyProduct(a.transpose());
    reference.template triangularView<StrictlyLower>().setZero();
    VERIFY((result - reference).norm() <= syrk_bound);
  }
}

EIGEN_DECLARE_TEST(gemm_packing) {
  CALL_SUBTEST((packing_storage<float, double>()));
  CALL_SUBTEST((packing_storage<double, std::complex<double>>()));
  CALL_SUBTEST((packing_alignment_fallback<OverAlignedScalar, float>()));
  CALL_SUBTEST((packing_alignment_fallback<float, OverAlignedScalar>()));
  CALL_SUBTEST(packing_lifetimes());
  CALL_SUBTEST((packing_products<float, ColMajor>()));
  CALL_SUBTEST((packing_products<double, RowMajor>()));
  CALL_SUBTEST((packing_products<std::complex<double>, ColMajor>()));
  CALL_SUBTEST((packing_products<std::complex<double>, RowMajor>()));
}
