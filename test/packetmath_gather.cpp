// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "packetmath_test_shared.h"
#include <memory>

// SVE's 32-bit gathers form the lane offsets the same way and still overflow; see pgather<float, PacketXf>.
#if defined(__unix__) && (defined(__LP64__) || defined(_LP64)) && !defined(EIGEN_VECTORIZE_SVE)
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#define EIGEN_TEST_LARGE_STRIDE_GATHER
#endif

namespace Eigen {
namespace test {

#ifdef EIGEN_TEST_LARGE_STRIDE_GATHER
// Lane i at from[i * stride] for a stride near the limit of 32-bit lane offsets (lanes - 1) * stride. Only the pages
// holding the lanes set in `mapped` are mapped, so the reserved range costs address space but no memory.
template <typename Scalar>
struct large_stride_lanes {
  Index stride;
  std::size_t bytes = 0;
  void* reserved = MAP_FAILED;
  Scalar* from = nullptr;

  large_stride_lanes(int lanes, Index stride_, std::uint32_t mapped = ~std::uint32_t(0)) : stride(stride_) {
    const std::uintptr_t page = std::uintptr_t(sysconf(_SC_PAGESIZE));
    const Index span = (lanes - 1) * numext::abs(stride);
    bytes = std::size_t(span + 1) * sizeof(Scalar) + page;
    reserved = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (reserved == MAP_FAILED) {
      std::cerr << "Skipping the gather test at stride " << stride << ": cannot reserve " << bytes << " bytes.\n";
      return;
    }
    from = static_cast<Scalar*>(reserved) + (stride < 0 ? span : 0);
    for (Index i = 0; i < lanes; ++i) {
      if (((mapped >> i) & 1) == 0) continue;
      void* lane_page = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(from + i * stride) & ~(page - 1));
      VERIFY_IS_EQUAL(mprotect(lane_page, page, PROT_READ | PROT_WRITE), 0);
      from[i * stride] = Scalar(i + 1);
    }
  }
  ~large_stride_lanes() {
    if (reserved != MAP_FAILED) VERIFY_IS_EQUAL(munmap(reserved, bytes), 0);
  }
};

// The largest stride whose lane offsets fit in an int, the next one, and its negation.
template <typename Packet>
std::vector<Index> large_strides() {
  const int last_lane = numext::maxi<int>(1, internal::unpacket_traits<Packet>::size - 1);
  const Index fits = Index(NumTraits<int>::highest()) / last_lane;
  return {fits, fits + 1, -(fits + 1)};
}

template <typename Scalar, typename Packet>
void pgather_pscatter_large_stride() {
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  if (PacketSize < 2) return;
  for (Index s : large_strides<Packet>()) {
    // Opaque, as in production: clang 14-20 merge _mm_set_pd(p[0x7fffffff], p[0]) into one load at a constant stride.
    volatile Index runtime_stride = s;
    const Index stride = runtime_stride;
    large_stride_lanes<Scalar> lanes(PacketSize, stride);
    if (lanes.from == nullptr) continue;
    Scalar actual[PacketSize], expected[PacketSize];
    for (Index i = 0; i < PacketSize; ++i) expected[i] = Scalar(i + 1);
    internal::pstoreu(actual, internal::pgather<Scalar, Packet>(lanes.from, stride));
    VERIFY(areEqualBits(expected, actual, PacketSize, false));
    for (Index i = 0; i < PacketSize; ++i) expected[i] = Scalar(2 * i + 3);
    internal::pscatter<Scalar, Packet>(lanes.from, internal::ploadu<Packet>(expected), stride);
    for (Index i = 0; i < PacketSize; ++i) actual[i] = lanes.from[i * stride];
    VERIFY(areEqualBits(expected, actual, PacketSize, false));
  }
}

#ifdef EIGEN_VECTORIZE_AVX512
// The masked forms, which the AVX-512 GEMM kernel uses on a strided result. The masked-off lanes stay unmapped, so
// touching one faults.
template <typename Scalar, typename Packet>
void masked_pgather_pscatter_large_stride() {
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  const auto mask = static_cast<typename internal::unpacket_traits<Packet>::mask_t>(0x5555);
  for (Index s : large_strides<Packet>()) {
    volatile Index runtime_stride = s;  // see pgather_pscatter_large_stride
    const Index stride = runtime_stride;
    large_stride_lanes<Scalar> lanes(PacketSize, stride, mask);
    if (lanes.from == nullptr) continue;
    Scalar actual[PacketSize], expected[PacketSize];
    for (Index i = 0; i < PacketSize; ++i) expected[i] = (i % 2 == 0) ? Scalar(i + 1) : Scalar(-1);
    internal::pstoreu(actual, internal::pgather(internal::pset1<Packet>(Scalar(-1)), lanes.from, stride, mask));
    VERIFY(areEqualBits(expected, actual, PacketSize, false));
    internal::pscatter(lanes.from, internal::pset1<Packet>(Scalar(0)), stride, mask);
    for (Index i = 0; i < PacketSize; i += 2) VERIFY_IS_EQUAL(lanes.from[i * stride], Scalar(0));
  }
}
#endif
#endif

template <typename Scalar, typename Packet>
struct runall<Scalar, Packet, false, false> {
  static void run() {
    constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
    auto samples = special_values<Scalar>();
    using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
    const Bits nan_payload = numext::bit_cast<Bits>(NumTraits<Scalar>::quiet_NaN()) | Bits(0x123);
    samples.push_back(numext::bit_cast<Scalar>(nan_payload));
    Scalar actual[PacketSize], expected[PacketSize];
    for (Index stride : {Index(-7), Index(-2), Index(-1), Index(0), Index(1), Index(2), Index(3), Index(7)}) {
      const Index span = (PacketSize - 1) * numext::abs(stride) + 1;
      for (Index offset = 0; offset < PacketSize; ++offset) {
        // Exact-size new[] lets sanitizers detect reads past the last gathered coefficient.
        std::unique_ptr<Scalar[]> storage(new Scalar[span + offset]);
        Scalar* data = storage.get() + offset;
        const Scalar* from = stride < 0 ? data + span - 1 : data;
        for (Index rotation = 0; rotation < Index(samples.size()); ++rotation) {
          for (Index i = 0; i < span; ++i) data[i] = samples[(i + rotation) % samples.size()];
          for (Index i = 0; i < PacketSize; ++i) expected[i] = from[i * stride];
          volatile Index runtime_stride = stride;
          internal::pstoreu(actual, internal::pgather<Scalar, Packet>(from, runtime_stride));
          VERIFY(areEqualBits(expected, actual, PacketSize, false));
          if (stride == 2) {
            internal::pstoreu(actual, internal::pgather<Scalar, Packet>(from, 2));
            VERIFY(areEqualBits(expected, actual, PacketSize, false));
          }
        }
      }
    }
#ifdef EIGEN_TEST_LARGE_STRIDE_GATHER
    pgather_pscatter_large_stride<Scalar, Packet>();
#endif
  }
};

}  // namespace test
}  // namespace Eigen

EIGEN_DECLARE_TEST(packetmath_gather) {
  CALL_SUBTEST_1(test::runner<float>::run());
  CALL_SUBTEST_2(test::runner<double>::run());
#ifdef EIGEN_TEST_LARGE_STRIDE_GATHER
  CALL_SUBTEST_1((test::pgather_pscatter_large_stride<int32_t, internal::packet_traits<int32_t>::type>()));
  CALL_SUBTEST_2((test::pgather_pscatter_large_stride<int64_t, internal::packet_traits<int64_t>::type>()));
#ifdef EIGEN_VECTORIZE_AVX512
  CALL_SUBTEST_1((test::masked_pgather_pscatter_large_stride<float, internal::Packet16f>()));
  CALL_SUBTEST_2((test::masked_pgather_pscatter_large_stride<double, internal::Packet8d>()));
#endif
#endif
}
