// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_GUARD_PAGE_BUFFER_H
#define EIGEN_TEST_GUARD_PAGE_BUFFER_H

// Include after main.h.
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define EIGEN_TEST_HAS_GUARD_PAGES 1
#else
#define EIGEN_TEST_HAS_GUARD_PAGES 0
#endif

namespace Eigen {
namespace test {

// Room for max_bytes followed by a PROT_NONE page, so that an element range placed at the end of the room faults on
// any excess read or write. Where there is no mmap the room is plain heap memory and nothing faults.
class GuardPageBuffer {
 public:
  explicit GuardPageBuffer(std::size_t max_bytes) {
#if EIGEN_TEST_HAS_GUARD_PAGES
    page_ = std::size_t(sysconf(_SC_PAGESIZE));
    VERIFY(page_ > 0);
    const std::size_t data_pages = (std::max)(std::size_t(1), (max_bytes + page_ - 1) / page_);
    map_bytes_ = (data_pages + 1) * page_;
    end_ = data_pages * page_;
    raw_ = static_cast<char*>(mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    VERIFY(raw_ != MAP_FAILED);
    VERIFY_IS_EQUAL(mprotect(raw_ + end_, page_, PROT_NONE), 0);
#else
    end_ = max_bytes + 256;
    storage_.resize(end_);
    raw_ = storage_.data();
#endif
  }

  ~GuardPageBuffer() {
#if EIGEN_TEST_HAS_GUARD_PAGES
    VERIFY_IS_EQUAL(munmap(raw_, map_bytes_), 0);
#endif
  }

  // The last start aligned to `alignment` bytes from which n_elems fit before the guard page.
  template <typename Scalar>
  Scalar* place_at_end(std::size_t n_elems, std::size_t alignment = alignof(Scalar)) const {
    VERIFY(n_elems * sizeof(Scalar) <= end_);
    const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(raw_ + end_ - n_elems * sizeof(Scalar));
    return reinterpret_cast<Scalar*>(start - start % alignment);
  }

  GuardPageBuffer(const GuardPageBuffer&) = delete;
  GuardPageBuffer& operator=(const GuardPageBuffer&) = delete;

 private:
  char* raw_ = nullptr;
  std::size_t end_ = 0;
#if EIGEN_TEST_HAS_GUARD_PAGES
  std::size_t page_ = 0;
  std::size_t map_bytes_ = 0;
#else
  std::vector<char> storage_;
#endif
};

}  // namespace test
}  // namespace Eigen

#undef EIGEN_TEST_HAS_GUARD_PAGES

#endif  // EIGEN_TEST_GUARD_PAGE_BUFFER_H
