// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#if EIGEN_MAX_ALIGN_BYTES > 0
#define ALIGNMENT EIGEN_MAX_ALIGN_BYTES
#else
#define ALIGNMENT 1
#endif

typedef Matrix<float, 16, 1> Vector16f;
typedef Matrix<float, 8, 1> Vector8f;

void check_handmade_aligned_malloc() {
  // Hand-make alignment needs at least sizeof(void*) to store the offset.
  constexpr int alignment = (std::max<int>)(EIGEN_DEFAULT_ALIGN_BYTES, sizeof(void *));

  for (int i = 1; i < 1000; i++) {
    char *p = (char *)internal::handmade_aligned_malloc(i, alignment);
    VERIFY(std::uintptr_t(p) % ALIGNMENT == 0);
    // if the buffer is wrongly allocated this will give a bad write --> check with valgrind
    for (int j = 0; j < i; j++) p[j] = 0;
    internal::handmade_aligned_free(p);
  }

  // A request that would exceed PTRDIFF_MAX bytes with its padding fails, and one within `alignment` of SIZE_MAX does
  // not wrap around to a tiny block.
  const std::size_t max_object = static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  for (std::size_t huge : {max_object - std::size_t(alignment) + 1, (std::numeric_limits<std::size_t>::max)()}) {
    char *p = static_cast<char *>(internal::handmade_aligned_malloc(huge, alignment));
    VERIFY(p == nullptr);
    internal::handmade_aligned_free(p);
    p = static_cast<char *>(internal::handmade_aligned_malloc(8, alignment));
    for (int j = 0; j < 8; j++) p[j] = char(j);
    char *q = static_cast<char *>(internal::handmade_aligned_realloc(p, huge, 8, alignment));
    VERIFY(q == nullptr);
    for (int j = 0; j < 8; j++) VERIFY_IS_EQUAL(p[j], char(j));
    internal::handmade_aligned_free(q == nullptr ? p : q);
  }
}

void check_aligned_malloc() {
  for (int i = ALIGNMENT; i < 1000; i++) {
    char *p = (char *)internal::aligned_malloc(i);
    VERIFY(std::uintptr_t(p) % ALIGNMENT == 0);
    // if the buffer is wrongly allocated this will give a bad write --> check with valgrind
    for (int j = 0; j < i; j++) p[j] = 0;
    internal::aligned_free(p);
  }
}

void check_aligned_new() {
  for (int i = ALIGNMENT; i < 1000; i++) {
    float *p = internal::aligned_new<float>(i);
    VERIFY(std::uintptr_t(p) % ALIGNMENT == 0);
    // if the buffer is wrongly allocated this will give a bad write --> check with valgrind
    for (int j = 0; j < i; j++) p[j] = 0;
    internal::aligned_delete(p, i);
  }
}

void check_aligned_stack_alloc() {
  for (int i = ALIGNMENT; i < 400; i++) {
    ei_declare_aligned_stack_constructed_variable(float, p, i, 0);
    VERIFY(std::uintptr_t(p) % ALIGNMENT == 0);
    // if the buffer is wrongly allocated this will give a bad write --> check with valgrind
    for (int j = 0; j < i; j++) p[j] = 0;
  }
}

// test compilation with both a struct and a class...
struct MyStruct {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  char dummychar;
  Vector16f avec;
};

class MyClassA {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  char dummychar;
  Vector16f avec;
};

template <typename T>
void check_dynaligned() {
  // TODO have to be updated once we support multiple alignment values
  if (T::SizeAtCompileTime % ALIGNMENT == 0) {
    T *obj = new T;
    VERIFY(T::NeedsToAlign == 1);
    VERIFY(std::uintptr_t(obj) % ALIGNMENT == 0);
    delete obj;
  }
}

template <typename T>
void check_custom_new_delete() {
  {
    T *t = new T;
    delete t;
  }

  {
    std::size_t N = internal::random<std::size_t>(1, 10);
    T *t = new T[N];
    delete[] t;
  }

  {
    T *t = new (std::nothrow) T;
    VERIFY(t != nullptr);
    delete t;
  }

  {
    std::size_t N = internal::random<std::size_t>(1, 10);
    T *t = new (std::nothrow) T[N];
    VERIFY(t != nullptr);
    delete[] t;
  }

#if EIGEN_MAX_ALIGN_BYTES > 0 && (!EIGEN_HAS_CXX17_OVERALIGN)
  {
    T *t = static_cast<T *>((T::operator new)(sizeof(T)));
    (T::operator delete)(t, sizeof(T));
  }

  {
    T *t = static_cast<T *>((T::operator new)(sizeof(T)));
    (T::operator delete)(t);
  }
#endif
}

template <int Alignment>
struct alignas(Alignment) ScratchScalar {
  static int alive;
  int value;
  ScratchScalar() : value(17) {
    VERIFY(std::uintptr_t(this) % Alignment == 0);
    ++alive;
  }
  ~ScratchScalar() { --alive; }
};
template <int Alignment>
int ScratchScalar<Alignment>::alive = 0;

template <int Alignment>
void check_typed_scratch() {
  using T = ScratchScalar<Alignment>;
  STATIC_CHECK((std::is_same<decltype(internal::scratch_malloc<T>(sizeof(T))), T *>::value));
  const Index sizes[] = {1, 3, EIGEN_STACK_ALLOCATION_LIMIT / sizeof(T) + 1};
  for (Index size : sizes) {
    T *raw = internal::scratch_malloc<T>(sizeof(T) * size);
    VERIFY(std::uintptr_t(raw) % Alignment == 0);
    VERIFY_IS_EQUAL(T::alive, 0);
    std::memset(static_cast<void *>(raw), 0, sizeof(T) * size);
    internal::scratch_free<T>(raw);
    {
      ei_declare_aligned_stack_constructed_variable(T, data, size, nullptr);
      VERIFY_IS_EQUAL(T::alive, size);
      for (Index i = 0; i < size; ++i) VERIFY_IS_EQUAL(data[i].value, 17);
    }
    VERIFY_IS_EQUAL(T::alive, 0);
    T *data = internal::scratch_new<T>(size);
    VERIFY_IS_EQUAL(T::alive, size);
    internal::scratch_delete(data, size);
    VERIFY_IS_EQUAL(T::alive, 0);
  }
  internal::scratch_free<T>(nullptr);
#ifdef EIGEN_EXCEPTIONS
  EIGEN_IF_CONSTEXPR (alignof(T) > EIGEN_STACK_ALIGN_BYTES &&
                      (EIGEN_STACK_ALIGN_BYTES != 0 || alignof(T) > alignof(std::max_align_t))) {
    const std::size_t overhead = alignof(T) - 1 + sizeof(void *);
    const std::size_t max_size = (std::numeric_limits<std::size_t>::max)();
    for (std::size_t bytes : {max_size, max_size - overhead + 1}) {
      bool threw = false;
      try {
        T *raw = internal::scratch_malloc<T>(bytes);
        internal::scratch_free<T>(raw);
      } catch (const std::bad_alloc &) {
        threw = true;
      }
      VERIFY(threw);
    }
  }
#endif
  {
    T supplied[2];
    {
      ei_declare_aligned_stack_constructed_variable(T, data, 2, supplied);
      VERIFY(data == supplied);
      VERIFY_IS_EQUAL(T::alive, 2);
    }
    VERIFY_IS_EQUAL(T::alive, 2);
  }
  VERIFY_IS_EQUAL(T::alive, 0);
#if defined(EIGEN_ALLOCA) && !defined(EIGEN_NO_ALLOCA)
  {
    T supplied[3];
    using Vector = Matrix<T, Dynamic, 1>;
    using View = Map<Vector, Unaligned>;
    View input(supplied, 3);
    internal::local_nested_eval_wrapper<View, 2, true> copy(input, nullptr);
    VERIFY(std::uintptr_t(copy.object.data()) % Alignment == 0);
    VERIFY_IS_EQUAL(copy.object(0).value, 17);
  }
  VERIFY_IS_EQUAL(T::alive, 0);
#endif
}

EIGEN_DECLARE_TEST(dynalloc) {
  CALL_SUBTEST(check_typed_scratch<16>());
  CALL_SUBTEST(check_typed_scratch<64>());
  CALL_SUBTEST(check_typed_scratch<512>());
  // low level dynamic memory allocation
  CALL_SUBTEST(check_handmade_aligned_malloc());
  CALL_SUBTEST(check_aligned_malloc());
  CALL_SUBTEST(check_aligned_new());
  CALL_SUBTEST(check_aligned_stack_alloc());

  for (int i = 0; i < g_repeat * 100; ++i) {
    CALL_SUBTEST(check_custom_new_delete<Vector4f>());
    CALL_SUBTEST(check_custom_new_delete<Vector2f>());
    CALL_SUBTEST(check_custom_new_delete<Matrix4f>());
    CALL_SUBTEST(check_custom_new_delete<MatrixXi>());
  }

// check static allocation, who knows ?
#if EIGEN_MAX_STATIC_ALIGN_BYTES
  for (int i = 0; i < g_repeat * 100; ++i) {
    CALL_SUBTEST(check_dynaligned<Vector4f>());
    CALL_SUBTEST(check_dynaligned<Vector2d>());
    CALL_SUBTEST(check_dynaligned<Matrix4f>());
    CALL_SUBTEST(check_dynaligned<Vector4d>());
    CALL_SUBTEST(check_dynaligned<Vector4i>());
    CALL_SUBTEST(check_dynaligned<Vector8f>());
    CALL_SUBTEST(check_dynaligned<Vector16f>());
  }

  {
    MyStruct foo0;
    VERIFY(std::uintptr_t(foo0.avec.data()) % ALIGNMENT == 0);
    MyClassA fooA;
    VERIFY(std::uintptr_t(fooA.avec.data()) % ALIGNMENT == 0);
  }

  // dynamic allocation, single object
  for (int i = 0; i < g_repeat * 100; ++i) {
    MyStruct *foo0 = new MyStruct();
    VERIFY(std::uintptr_t(foo0->avec.data()) % ALIGNMENT == 0);
    MyClassA *fooA = new MyClassA();
    VERIFY(std::uintptr_t(fooA->avec.data()) % ALIGNMENT == 0);
    delete foo0;
    delete fooA;
  }

  // dynamic allocation, array
  const int N = 10;
  for (int i = 0; i < g_repeat * 100; ++i) {
    MyStruct *foo0 = new MyStruct[N];
    VERIFY(std::uintptr_t(foo0->avec.data()) % ALIGNMENT == 0);
    MyClassA *fooA = new MyClassA[N];
    VERIFY(std::uintptr_t(fooA->avec.data()) % ALIGNMENT == 0);
    delete[] foo0;
    delete[] fooA;
  }
#endif
}
