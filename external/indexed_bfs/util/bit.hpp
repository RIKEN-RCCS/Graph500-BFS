//
// Facilitate bitwise operations.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-22.
//
#pragma once

#include "iterator.hpp"
#include <atomic>
#include <bitset>
#include <cassert>
#include <vector>

namespace indexed_bfs {
namespace util {
namespace bit {
namespace detail {

using namespace util;

////////////////////////////////////////////////////////////////////////////////
//
// Backporting from C++20 or later.
//
////////////////////////////////////////////////////////////////////////////////

// TODO: provide overloads for unsigned integers.
template <typename T> constexpr int countl_zero(T x) noexcept;

// TODO: provide overloads for unsigned integers.
template <typename T> constexpr int countl_one(T x) noexcept;

template <typename T> constexpr int countr_zero(T x) noexcept;

template <> constexpr int countr_zero(const unsigned int x) noexcept {
  return __builtin_ctz(x);
}

template <> constexpr int countr_zero(const unsigned long x) noexcept {
  return __builtin_ctzl(x);
}

template <> constexpr int countr_zero(const unsigned long long x) noexcept {
  return __builtin_ctzll(x);
}

// TODO: provide overloads for unsigned integers.
template <typename T> constexpr int countr_one(T x) noexcept;

template <typename T> constexpr int popcount(T x) noexcept;

template <> constexpr int popcount(const unsigned int x) noexcept {
  return __builtin_popcount(x);
}

template <> constexpr int popcount(const unsigned long x) noexcept {
  return __builtin_popcountl(x);
}

template <> constexpr int popcount(const unsigned long long x) noexcept {
  return __builtin_popcountll(x);
}

////////////////////////////////////////////////////////////////////////////////
//
// Iterators over `std::bitset`
//
////////////////////////////////////////////////////////////////////////////////

//
// Note: currently `N` greater than 64 is not supported.
//
template <size_t N> struct one_iterator {
  using this_type = one_iterator<N>;
  using iterator_category = std::input_iterator_tag;
  using difference_type = size_t;
  using value_type = size_t;
  using pointer = value_type *;
  using reference = const value_type &;

  std::bitset<N> b;
  size_t i;

  reference operator*() const { return this->i; }

  this_type &operator++() {
    this->b.reset(i);
    if (this->b.any()) {
      this->i = countr_zero(this->b.to_ullong());
    }
    return *this;
  }

  this_type operator++(int) {
    const auto tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const this_type &x, const this_type &y) {
    return x.b == y.b;
  }

  friend bool operator!=(const this_type &x, const this_type &y) {
    return !(x == y);
  }
};

template <size_t N>
using one_iterator_range = iterator::iterator_range<one_iterator<N>>;

template <size_t N> one_iterator<N> make_one_iterator(const std::bitset<N> b) {
  return one_iterator<N>{b, static_cast<size_t>(countr_zero(b.to_ullong()))};
}

template <size_t N> one_iterator_range<N> iterate_ones(const std::bitset<N> b) {
  return iterator::make_iterator_range(make_one_iterator(b),
                                       make_one_iterator(std::bitset<N>()));
}

////////////////////////////////////////////////////////////////////////////////
//
// Bitwise operations for unsigned integers
//
////////////////////////////////////////////////////////////////////////////////

template <typename Integral>
Integral with_bit(const Integral v, const size_t i) {
  static_assert(std::is_unsigned<Integral>::value, "");
  assert(i < sizeof(Integral) * 8);

  return v | (static_cast<Integral>(1) << i);
}

template <typename Integral>
Integral without_bit(const Integral v, const size_t i) {
  static_assert(std::is_unsigned<Integral>::value, "");
  assert(i < sizeof(Integral) * 8);

  return v & ~(static_cast<Integral>(1) << i);
}

template <typename Integral> bool get_bit(const Integral v, const size_t i) {
  static_assert(std::is_unsigned<Integral>::value, "");
  assert(i < sizeof(Integral) * 8);

  return v & (static_cast<Integral>(1) << i);
}

////////////////////////////////////////////////////////////////////////////////
//
// `bit_vector`: Thread-safe version of `std::vector<bool>`
//
////////////////////////////////////////////////////////////////////////////////

struct bit_vector {
  using element_type = uint64_t;
  static constexpr int element_bits = sizeof(element_type) * 8;

  std::vector<std::atomic<element_type>> vec;
  size_t bit_length;

  bool operator[](const size_t i) const {
    const auto i_elem = i / element_bits;
    const auto i_bit = i % element_bits;
    return get_bit(vec[i_elem].load(), i_bit);
  }

  void clear() { vec.clear(); }

  size_t count() const {
    size_t n = 0;
    for (const auto &e : vec) {
      n += popcount(e.load());
    }
    return n;
  }

  void set(const size_t i) {
    assert(i < this->bit_length);

    const auto i_elem = i / element_bits;
    const auto i_bit = i % element_bits;
    auto &e = vec[i_elem];

    for (;;) {
      auto expected = e.load();
      auto desired = with_bit(expected, i_bit);
      if (e.compare_exchange_strong(expected, desired)) {
        break;
      }
    }
  }

  size_t size() const { return bit_length; }
};

bit_vector make_bit_vector(const size_t bit_length) {
  const auto element_bits = sizeof(bit_vector::element_type) * 8;
  const auto n = (bit_length + element_bits - 1) / element_bits;
  return bit_vector{std::vector<std::atomic<bit_vector::element_type>>(n),
                    bit_length};
}

} // namespace detail

using detail::bit_vector;
using detail::countl_one;
using detail::countl_zero;
using detail::countr_one;
using detail::countr_zero;
using detail::get_bit;
using detail::iterate_ones;
using detail::make_bit_vector;
using detail::one_iterator;
using detail::popcount;
using detail::with_bit;
using detail::without_bit;

} // namespace bit
} // namespace util
} // namespace indexed_bfs
