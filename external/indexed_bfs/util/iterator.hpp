//
// Kind of itertools in C++14.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-30.
//
#pragma once

#include "types.hpp"
#include <iterator>

namespace indexed_bfs {
namespace util {
namespace iterator {
namespace detail {

using namespace indexed_bfs::util;

////////////////////////////////////////////////////////////////////////////////
//
// iterator_range
//
////////////////////////////////////////////////////////////////////////////////

//
// Range represented by the first and last iterators.
//
template <typename InputIterator> struct iterator_range {
  InputIterator first, last;

  InputIterator begin() { return first; }
  InputIterator begin() const { return first; }
  InputIterator end() { return last; }
  InputIterator end() const { return last; }

  //
  // Returns the distance between `first` and `last`.
  //
  // This function is defined only if `InputIterator` has `operator-()`, which
  // is required for `LegacyRandomAccessIterator`.
  // Thus, the complexity is expected to be O(1).
  //
  template <typename T = InputIterator>
  std::enable_if_t<types::has_sub_operator<T>::value,
                   typename std::iterator_traits<T>::difference_type>
  size() const {
    return last - first;
  }
};

template <typename InputIterator>
static iterator_range<InputIterator>
make_iterator_range(const InputIterator first, const InputIterator last) {
  return {first, last};
}

////////////////////////////////////////////////////////////////////////////////
//
// forward_iterator_builder
//
////////////////////////////////////////////////////////////////////////////////

//
// Build a forward iterator from the minimum definition.
//
// `Base` should have the following definitions:
// - `value_type`
// - Default constructor
// - `operator*()`
// - `operator++()`
// - `operator==()`
//
template <typename Base> struct forward_iterator_builder {
  using self_type = forward_iterator_builder<Base>;

  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = typename Base::value_type;
  using pointer = std::add_pointer_t<value_type>;
  using reference = std::add_lvalue_reference_t<value_type>;

  Base t;

  forward_iterator_builder() : t() {}

  template <typename... Args>
  forward_iterator_builder(Args... args) : t(std::forward<Args>(args)...) {}

  decltype(*t) operator*() { return *t; }

  template <typename T = Base>
  std::enable_if_t<
      std::is_lvalue_reference<decltype(*std::declval<T>())>::value, pointer>
  operator->() {
    return &*t;
  }

  self_type &operator++() {
    ++t;
    return *this;
  }

  self_type operator++(int) {
    self_type old = *this;
    ++(*this);
    return old;
  }

  bool operator==(const self_type &x) const { return t == x.t; }

  bool operator!=(const self_type &x) const { return !(*this == x); }
};

////////////////////////////////////////////////////////////////////////////////
//
// counting_iterator
//
////////////////////////////////////////////////////////////////////////////////

//
// Iterator for counting up a number.
//
// TODO: rewrite using iterator builder
//
template <typename Int> struct counting_iterator {
  using self_type = counting_iterator<Int>;
  using iterator_category = std::input_iterator_tag;
  using difference_type = Int;
  using value_type = const Int;
  using pointer = value_type *;
  using reference = value_type &;

  Int value;

  reference operator*() const { return this->value; }

  self_type &operator++() {
    this->value += 1;
    return *this;
  }

  self_type operator++(int) {
    const auto tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const self_type &x, const self_type &y) {
    return x.value == y.value;
  }

  friend bool operator!=(const self_type &x, const self_type &y) {
    return !(x == y);
  }
};

template <typename Int>
counting_iterator<Int> make_counting_iterator(const Int value) {
  return counting_iterator<Int>{value};
}

} // namespace detail

using detail::counting_iterator;
using detail::forward_iterator_builder;
using detail::iterator_range;
using detail::make_counting_iterator;
using detail::make_iterator_range;

} // namespace iterator
} // namespace util
} // namespace indexed_bfs
