//
// Kind of itertools in C++14.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-30.
//
#pragma once

#include <iterator>

namespace indexed_bfs {
namespace util {
namespace iterator {
namespace detail {

//
// Range represented by the first and last iterators.
//
template <typename InputIterator> struct iterator_range {
  InputIterator first, last;

  InputIterator begin() { return first; }
  InputIterator begin() const { return first; }
  InputIterator end() { return last; }
  InputIterator end() const { return last; }

  // Returns the distance between `first` and `last`.
  //
  // If `InputIterator` meets the requirements of `LegacyRandomAccessIterator`,
  // the complexity is constant; otherwise, it is linear.
  typename std::iterator_traits<InputIterator>::difference_type size() {
    return std::distance(first, last);
  }
};

template <typename InputIterator>
static iterator_range<InputIterator>
make_iterator_range(const InputIterator first, const InputIterator last) {
  return {first, last};
}

//
// Iterator for counting up a number.
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
using detail::iterator_range;
using detail::make_counting_iterator;
using detail::make_iterator_range;

} // namespace iterator
} // namespace util
} // namespace indexed_bfs
