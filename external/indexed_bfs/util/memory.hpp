//
// Memory management.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-22.
//
#pragma once

#include <vector>

namespace indexed_bfs {
namespace util {
namespace memory {
namespace detail {

template <typename Container>
static Container make_with_capacity(const size_t n) {
  Container c;
  c.reserve(n);
  return c;
}

//
// Returns the heap memory size in bytes consumed by the given object.
//
// This function is not declared as
// ```
// template <typename T> static size_t heap_size_of(const T &x)
// ```
// and defined by partial specialization because it requires overloads for every
// template type with a specific template parameters (e.g., `std::vector<int>`
// instead of `std::vector<T>`).
//
template <typename T> static size_t heap_size_of(const std::vector<T> &x) {
  return sizeof(T) * x.capacity();
}

} // namespace detail

using detail::heap_size_of;
using detail::make_with_capacity;

} // namespace memory
} // namespace util
} // namespace indexed_bfs
