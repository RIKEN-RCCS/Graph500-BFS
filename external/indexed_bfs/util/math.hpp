//
// Authored by ARAI Junya <araijn@gmail.com> on 2024-02-13
//
#pragma once

#include "bit.hpp"
#include <type_traits>

namespace indexed_bfs {
namespace util {
namespace math {
namespace detail {

//
// Incomplete backport of `std::gcd()`
//
template <typename M, typename N>
static std::common_type_t<M, N> gcd(const M m, const N n) {
  if (n == 0) {
    return m;
  } else {
    return gcd(n, m % n);
  }
}

//
// Incomplete backport of `std::lcm()`
//
template <typename M, typename N>
static std::common_type_t<M, N> lcm(const M m, const N n) {
  return m / gcd(m, n) * n;
}

template <typename M, typename N>
static std::enable_if_t<std::is_integral<M>::value &&
                            std::is_integral<N>::value,
                        std::common_type_t<M, N>>
div_ceil(const M divend, const N divisor) {
  return (divend + divisor - 1) / divisor;
}

static int log2_floor(const uint64_t x) {
  assert(x > 0);
  return 63 - bit::countr_zero(x);
}

} // namespace detail

using detail::div_ceil;
using detail::gcd;
using detail::lcm;
using detail::log2_floor;

} // namespace math
} // namespace util
} // namespace indexed_bfs
