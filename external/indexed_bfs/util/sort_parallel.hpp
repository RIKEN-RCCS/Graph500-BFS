//
// Parallel sort based on OpenMP.
// Note: In C++17 or later, use `std::sort` with the parallel execution policy
// instead of this code.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-12-28.
//
#pragma once

#include <algorithm>
#include <omp.h>
#include <vector>

namespace indexed_bfs {
namespace util {
namespace sort_parallel {
namespace detail {

template <typename RandomAccessIterator, typename Compare>
static void sort_parallel(const RandomAccessIterator first,
                          const RandomAccessIterator last, Compare comp) {
  const int n_threads = omp_get_max_threads();
  const size_t n = std::distance(first, last);
  size_t segment_len = (n + n_threads - 1) / n_threads;

  // Sort each segment
#pragma omp parallel for
  for (size_t left = 0; left < n; left += segment_len) {
    const size_t right = std::min(left + segment_len, n);
    std::sort(first + left, first + right, comp);
  }

  // Repeatedly merge two segments
  for (; segment_len < n; segment_len *= 2) {
#pragma omp parallel for
    for (size_t left = 0; left < n; left += 2 * segment_len) {
      const size_t mid = std::min(left + segment_len, n);
      const size_t right = std::min(left + 2 * segment_len, n);
      std::inplace_merge(first + left, first + mid, first + right, comp);
    }
  }
}

template <typename RandomAccessIterator>
static void sort_parallel(const RandomAccessIterator first,
                          const RandomAccessIterator last) {
  sort_parallel(first, last, [](auto &x, auto &y) { return x < y; });
}

} // namespace detail

using detail::sort_parallel;

} // namespace sort_parallel
} // namespace util
} // namespace indexed_bfs
