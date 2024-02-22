//
// Parallel sort based on OpenMP.
// Note: In C++17 or later, use `std::sort` with the parallel execution policy
// instead of this code.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-12-28.
//
#pragma once

#include "bit.hpp"
#include "math.hpp"
#include <algorithm>
#include <future>
#include <omp.h>
#include <vector>

namespace indexed_bfs {
namespace util {
namespace sort {
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

template <typename RandomAccessIterator>
RandomAccessIterator median_iterator(RandomAccessIterator a,
                                     RandomAccessIterator b,
                                     RandomAccessIterator c) {
  if (*a < *b) {
    if (*b < *c)
      return b; // a < b < c
    else if (*a < *c)
      return c; // a < c <= b
    else
      return a; // c <= a < b
  } else {
    if (*a < *c)
      return a; // b <= a < c
    else if (*b < *c)
      return c; // b < c <= a
    else
      return b; // c <= b <= a
  }
}

template <typename RandomAccessIterator, typename BinaryFunction>
struct compact_parallel_sorter {
  BinaryFunction is_less;
  const size_t cutoff_len;

  compact_parallel_sorter(BinaryFunction _is_less, const size_t _cutoff_len)
      : is_less(std::move(_is_less)), cutoff_len(_cutoff_len) {}

  void go(const RandomAccessIterator first, const RandomAccessIterator last,
          const int remaining_recs) {
    const auto n = last - first;
    if (n <= to_sig(cutoff_len) || remaining_recs == 0) {
      std::sort(first, last, is_less);
      return;
    }

    // Choose the median as a pivot
    auto p = median_iterator(first, first + n / 2, last - 1);

    // Move the pivot to the end of the range
    std::iter_swap(p, last - 1);
    p = last - 1;

    const auto mid =
        std::partition(first, last - 1, [p](const auto &x) { return x < *p; });
    std::iter_swap(mid, p);

#pragma omp task
    go(first, mid, remaining_recs - 1);
#pragma omp task
    go(mid + 1, last, remaining_recs - 1);
  }
};

template <typename RandomAccessIterator, typename BinaryFunction>
static void
compact_parallel_sort(const RandomAccessIterator first,
                      const RandomAccessIterator last, BinaryFunction is_less,
                      const size_t cutoff_len, const int max_recursions) {
  compact_parallel_sorter<RandomAccessIterator, BinaryFunction> sorter(
      std::move(is_less), cutoff_len);

#pragma omp parallel
#pragma omp single nowait
#pragma omp taskgroup
  sorter.go(first, last, max_recursions);

  assert(std::is_sorted(first, last, is_less));
}

template <typename RandomAccessIterator, typename BinaryFunction>
static void compact_parallel_sort(const RandomAccessIterator first,
                                  const RandomAccessIterator last,
                                  BinaryFunction is_less) {
  // Needs to be tuned.
  const size_t cutoff_len =
      65536 /
      sizeof(typename std::iterator_traits<RandomAccessIterator>::value_type);
  const int max_recursions =
      math::log2_floor(std::thread::hardware_concurrency() * 8);
  compact_parallel_sort(first, last, is_less, cutoff_len, max_recursions);
}

template <typename RandomAccessIterator>
static void compact_parallel_sort(const RandomAccessIterator first,
                                  const RandomAccessIterator last) {
  compact_parallel_sort(
      first, last,
      std::less<
          typename std::iterator_traits<RandomAccessIterator>::value_type>());
}

} // namespace detail

using detail::compact_parallel_sort;
using detail::sort_parallel;

} // namespace sort
} // namespace util
} // namespace indexed_bfs
