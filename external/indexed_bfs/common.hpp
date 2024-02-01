//
// Provide definitions commonly used in this project.
// Note that project-agnostic definitions should be in the `util` directory.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-22.
//
#pragma once

#define OR_DIE(cond)                                                           \
  indexed_bfs::common::detail::or_die((cond), #cond, __FILE__, __LINE__)

// DO NOT INCLUDE ANY HEADERS IN `src/` except `src/utii/`.
#include "util/bit.hpp"
#include "util/iterator.hpp"
#include "util/log.hpp"
#include "util/memory.hpp"
#include "util/show.hpp"
#include "util/sort_parallel.hpp"
#include "util/time.hpp"
#include "util/types.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// The header of OpenMPI generates warnings
#if defined(__GNUC__) && !defined(__FUJITSU)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#pragma GCC diagnostic pop
#else
#include <mpi.h>
#endif

////////////////////////////////////////////////////////////////////////////////
//
// Macros
//
////////////////////////////////////////////////////////////////////////////////

// Make a struct noncopyable and movable.
//
// Note that use of macros should be avoided, and Boost achieve the same
// functionality by inheriting `boost::noncopyable`. However, inheriting another
// class prevents the subclass from aggregate initialization (`{ ... }`) until
// C++17. Therefore, we resort to a macro.
#define MOVE_ONLY(struct_name)                                                 \
  struct_name(const struct_name &) = delete;                                   \
  struct_name &operator=(const struct_name &) = delete;                        \
  struct_name(struct_name &&) = default;                                       \
  struct_name &operator=(struct_name &&) = default;

namespace indexed_bfs {
namespace common {
namespace detail {

////////////////////////////////////////////////////////////////////////////////
//
// Constant parameters
//
////////////////////////////////////////////////////////////////////////////////

constexpr uint64_t random_seed1 = 2;
constexpr uint64_t random_seed2 = 3;
constexpr int edge_factor = 16;

////////////////////////////////////////////////////////////////////////////////
//
// Debugging and error checking
//
////////////////////////////////////////////////////////////////////////////////

static void die() { MPI_Abort(MPI_COMM_WORLD, 1); }

// Used from the `OR_DIE` macro
static void or_die(const bool expected, const char *cond, const char *file,
                   const int line) {
  if (!expected) {
    std::cerr << "OR_DIE: condition not satisfied at " << file << ":" << line
              << ": " << cond << std::endl;
    die();
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Miscellaneous
//
////////////////////////////////////////////////////////////////////////////////

// Evenly splits the range of length `n_elems` into `n_parts` and returns a pair
// of the begging index and the ending index of the `nth`-th part.
static std::pair<size_t, size_t> split(const size_t n_elems,
                                       const size_t n_parts, const size_t nth) {
  const auto each = (n_elems + n_parts - 1) / n_parts;
  const auto first = each * nth;
  const auto last = std::min(n_elems, each * (nth + 1));
  return std::make_pair(first, last);
}

} // namespace detail

using detail::die;
using detail::edge_factor;
using detail::random_seed1;
using detail::random_seed2;
using detail::split;

} // namespace common
} // namespace indexed_bfs
