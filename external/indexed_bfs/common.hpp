//
// Provide definitions commonly used in this project.
// Note that project-agnostic definitions should be in the `util` directory.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-22.
//
#pragma once

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
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tl/optional.hpp>
#include <unistd.h>
#include <vector>

#define OR_DIE(cond)                                                           \
  indexed_bfs::common::detail::or_die((cond), #cond, __FILE__, __LINE__)

#define LOG_RSS()                                                              \
  do {                                                                         \
    const auto gds = indexed_bfs::common::summarize_rss_gb();                  \
    LOG_I << "Five-number summary of RSSs in GiB: " << show::show(gds);        \
  } while (false)

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

using size_t = std::size_t;

namespace indexed_bfs {
namespace common {
namespace detail {

using namespace indexed_bfs::util;
using indexed_bfs::util::types::to_unsig;

template <typename InputIterator>
auto five_number_summary(const InputIterator first, const InputIterator last)
    -> std::array<typename std::iterator_traits<InputIterator>::value_type, 5>;

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
// Platform Information
//
////////////////////////////////////////////////////////////////////////////////

struct platform_info {
  bool fugaku;
};

const platform_info &platform() {
  static platform_info info = {
// fugaku
#ifdef FUGAKU
      true
#else
      false
#endif
  };

  return info;
}

////////////////////////////////////////////////////////////////////////////////
//
// Integers of Various Bit-Lengths
//
////////////////////////////////////////////////////////////////////////////////

//
// All of these integers are POD types, and so cannot have a constructor.
// Use `make_*()` functions instead (e.g., `make_uint48()`).
//

//
// 40-bit unsigned integer.
//
struct __attribute__((packed)) uint40 {
  static constexpr uint64_t max() { return 0xfffffffff; }
  static constexpr uint64_t min() { return 0; }
  static constexpr uint40 zero();

  uint32_t low;
  uint8_t high;

  bool operator==(const uint40 &x) const {
    return low == x.low && high == x.high;
  }

  bool operator<(const uint40 &x) const { return get() < x.get(); }

  uint64_t get() const { return (static_cast<uint64_t>(high) << 32) | low; }
};
static_assert(sizeof(uint40) == 5, "");

constexpr uint40 uint40::zero() { return {0, 0}; }

static uint40 make_uint40(const uint64_t x) {
  const uint32_t low = static_cast<uint32_t>(x);
  const uint8_t high = static_cast<uint8_t>(x >> 32);
  return {low, high};
}

//
// 40-bit signed integer.
//
struct int40 {
  static constexpr int64_t max() { return 0x7fffffffff - 1; }
  static constexpr int64_t min() { return -0x7fffffffff; }
  static constexpr int40 zero();

  uint40 u;

  bool operator==(const int40 &x) const { return u == x.u; }

  bool operator<(const int40 &x) const { return u < x.u; }

  int64_t get() const {
    // Safe to cast to `int64_t` because `u.get()` < 2^40
    const int64_t y = static_cast<int64_t>(u.get());
    return y <= max() ? y : y - 0xffffffffff;
  }
};
static_assert(sizeof(int40) == 5, "");

constexpr int40 int40::zero() { return {uint40::zero()}; }

int40 make_int40(const int64_t x) {
  assert(x >= int40::min());
  assert(x <= int40::max());

  const uint64_t y = x >= 0 ? to_unsig(x) : to_unsig(x + 0xffffffffff);
  return {make_uint40(y)};
}

//
// 48-bit unsigned integer.
//
struct __attribute__((packed)) uint48 {
  static constexpr uint64_t max() { return 0xffffffffffff; }
  static constexpr uint64_t min() { return 0; }
  static constexpr uint48 zero();

  uint32_t low;
  uint16_t high;

  uint64_t get() const { return (static_cast<uint64_t>(high) << 32) | low; }
};
static_assert(sizeof(uint48) == 6, "");

constexpr uint48 uint48::zero() { return {0, 0}; }

uint48 make_uint48(const uint64_t x) {
  const uint32_t low = static_cast<uint32_t>(x);
  const uint16_t high = static_cast<uint16_t>(x >> 32);
  return {low, high};
}

//
// 48-bit signed integer.
//
struct int48 {
  static constexpr int64_t max() { return 0x7fffffffffff - 1; }
  static constexpr int64_t min() { return -0x7fffffffffff; }
  static constexpr int48 zero();

  uint48 u;

  int64_t get() const {
    // Safe to cast to `int64_t` because `u.get()` < 2^48
    const int64_t y = static_cast<int64_t>(u.get());
    return y < 0x7fffffffffff ? y : y - 0xffffffffffff;
  }
};
static_assert(sizeof(int48) == 6, "");

constexpr int48 int48::zero() { return {uint48::zero()}; }

int48 make_int48(const int64_t x) {
  assert(x >= int48::min());
  assert(x <= int48::max());

  const uint64_t y = x >= 0 ? to_unsig(x) : to_unsig(x + 0xffffffffffff);
  return {make_uint48(y)};
}

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

//
// Returns a RSS in bytes.
//
// Example of `status`:
// - `VmRSS:       864 kB`
// - `HugetlbPages:          0 kB`
//
static size_t parse_memory_status(const std::string &status) {
  std::istringstream ss(status);
  std::string ign, unit;
  size_t rss;

  ss >> ign;  // Discard "VmRSS" or "HugetlbPages"
  ss >> rss;  // Human-readable size
  ss >> unit; // "kB", "MB", "GB", ...
  if (ss.fail()) {
    LOG_E << "Invalid format: " << status;
    die();
    return 0;
  }

  if (unit == "kB") {
    return rss * 1024;
  } else if (unit == "MB") {
    return rss * 1024 * 1024;
  } else if (unit == "GB") {
    return rss * 1024 * 1024 * 1024;
  } else {
    LOG_E << "Unsupported unit: " << unit;
    die();
    return 0;
  }
}

static size_t read_rss() {
  std::ifstream f("/proc/self/status");
  OR_DIE(f.is_open());

  std::string ln;
  size_t rss = 0;
  while (std::getline(f, ln)) {
    if (ln.rfind("VmRSS", 0) == 0) {
      rss += parse_memory_status(ln);
    } else if (platform().fugaku && ln.rfind("HugetlbPages", 0) == 0) {
      rss += parse_memory_status(ln);
    }
  }

  return rss;
}

static std::array<size_t, 5> summarize_rss() {
  const unsigned long rss = read_rss();

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  std::vector<unsigned long> recv(nproc);

  MPI_Gather(&rss, 1, MPI_UNSIGNED_LONG, recv.data(), 1, MPI_UNSIGNED_LONG, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    return five_number_summary(recv.begin(), recv.end());
  } else {
    return std::array<size_t, 5>{};
  }
}

static std::array<double, 5> summarize_rss_gb() {
  const std::array<size_t, 5> bytes = summarize_rss();

  std::array<double, 5> gbs;
  std::transform(bytes.begin(), bytes.end(), gbs.begin(),
                 [](auto b) { return b / (1024 * 1024 * 1024.0); });

  return gbs;
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

template <typename InputIterator>
auto five_number_summary(const InputIterator first, const InputIterator last)
    -> std::array<typename std::iterator_traits<InputIterator>::value_type, 5> {
  using value_type = typename std::iterator_traits<InputIterator>::value_type;

  std::vector<value_type> vec(first, last);
  sort_parallel::sort_parallel(vec.begin(), vec.end());

  const auto n = vec.size();
  const auto q0 = vec[0];
  const auto q1 = vec[static_cast<size_t>(n * 0.25)];
  const auto q2 = vec[static_cast<size_t>(n * 0.50)];
  const auto q3 = vec[static_cast<size_t>(n * 0.75)];
  const auto q4 = vec[n - 1];

  return {q0, q1, q2, q3, q4};
}

template <typename Container>
auto five_number_summary(const Container &c)
    -> std::array<types::range_value_t<Container>, 5> {
  return five_number_summary(std::begin(c), std::end(c));
}

} // namespace detail

using detail::die;
using detail::edge_factor;
using detail::five_number_summary;
using detail::int40;
using detail::int48;
using detail::make_int40;
using detail::make_int48;
using detail::make_uint40;
using detail::make_uint48;
using detail::platform;
using detail::random_seed1;
using detail::random_seed2;
using detail::split;
using detail::summarize_rss_gb;
using detail::uint40;
using detail::uint48;

} // namespace common
} // namespace indexed_bfs
