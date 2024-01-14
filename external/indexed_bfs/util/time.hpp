//
// Utilities for time measurement.
//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-07-18.
//
#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>

//
// Define a timed scope.
// The total times consumed in each scope are shown by `print_all_scopes()`.
//
#define INDEXED_BFS_TIMED_SCOPE(scope_name)                                    \
  static indexed_bfs::util::time::detail::record record_##scope_name(          \
      #scope_name);                                                            \
  const indexed_bfs::util::time::detail::entry scope_##scope_name(             \
      &record_##scope_name)

namespace indexed_bfs {
namespace util {
namespace time {
namespace detail {

static std::chrono::high_resolution_clock::time_point now() {
  return std::chrono::high_resolution_clock::now();
}

static double
elapsed_secs(const std::chrono::high_resolution_clock::time_point &origin) {
  return std::chrono::duration<double>(now() - origin).count();
}

template <typename F>
static auto time(F f)
    -> std::enable_if_t<std::is_same<decltype(f()), void>::value, double> {
  const auto s = now();
  f();
  return elapsed_secs(s);
}

template <typename F>
static auto time(F f)
    -> std::enable_if_t<!std::is_same<decltype(f()), void>::value,
                        std::pair<double, decltype(f())>> {
  const auto s = now();
  auto x = f();
  return std::make_pair(elapsed_secs(s), std::move(x));
}

////////////////////////////////////////////////////////////////////////////////
//
// Definitions for INDEXED_BFS_TIMED_SCOPE
//
////////////////////////////////////////////////////////////////////////////////

struct record;

static std::vector<record *> g_records;

struct record {
  const char *const name;
  double total_secs;
  size_t entry_count;

  record(const char *const _name) : name(_name), total_secs(), entry_count() {
    g_records.push_back(this);
  }
};

struct entry {
  record *const target;
  std::chrono::high_resolution_clock::time_point start;

  entry(record *const _target) : target(_target), start(now()) {}

  ~entry() {
    target->total_secs += elapsed_secs(start);
    target->entry_count += 1;
  }
};

//
// Prints information of all the timed scopes to stderr.
// This is useful for calling at exit with `std::atexit()`.
//
static void print_all_scopes() {
  // Make a buffer to avoid interleaving outputs from multiple threads
  std::stringstream ss;
  ss << "timed_scopes:\n";

  for (const auto &r : g_records) {
    ss << "  ";
    ss << r->name << ": {total_secs: " << r->total_secs;
    ss << ", entry_count: " << r->entry_count;
    ss << ", average_secs: " << r->total_secs / r->entry_count << "}\n";
  }

  std::cerr << ss.str();
}

} // namespace detail

using detail::elapsed_secs;
using detail::now;
using detail::print_all_scopes;
using detail::time;

} // namespace time
} // namespace util
} // namespace indexed_bfs
