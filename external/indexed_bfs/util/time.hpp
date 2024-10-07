//
// Utilities for time measurement.
//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-07-18.
//
#pragma once

#include "macro.hpp"
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
  const auto indexed_bfs_entry##__COUNTER__ =                                  \
      indexed_bfs::util::time::detail::make_entry(__PRETTY_FUNCTION__,         \
                                                  scope_name, []() {})

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
template <typename UniqueTag> struct entry;
static void print_all_scopes();

static std::vector<record *> g_records;

struct record {
  const char *const pretty_function;
  const char *const scope_name;
  std::atomic<uint64_t> total_nanos; // Total duration in nanoseconds
  std::atomic<uint64_t> entry_count;

  record(const char *const _pretty_function, const char *const _scope_name)
      : pretty_function(_pretty_function), scope_name(_scope_name),
        total_nanos(), entry_count() {
    g_records.push_back(this);
  }
};

template <typename UniqueTag>
static entry<UniqueTag> make_entry(const char *const pretty_function,
                                   const char *const scope_name, UniqueTag) {
  static record r(pretty_function, scope_name);
  return entry<UniqueTag>(&r);
}

template <typename UniqueTag> class entry {
  record *const record_;
  std::chrono::high_resolution_clock::time_point start_;

public:
  ~entry() {
    const uint64_t ns = std::chrono::nanoseconds(now() - start_).count();
    record_->total_nanos.fetch_add(ns);
    record_->entry_count.fetch_add(1);
  }

private:
  entry(record *const r) : record_(r), start_(now()) {}

  friend void print_all_scopes();

  template <typename T>
  friend entry<T> make_entry(const char *const pretty_function,
                             const char *const scope_name, T);
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
    const double secs = 0.000000001 * r->total_nanos;
    const std::string func =
        macro::pretty_function_name_with_namespace(r->pretty_function);
    ss << "- {function: '" << func;
    if (r->scope_name == nullptr) {
      ss << "', scope: null";
    } else {
      ss << "', scope: '" << r->scope_name << '\'';
    }
    ss << ", entry_count: " << r->entry_count;
    ss << ", total_secs: " << secs;
    ss << ", average_secs: " << secs / r->entry_count << "}\n";
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
