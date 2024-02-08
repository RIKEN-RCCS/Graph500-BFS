//
// Authored by ARAI Junya <araijn@gmail.com> on 2024-02-13
//
#pragma once

#include <string>

namespace indexed_bfs {
namespace util {
namespace macro {
namespace detail {

//
// Returns a function name in the `__PRETTY_FUNCTION__` macro, including the
// namespace of that function.
//
static std::string
pretty_function_name_with_namespace(const char *const pretty_function) {
  const char *first = pretty_function;
  for (const char *p = pretty_function;; ++p) {
    if (*p == ' ') {
      first = p + 1;
    } else if (*p == '(') {
      return std::string(first, p);
    }
  }
}

} // namespace detail

using detail::pretty_function_name_with_namespace;

} // namespace macro
} // namespace util
} // namespace indexed_bfs
