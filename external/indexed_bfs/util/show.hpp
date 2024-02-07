//
// `show()`: Wrap values for pretty printing to a `std::ostream`.
//
// Note: The same functionality can be implemented by directly overloading
// `operator<<()`, but it may cause unintentional overload resolution.
//
// Examples:
// ```
// std::cout << show(std::vector<int>({0, 1, 2}));
// ```
// ==> "{0, 1, 2}"
//
// ```
// std::cout << show(1);
// ```
// ==> "1" (same as "std::cout << 1")
//
// Authored by ARAI Junya <araijn@gmail.com> in 2011.
//
#pragma once

#include <iostream>
#include <tuple>

namespace indexed_bfs {
namespace util {
namespace show {
namespace detail {

template <typename T> struct shown {
  const T &x;
};

template <typename T>
std::ostream &operator<<(std::ostream &, const util::show::detail::shown<T>);

template <typename T> shown<T> show(const T &x) { return {x}; }

// If `T` is printable by default, `show` just returns the value
template <typename T, typename = decltype(std::cout << std::declval<T>())>
void pretty_print(const T &x, std::ostream *const os, int, ...) {
  *os << x;
}

template <typename T1, typename T2>
void pretty_print(const std::pair<T1, T2> &x, std::ostream *const os, ...) {
  *os << '(' << show(x.first) << ", " << show(x.second) << ')';
}

template <typename SinglePassRange>
void pretty_print(const SinglePassRange &x, std::ostream *const os, ...) {
  *os << '{';
  auto it = std::begin(x);
  const auto end = std::end(x);
  if (it != end) {
    *os << show(*it++);
    for (; it != end; ++it)
      *os << ", " << show(*it);
  }
  *os << '}';
}

//
// Pretty-print for std::tuple, retrived from
// http://stackoverflow.com/questions/6245735/pretty-print-stdtuple
//
template <std::size_t...> struct seq {};

template <std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <std::size_t... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <class Tuple, std::size_t... Is>
void print_tuple(Tuple const &t, seq<Is...>, std::ostream *const os) {
  using swallow = int[];
  *os << '(';
  (void)swallow{0,
                (void(*os << (Is ? ", " : "") << show(std::get<Is>(t))), 0)...};
  *os << ')';
}

template <typename... Types>
void pretty_print(const std::tuple<Types...> &x, std::ostream *const os, ...) {
  print_tuple(x, gen_seq<sizeof...(Types)>(), os);
}

// Overload `operator<<()` to print `show(...)`
template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const util::show::detail::shown<T> s) {
  util::show::detail::pretty_print(s.x, &os, 0);
  return os;
}

} // namespace detail

using detail::show;

} // namespace show
} // namespace util
} // namespace indexed_bfs
