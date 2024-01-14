//
// Project-agnostic definitions regarding types.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-10-30.
//
#pragma once

#include <cassert>
#include <iterator>
#include <limits>
#include <type_traits>

namespace indexed_bfs {
namespace util {
namespace types {
namespace detail {

using namespace indexed_bfs;

////////////////////////////////////////////////////////////////////////////////
//
// Backports from C++17
//
////////////////////////////////////////////////////////////////////////////////

//
// `std::void_t`
//
// We employ the definition shown in
// https://en.cppreference.com/w/cpp/types/void_t.
//
template <typename... Ts> struct make_void { typedef void type; };

template <typename... Ts> using void_t = typename make_void<Ts...>::type;

////////////////////////////////////////////////////////////////////////////////
//
// Backports from C++20
//
////////////////////////////////////////////////////////////////////////////////

//
// `is_range`, a substitute for the `std::ranges::range` concept in C++20.
//
template <typename T, typename = void> struct is_range : std::false_type {};

template <typename T>
struct is_range<T, void_t<decltype(std::begin(std::declval<T>())),
                          decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T> constexpr bool is_range_v = is_range<T>::value;

//
// `std::ranges::iterator_t`
//
template <typename T, typename = void> struct range_iterator {};

template <typename T>
struct range_iterator<T, std::enable_if_t<is_range<T>::value>> {
  using type = decltype(std::begin(std::declval<T>()));
};

template <typename T> using range_iterator_t = typename range_iterator<T>::type;

//
// `std::ranges::range_value_t`
//
template <typename T>
using range_value_t =
    typename std::iterator_traits<range_iterator_t<T>>::value_type;

////////////////////////////////////////////////////////////////////////////////
//
// Property Checks
//
////////////////////////////////////////////////////////////////////////////////

//
// Checks if `T` has an operator `<`.
//
template <typename T, typename = void>
struct has_lt_operator : std::false_type {};

template <typename T>
struct has_lt_operator<T,
                       void_t<decltype(std::declval<T>() < std::declval<T>())>>
    : std::true_type {};

//
// Checks if `T` has an operator `<=`.
//
template <typename T, typename = void>
struct has_le_operator : std::false_type {};

template <typename T>
struct has_le_operator<T,
                       void_t<decltype(std::declval<T>() <= std::declval<T>())>>
    : std::true_type {};

//
// Checks if `T` has an operator `>`.
//
template <typename T, typename = void>
struct has_gt_operator : std::false_type {};

template <typename T>
struct has_gt_operator<T,
                       void_t<decltype(std::declval<T>() > std::declval<T>())>>
    : std::true_type {};

//
// Checks if `T` has an operator `>=`.
//
template <typename T, typename = void>
struct has_ge_operator : std::false_type {};

template <typename T>
struct has_ge_operator<T,
                       void_t<decltype(std::declval<T>() >= std::declval<T>())>>
    : std::true_type {};

//
// Checks if `T` has an operator `==`.
//
template <typename T, typename = void>
struct has_eq_operator : std::false_type {};

template <typename T>
struct has_eq_operator<T,
                       void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};

////////////////////////////////////////////////////////////////////////////////
//
// wrapper
//
////////////////////////////////////////////////////////////////////////////////

//
// Wrapper for strong typedef.
// `Tag` should be a unique type for distinguishing strongly-typedef'ed type.
//
template <typename T, typename Tag> class wrapper {
public:
  using inner_type = T;
  using tag_type = Tag;

  T t; // Name is derived from `BOOST_STRONG_TYPEDEF`

  wrapper() {}

  wrapper(const T &_t) : t(std::move(_t)) {}

  template <typename U = T>
  std::enable_if_t<has_lt_operator<U>::value, bool>
  operator<(const wrapper<T, Tag> &x) const {
    return t < x.t;
  }

  template <typename U = T>
  std::enable_if_t<has_le_operator<U>::value, bool>
  operator<=(const wrapper<T, Tag> &x) const {
    return t <= x.t;
  }

  template <typename U = T>
  std::enable_if_t<has_gt_operator<U>::value, bool>
  operator>(const wrapper<T, Tag> &x) const {
    return t > x.t;
  }

  template <typename U = T>
  std::enable_if_t<has_ge_operator<U>::value, bool>
  operator>=(const wrapper<T, Tag> &x) const {
    return t >= x.t;
  }

  template <typename U = T>
  std::enable_if_t<has_eq_operator<U>::value, bool>
  operator==(const wrapper<T, Tag> &x) const {
    return t == x.t;
  }

  template <typename U = T>
  std::enable_if_t<has_eq_operator<U>::value, bool>
  operator!=(const wrapper<T, Tag> &x) const {
    return !(t == x.t);
  }
};

template <typename T, typename Tag>
std::ostream &operator<<(std::ostream &os, const wrapper<T, Tag> &w) {
  os << w.t;
  return os;
}

////////////////////////////////////////////////////////////////////////////////
//
// Miscellaneous
//
////////////////////////////////////////////////////////////////////////////////

//
// Base class to prohibit a copy. (cf. `boost::noncopyable`)
//
class noncopyable {
public:
  noncopyable() = default;
  noncopyable(const noncopyable &) = delete;
  noncopyable(noncopyable &&) = default;
  noncopyable &operator=(const noncopyable &) = delete;
  noncopyable &operator=(noncopyable &&) = default;
};

//
// Converts an integer to a signed integer.
//
template <typename Unsigned>
static constexpr auto to_sig(const Unsigned x) -> std::make_signed_t<Unsigned> {
  using signed_type = std::remove_cv_t<std::make_signed_t<Unsigned>>;
  assert(x <= std::numeric_limits<signed_type>::max());
  return static_cast<signed_type>(x);
}

//
// Converts an integer to an unsigned integer.
//
template <typename Signed>
static constexpr auto to_unsig(const Signed x) -> std::make_unsigned_t<Signed> {
  using unsigned_type = std::remove_cv_t<std::make_unsigned_t<Signed>>;
  assert(x >= 0);
  return static_cast<unsigned_type>(x);
}

} // namespace detail

using detail::is_range;
using detail::is_range_v;
using detail::noncopyable;
using detail::range_iterator;
using detail::range_iterator_t;
using detail::range_value_t;
using detail::to_sig;
using detail::to_unsig;
using detail::wrapper;

} // namespace types
} // namespace util
} // namespace indexed_bfs
