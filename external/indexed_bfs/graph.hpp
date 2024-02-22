//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-10-27.
//
#pragma once

#include "common.hpp"
#include "generator/graph_generator.h"
#include "net.hpp"

namespace indexed_bfs {
namespace graph {
namespace detail {

using namespace util;
using namespace util::types;
using namespace util::iterator;
using namespace common;
using util::memory::make_with_capacity;

////////////////////////////////////////////////////////////////////////////////
//
// vertex and global_vertex
//
////////////////////////////////////////////////////////////////////////////////

struct vertex_tag;
using vertex_int = int32_t;
using vertex = newtype<vertex_int, vertex_tag>;

struct global_vertex_tag;
using global_vertex_int = int64_t;
using global_vertex = newtype<global_vertex_int, global_vertex_tag>;

//
// Make a `vertex` from an integer.
//
// This function is useful for making a `vertex` from an integer type that is
// unsigned or larger than `vertex_int` (e.g., `size_t`).
//
template <typename Int> static vertex make_vertex_from(const Int x) {
  // `to_sig(x)` has no effect because it is called only if `Int` is signed
  assert(std::is_unsigned<Int>::value ||
         to_sig(x) >= std::numeric_limits<vertex_int>::min());
  assert(x <= std::numeric_limits<vertex_int>::max());
  return vertex{static_cast<vertex_int>(x)};
}

//
// Make a `global_vertex` from an integer.
//
// This function is useful for making a `vertex` from an integer type that is
// unsigned or larger than `vertex_int` (e.g., `size_t`).
//
template <typename Int>
static global_vertex make_global_vertex_from(const Int x) {
  // `to_sig(x)` has no effect because it is called only if `Int` is signed
  assert(std::is_unsigned<Int>::value ||
         to_sig(x) >= std::numeric_limits<global_vertex_int>::min());
  assert(x <= std::numeric_limits<global_vertex_int>::max());
  return global_vertex{static_cast<global_vertex_int>(x)};
}

////////////////////////////////////////////////////////////////////////////////
//
// struct edge
//
////////////////////////////////////////////////////////////////////////////////

struct edge;
static edge make_edge(const global_vertex source, const global_vertex target);

struct edge {
  packed_edge t;

  bool operator<(const edge &other) const {
    return source().t < other.source().t ||
           (source().t == other.source().t && target().t < other.target().t);
  }

  bool operator==(const edge &other) const {
    return t.v0_low == other.t.v0_low && t.v1_low == other.t.v1_low &&
           t.high == other.t.high;
  }

  global_vertex source() const { return global_vertex{get_v0_from_edge(&t)}; }

  global_vertex target() const { return global_vertex{get_v1_from_edge(&t)}; }

  edge reverse() const { return make_edge(target(), source()); }
};
static_assert(sizeof(edge) == 12, "");

static std::ostream &operator<<(std::ostream &os, const edge &e) {
  os << "(" << e.source().t << ", " << e.target().t << ")";
  return os;
}

static edge make_edge(const global_vertex source, const global_vertex target) {
  edge e;
  write_edge(&e.t, source.t, target.t);
  return e;
}

////////////////////////////////////////////////////////////////////////////////
//
// struct jag_array
//
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename OffsetType = size_t> struct jag_array {
  std::vector<OffsetType> offsets_;
  std::vector<T> values_;

  template <typename BidirectionalIt, typename Key, typename Value>
  static jag_array from_groups_by(const BidirectionalIt first,
                                  const BidirectionalIt last,
                                  const size_t n_slices, Key key, Value value) {
    if (first == last) {
      return jag_array<T, OffsetType>();
    }

    OffsetType offset = 0;
    decltype(key(*first)) k = 0;
    auto offsets = make_with_capacity<std::vector<OffsetType>>(n_slices + 1);
    auto values =
        make_with_capacity<std::vector<T>>(std::distance(first, last));

    offsets.push_back(0);
    for (BidirectionalIt it = first; it != last; ++it) {
      const auto k_next = key(*it);
      assert(k <= k_next);
      for (; k < k_next; ++k) {
        offsets.push_back(offset);
      }
      values.push_back(std::move(value(*it)));
      ++offset;
    }

    assert(offsets.size() < n_slices + 1);
    assert(to_sig(values.size()) == std::distance(first, last));

    while (offsets.size() < n_slices + 1) {
      offsets.push_back(offset);
    }

    jag_array ret;
    ret.offsets_ = std::move(offsets);
    ret.values_ = std::move(values);
    return ret;
  }

  size_t slice_count() const { return offsets_.size() - 1; }

  iterator_range<const T *> operator[](const size_t i) const {
    return make_iterator_range(&values_[offsets_[i]],
                               &values_[offsets_[i + 1]]);
  }

  iterator_range<T *> operator[](const size_t i) {
    return make_iterator_range(&values_[offsets_[i]],
                               &values_[offsets_[i + 1]]);
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Accessing `jag_array` as a Graph
//
////////////////////////////////////////////////////////////////////////////////

class vertex_iterator
    : public std::iterator<std::forward_iterator_tag, vertex> {
  vertex u_;

public:
  vertex_iterator(const vertex u) : u_(u) {}

  const vertex &operator*() const { return u_; }

  vertex &operator*() { return u_; }

  bool operator==(const vertex_iterator &other) const { return u_ == other.u_; }

  bool operator!=(const vertex_iterator &other) const {
    return !(*this == other);
  }

  iterator &operator++() {
    ++u_.t;
    return *this;
  }
};

template <typename T, typename OffsetType>
static size_t vertex_count(const jag_array<T, OffsetType> &g) {
  return g.slice_count();
}

template <typename T, typename OffsetType>
static size_t degree(const jag_array<T, OffsetType> &g, const vertex u) {
  return g[u.t].size();
}

template <typename T, typename OffsetType>
static vertex_iterator vertices_begin(const jag_array<T, OffsetType> &) {
  return vertex_iterator(vertex{0});
}

template <typename T, typename OffsetType>
static vertex_iterator vertices_end(const jag_array<T, OffsetType> &g) {
  return vertex_iterator(vertex{static_cast<vertex_int>(vertex_count(g))});
}

template <typename T, typename OffsetType>
static iterator::iterator_range<vertex_iterator>
vertices(const jag_array<T, OffsetType> &g) {
  return iterator::make_iterator_range(vertices_begin(g), vertices_end(g));
}

////////////////////////////////////////////////////////////////////////////////
//
// Utility functions
//
////////////////////////////////////////////////////////////////////////////////

static size_t global_vertex_count(const int scale) {
  return static_cast<size_t>(1) << scale;
}

static size_t global_edge_count(const int scale) {
  return global_vertex_count(scale) * common::edge_factor;
}

} // namespace detail

using detail::degree;
using detail::edge;
using detail::global_edge_count;
using detail::global_vertex;
using detail::global_vertex_count;
using detail::global_vertex_int;
using detail::jag_array;
using detail::make_edge;
using detail::make_vertex_from;
using detail::vertex;
using detail::vertex_count;
using detail::vertex_int;
using detail::vertex_iterator;
using detail::vertices;
using detail::vertices_begin;
using detail::vertices_end;

} // namespace graph

template <> MPI_Datatype net::detail::datatype_of<graph::global_vertex>() {
  return net::datatype_of<graph::global_vertex::inner_type>();
}

template <> MPI_Datatype net::detail::datatype_of<graph::vertex>() {
  return net::datatype_of<graph::vertex::inner_type>();
}

template <> MPI_Datatype net::detail::datatype_of<graph::edge>() {
  static MPI_Datatype t = MPI_DATATYPE_NULL;
  if (t == MPI_DATATYPE_NULL) {
    t = net::commit_contiguous(3, net::datatype_of<uint32_t>());
  }
  return t;
}

} // namespace indexed_bfs
