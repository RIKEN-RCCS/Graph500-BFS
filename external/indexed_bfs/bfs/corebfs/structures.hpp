//
// Compressed sparse array for storing parents of the tree vertices.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2024-01-11.
//
#pragma once

#include "../../argument.hpp"
#include "../../common.hpp"
#include "../../graph.hpp"
#include "../../graph500.hpp"
#include "../../net.hpp"
#include <array>

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace structures {
namespace detail {

using namespace graph;
using namespace util;
using indexed_bfs::common::int48;
using indexed_bfs::common::make_int48;
using indexed_bfs::util::memory::heap_size_of;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

////////////////////////////////////////////////////////////////////////////////
//
// parent_array
//
////////////////////////////////////////////////////////////////////////////////

struct parent_array;

static void test_parent_array(const parent_array &result,
                              const std::vector<global_vertex> &original);

template <size_t N>
static size_t count_ones_before(const std::bitset<N> &b, const size_t pos) {
  assert(pos < N);
  // Example: N = 8, b = 0b01101001, pos = 6
  //  ~std::bitset<N>()                   = 0x11111111
  //  ~std::bitset<N>() >> (N - pos)      = 0x00111111
  // (~std::bitset<N>() >> (N - pos)) & b = 0x00101001
  // ==> result: 3
  return ((~std::bitset<N>() >> (N - pos)) & b).count();
}

struct parent_array : types::noncopyable {
public:
  constexpr static size_t word_bits = 64;

  class iterator : public std::iterator<std::forward_iterator_tag,
                                        std::pair<vertex, global_vertex>> {
    const parent_array &me_;
    vertex u_;

  public:
    iterator(const parent_array &me, const vertex u) : me_(me), u_(u) {}

    value_type operator*() const {
      assert(me_.contains(u_));
      return std::make_pair(u_, me_[u_]);
    }

    bool operator==(const iterator &other) const {
      return u_ == other.u_ && &me_ == &other.me_;
    }

    bool operator!=(const iterator &other) const { return !(*this == other); }

    iterator &operator++() {
      u_ = next_vertex(me_, u_);
      assert(u_.t <= to_sig(me_.size()));
      return *this;
    }

  private:
    static vertex next_vertex(const parent_array &me, const vertex u) {
      size_t i = u.t / word_bits;
      const size_t j = u.t % word_bits;

      // Cannot advance
      if (u.t >= to_sig(me.size()) || i >= me.metas_.size()) {
        return make_vertex_from(me.size());
      }

      // Advance to the next vertex in the current metadata block
      const auto rest_contains = me.metas_[i].contains >> (j + 1);
      if (rest_contains.any()) {
        return vertex(u.t + 1 + bit::countr_zero(rest_contains.to_ullong()));
      }

      // Search subsequent blocks for a present vertex
      while (i + 1 < me.metas_.size()) {
        ++i;
        const auto &m = me.metas_[i];
        if (m.contains.any()) {
          const auto j = bit::countr_zero(m.contains.to_ullong());
          return vertex(i * word_bits + j);
        }
      }

      // Reached the end
      return make_vertex_from(me.size());
    }
  };

  struct metadata {
    std::bitset<word_bits> contains;
    uint32_t prefix_sum;
  };

private:
  // Number of vertices in this array.
  // `operator[](u)` returns a value for `u` s.t. `0 <= u.t < size_`.
  size_t size_;
  // Metadata for each block of `word_bits` (maybe 64) vertices.
  // The metadata for vertex `u` is stored in `metas_[u.t / word_bits]`.
  std::vector<metadata> metas_;
  // Parents of vertices that have parents.
  std::vector<int48> parents_;

public:
  parent_array() : size_(), metas_(), parents_() {}
  parent_array(parent_array &&) = default;
  parent_array &operator=(parent_array &&) = default;

  parent_array(const std::vector<global_vertex> &parents)
      : size_(parents.size()), metas_(), parents_() {
    //
    // Initialize metas_
    //
    metas_.resize((size_ + word_bits - 1) / word_bits);

    uint32_t sum = 0;
    for (size_t i = 0; i < metas_.size(); ++i) {
      uint64_t b = 0;
      for (size_t j = 0; j < 64 && i * 64 + j < size_; ++j) {
        const uint64_t t = parents[i * 64 + j].t >= 0 ? 1 : 0;
        b |= t << j;
      }
      metas_[i] = {b, sum};
      sum += bit::popcount(b);
    }

    //
    // Initialize parents_
    //
    parents_.resize(sum);
    size_t n_parents = 0;
    for (const auto u : parents) {
      if (u.t >= 0) {
        parents_[n_parents] = make_int48(u.t);
        ++n_parents;
      }
    }

    test_parent_array(*this, parents);
  }

  //
  // Returns a parent (non-negative vertex ID) if `u` is in the tree; otherwise,
  // returns `vertex{-1}`.
  //
  global_vertex operator[](const vertex u) const {
    assert(u.t >= 0);
    assert(u.t < to_sig(size_));

    const size_t i = u.t / word_bits;
    const size_t j = u.t % word_bits;
    if (metas_[i].contains[j]) {
      const size_t n = count_ones_before(metas_[i].contains, j);
      return global_vertex(parents_[metas_[i].prefix_sum + n].get());
    } else {
      return global_vertex(-1);
    }
  }

  iterator begin() const {
    iterator it(*this, vertex{0});
    if (!contains(vertex{0})) {
      ++it;
    }
    return it;
  }

  iterator end() const { return iterator(*this, make_vertex_from(size_)); }

  bool contains(const vertex u) const {
    assert(u.t >= 0);
    assert(u.t < to_sig(size_));

    return metas_[u.t / word_bits].contains[u.t % word_bits];
  }

  //
  // Decompress the parent array and callback for each element in parallel.
  //
  template <typename F> void for_each_parallel(F f) const {
#pragma omp parallel for
    for (size_t i_meta = 0; i_meta < metas_.size(); ++i_meta) {
      auto *const m = &metas_[i_meta];
      size_t i_parent = m->prefix_sum;

      for (const size_t bit_pos : bit::iterate_ones(m->contains)) {
        const vertex child = make_vertex_from(i_meta * word_bits + bit_pos);
        const global_vertex parent(parents_[i_parent].get());
        f(child, parent);
        ++i_parent;
      }
    }
  }

  //
  // Decompress the parent array into a simple array in parallel.
  //
  void dump(global_vertex *const parents) const {
#pragma omp parallel for
    for (size_t i_meta = 0; i_meta < metas_.size(); ++i_meta) {
      auto *const m = &metas_[i_meta];
      size_t i_parent = m->prefix_sum;

      for (const size_t bit_pos : bit::iterate_ones(m->contains)) {
        const auto child = make_vertex_from(i_meta * word_bits + bit_pos);
        const auto parent = global_vertex(parents_[i_parent].get());
        parents[child.t] = parent;
        ++i_parent;
      }
    }
  }

  //
  // Applies a function to the parent of every vertices contained in this array.
  //
  // Mapper: `global_vertex m(vertex child, global_vertex parent)`
  //
  template <typename Mapper> void map(Mapper m) {
    size_t i = 0;
    for (const auto p : *this) {
      const vertex child = p.first;
      const global_vertex parent = p.second;
      parents_[i] = make_int48(m(child, parent).t);
      ++i;
    }
  }

  //
  // Retains parents of vertices that `pred` returned `true`, and removes the
  // others.
  //
  // Predicate: `bool pred(vertex child, global_vertex parent)`
  //
  template <typename Predicate> void retain(Predicate pred) {
    size_t i_get = 0;
    size_t i_put = 0;
    size_t n_removed = 0;

    for (size_t i_meta = 0; i_meta < metas_.size(); ++i_meta) {
      auto *const m = &metas_[i_meta];
      assert(m->prefix_sum == i_get);
      m->prefix_sum -= n_removed;

      for (const size_t bit_pos : bit::iterate_ones(m->contains)) {
        const auto child = make_vertex_from(i_meta * word_bits + bit_pos);
        const auto parent = global_vertex(parents_[i_get].get());
        if (pred(child, parent)) {
          parents_[i_put] = parents_[i_get];
          ++i_put;
        } else {
          m->contains.reset(bit_pos);
          ++n_removed;
        }
        ++i_get;
      }
    }
  }

  size_t size() const { return size_; }

  friend size_t heap_size_of(const parent_array &m) {
    return memory::heap_size_of(m.metas_) + heap_size_of(m.parents_);
  }
};

static void test_parent_array(const parent_array &result,
                              const std::vector<global_vertex> &original) {
  // Suppress the unused warning when `NDEBUG` is defined
  static_cast<void>(result);
  static_cast<void>(original);

  // Test:
  // All the parents obtained from `parent_array` must be equal to that of
  // `parents`.
  assert([&]() {
    for (vertex u{0}; u.t < to_sig(original.size()); ++u.t) {
      if (result[u] != original[u.t]) {
        return false;
      }
    }
    return true;
  }());

  // The iterator must returns all the valid parent values
  assert([&]() {
    std::vector<global_vertex> restored(result.size(), global_vertex{-1});
    for (const auto p : result) {
      restored[p.first.t] = p.second;
    }
    return std::equal(restored.begin(), restored.end(), original.begin(),
                      original.end());
  }());
}

} // namespace detail

using detail::parent_array;

} // namespace structures
} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
