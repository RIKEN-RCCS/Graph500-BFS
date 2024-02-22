//
// Authored by ARAI Junya <araijn@gmail.com> in 2024-02-02.
//
#pragma once

#include "../../common.hpp"
#include "../../graph.hpp"
#include <cmath>

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace distribution {
namespace detail {

using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using namespace indexed_bfs::common;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

////////////////////////////////////////////////////////////////////////////////
//
// source_vertex and target_vertex
//
////////////////////////////////////////////////////////////////////////////////

struct source_vertex_tag {};
using source_vertex = types::newtype<int64_t, source_vertex_tag>;

struct target_vertex_tag {};
using target_vertex = types::newtype<int64_t, target_vertex_tag>;

template <typename Int> source_vertex make_source_vertex_from(const Int x) {
  return static_cast<source_vertex::inner_type>(x);
}

template <typename Int> target_vertex make_target_vertex_from(const Int x) {
  return static_cast<target_vertex::inner_type>(x);
}

////////////////////////////////////////////////////////////////////////////////
//
// yoo
//
////////////////////////////////////////////////////////////////////////////////

//
// We employ Yoo's distribution.
// The following figure shows the example for R = 4 and C = 3.
// A *unit* is the smallest vertex set of distribution.
// A unit ID for global vertex `u` is obtained by `unit(u)`.
// Vertices are ordered in each unit by `index_in_unit(u)`.
//
// The rank of P(i, j) in the universal communicator is `i + j * column_.size()`
// (i.e., column major) (derived from the Fugaku impl).
//
//                 --- Row rank 0 ---- --- Row rank 1 ---- --- Row rank 2 ----
//      Unit ID ->   0 .  1 .  2 .  3 .  4 .  5 .  6 .  7 .  8 .  9 . 10 . 11
//                +----+----+----+----+----+----+----+----+----+----+----+----+
//              0 |   P(0, 0) =  0    |   P(0, 1) =  4    |   P(0, 2) =  8    |
//                + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              1 |   P(1, 0) =  1    |   P(1, 1) =  5    |   P(1, 2) =  9    |
// Row cycle 0    + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              2 |   P(2, 0) =  2    |   P(2, 1) =  6    |   P(2, 2) = 10    |
//                + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              3 |   P(3, 0) =  3    |   P(3, 1) =  7    |   P(3, 2) = 11    |
//    - - -       +----+----+----+----+----+----+----+----+----+----+----+----+
//              4 |   P(0, 0) =  0    |   P(0, 1) =  4    |   P(0, 2) =  8    |
//                + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              5 |   P(1, 0) =  1    |   P(1, 1) =  5    |   P(1, 2) =  9    |
// Row cycle 1    + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              6 |   P(2, 0) =  2    |   P(2, 1) =  6    |   P(2, 2) = 10    |
//                + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              7 |   P(3, 0) =  3    |   P(3, 1) =  7    |   P(3, 2) = 11    |
//    - - -       +----+----+----+----+----+----+----+----+----+----+----+----+
//              8 |   P(0, 0) =  0    |   P(0, 1) =  4    |   P(0, 2) =  8    |
//                + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//              9 |   P(1, 0) =  1    |   P(1, 1) =  5    |   P(1, 2) =  9    |
// Row cycle 2    + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//             10 |   P(2, 0) =  2    |   P(2, 1) =  6    |   P(2, 2) = 10    |
//                + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- + -- +
//             11 |   P(3, 0) =  3    |   P(3, 1) =  7    |   P(3, 2) = 11    |
//    Unit ID --^ +----+----+----+----+----+----+----+----+----+----+----+----+
//
// Be careful to confusing notations:
//   # of rows          = column_.size() (= R = 3)
//   # of columns       = row_.size()    (= C = 4)
//   i (vertical pos)   = column_.rank()
//   j (horizontal pos) = row_.rank()
// These values are not rephrased in the code to eliminate ambiguity in a
// natural language (e.g., does "row_size" mean whether the size of the row
// communicator or the number of rows?).
//
class yoo : types::noncopyable {
  int scale_;
  net::comm all_;
  net::comm row_;
  net::comm column_;
  uint32_t unit_size_;

public:
  yoo(const int scale, net::comm all, net::comm row, net::comm column)
      : scale_(scale), all_(std::move(all)), row_(std::move(row)),
        column_(std::move(column)) {
    assert(all_.size() == row_.size() * column_.size());
    assert(all_.rank() == column_.rank() + column_.size() * row_.rank());

    const size_t n = graph::global_vertex_count(scale_);
    unit_size_ = (n + all_.size() - 1) / all_.size();
  }

  yoo(yoo &&) = default;
  yoo &operator=(yoo &&) = default;

  const net::comm &all() const { return all_; }
  const net::comm &row() const { return row_; }
  const net::comm &column() const { return column_; }

  //
  // The number of vertices computed from the scale
  //
  size_t global_vertex_count() const {
    return graph::global_vertex_count(scale_);
  }

  size_t local_vertex_count() const {
    const auto n_global = global_vertex_count();
    if (to_unsig(all_.rank()) < n_global % all_.size()) {
      return n_global / all_.size() + 1;
    } else {
      return n_global / all_.size();
    }
  }

  // TODO: unify `local_vertex_count` and `unit_size`
  size_t unit_size() const {
    assert(local_vertex_count() <= unit_size_);
    return unit_size_;
  }

  //
  // [Vertex distribution]
  // Returns a rank to which `u` is distributed.
  //
  int owner_rank(const global_vertex u) const {
    return static_cast<int>(unit(u));
  }

  //
  // [Vertex distribution]
  // Converts a global vertex into the local vertex in the corresponding rank.
  //
  vertex to_local_remote(const global_vertex u) const {
    return make_vertex_from(index_in_unit(u));
  }

  //
  // [Vertex distribution]
  // Converts a global vertex into the local vertex in the corresponding rank.
  // Calling this function on non-owner ranks is prohibited.
  //
  vertex to_local(const global_vertex u) const {
    assert(all_.rank() == owner_rank(u));
    return to_local_remote(u);
  }

  //
  // [Vertex distribution]
  // Converts a local vertex on the given rank into the global vertex.
  //
  global_vertex to_global(const vertex u, const int rank) const {
    return ununit(rank, u.t);
  }

  //
  // [Vertex distribution]
  // Converts a local vertex into the global vertex, assuming that the caller
  // rank is the corresponding owner.
  //
  global_vertex to_global(const vertex u) const {
    return to_global(u, all_.rank());
  }

  //
  // [Edge distribution]
  // Returns a rank in the column communicator to which `source` is distributed.
  //
  int column_rank(const global_vertex source) const {
    return unit(source) % column_.size();
  }

  //
  // [Edge distribution]
  // Returns a rank in the row communicator to which `target` is distributed.
  //
  int row_rank(const global_vertex target) const {
    return unit(target) / column_.size();
  }

  //
  // [Edge distribution]
  // Returns a rank in communicator `all_` to which `e` is distributed.
  //
  int owner_rank(const edge &e) const {
    const int i = column_rank(e.source());
    const int j = row_rank(e.target());
    // Column major
    return i + j * column_.size();
  }

  source_vertex to_source_remote(const global_vertex source) const {
    return unit(source) / column_.size() * unit_size_ + index_in_unit(source);
  }

  source_vertex to_source(const global_vertex source) const {
    assert(column_.rank() == column_rank(source));
    return to_source_remote(source);
  }

  target_vertex to_target_remote(const global_vertex target) const {
    return (unit(target) % column_.size()) * unit_size_ + index_in_unit(target);
  }

  target_vertex to_target(const global_vertex target) const {
    assert(row_.rank() == row_rank(target));
    return to_target_remote(target);
  }

  global_vertex to_global(const source_vertex s) const {
    const auto unit_id = column_.size() * (s.t / unit_size_) + column_.rank();
    const auto index = s.t % unit_size_;
    const global_vertex ret = ununit(unit_id, index);
    assert(to_source_remote(ret) == s);
    return ret;
  }

  global_vertex to_global(const target_vertex t) const {
    const auto unit_id = column_.size() * row_.rank() + t.t / unit_size_;
    const auto index = t.t % unit_size_;
    const global_vertex ret = ununit(unit_id, index);
    assert(to_target_remote(ret) == t);
    return ret;
  }

  size_t source_length() const {
    return unit_size_ * row_.size(); // unit_size_ * (#row cycles)
  }

  size_t target_length() const { return unit_size_ * column_.size(); }

  //
  // Returns `[first, last)` that represents the range of overlapping indices
  // of local vertices within source vertices.
  //
  std::pair<size_t, size_t> local_in_source() const {
    const int rank = row_.rank();
    const size_t first = unit_size_ * rank;
    const size_t last = first + local_vertex_count();
    return std::make_pair(first, last);
  }

  template <typename T>
  void transpose_to_source(const std::vector<T> &target_data,
                           T *const source_data) const {
    assert(target_data.size() == target_length());

    net::allgather(&target_data[unit_size_ * column_.rank()], unit_size_,
                   source_data, row_);
  }

private:
  uint32_t unit(const global_vertex u) const {
    assert(u.t >= 0);
    return static_cast<uint32_t>(u.t % all_.size());
  }

  uint32_t index_in_unit(const global_vertex u) const {
    assert(u.t >= 0);
    assert(to_unsig(u.t) / all_.size() <= std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(u.t / all_.size());
  }

  global_vertex ununit(const uint32_t unit_id, const uint32_t index) const {
    const global_vertex ret(
        static_cast<global_vertex_int>(index) * all_.size() + unit_id);
    assert(unit(ret) == unit_id);
    assert(index_in_unit(ret) == index);
    return ret;
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// edge_2d
//
////////////////////////////////////////////////////////////////////////////////

// POD type
struct __attribute__((packed)) edge_2d {
  int40 source_;
  int40 target_;

  bool operator<(const edge_2d &x) const {
    return source_ < x.source_ || (source_ == x.source_ && target_ < x.target_);
  }

  bool operator==(const edge_2d &x) const {
    return source_ == x.source_ && target_ == x.target_;
  }

  source_vertex source() const { return source_.get(); }

  void set_source(const source_vertex s) { source_ = make_int40(s.t); }

  target_vertex target() const { return target_.get(); }

  void set_target(const target_vertex t) { target_ = make_int40(t.t); }
};
static_assert(std::is_pod<edge_2d>::value, "");
static_assert(sizeof(edge_2d) == sizeof(int40) * 2, "");

static edge_2d make_edge_2d(const source_vertex s, const target_vertex t) {
  // -1 might be used to represent invalid vertices
  assert(s.t >= -1);
  assert(t.t >= -1);
  return {make_int40(s.t), make_int40(t.t)};
}

static edge_2d make_edge_2d_remote(const yoo &d, const edge &e) {
  return make_edge_2d(d.to_source_remote(e.source()),
                      d.to_target_remote(e.target()));
}

static edge_2d make_edge_2d(const yoo &d, const edge &e) {
  return make_edge_2d(d.to_source(e.source()), d.to_target(e.target()));
}

////////////////////////////////////////////////////////////////////////////////
//
// distributor
//
////////////////////////////////////////////////////////////////////////////////

template <typename T>
static std::vector<std::atomic<T>>
copy_atomic_vector(const std::vector<T> &orig) {
  std::vector<std::atomic<T>> ret(orig.size());
  for (size_t i = 0; i < orig.size(); ++i) {
    ret[i].store(orig[i]);
  }
  return ret;
}

template <typename T>
static void deduplicate(std::vector<edge_2d> *const sorted_edges) {
  const auto last = std::unique(sorted_edges->begin(), sorted_edges->end());
  sorted_edges->resize(last - sorted_edges->begin());
}

class distributor : types::noncopyable {
  const yoo &dist_;
  std::vector<edge_2d> edges_;

public:
  distributor(const yoo &d, const size_t edge_capacity) : dist_(d), edges_() {
    edges_.reserve(edge_capacity);
  }

  //
  // 1. Transfers edges to the owner of each of them and appends the reveiced
  //    edges to `edge_2ds`.
  // 2. Removes self-loops.
  // 3. Symmetrize edges.
  //
  // `convert` is used to convert the value type of `RandomAccessIterator` to
  // `graph::edge`.
  // Wrapping the iterator with a something like `boost::transform_iterator` or
  // `std::ranges::views::transform` might be better in terms of generality, but
  // doing it in C++14 without Boost is painful...
  //
  template <typename RandomAccessIterator,
            typename EdgeConverter = types::identity>
  void feed(RandomAccessIterator first, RandomAccessIterator last,
            EdgeConverter convert = types::identity{}) {
    INDEXED_BFS_TIMED_SCOPE(nullptr);

    using difference_type =
        typename std::iterator_traits<RandomAccessIterator>::difference_type;
    const difference_type n = last - first;

    std::vector<std::atomic<int>> send_counts(dist_.all().size());
    {
      INDEXED_BFS_TIMED_SCOPE("send_counts");
#pragma omp parallel for schedule(static)
      for (difference_type i = 0; i < n; ++i) {
        const edge &e = convert(first[i]);
        // Remove self-loops
        if (e.source() != e.target()) {
          send_counts[dist_.owner_rank(e)].fetch_add(1);
          send_counts[dist_.owner_rank(e.reverse())].fetch_add(1);
        }
      }
    }

    const std::vector<int> send_displs = net::make_displacements(send_counts);

    std::vector<edge_2d> send_data(send_displs.back() + send_counts.back() + 1);
    std::vector<std::atomic<int>> n_puts = copy_atomic_vector(send_displs);
    {
      INDEXED_BFS_TIMED_SCOPE("send_data");
#pragma omp parallel for schedule(static)
      for (difference_type i = 0; i < n; ++i) {
        const edge &e = convert(first[i]);
        if (e.source() != e.target()) {
          const int j = n_puts[dist_.owner_rank(e)].fetch_add(1);
          send_data[j] = make_edge_2d_remote(dist_, e);
          const int k = n_puts[dist_.owner_rank(e.reverse())].fetch_add(1);
          send_data[k] = make_edge_2d_remote(dist_, e.reverse());
        }
      }
    }

    const std::vector<std::atomic<int>> recv_counts =
        net::alltoall(send_counts, dist_.all());
    const std::vector<int> recv_displs = net::make_displacements(recv_counts);

    const size_t orig_len = edges_.size();
    const size_t recv_total = recv_displs.back() + recv_counts.back();
    const size_t req_len = edges_.size() + recv_total;
    if (edges_.capacity() < req_len) {
      LOG_W << "Reallocation: {capacity: " << edges_.capacity()
            << ", required: " << req_len << "}";
    }
    edges_.resize(req_len);

    {
      INDEXED_BFS_TIMED_SCOPE("alltoallv");
      net::alltoallv(send_data.data(),
                     reinterpret_cast<const int *>(send_counts.data()),
                     send_displs.data(), edges_.data() + orig_len,
                     reinterpret_cast<const int *>(recv_counts.data()),
                     recv_displs.data(), dist_.all());
    }

    assert(std::all_of(edges_.begin() + orig_len, edges_.end(), [&](auto &e) {
      return e.source().t < to_sig(dist_.source_length()) &&
             e.target().t < to_sig(dist_.target_length());
    }));
  }

  std::vector<edge_2d> drain() {
    INDEXED_BFS_TIMED_SCOPE(nullptr);

    //
    // `util::sort::sort_parallel()` requires additional memory at
    // `std::inplace_merge()` (empirically, its size seems almost the half of
    // the size of `edges_`, which is critically large), and so we use Boost
    // Sort Parallel instead.
    //
    LOG_I << "Sorting...";
    sort::compact_parallel_sort(edges_.begin(), edges_.end());

    //
    // We avoid to use the erase-remove idiom because `erase()` requires a
    // significant amount of memory for some reason (copying the contents to
    // new memory region?).
    // We also do not call `shrink_to_fit()` because of its additional memory
    // consumption.
    //
    LOG_I << "Deduplicating...";
    const auto last = std::unique(edges_.begin(), edges_.end());
    edges_.resize(last - edges_.begin());

    INDEXED_BFS_LOG_RSS();
    return std::move(edges_);
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Free Functions
//
////////////////////////////////////////////////////////////////////////////////

static std::pair<int, int> determine_sizes(const net::comm &c) {
  int col = static_cast<int>(std::sqrt(c.size()));
  // Simple stupid algorithm...
  for (; col > 1 && c.size() % col != 0; --col)
    ;
  return std::make_pair(c.size() / col, col);
}

static std::pair<net::comm, net::comm>
split_comm(const net::comm &c, const int row_size, const int column_size) {
  assert(c.size() == row_size * column_size);
  static_cast<void>(column_size); // Suppress warning

  const int rank_row = c.rank() % row_size;
  const int rank_col = c.rank() / row_size;
  net::comm row = c.split(rank_row, rank_col);
  net::comm col = c.split(rank_col, rank_row);
  return std::make_pair(std::move(row), std::move(col));
}

static yoo make_distribution(const int scale) {
  net::comm all = net::world().dup();

  int n_rows, n_cols;
  std::tie(n_rows, n_cols) = determine_sizes(all);
  LOG_I << "row_count: " << n_rows;
  LOG_I << "column_count: " << n_cols;

  net::comm row, col;
  std::tie(row, col) = split_comm(all, n_rows, n_cols);

  // Confusing...
  assert(row.size() == n_cols);
  assert(col.size() == n_rows);

  return yoo(scale, std::move(all), std::move(row), std::move(col));
}

} // namespace detail

using detail::distributor;
using detail::edge_2d;
using detail::make_distribution;
using detail::make_source_vertex_from;
using detail::make_target_vertex_from;
using detail::source_vertex;
using detail::target_vertex;
using detail::yoo;

} // namespace distribution
} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
