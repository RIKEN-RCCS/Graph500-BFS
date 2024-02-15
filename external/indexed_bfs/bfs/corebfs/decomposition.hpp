//
// Authored by ARAI Junya <araijn@gmail.com> in 2024-01-29.
//
#pragma once

#include "../../common.hpp"
#include "../../graph.hpp"
#include "distribution.hpp"
#include <array>
#include <cmath>
#include <unordered_set>

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace decomposition {
namespace detail {

using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using namespace indexed_bfs::bfs::corebfs::distribution;
using namespace indexed_bfs::common;
using indexed_bfs::util::bit::bit_vector;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::sort::sort_parallel;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

////////////////////////////////////////////////////////////////////////////////
//
// Free Functions
//
////////////////////////////////////////////////////////////////////////////////

static size_t bit_vector_aligned_unit_size(const yoo &d) {
  return (d.unit_size() + bit_vector::element_bits - 1) /
         bit_vector::element_bits * bit_vector::element_bits;
}

static size_t bit_vector_aligned_index(const yoo &d, const target_vertex t) {
  return (t.t / d.unit_size()) * bit_vector_aligned_unit_size(d) +
         (t.t % d.unit_size());
}

static std::vector<edge_2d>::const_iterator
find_incident(const std::vector<edge_2d> &edges, const source_vertex s) {
  return std::lower_bound(edges.begin(), edges.end(), s,
                          [&](auto &e, auto s) { return e.source() < s; });
}

static std::vector<edge_2d>::iterator
find_incident(const source_vertex s, std::vector<edge_2d> *const edges) {
  return std::lower_bound(edges->begin(), edges->end(), s,
                          [&](auto &e, auto s) { return e.source() < s; });
}

static std::vector<std::atomic<size_t>>
count_degrees(const yoo &d, const std::vector<edge_2d> &edges_2d) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

  std::vector<std::atomic<size_t>> degs_loc;

  //
  // R = 3, C = 2:
  //
  //   low --> +----+----+----+----+----+----+
  //  P(0, 0)  |   P(0, 0) <--|-- P(0, 1)    |  Processed at row_rank = 0
  //  high --> + -- + -- + -- + -- + -- + -- +
  //           :              :              :
  //           +----+----+----+----+----+----+
  //  P(0, 1)  |   P(0, 0) ---|-> P(0, 1)    |  Processed at row_rank = 1
  //           + -- + -- + -- + -- + -- + -- +
  //           :              :              :
  //
  for (int row_rank = 0; row_rank < d.row().size(); ++row_rank) {
    LOG_I << "Counting degrees for row rank " << row_rank;

    std::vector<std::atomic<size_t>> degs_remote(d.unit_size());

    const auto low = make_source_vertex_from(d.unit_size() * row_rank);
    const auto high = make_source_vertex_from(d.unit_size() * (row_rank + 1));

    const auto first = find_incident(edges_2d, low);
    const auto last = find_incident(edges_2d, high);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < last - first; ++i) {
      const vertex u = d.to_local_remote(d.to_global(first[i].source()));
      degs_remote[u.t].fetch_add(1);
    }

    INDEXED_BFS_LOG_RSS();
    net::reduce_inplace(degs_remote.size(), MPI_SUM, row_rank, d.row(),
                        degs_remote.data());
    INDEXED_BFS_LOG_RSS();
    malloc_trim(0);
    INDEXED_BFS_LOG_RSS();

    if (d.row().rank() == row_rank) {
      degs_loc = std::move(degs_remote);
    }
  }

  // Every rank should have a result
  assert(degs_loc.size() == d.unit_size());
  // The sum of degrees should equal to the number of edges
  assert([&]() {
    const size_t d_local = std::accumulate(
        degs_loc.begin(), degs_loc.end(), static_cast<size_t>(0),
        [](auto x, auto &y) { return x + y.load(); });
    const size_t d_global = net::reduce(d_local, MPI_SUM, 0, d.all());
    const size_t m = net::reduce(edges_2d.size(), MPI_SUM, 0, d.all());
    return d.all().rank() != 0 || d_global == m;
  }());

  return degs_loc;
}

static bit_vector
make_exist_mask(const yoo &d,
                const std::vector<std::atomic<size_t>> &degs_loc) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

  assert(degs_loc.size() == d.unit_size());

  bit_vector exists_loc(bit_vector_aligned_unit_size(d));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < degs_loc.size(); ++i) {
    if (degs_loc[i].load() > 0) {
      exists_loc.set(i);
    }
  }
  return exists_loc;
}

static std::vector<source_vertex>
find_leaves(const yoo &d, const std::vector<std::atomic<size_t>> &degs_loc) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

  // Leaves are fewer than local vertices
  auto leaves = make_with_capacity<std::vector<source_vertex>>(d.unit_size());
  for (vertex u(0); u.t < to_sig(degs_loc.size()); ++u.t) {
    if (degs_loc[u.t].load() == 1) {
      leaves.push_back(d.to_source(d.to_global(u)));
    }
  }
  leaves.shrink_to_fit();
  return leaves;
}

//
// Returns all the edges incident to `leaves` and marks those edges to be
// removed later.
//
// Since twigs are exchanged between other nodes, the vertex IDs of twigs are
// converted to their global IDs.
//
static std::vector<edge>
find_mark_twigs(const yoo &d, const std::vector<source_vertex> &leaves,
                const bit_vector &exists_aligned_tgt,
                std::vector<edge_2d> *const edges_2d) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

  // Leaves have only one neighbor, and so twigs are fewer than them.
  auto twigs = make_with_capacity<std::vector<edge>>(leaves.size());

#pragma omp parallel
  {
    std::deque<edge> es;

#pragma omp for schedule(static)
    for (size_t i = 0; i < leaves.size(); ++i) {
      const source_vertex s = leaves[i];
      auto it = find_incident(s, edges_2d);
      for (; it != edges_2d->end() && it->source() == s; ++it) {
        const size_t i = bit_vector_aligned_index(d, it->target());
        if (!exists_aligned_tgt[i]) {
          continue;
        }

        // Add a found twig edge
        const global_vertex u = d.to_global(it->source());
        const global_vertex v = d.to_global(it->target());
        es.push_back(make_edge(u, v));

        // Mark the edge to remove by replacing the target with -1.
        // This keeps `edges_2d` sorted because `it->source()` has only one
        // incident edge, and so the value of `it->target()` has nothing in the
        // ordering.
        it->set_target(target_vertex(-1));

        break;
      }
    }

#pragma omp critical
    std::copy(es.begin(), es.end(), std::back_inserter(twigs));
  } // #pragma omp parallel

  assert(twigs.size() <= leaves.size());
  return twigs;
}

static bool
should_match_leaves_and_sources(const yoo &d,
                                const std::vector<source_vertex> &leaves,
                                const std::vector<edge> &twigs_loc) {
  bool success = true;
  auto leaf_set = make_with_capacity<std::unordered_set<vertex>>(leaves.size());

  // All the leaves are local vertices of this rank
  for (const source_vertex &s : leaves) {
    if (!leaf_set.insert(d.to_local(d.to_global(s))).second) {
      LOG_E << "Duplicated leaf: " << d.to_global(s);
      success = false;
    }
  }

  // All the sources are local vertices of this rank
  for (const edge &e : twigs_loc) {
    const vertex u = d.to_local(e.source());
    if (leaf_set.erase(u) == 0) {
      LOG_E << "Non-leaf twig source: " << d.to_global(u);
      success = false;
    }
  }

  return success;
}

static void update_by_source(const yoo &d, const std::vector<edge> &twigs_2d,
                             const int row_rank,
                             const std::vector<source_vertex> &leaves,
                             std::vector<std::atomic<size_t>> *const degs_loc,
                             std::vector<global_vertex> *const parents_loc,
                             bit_vector *const exists_loc) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

  // The number of the gathered twigs does not exceed `leaves.size() * 2`
  // because every source of the twigs are leaves.
  // `* 2` is necessary for the case that all the twigs are a two-vertex
  // connected component and the both endpoint vertices are owned by this rank.
  const std::vector<edge> twigs_loc = net::gatherv(twigs_2d, row_rank, d.row());
  if (d.row().rank() != row_rank) {
    return;
  }

  assert(twigs_loc.size() <= leaves.size() * 2);
  assert(should_match_leaves_and_sources(d, leaves, twigs_loc));
  static_cast<void>(leaves); // Suppress warnings

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < twigs_loc.size(); ++i) {
    const edge &e = twigs_loc[i];
    const vertex s = d.to_local(e.source());
    // The source of every twig is a leaf, and the degree of a leaf is one.
    // However, if `s` is in a two-vertex connected component and the other
    // endpoint also appered in `twigs_loc`, its degree may be zero.
    assert((*degs_loc)[s.t].load() <= 1);
    (*degs_loc)[s.t].store(0);
    (*parents_loc)[s.t] = e.target();
    exists_loc->set(s.t, false);
  }
}

static void update_by_target(const yoo &d, const std::vector<edge> &twigs,
                             std::vector<std::atomic<size_t>> *const degs_loc,
                             bit_vector *const exists_loc) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

  const std::vector<vertex> twig_targets_loc =
      net::alltoallv_by(
          twigs,
          [&](auto &e) { return d.to_target(e.target()).t / d.unit_size(); },
          [&](auto &e) { return d.to_local_remote(e.target()); }, d.column())
          .data;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < twig_targets_loc.size(); ++i) {
    const vertex u = twig_targets_loc[i];

    // Decrease the degree of target vertices of twigs.
    // Note that the degree of a target may be zero; this happens if `e` is an
    // isolated connected component (both endpoints are leaves)
    if ((*degs_loc)[u.t].load() > 0) {
      const size_t d = (*degs_loc)[u.t].fetch_sub(1);
      assert(d > 0); // Underflow if `d == 0`
      if (d == 1) {  // Now `(*degs_loc)[u.t]` is zero
        exists_loc->set(u.t, false);
      }
    }
  }
}

static void remove_removed(const yoo &d, const bit_vector &exists_aligned_tgt,
                           std::vector<edge_2d> *const edges_2d) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < edges_2d->size(); ++i) {
    edge_2d *const e = &(*edges_2d)[i];
    // Check if `e` is not yet marked for removal and if its target is removed
    if (e->target().t >= 0 &&
        !exists_aligned_tgt[bit_vector_aligned_index(d, e->target())]) {
      e->set_target(target_vertex(-1));
    }
  }

  const auto it = std::remove_if(edges_2d->begin(), edges_2d->end(),
                                 [](auto &e) { return e.target().t < 0; });
  edges_2d->erase(it, edges_2d->end());
}

//
// Removes edges in tree parts (i.e., not in 2-cores) of the graph.
//
// This function iteratively removes the edges incident to degree-1 vertices
// until convergence. We refer to a degree-1 vertex and its incident edge as a
// *leaf* and a *twig*, respectively.
//
// Letting $\bm{A}$ be the adjacency matrix of the given graph, This algorithm
// can be described as follows in linear algebra:
//
// - Compute $\bm{d}$, a column vector of vertex degrees
// - while true do
//     - Compute column vector $\bm{t}$ s.t. $t_i = 1$ if $d_i = 1$; otherwise,
//       $t_i = 0$.
//     - $\bm{\delta} \gets \bm{A} \bm{t}$
//     - if $\bm{\delta} = \bm{0}$ then break
//     - $\bm{d} \gets \max \(\bm{0}, \bm{d} - \bm{t} - \bm{\delta}\)$
//
// Estimated memory consumption for SCALE = 43, R = 528, C = 288:
//
//                                  exists_aligned_tgt: 3.6 GiB
//                                <---      1 bit/target     --->
//                               +----+----+----+----+......+----+
//                             i |                               |
//                               +-                             -+
//                           i+R |                               |
//                               +-           P(i, j)           -+
//  exists_loc: 7.2 MiB          :                               :
//   1 bit/local vertex  v--     +-      edges_2d: 17.2 GiB     -+
//    degs_loc: 0.4 GiB  | i+R*j |         40x2 bit/edge         |
// parents_loc: 0.4 GiB  ^--     +-                             -+
//  64 bit/local vertex          :                               :
//                               +-                             -+
//                     i+R*(C-1) |                               |
//                   Unit ID--^  +----+----+----+----+......+----+
//
static std::vector<global_vertex>
prune_trees(const yoo &d, std::vector<edge_2d> *const edges_2d) {
  INDEXED_BFS_TIMED_SCOPE(nullptr);
  INDEXED_BFS_LOG_RSS();

  // `vertex` -> degree
  std::vector<std::atomic<size_t>> degs_loc = count_degrees(d, *edges_2d);
  INDEXED_BFS_LOG_RSS();

  // `vertex` -> parent
  std::vector<global_vertex> parents_loc(d.local_vertex_count(),
                                         global_vertex(-1));
  INDEXED_BFS_LOG_RSS();

  // `vertex` -> (does it have non-zero degree?)
  //
  // For directly using this data as a send buffer in communications, the size
  // is aligned to 64-bit boundary.
  bit_vector exists_loc = make_exist_mask(d, degs_loc);
  INDEXED_BFS_LOG_RSS();

  // `target` -> (does it exist?)
  //
  // Bacause of a 64-bit alignment, the bit for target `t` is not at
  // `exists_aligned_tgt[t.t]`. Use `bit_vector_aligned_index()` for access.
  //
  //                    |    Row rank 0's data    |    Row rank 1's data    |
  //                    |<-- d.unit_size() -->    |<-- d.unit_size() -->    |
  //                    |<-- aligned unit size -->|<-- aligned unit size -->|
  // exists_aligned_tgt [......................................................]
  //
  bit_vector exists_aligned_tgt(bit_vector_aligned_unit_size(d) *
                                d.column().size());
  INDEXED_BFS_LOG_RSS();

  const size_t n_connected_global =
      net::count_if(degs_loc.begin(), degs_loc.end(), d.all(),
                    [](auto &d) { return d.load() > 0; });
  if (d.all().rank() == 0) {
    const double p = static_cast<double>(n_connected_global) / d.global_vertex_count();
    LOG_I << "connected_vertex_count: " << n_connected_global;
    LOG_I << "connected_vertex_proportion: " << p;
  }

  // Total number of removed vertices (leaves)
  size_t n_removed_global = 0;

  for (int n_iter = 1;; ++n_iter) {
    // Condition to avoid infinite loop; the diameter of Kronecker graphs is
    // usually small.
    OR_DIE(n_iter <= 32);

    // Find leaves in the outside of a loop over `row_rank` for load balancing
    std::vector<source_vertex> leaves_loc = find_leaves(d, degs_loc);
    INDEXED_BFS_LOG_RSS();

    //
    // R = 3, C = 2
    //
    // Unit owner
    // ----------  +----+----+----+----+----+----+  ---
    //    P(0, 0)  |   P(0, 0)    |   P(0, 1)    |   ^
    //             + -- + -- + -- + -- + -- + -- +   | Processed at the loop
    //    P(1, 0)  |   P(1, 0)    |   P(1, 1)    |   | for row_rank = 0
    //             + -- + -- + -- + -- + -- + -- +   |
    //    P(2, 0)  |   P(2, 0)    |   P(2, 1)    |   v
    //             +----+----+----+----+----+----+  ---
    //    P(0, 1)  |   P(0, 0)    |   P(0, 1)    |   ^
    //             + -- + -- + -- + -- + -- + -- +   | Processed at the loop
    //    P(1, 1)  |   P(1, 0)    |   P(1, 1)    |   | for row_rank = 1
    //             + -- + -- + -- + -- + -- + -- +   |
    //    P(2, 1)  |   P(2, 0)    |   P(2, 1)    |   v
    //             +----+----+----+----+----+----+  ---
    //
    // To reduce the peak size of communication volume, horizontally divide the
    // adjacency matrix into the `d.row().size()` parts and process one by one.
    //
    bool updated = false;
    for (int row_rank = 0; row_rank < d.row().size(); ++row_rank) {
      LOG_I << "Iteration " << n_iter << " for row rank " << row_rank;
      INDEXED_BFS_LOG_RSS();

      //
      //           +----+----+----+----+----+----+
      //  P(0, 0)  |loc   ^    ^  |              |
      //           + |    |    |  + -- + -- + -- +
      //  P(1, 0)  | v   loc   ^  |              |
      //           + |    |    |  + -- + -- + -- +
      //  P(2, 0)  | v    v   loc |              |
      //           +----+----+----+----+----+----+
      //  P(0, 1)  |              |loc   ^    ^  |
      //           + -- + -- + -- + |    |    |  +
      //  P(1, 1)  |              | v   loc   ^  |
      //           + -- + -- + -- + |    |    |  +
      //  P(2, 1)  |              | v    v   loc |
      //           +----+----+----+----+----+----+
      //
      // Exchange `exists_loc` in each column to collect the latest data of
      // target vertices in each process.
      //
      net::allgather(exists_loc.data(), exists_loc.element_count(),
                     exists_aligned_tgt.data(), d.column());

      //
      // In row_rank = 0:
      //           +----+----+----+----+----+----+
      //  P(0, 0)  |   P(0, 0) ---|-> P(0, 1)    |
      //           + -- + -- + -- + -- + -- + -- +  In each row, bcast vertices
      //  P(1, 0)  |   P(1, 0) ---|-> P(1, 1)    |  whose degrees are one
      //           + -- + -- + -- + -- + -- + -- +  (we call them "leaves")
      //  P(2, 0)  |   P(2, 0) ---|-> P(2, 1)    |
      //           +----+----+----+----+----+----+
      //           |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //           |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //           |              |              |
      //           +----+----+----+----+----+----+
      //
      std::vector<source_vertex> leaves;
      if (d.row().rank() == row_rank) {
        leaves = std::move(leaves_loc);
      }
      net::bcast(row_rank, d.row(), &leaves);
      updated = updated || leaves.size() > 0;

      const size_t n_leaves_global =
        net::reduce(leaves.size(), MPI_SUM, 0, d.column());
      n_removed_global += n_leaves_global;

      if (d.all().rank() == 0) {
        LOG_I << "n_leaves_global: " << n_leaves_global;
        LOG_I << "n_removed_global: " << n_removed_global;
      }
      assert(n_removed_global <= n_connected_global);

      //
      // In row_rank = 0:
      //           +----+----+----+----+----+----+
      //  P(0, 0)  |                             |
      //           +   Locally find edges from   +
      //  P(1, 0)  |   leaves ("twigs") in this  |
      //           +        matrix block -- + -- +
      //  P(2, 0)  |              |           <--|-- `twigs` in P(2, 1)
      //           +----+----+----+----+----+----+
      //           |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //           |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //           |              |              |
      //           +----+----+----+----+----+----+
      //
      // `_2d` indicates that twigs found in a 2D partitioned matrix.
      const std::vector<edge> twigs_2d =
          find_mark_twigs(d, leaves, exists_aligned_tgt, edges_2d);

      //
      // In row_rank = 0:
      //           +----+----+----+----+----+----+
      //  P(0, 0)  |   P(0, 0) <--|-- P(0, 1)    |
      //           + -- + -- + -- + -- + -- + -- +  In each row, gather the
      //  P(1, 0)  |   P(1, 0) <--|-- P(1, 1)    |  found twigs to the owner of
      //           + -- + -- + -- + -- + -- + -- +  each of their sources
      //  P(2, 0)  |   P(2, 0) <--|-- P(2, 1)    |
      //           +----+----+----+----+----+----+
      //           |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //           |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //           |              |              |
      //           +----+----+----+----+----+----+
      //
      update_by_source(d, twigs_2d, row_rank, leaves, &degs_loc, &parents_loc,
                       &exists_loc);

      //
      // In row_rank = 0:
      //                P(1,0)    P(0,1)    P(2,1) <-- Transposed positions of
      //           P(0,0)    P(2,0)    P(1,1)          owners
      //           +----+----+----+----+----+----+
      //  P(0, 0)  | *    |    |    *    |    |  |
      //           + ^    v    |    ^    v    |  +  In each column, gather the
      //  P(1, 0)  | |    *    |    |    *    |  |  twigs to the owner of each
      //           + |    ^    v    |    ^    v  +  of their targets
      //  P(2, 0)  | |    |    *    |    |    *  |
      //           +----+----+----+----+----+----+
      //  P(0, 1)  |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //  P(1, 1)  |              |              |
      //           + -- + -- + -- + -- + -- + -- +
      //  P(2, 1)  |              |              |
      //           +----+----+----+----+----+----+
      //
      update_by_target(d, twigs_2d, &degs_loc, &exists_loc);
    }

    // Exit if no leaves are found in every rank
    if (!net::allreduce(updated, MPI_LOR, d.column())) {
      break;
    }
  }

  LOG_I << "Removing tree edges...";
  const size_t m_orig = net::reduce(edges_2d->size(), MPI_SUM, 0, d.all());
  remove_removed(d, exists_aligned_tgt, edges_2d);
  const size_t m_core = net::reduce(edges_2d->size(), MPI_SUM, 0, d.all());

  const size_t n_core = net::count_if(degs_loc.begin(), degs_loc.end(), d.all(),
                                      [](auto &d) { return d.load() > 0; });
  if (d.all().rank() == 0) {
    const double pn = static_cast<double>(n_core) / d.global_vertex_count();
    LOG_I << "core_vertex_count: " << n_core;
    LOG_I << "core_vertex_proportion: " << pn;

    const double pm = static_cast<double>(m_core) / m_orig;
    LOG_I << "original_edge_count: " << m_orig;
    LOG_I << "core_edge_count: " << m_core;
    LOG_I << "core_edge_proportion: " << pm;
  }

  return parents_loc;
}

} // namespace detail

using detail::distributor;
using detail::edge_2d;
using detail::prune_trees;

} // namespace decomposition
} // namespace corebfs
} // namespace bfs

template <>
MPI_Datatype
net::detail::datatype_of<bfs::corebfs::decomposition::detail::source_vertex>() {
  return net::datatype_of<
      bfs::corebfs::decomposition::detail::source_vertex::inner_type>();
}

template <>
MPI_Datatype
net::detail::datatype_of<bfs::corebfs::decomposition::detail::target_vertex>() {
  return net::datatype_of<
      bfs::corebfs::decomposition::detail::target_vertex::inner_type>();
}

template <>
MPI_Datatype net::detail::datatype_of<bfs::corebfs::decomposition::edge_2d>() {
  using bfs::corebfs::decomposition::edge_2d;
  using common::int40;

  static MPI_Datatype t = MPI_DATATYPE_NULL;
  if (t == MPI_DATATYPE_NULL) {
    // Note: `edge_2d` is a packed struct
    static_assert(sizeof(edge_2d) == sizeof(int40) * 2, "");

    constexpr int n = 2;
    const int lengths[n] = {1, 1};
    // Taking addresses of packed members is warned due to unalignedness
    const MPI_Aint displs[n] = {0, sizeof(int40)};
    const MPI_Datatype types[n] = {
        net::datatype_of<int40>(),
        net::datatype_of<int40>(),
    };

    auto dt = net::datatype::create_struct(n, lengths, displs, types);
    dt = dt.create_resized(0, sizeof(edge_2d)); // Force the packed size
    dt.commit();
    t = dt.release();
  }
  return t;
}

} // namespace indexed_bfs
