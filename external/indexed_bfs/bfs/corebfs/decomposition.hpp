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
using indexed_bfs::common::int40;
using indexed_bfs::common::make_int40;
using indexed_bfs::util::bit::bit_vector;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::sort_parallel::sort_parallel;
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

static std::vector<size_t> count_degrees(const yoo &d,
                                         const std::vector<edge_2d> &edges_2d) {
  std::vector<size_t> degs_loc;

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

    std::vector<size_t> degs_remote(d.unit_size());

    const auto low = make_source_vertex_from(d.unit_size() * row_rank);
    const auto high = make_source_vertex_from(d.unit_size() * (row_rank + 1));

    const auto first = find_incident(edges_2d, low);
    const auto last = find_incident(edges_2d, high);

    for (auto it = first; it != last; ++it) {
      const vertex u = d.to_local_remote(d.to_global(it->source()));
      degs_remote[u.t] += 1;
    }

    net::reduce_inplace(degs_remote.size(), MPI_SUM, row_rank, d.row(),
                        degs_remote.data());

    if (d.row().rank() == row_rank) {
      degs_loc = std::move(degs_remote);
    }
  }

  // All ranks should have a result
  assert(degs_loc.size() == d.unit_size());

  return degs_loc;
}

static bit_vector make_exist_mask(const yoo &d,
                                  const std::vector<size_t> &degs_loc) {
  assert(degs_loc.size() == d.unit_size());

  bit_vector exists_loc(bit_vector_aligned_unit_size(d));
  for (size_t i = 0; i < degs_loc.size(); ++i) {
    if (degs_loc[i] > 0) {
      exists_loc.set(i);
    }
  }
  return exists_loc;
}

static std::vector<source_vertex>
bcast_leaves(const yoo &d, const std::vector<size_t> &degs_loc,
             const int row_root_rank) {
  std::vector<source_vertex> leaves;

  if (d.row().rank() == row_root_rank) {
    leaves.reserve(d.unit_size()); // Leaves are fewer than local vertices
    for (vertex u(0); u.t < to_sig(degs_loc.size()); ++u.t) {
      if (degs_loc[u.t] == 1) {
        leaves.push_back(d.to_source(d.to_global(u)));
      }
    }
  }

  net::bcast(row_root_rank, d.row(), &leaves);
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
  // Leaves have only one neighbor, and so twigs are fewer than them.
  auto twigs = make_with_capacity<std::vector<edge>>(leaves.size());

  for (const source_vertex s : leaves) {
    auto it = find_incident(s, edges_2d);
    for (; it != edges_2d->end() && it->source() == s; ++it) {
      const size_t i = bit_vector_aligned_index(d, it->target());
      if (exists_aligned_tgt[i]) {
        // Add a found twig edge
        const global_vertex u = d.to_global(it->source());
        const global_vertex v = d.to_global(it->target());
        twigs.push_back(make_edge(u, v));

        // Mark the edge to remove by replacing the target with -1
        // Note: do not replace the source since it makes the edge list unsorted
        *it = make_edge_2d(it->source(), target_vertex(-1));

        // The rest of the neighbors of `s` should be nonexistent
        assert([&]() {
          auto jt = it + 1;
          for (; jt != edges_2d->end() && jt->source() == s; ++jt) {
            const size_t j = bit_vector_aligned_index(d, jt->target());
            if (exists_aligned_tgt[j]) {
              return false;
            }
          }
          return true;
        }());

        break;
      }
    }
  }

  assert(twigs.size() <= leaves.size());
  return twigs;
}

static bool
should_match_leaves_and_sources(const yoo &d,
                                const std::vector<source_vertex> &leaves,
                                const std::vector<edge> &twigs_loc) {
  bool success = true;
  auto leaf_set = make_with_capacity<std::unordered_set<vertex>>(leaves.size());
  auto source_set =
      make_with_capacity<std::unordered_set<vertex>>(twigs_loc.size());

  // All the leaves are local vertices of this rank
  for (const source_vertex &s : leaves) {
    if (!leaf_set.insert(d.to_local(d.to_global(s))).second) {
      LOG_E << "Duplicated leaf: " << d.to_global(s);
      success = false;
    }
  }

  // All the sources are local vertices of this rank
  for (const edge &e : twigs_loc) {
    if (!source_set.insert(d.to_local(e.source())).second) {
      LOG_E << "Duplicated twig source: " << e.source();
      success = false;
    }
  }

  if (leaf_set != source_set) {
    success = false;
  }

  return success;
}

static void update_by_source(const yoo &d, const std::vector<edge> &twigs_2d,
                             const int row_rank,
                             const std::vector<source_vertex> &leaves,
                             std::vector<size_t> *const degs_loc,
                             std::vector<global_vertex> *const parents_loc,
                             bit_vector *const exists_loc) {
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

  for (const auto &e : twigs_loc) {
    const vertex s = d.to_local(e.source());
    // The source of every twig is a leaf, and the degree of a leaf is one.
    // However, if `s` is in a two-vertex connected component and the other
    // endpoint also appered in `twigs_loc`, its degree may be zero.
    assert((*degs_loc)[s.t] <= 1);
    (*degs_loc)[s.t] = 0;
    (*parents_loc)[s.t] = e.target();
    exists_loc->set(s.t, false);
  }
}

static void update_by_target(const yoo &d, const std::vector<edge> &twigs,
                             std::vector<size_t> *const degs_loc,
                             bit_vector *const exists_loc) {
  const std::vector<edge> twigs_loc =
      net::alltoallv_by(
          twigs,
          [&](auto &e) { return d.to_target(e.target()).t / d.unit_size(); },
          d.column())
          .data;

  for (const auto &e : twigs_loc) {
    const vertex u = d.to_local(e.target());

    // `e.source()` is removed, and so the degree of `e.target()` decreases.
    // Note that the degree of `e.target()` may be zero; this happens if `e` is
    // an isolated connected component (both endpoints are leaves)
    if ((*degs_loc)[u.t] > 0) {
      (*degs_loc)[u.t] -= 1;
      if ((*degs_loc)[u.t] == 0) {
        exists_loc->set(u.t, false);
      }
    }
  }
}

static void remove_removed(const yoo &d, const bit_vector &exists_aligned_tgt,
                           std::vector<edge_2d> *const edges_2d) {
  size_t i_put = 0;
  for (size_t i_get = 0; i_get < edges_2d->size(); ++i_get) {
    // Check if this edge is marked for removal
    if ((*edges_2d)[i_get].target().t < 0) {
      continue;
    }

    // Check if the target is removed
    const size_t t_aligned =
        bit_vector_aligned_index(d, (*edges_2d)[i_get].target());
    if (!exists_aligned_tgt[t_aligned]) {
      continue;
    }

    (*edges_2d)[i_put] = (*edges_2d)[i_get];
    ++i_put;
  }

  edges_2d->resize(i_put);
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
  LOG_I << "Start finding 2-core";
  LOG_RSS();

  // `vertex` -> degree
  std::vector<size_t> degs_loc = count_degrees(d, *edges_2d);
  LOG_RSS();

  const size_t n =
      net::reduce(std::accumulate(degs_loc.begin(), degs_loc.end(), 0), MPI_SUM,
                  0, d.all());
  const size_t m = net::reduce(edges_2d->size(), MPI_SUM, 0, d.all());
  if (d.all().rank() == 0) {
    LOG_E << "n = " << n << ", m = " << m;
  }

  //  // The sum of degrees should equal to the number of edges
  //  assert([&]() {
  //    const size_t n = net::reduce(
  //      std::accumulate(degs_loc.begin(), degs_loc.end(), 0),
  //      MPI_SUM,
  //      0,
  //      d.all()
  //    );
  //    const size_t m = net::reduce(
  //      edges_2d->size(),
  //      MPI_SUM,
  //      0,
  //      d.all()
  //    );
  //    if (d.all().rank() == 0) {
  //      if (n != m) {
  //        LOG_E << "n = " << n << ", m = " << m;
  //      }
  //      return n == m;
  //    } else {
  //      return true;
  //    }
  //  }());

  // `vertex` -> parent
  std::vector<global_vertex> parents_loc(d.local_vertex_count(),
                                         global_vertex(-1));
  LOG_RSS();

  // `vertex` -> (does it have non-zero degree?)
  //
  // For directly using this data as a send buffer in communications, the size
  // is aligned to 64-bit boundary.
  bit_vector exists_loc = make_exist_mask(d, degs_loc);
  LOG_RSS();

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
  LOG_RSS();

  const size_t n_connected = net::count_if(
      degs_loc.begin(), degs_loc.end(), d.all(), [](auto d) { return d > 0; });
  if (d.all().rank() == 0) {
    const double p = static_cast<double>(n_connected) / d.global_vertex_count();
    LOG_I << "connected_vertex_count: " << n_connected;
    LOG_I << "connected_vertex_proportion: " << p;
  }

  for (int n_iter = 1;; ++n_iter) {
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
      LOG_RSS();

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
      const std::vector<source_vertex> leaves =
          bcast_leaves(d, degs_loc, row_rank);
      updated = updated || leaves.size() > 0;

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

  const size_t m_orig = net::reduce(edges_2d->size(), MPI_SUM, 0, d.all());
  remove_removed(d, exists_aligned_tgt, edges_2d);
  const size_t m_core = net::reduce(edges_2d->size(), MPI_SUM, 0, d.all());

  const size_t n_core = net::count_if(degs_loc.begin(), degs_loc.end(), d.all(),
                                      [](auto d) { return d > 0; });
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

template <> MPI_Datatype net::detail::datatype_of<std::atomic<uint64_t>>() {
  static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "");
  return net::datatype_of<uint64_t>();
}

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
    static_assert(sizeof(edge_2d) == sizeof(int40) * 2);

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
