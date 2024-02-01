//
// Definitions regarding core decomposition and tree handling.
//
// Authored by ARAI Junya <araijn@gmail.com> in 2024-01-14.
//
#pragma once

#include "../../net.hpp"
#include "dist_graph.hpp"
#include "parent_array.hpp"

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace decomposition {
namespace detail {

using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using namespace indexed_bfs::bfs::corebfs;
using namespace indexed_bfs::bfs::corebfs::dist_graph;
using indexed_bfs::bfs::corebfs::parent_array::parent_array;
using indexed_bfs::net::gathered_data;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::sort_parallel::sort_parallel;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

//
// Retains only the edges in the 2-core of `g`; in other words, removes all the
// other edges.
//
// Returns `parents`, where `parents[u]` is the parent of vertex `u` in the
// BFS-tree.
// If `u` is isolated (not connected) or in the 2-core, `parent[u]` is -1.
//
// DO NOT FORGET TO APPLY `reset_unreachable()`.
// Although the Graph500 specification requires the parent of an unreachable
// vertex is set to -1, `parents[u]` will not be -1 even if `u` is not in the
// greatest connected component (GCC).
//
static std::vector<global_vertex> retain_core(binary_csr *const g) {
  const size_t n_local = g->slice_count();
  std::vector<global_vertex> parents(n_local, {-1});

  // We consider an edge corresponding to `g->values[i]` is removed if
  // `removed_neighbors[i]` is true
  auto removed_neighbors = bit::make_bit_vector(g->values_.size());

  // Make a degree array and add initial twigs.
  // A twig is an edge to be removed.
  // The direction of a twig must be from a vertex that is close to or inside of
  // the 2-core to a vertex farther to the 2-core.
  auto degs_local = make_with_capacity<std::vector<size_t>>(n_local);
  auto twigs_local = make_with_capacity<std::vector<edge>>(n_local);
  for (vertex u = {0}; u.t < to_sig(n_local); ++u.t) {
    degs_local.push_back(degree(*g, u));
    if (degree(*g, u) == 1) {
      twigs_local.push_back(make_edge(to_global(u), global_vertex{-1}));
    }
  }

  // Iterate while twigs are found
  for (int step = 0;; ++step) {
    LOG_I << "Step: " << step;

    //
    // (1) Find twigs for the next step (`twigs_remote`)
    //
    // We may also loop over each local vertex instead, but we iterate over
    // `twigs_local` because it is expected to be much smaller.
    auto twigs_remote = make_with_capacity<std::vector<edge>>(n_local);
    for (auto it = twigs_local.begin(); it != twigs_local.end(); ++it) {
      const auto s = to_remote_local(it->source());
      if (degs_local[s.t] != 1) {
        // Note: the degree of `s` can be more than one but also zero because
        // `s` may appear in `twigs_local` multiple times.
        continue;
      }

      // Locate the index of a remaining neighbor
      size_t i = g->offsets_[s.t];
      for (; i < g->offsets_[s.t + 1] && removed_neighbors[i]; ++i)
        ;
      assert(i < g->offsets_[s.t + 1]); // Must be found

      assert(parents[s.t].t == -1);
      parents[s.t] = g->values_[i];

      // Remove the edge from `s` to its parent
      removed_neighbors.set(i);
      degs_local[s.t] = 0;
      twigs_remote.push_back(make_edge(g->values_[i], it->source()));
    }

    // Break if no more twigs are found
    if (net::allreduce(twigs_remote.size(), MPI_SUM, net::world()) == 0) {
      break;
    }

    //
    // (2) Exchange twigs
    //
    twigs_local =
        net::alltoallv_inplace_by(
            &twigs_remote, [](auto &e) { return owner_rank(e.source()); },
            net::world())
            .data;

    //
    // (3) Remove twigs in the graph
    //
    // Group `twigs_local` by the source as a key
    sort_parallel(twigs_local.begin(), twigs_local.end());
    global_vertex key;
    for (auto it = twigs_local.begin(); it != twigs_local.end();) {
      key = it->source();
      for (; it != twigs_local.end() && key == it->source(); ++it) {
        const auto s = to_remote_local(it->source());

        // `degs_local[s.t]` can be zero if `it->source()` and `it->target()`
        // compose a two-vertex connected component.
        // This is because both source-target and target-source edges are in
        // `twigs_remote` and those edges have been removed before the insertion
        // to `twigs_remote`.
        if (degs_local[s.t] == 0) {
          continue;
        }

        // Assume that the neighbors are sorted
        const auto p =
            std::lower_bound(&g->values_[g->offsets_[s.t]],
                             &g->values_[g->offsets_[s.t + 1]], it->target());
        assert(p != &g->values_[g->offsets_[s.t + 1]]); // Must be found

        const auto i = p - &g->values_[0];
        assert(!removed_neighbors[i]);
        assert(degs_local[s.t] > 0);
        removed_neighbors.set(i);
        degs_local[s.t] -= 1;
      }
    }
  }

  size_t i_put = 0, i_get = 0;
  for (size_t u = 0; u + 1 < g->offsets_.size(); ++u) {
    g->offsets_[u] = i_put;
    for (; i_get < g->offsets_[u + 1]; ++i_get) {
      if (!removed_neighbors[i_get]) {
        g->values_[i_put] = g->values_[i_get];
        ++i_put;
      }
    }
  }
  g->offsets_.back() = i_put;
  g->values_.resize(i_put);

  assert([&]() {
    for (vertex u{0}; u.t < to_sig(n_local); ++u.t) {
      if (degs_local[u.t] != degree(*g, u)) {
        LOG_E << "u.t = " << u.t << ", degs_local[u.t] = " << degs_local[u.t]
              << ", degree(*g, u) = " << degree(*g, u);
        return false;
      }
    }
    return true;
  }());

  // All vertices that is connected and not in 2-core must have their parent
  assert(std::all_of(iterator::make_counting_iterator<size_t>(0),
                     iterator::make_counting_iterator(n_local), [&](auto u) {
                       return !((*g)[u].size() > 0 && degs_local[u] == 0) ||
                              parents[u].t >= 0;
                     }));

  return parents;
}

static gathered_data<vertex>
exchange_parents(const parent_array &tree_parents_local,
                 const std::vector<bool> &reach_local) {
  std::vector<int> counts(net::world().size());
  for (const auto p : tree_parents_local) {
    const auto child = p.first;
    const auto parent = p.second;
    if (!reach_local[child.t]) {
      counts[owner_rank(parent)] += 1;
    }
  }

  std::vector<int> displs = net::make_displacements(counts);
  const size_t n_parents = displs.back() + counts.back();

  // Pre-convert a global ID to its local ID to reduce communication volume and
  // peak memory consumption.
  std::vector<vertex> parents(n_parents);
  std::vector<int> is = displs; // Copy
  for (const auto p : tree_parents_local) {
    const auto child = p.first;
    const auto parent = p.second;
    if (!reach_local[child.t]) {
      const int r = owner_rank(parent);
      parents[is[r]] = to_remote_local(parent);
      is[r] += 1;
    }
  }

  return net::alltoallv(parents, counts, displs, net::world());
}

static gathered_data<uint8_t>
exchange_reachabilities(const parent_array &tree_parents_local,
                        const std::vector<bool> &reach_local) {
  auto rx = exchange_parents(tree_parents_local, reach_local);

  std::vector<uint8_t> tx_reach(rx.data.size());
#pragma omp parallel for
  for (size_t i = 0; i < rx.data.size(); ++i) {
    tx_reach[i] = reach_local[rx.data[i].t];
  }

  rx.data = std::vector<vertex>(); // Release memory before receiving data
  return net::alltoallv(tx_reach, rx.counts, rx.displacements, net::world());
}

static bool update_reachabilities(const parent_array &tree_parents_local,
                                  std::vector<bool> *const reach_local) {
  // Separate functions to release memory by scopes
  auto rx = exchange_reachabilities(tree_parents_local, *reach_local);

  // Read `rx.data` in the original ordering of `tree_parents`
  bool changed = false;
  for (const auto p : tree_parents_local) {
    const auto child = p.first;
    const auto parent = p.second;

    if (!(*reach_local)[child.t]) {
      const int r = owner_rank(parent);
      if (rx.data[rx.displacements[r]] != 0) {
        assert(!(*reach_local)[child.t]);
        (*reach_local)[child.t] = true;
        changed = true;
      }
      rx.displacements[r] += 1;
    }
  }

  return changed;
}

//
// Returns reachability map for every local vertex.
//
// `core_parents_local` should give a parent for every core vertex in the giant
// connected component and have the length equal to the number of local
// vertices.
//
static std::vector<bool>
propagate_reachability(const parent_array &tree_parents,
                       const global_vertex *const core_parents_local) {
  const size_t n_local = tree_parents.size();
  std::vector<bool> reach_local(n_local);
  // Initial state: the core vertices in the GCC are reachable
  for (vertex u{0}; u.t < to_sig(n_local); ++u.t) {
    if (core_parents_local[u.t].t >= 0) {
      reach_local[u.t] = true;
    }
  }

  // Iterative process: propagate reachability in a bottom-up manner
  bool changed_global = true;
  for (int i = 0; changed_global; ++i) {
    LOG_I << "Removing unreachable tree parents: iteration " << i;
    const bool changed_local =
        update_reachabilities(tree_parents, &reach_local);
    changed_global = net::allreduce(changed_local, MPI_LOR, net::world());
  }

  return reach_local;
}

//
// Replaces the parents of the unreachable vertices with -1.
//
static void reset_unreachable(const global_vertex *const core_parents_local,
                              parent_array *const tree_parents) {
  const std::vector<bool> reach_local =
      propagate_reachability(*tree_parents, core_parents_local);

  size_t n_unreach_local = 0;
  tree_parents->map([&](const vertex child, const global_vertex parent) {
    if (reach_local[child.t]) {
      return parent;
    } else {
      ++n_unreach_local;
      return global_vertex{-1};
    }
  });

  const size_t n_unreach_global =
      net::reduce(n_unreach_local, MPI_SUM, 0, net::world());
  if (net::world().rank() == 0) {
    LOG_I << "Number of removed parents: " << n_unreach_global;
  }
}

} // namespace detail

using detail::reset_unreachable;
using detail::retain_core;

} // namespace decomposition
} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
