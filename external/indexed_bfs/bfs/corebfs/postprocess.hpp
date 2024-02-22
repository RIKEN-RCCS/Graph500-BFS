//
// Authored by ARAI Junya <araijn@gmail.com> in 2024-01-14.
//
#pragma once

#include "../../net.hpp"
#include "distribution.hpp"
#include "structures.hpp"

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace postprocess {
namespace detail {

using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using indexed_bfs::bfs::corebfs::distribution::yoo;
using indexed_bfs::bfs::corebfs::structures::parent_array;
using indexed_bfs::net::gathered_data;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::sort::sort_parallel;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

static gathered_data<vertex>
exchange_parents(const yoo &d, const parent_array &tree_parents_local,
                 const std::vector<bool> &reach_local) {
  std::vector<int> counts(d.all().size());
  for (const auto p : tree_parents_local) {
    const auto child = p.first;
    const auto parent = p.second;
    if (!reach_local[child.t]) {
      counts[d.owner_rank(parent)] += 1;
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
      const int r = d.owner_rank(parent);
      parents[is[r]] = d.to_local_remote(parent);
      is[r] += 1;
    }
  }

  return net::alltoallv(parents, counts, displs, d.all());
}

static gathered_data<uint8_t>
exchange_reachabilities(const yoo &d, const parent_array &tree_parents_local,
                        const std::vector<bool> &reach_local) {
  auto rx = exchange_parents(d, tree_parents_local, reach_local);

  std::vector<uint8_t> tx_reach(rx.data.size());
#pragma omp parallel for
  for (size_t i = 0; i < rx.data.size(); ++i) {
    tx_reach[i] = reach_local[rx.data[i].t];
  }

  rx.data = std::vector<vertex>(); // Release memory before receiving data
  return net::alltoallv(tx_reach, rx.counts, rx.displacements, d.all());
}

static bool update_reachabilities(const yoo &d,
                                  const parent_array &tree_parents_local,
                                  std::vector<bool> *const reach_local) {
  // Separate functions to release memory by scopes
  auto rx = exchange_reachabilities(d, tree_parents_local, *reach_local);

  // Read `rx.data` in the original ordering of `tree_parents`
  bool changed = false;
  for (const auto p : tree_parents_local) {
    const auto child = p.first;
    const auto parent = p.second;

    if (!(*reach_local)[child.t]) {
      const int r = d.owner_rank(parent);
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
propagate_reachability(const yoo &d, const parent_array &tree_parents,
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
        update_reachabilities(d, tree_parents, &reach_local);
    changed_global = net::allreduce(changed_local, MPI_LOR, d.all());
  }

  return reach_local;
}

//
// Replaces the parents of the unreachable vertices with -1.
//
static void reset_unreachable(const yoo &d,
                              const global_vertex *const core_parents_local,
                              parent_array *const tree_parents) {
  const std::vector<bool> reach_local =
      propagate_reachability(d, *tree_parents, core_parents_local);

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
      net::reduce(n_unreach_local, MPI_SUM, 0, d.all());
  if (d.all().rank() == 0) {
    LOG_I << "Number of removed parents: " << n_unreach_global;
  }
}

} // namespace detail

using detail::reset_unreachable;

} // namespace postprocess
} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
