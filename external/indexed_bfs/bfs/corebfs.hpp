//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-11-02.
//
#pragma once

#include "../argument.hpp"
#include "../common.hpp"
#include "../graph.hpp"
#include "../graph500.hpp"
#include "../net.hpp"
#include "corebfs/csr_1d.hpp"
#include "corebfs/decomposition.hpp"
#include "corebfs/postprocess.hpp"
#include "corebfs/structures.hpp"
#include <array>
#include <cstddef>

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace detail {

using namespace indexed_bfs::util;
using namespace indexed_bfs::graph;
using namespace indexed_bfs::bfs::corebfs::csr_1d;
using namespace indexed_bfs::bfs::corebfs::distribution;
using indexed_bfs::bfs::corebfs::distribution::make_distribution;
using indexed_bfs::bfs::corebfs::distribution::yoo;
using indexed_bfs::bfs::corebfs::postprocess::reset_unreachable;
using indexed_bfs::bfs::corebfs::structures::parent_array;
using indexed_bfs::net::gathered_data;
using indexed_bfs::util::memory::heap_size_of;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::show::show;
using indexed_bfs::util::sort::sort_parallel;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

struct bfs_index {
  yoo dist;
  // 2-core of a given graph.
  // Every connected vertex in an original, generated graph has a non-zero
  // degree in `core` or is contained in `tree_parents` (though its parent can
  // be `-1`).
  binary_csr core;
  // Parents of vertices in the tree parts of a given graph.
  parent_array tree_parents;
};

struct bfs_state {
  const bfs_index &ix;
  std::vector<global_vertex> parents_local;
};

static std::vector<edge_2d> distribute(const yoo &d,
                                       std::vector<edge> &&edges) {
  using namespace decomposition;

  // Assume that each process has almost even number of edges
  // x2 for symmetrization, and x1.2 for anticipating imbalance in distribution
  const size_t m_estimated = static_cast<size_t>(edges.size() * 2 * 1.2);
  distributor distr(d, m_estimated);
  INDEXED_BFS_LOG_RSS();

  const size_t giga_bytes = static_cast<size_t>(1) << 30;
  // x2 for symmetrization
  const size_t chunk_len = giga_bytes / (sizeof(edge_2d) * 2);

  for (size_t i = 0; i < edges.size(); i += chunk_len) {
    const size_t ilast = std::min(i + chunk_len, edges.size());
    distr.feed(edges.begin() + i, edges.begin() + ilast);
    LOG_I << "Completed " << ilast << "/" << edges.size() << " edges";
    INDEXED_BFS_LOG_RSS();
  }

  auto edges_2d = distr.drain();
  const size_t m = net::reduce(edges_2d.size(), MPI_SUM, 0, d.all());
  LOG_I << "preprocessed_edge_count: " << m;

  return edges_2d;
}

static bfs_index construct(const int scale, std::vector<edge> &&edges) {
  using namespace decomposition;

  auto d = make_distribution(scale);

  LOG_I << "Distributing edges...";
  auto edges_2d = distribute(d, std::move(edges));
  INDEXED_BFS_LOG_RSS();

  LOG_I << "Finding 2-core...";
  std::vector<global_vertex> parents_local = prune_trees(d, &edges_2d);
  assert(parents_local.size() == d.local_vertex_count());

  LOG_I << "Compressing a parent array...";
  parent_array tree_parents(std::move(parents_local));
  INDEXED_BFS_LOG_RSS();

  auto core = csr_1d::build(d, std::vector<edge>());
  //  LOG_I << "Constructing 1D-distribution CSR...";
  //  for (const auto &e : edges_2d) {
  //    const auto f = make_edge(d.to_global(e.source()),
  //    d.to_global(e.target())); if (f.source() < f.target()) {
  //      edges.push_back(f);
  //    }
  //  }
  //  edges_2d = std::vector<edge_2d>(); // Release memory
  //  auto core = csr_1d::build(d, std::move(edges));
  //
  //  const auto parent_array_sizes =
  //      net::gather(heap_size_of(tree_parents), 0, d.all());
  //  if (d.all().rank() == 0) {
  //    const auto s = common::five_number_summary(parent_array_sizes);
  //    LOG_I << "Five-number summary of the heap size of parent_array per "
  //             "process: "
  //          << show(s);
  //  }

  return {std::move(d), std::move(core), std::move(tree_parents)};
}

// `Callback` should be `void c(vertex u, global_vertex parent)`.
// It is called when the parent of vertex `u` is found.
template <typename Callback>
static global_vertex
bfs_tree_with_callback(const yoo &d, const parent_array &tree_parents,
                       const global_vertex root, Callback c) {
  global_vertex u = root;
  global_vertex p = root; // Precomputed parent of `u`; first, it's `u` itself

  //
  // Ascend to the 2-core.
  // Except the first iteration, negative `p` means that `u` is in the 2-core.
  //
  // Be careful that this may loop forever if `tree_parents` is set for
  // connected components that do not contain 2-core.
  // It is supposed not to occur after the parents of vertices not in the giant
  // connected component are removed from `tree_parents`.
  //
  while (p.t >= 0) {
    if (d.owner_rank(p) == d.all().rank()) {
      c(d.to_local_remote(p), u);
    }
    u = p; // Move to the parent

    // Get the parent of `u`
    if (d.owner_rank(u) == d.all().rank()) {
      p = tree_parents[d.to_local_remote(u)];
    }
    p = net::bcast(p, d.owner_rank(u), d.all());
  }

  return u;
}

static global_vertex bfs_tree(bfs_state *const s, const global_vertex root) {
  return bfs_tree_with_callback(
      s->ix.dist, s->ix.tree_parents, root,
      [&](vertex u, global_vertex p) { s->parents_local[u.t] = p; });
}

//
// `core_root` must be in the 2-core.
//
static void bfs_core(bfs_state *const, const global_vertex) {}

static void bfs(bfs_state *const s, const global_vertex root) {
  if (net::world().rank() == 0) {
    std::cout << "  root: " << root << std::endl;
  }

  const global_vertex core_root = bfs_tree(s, root);
  bfs_core(s, core_root);
}

//
// For debugging; cannot handle large graphs
//
static binary_csr make_global_csr(const int scale,
                                  std::vector<edge> edges_global) {
  LOG_I << "Remove loop";
  // Remove loop
  edges_global.erase(
      std::remove_if(edges_global.begin(), edges_global.end(),
                     [](auto &e) { return e.source() == e.target(); }),
      edges_global.end());

  LOG_I << "Symmetrize";
  // Symmetrize (bi-directionalize)
  const size_t m_old = edges_global.size();
  edges_global.reserve(edges_global.size() * 2);
  for (size_t i = 0; i < m_old; ++i) {
    const auto &e = edges_global[i];
    edges_global.push_back(make_edge(e.target(), e.source()));
  }

  // Sort (expensive)
  LOG_I << "Start sorting edges";
  sort_parallel(edges_global.begin(), edges_global.end());
  LOG_I << "Completed sorting edges";

  // Deduplicate
  edges_global.erase(std::unique(edges_global.begin(), edges_global.end()),
                     edges_global.end());
  edges_global.shrink_to_fit();

  // Build a CSR
  return binary_csr::from_groups_by(
      edges_global.begin(), edges_global.end(), global_vertex_count(scale),
      [](auto &e) { return e.source().t; }, [](auto &e) { return e.target(); });
}

static std::pair<std::vector<global_vertex>, std::vector<int8_t>>
bfs_local(const binary_csr &csr, const global_vertex root) {
  std::vector<global_vertex> parents(vertex_count(csr), global_vertex{-1});
  std::vector<int8_t> dists(vertex_count(csr), -1);
  bit::bit_vector front(vertex_count(csr));
  bit::bit_vector next(vertex_count(csr));
  size_t n_front = 0;

  parents[root.t] = root;
  dists[root.t] = 0;

  for (const auto v : csr[root.t]) {
    parents[v.t] = root;
    dists[v.t] = 1;
    front.set(v.t);
  }
  n_front = degree(csr, make_vertex_from(root.t));

  for (int d = 2; n_front > 0; ++d) {
    next.clear();
    std::atomic<size_t> n_next(0);

#pragma omp parallel for schedule(static, 1)
    for (vertex_int i = 0; i < to_sig(vertex_count(csr)); ++i) {
      if (!front[i]) {
        continue;
      }

      const auto u = global_vertex{i};
      for (const auto v : csr[u.t]) {
        if (dists[v.t] < 0) {
          parents[v.t] = u;
          dists[v.t] = d;
          next.set(v.t);
          n_next.fetch_add(1);
        }
      }
    }

    std::swap(front, next);
    n_front = n_next.load();
  }

  return std::make_pair(std::move(parents), std::move(dists));
}

static global_vertex find_highest_degree_vertex(const binary_csr &g_global) {
  // Find the vertex of the greatest degree
  global_vertex u{0};
  for (global_vertex v{0}; v.t < to_sig(vertex_count(g_global)); ++v.t) {
    const size_t d_u = degree(g_global, make_vertex_from(u.t));
    const size_t d_v = degree(g_global, make_vertex_from(v.t));
    if (d_u < d_v) {
      u = v;
    }
  }
  return u;
}

static std::vector<global_vertex> emulate_dist_bfs(const yoo &d,
                                                   const binary_csr &g_global,
                                                   const global_vertex root) {

  const auto parents_global = bfs_local(g_global, root).first;

  auto parents_local =
      make_with_capacity<std::vector<global_vertex>>(d.local_vertex_count());
  for (vertex u{0}; u.t < to_sig(d.local_vertex_count()); ++u.t) {
    parents_local.push_back(parents_global[d.to_global(u).t]);
  }

  return parents_local;
}

static bool validate_tree_parents(const bfs_index &ix,
                                  const binary_csr &g_global,
                                  const global_vertex gcc_root) {
  const auto true_parents_local = emulate_dist_bfs(ix.dist, g_global, gcc_root);

  bool success = true;
  for (vertex u(0); u.t < to_sig(ix.dist.local_vertex_count()); ++u.t) {
    const global_vertex u_global = ix.dist.to_global(u);

    //
    // Parents for any core vertices should not be contained
    //
    if (degree(ix.core, u) > 0 && ix.tree_parents[u].t >= 0) {
      LOG_E << "core_paernt: {u: " << u_global
            << ", value: " << ix.tree_parents[u]
            << ", core_degree: " << degree(ix.core, u) << "}";
      success = false;
    }

    //
    // Parents for every tree vertex should equal to the ground truth
    //
    const bool is_connected_tree =
        true_parents_local[u.t].t >= 0 && degree(ix.core, u) == 0;
    if (is_connected_tree && ix.tree_parents[u] != true_parents_local[u.t]) {
      LOG_E << "tree_parent_mismatch: {u: " << u_global
            << ", expected: " << true_parents_local[u.t]
            << ", actual: " << ix.tree_parents[u] << "}";
      success = false;
    }
  }
  return success;
}

static std::vector<global_vertex>
emulate_dist_bfs_core(const yoo &d, const binary_csr &g_global,
                      const global_vertex root, const binary_csr &core) {
  auto parents_local = emulate_dist_bfs(d, g_global, root);

  for (vertex u{0}; u.t < to_sig(d.local_vertex_count()); ++u.t) {
    if (degree(core, u) == 0) {
      parents_local[u.t] = global_vertex{-1};
    }
  }

  return parents_local;
}

static std::vector<global_vertex>
sample_roots(const bfs_index &ix, const int scale, const size_t n_roots) {
  LOG_I << "Start sampling search keys";
  const auto roots = graph500::sample_roots(
      scale, n_roots, ix.dist.all(), [&](global_vertex u) {
        return ix.dist.owner_rank(u) == ix.dist.all().rank() &&
               degree(ix.core, ix.dist.to_local(u)) > 0;
      });
  LOG_I << "Completed sampling search keys";

  return roots;
}

static std::vector<global_vertex>
gather_parents(const yoo &d, const std::vector<global_vertex> &parents_local) {
  const auto r = net::allgatherv(parents_local, d.all());

  std::vector<global_vertex> parents_global(d.global_vertex_count());
  for (int rank = 0; rank < d.all().size(); ++rank) {
    for (int i = 0; i < r.counts[rank]; ++i) {
      const global_vertex u = d.to_global(vertex{i}, rank);
      parents_global[u.t] = r.data[r.displacements[rank] + i];
    }
  }

  return parents_global;
}

static graph500::result run(const argument::arguments &args) {
  INDEXED_BFS_LOG_RSS();

  LOG_I << "Start generating edges";
  auto edges_local = graph500::generate_edges(args.scale);
  LOG_I << "Completed generating edges";
  INDEXED_BFS_LOG_RSS();

  //  LOG_I << "For test: start constructing a global CSR";
  //  const auto edges_global = net::allgatherv(edges_local, net::world()).data;
  //  const auto g_global = make_global_csr(args.scale, edges_global);
  //  LOG_I << "For test: completed constructing a global CSR";

  LOG_I << "Start constructing a graph";
  const auto construction_start = time::now();
  auto ix = construct(args.scale, std::move(edges_local));
  const auto construction_secs = time::elapsed_secs(construction_start);
  LOG_I << "Completed constructing a graph";
  INDEXED_BFS_LOG_RSS();

  return graph500::result{construction_secs};

  //  LOG_I << "Start removing parents outside of the GCC";
  //  const global_vertex gcc_root = find_highest_degree_vertex(g_global);
  //  reset_unreachable(
  //      ix.dist,
  //      emulate_dist_bfs_core(ix.dist, g_global, gcc_root, ix.core).data(),
  //      &ix.tree_parents);
  //  LOG_I << "Completed removing parents outside of the GCC";
  //
  //  OR_DIE(validate_tree_parents(ix, g_global, gcc_root));
  //
  //  const auto roots = sample_roots(ix, args.scale, args.root_count);
  //
  //  if (net::world().rank() == 0) {
  //    std::cout << "searches:" << std::endl;
  //  }
  //
  //  for (size_t i_root = 0; i_root < roots.size(); ++i_root) {
  //    if (net::world().rank() == 0) {
  //      std::cout << "- index: " << i_root << std::endl;
  //    }
  //
  //    bfs_state s{ix, std::vector<global_vertex>(vertex_count(ix.core),
  //                                               global_vertex{-1})};
  //
  //    for (const auto p : ix.tree_parents) {
  //      s.parents_local[p.first.t] = p.second;
  //    }
  //
  //    bfs(&s, roots[i_root]);
  //
  //    const auto true_parents_local =
  //        emulate_dist_bfs(ix.dist, g_global, roots[i_root]);
  //
  //    for (vertex u{0}; u.t < to_sig(ix.dist.local_vertex_count()); ++u.t) {
  //      if (degree(ix.core, u) == 0) {
  //        if (s.parents_local[u.t] != true_parents_local[u.t]) {
  //          LOG_E << "tree_parent_mismatch: u = " << u
  //                << ", s.parents_local[u.t] = " << s.parents_local[u.t]
  //                << ", true_parents_local[u.t] = " <<
  //                true_parents_local[u.t];
  //        }
  //      }
  //    }
  //  }
  //
  //  return graph500::result{construction_secs};
}

} // namespace detail

using detail::run;

// Exports for Fugaku
using detail::bfs_tree_with_callback;
using detail::binary_csr;
using detail::construct;
using detail::vertex_count;

} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
