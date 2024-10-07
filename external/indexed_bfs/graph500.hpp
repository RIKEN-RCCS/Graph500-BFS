//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-11-06.
//
#pragma once

#include "argument.hpp"
#include "common.hpp"
#include "graph.hpp"
#include "net.hpp"
#include <atomic>
#include <bitset>
#include <omp.h>
#include <random>

#include "generator/make_graph.h"
#include "generator/utils.h"

namespace indexed_bfs {
namespace graph500 {
namespace detail {

using namespace util;
using graph::edge;
using graph::global_vertex;
using graph::global_vertex_int;
using memory::make_with_capacity;
using types::to_sig;

// Since our algorithm cannot directly detect a loop, we assume that there
// exists a loop if a distance exceeds this number
static constexpr size_t max_distance = 64;

struct result {
  double construction_secs;
};

struct distance_calculator {
  MOVE_ONLY(distance_calculator)

  const global_vertex *parents_;
  size_t n_vertices_;
  std::vector<std::atomic<global_vertex_int>> dists_;

  //
  // Returns the positive distance between `u` and the search key if `u` is
  // reachable and the BFS tree does not contain any loop.
  // Returns -1 if `u` is unreachable and -2 if there exist loops.
  //
  global_vertex_int query(const global_vertex u, const global_vertex_int dist) {
    // Is `u` in a loop?
    if (dist > to_sig(max_distance)) {
      return -1;
    }

    // Is the distance of `u` already computed?
    const auto d = dists_[u.t].load();
    if (d >= 0) {
      return d;
    }

    // Is `u` unreachable?
    const auto p = parents_[u.t];
    if (p.t < 0) {
      return -1;
    }

    // Is `u` the search key (root)?
    if (p.t == u.t) {
      return 0;
    }

    // Recursive call to compute the distance of the parent of `u`
    const auto dp = query(p, dist + 1);
    if (dp < 0) {
      return dp; // -1 if unreachable, -2 if a loop exists
    }

    // Memorize the distance of `u`
    dists_[u.t].store(dp + 1);
    return dp + 1;
  }
};

distance_calculator make_distance_calculator(const global_vertex *const parents,
                                             const size_t n_vertices) {
  std::vector<std::atomic<global_vertex_int>> dists(n_vertices);
  for (auto &d : dists) {
    d.store(-1);
  }

  return distance_calculator{
      parents,
      n_vertices,
      std::move(dists),
  };
}

static std::vector<edge> generate_edges(const int scale) {
  //
  // Define `INDEXED_BFS_NO_EDGE_GENERATION` in Graph500-BFS; otherwise,
  // indexed_bfs requires functions defined in the Graph500 reference
  // implementation (i.e., `make_mrg_seed` and `generate_kronecker_range`)
  //
#ifdef INDEXED_BFS_NO_EDGE_GENERATION

  static_cast<void>(scale);
  assert(!"Unreachable");
  return std::vector<edge>();

#else

  const int n_procs = net::world().size();
  const int rank = net::world().rank();

  uint_fast32_t seed[5];
  make_mrg_seed(common::random_seed1, common::random_seed2, seed);

  const auto m_global = graph::global_edge_count(scale);
  const auto range_local = common::split(m_global, n_procs, rank);
  const auto m_local = range_local.second - range_local.first;

  std::vector<edge> edges(m_local);
#pragma omp parallel
  {
    const auto nt = omp_get_num_threads();
    const auto tid = omp_get_thread_num();
    const auto range_thread = common::split(m_local, nt, tid);

    generate_kronecker_range(
        seed, scale, range_local.first + range_thread.first,
        range_local.first + range_thread.second,
        reinterpret_cast<packed_edge *>(&edges[range_thread.first]));
  }

  return edges;

#endif // INDEXED_BFS_NO_EDGE_GENERATION
}

static size_t count_vertices(const std::vector<edge> &edges) {
  global_vertex u{-1};
  for (const auto &e : edges) {
    u = std::max(u, std::max(e.source(), e.target()));
  }
  return u.t + 1;
}

static bit::bit_vector check_connectivity(const std::vector<edge> &edges) {
  const auto n = count_vertices(edges);
  bit::bit_vector vec(n);

#pragma omp parallel for
  for (size_t i = 0; i < edges.size(); ++i) {
    vec.set(edges[i].source().t);
    vec.set(edges[i].target().t);
  }

  return vec;
}

//
// `is_connected(global_vertex u)` should return `true` on AT LEAT ONE MPI
// PROCESS if `u` has an incident edge.
// In other words, if the graph is partitioned among multiple processes,
// processes that do not have an edge incident to `u` need not to return `true`.
//
template <typename F>
static std::vector<global_vertex>
sample_roots(const int scale, const size_t n_roots, const net::comm &c,
             F is_connected) {
  std::seed_seq seed{common::random_seed1, common::random_seed2};
  std::mt19937 gen(seed);
  const global_vertex_int v_max =
      (static_cast<global_vertex_int>(1) << scale) - 1;
  std::uniform_int_distribution<global_vertex_int> dist(0, v_max);

  auto roots = make_with_capacity<std::vector<global_vertex>>(n_roots);
  for (;;) {
    // Generate 64 random vertices and a connectivity bitmap.
    // Note that all the MPI processes get the same `candidates` because they
    // use the same seeds.
    auto candidates = make_with_capacity<std::vector<global_vertex>>(64);
    uint64_t connected_local = 0;
    for (size_t i = 0; i < 64; ++i) {
      const auto u = global_vertex{dist(gen)};
      candidates.push_back(u);
      if (is_connected(u)) {
        connected_local |= static_cast<uint64_t>(1) << i;
      }
    }

    // Compute bit-wise OR of the connectivity bitmap
    const std::bitset<64> connected_global(
        net::allreduce(connected_local, MPI_BOR, c));

    for (size_t i = 0; i < 64; ++i) {
      if (connected_global[i]) {
        roots.push_back(candidates[i]);
        if (roots.size() >= n_roots) {
          return roots;
        }
      }
    }
  }
}

static std::vector<global_vertex> sample_roots(const int scale,
                                               const size_t n_roots,
                                               const std::vector<edge> &edges) {
  const auto connected = check_connectivity(edges);
  return sample_roots(scale, n_roots, net::world(), [&](auto u) {
    return u.t < to_sig(connected.size()) && connected[u.t];
  });
}

//
// Returns an array of distances computed from the BFS tree if it does not
// contain a cycle; otherwise, returns a vertex whose parents make a cycle.
// Note that the distances of unreachable vertices are set to -1.
//
static std::vector<global_vertex_int>
compute_distances(const global_vertex *const parents,
                  const global_vertex_int n_vertices,
                  global_vertex *const loop_vertex) {
  assert(loop_vertex->t < 0);

  std::vector<global_vertex_int> dists(n_vertices, -1);
  global_vertex_int u_cycle = n_vertices;
  auto dc = make_distance_calculator(parents, n_vertices);

#pragma omp parallel for reduction(min : u_cycle)
  for (global_vertex_int u = 0; u < n_vertices; ++u) {
    const auto d = dc.query(global_vertex{u}, 0);
    if (d == -2) {
      if (u_cycle == n_vertices) {
        u_cycle = u;
      }
    } else {
      assert(d >= -1);
      dists[u] = d;
    }
  }

  if (u_cycle < n_vertices) {
    *loop_vertex = global_vertex{u_cycle};
    return std::vector<global_vertex_int>();
  } else {
    return dists;
  }
}

static bool validate(const global_vertex /* root */,
                     const std::vector<edge> &edges,
                     const global_vertex *const parents) {
  const auto n = count_vertices(edges);

  global_vertex loop_vertex{-1};
  const auto dists = compute_distances(parents, n, &loop_vertex);
  if (loop_vertex.t >= 0) {
    LOG_E << "Parents of vertex " << loop_vertex.t << " form a loop";
    return false;
  }

  bool success = true;

  for (size_t i = 0; i < edges.size(); ++i) {
    const auto s = edges[i].source().t;
    const auto t = edges[i].target().t;

    if ((dists[s] < 0 && dists[t] >= 0) || (dists[s] >= 0 && dists[t] < 0)) {
      LOG_E << "Edge between visited and unvisited vertices: "
            << "vertex " << s << " (distance " << dists[s] << ") "
            << "and vertex " << t << " (distance " << dists[t] << ")";
      success = false;
      break;
    }

    if (std::abs(dists[s] - dists[t]) > 1) {
      LOG_E << "Edge between vertices of the distance more than 1: "
            << "vertex " << s << " (distance " << dists[s] << ") "
            << "and vertex " << t << " (distance " << dists[t] << ")";
      success = false;
      break;
    }
  }

  return success;
}

static void print_result(const argument::arguments &args, const result &r) {
  std::cout << "graph500_result:\n";
  std::cout << "  SCALE: " << args.scale << "\n";
  std::cout << "  edgefactor: " << common::edge_factor << "\n";
  std::cout << "  NBFS: " << args.root_count << "\n";
  std::cout << "  construction_time: " << r.construction_secs << "\n";
  std::cout << std::endl;
}

} // namespace detail

using detail::count_vertices;
using detail::generate_edges;
using detail::print_result;
using detail::result;
using detail::sample_roots;
using detail::validate;

} // namespace graph500
} // namespace indexed_bfs
