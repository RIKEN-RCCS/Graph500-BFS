//
// Adaptor between Graph500-BFS and CoreBFS
//
// Authored by ARAI Junya <araijn@gmail.com> on 2024-01-09.
//
#ifndef COREBFS_ADAPTOR_HPP
#define COREBFS_ADAPTOR_HPP

#include <mpi.h>

namespace indexed_bfs {
namespace net {
namespace detail {

// Wrapper of `MPI_Comm` with member functions corresponding to `MPI_Comm_*()`.
struct comm {
  MPI_Comm t;  // Named after an internal value of BOOST_STRONG_TYPEDEF
  int rank_;

  int rank() const { return rank_; }

  int size() const {
    int x;
    MPI_Comm_size(this->t, &x);
    return x;
  }
};

static const comm &world();

}  // namespace detail
}  // namespace net
}  // namespace indexed_bfs

#pragma push_macro("__builtin_ctz")
#pragma push_macro("__builtin_popcount")
#undef __builtin_ctz
#undef __builtin_popcount
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
// Block include of "make_graph.h" since it generates link errors
#define MAKE_GRAPH_H

#include "indexed_bfs/bfs/corebfs.hpp"

#pragma GCC diagnostic pop
#pragma pop_macro("__builtin_ctz")
#pragma pop_macro("__builtin_popcount")

// Headers of Graph500-BFS must be included after those of indexed_bfs
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"

#include <random>

static const indexed_bfs::net::detail::comm &indexed_bfs::net::detail::world() {
  static comm c = comm{mpi.comm_2d, mpi.rank_2d};
  return c;
}

namespace corebfs_adaptor {
namespace detail {

using namespace indexed_bfs::bfs::corebfs;
using namespace indexed_bfs::bfs;
using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using namespace indexed_bfs;
using indexed_bfs::bfs::corebfs::binary_csr;
using indexed_bfs::bfs::corebfs::dist_graph::to_global;
using indexed_bfs::bfs::corebfs::dist_graph::to_local;
using indexed_bfs::bfs::corebfs::parent_array::parent_array;
using util::memory::make_with_capacity;
using util::types::to_sig;
using util::types::to_unsig;

using edge_storage = EdgeListStorage<UnweightedPackedEdge>;

static void print_log(const log::record &r) {
  const char *const sev = log::severity_string(r.severity);
  print_with_prefix("%-5s [%s@%d] %s", sev, r.function, r.line, r.message);
}

static edge to_edge(const UnweightedPackedEdge &e) {
  return make_edge(global_vertex{e.v0()}, global_vertex{e.v1()});
}

static UnweightedPackedEdge make_unweighted_packed_edge(const global_vertex s,
                                                        const global_vertex t) {
  UnweightedPackedEdge e;
  e.set(s.t, t.t);
  return e;
}

//
// Reads all the edges in `stor` into a vector.
//
static std::vector<edge> read_all_edges(edge_storage *const stor) {
  auto vec = make_with_capacity<std::vector<edge>>(stor->edge_filled_size());
  const int n = stor->beginRead(false);
  for (int i = 0; i < n; ++i) {
    UnweightedPackedEdge *p;
    const int n = stor->read(&p);
    if (n <= 0) {
      // `n` can be zero because `beginRead()` returns the maximum number of
      // chunks among processes.
      break;
    }
    std::transform(p, p + n, std::back_inserter(vec), to_edge);
  }
  stor->endRead();

  assert(to_sig(vec.size()) <= stor->edge_filled_size());
  return vec;
}

//
// Writes all the edges in `g` to `output`.
// Note that `g` is assumed to be symmetric, and this function writes edge
// `(s, t)` s.t. `s < t` only.
//
static void write_all_edges(const binary_csr &g, edge_storage *output) {
  auto chunk = make_with_capacity<std::vector<UnweightedPackedEdge>>(
      edge_storage::CHUNK_SIZE);

  output->beginWrite();

  vertex s_local(0);
  size_t i_value = 0;
  while (s_local.t < to_sig(vertex_count(g))) {
    chunk.clear();

    for (; s_local.t < to_sig(vertex_count(g)); ++s_local.t) {
      for (; i_value < g.offsets_[s_local.t + 1]; ++i_value) {
        if (chunk.size() >= edge_storage::CHUNK_SIZE) {
          goto flush_chunk;
        }

        const global_vertex s = corebfs::to_global(s_local);
        const global_vertex t = g.values_[i_value];
        if (s < t) {
          chunk.push_back(make_unweighted_packed_edge(s, t));
        }
      }
    }

  flush_chunk:
    output->write(chunk.data(), chunk.size());
  }

  output->endWrite();
}

//
// Verifies the equality of the rank assignments and the vertex distribution
// between Graph500-BFS and indexed_bfs.
//
static void test_rank_mapping(const int scale) {
  using namespace std;

  OR_DIE(mpi.size_2d == net::world().size());
  OR_DIE(mpi.rank_2d == net::world().rank());

  const size_t n_vertices = static_cast<size_t>(1) << scale;
  random_device dev;
  mt19937_64 rng(dev());
  uniform_int_distribution<mt19937_64::result_type> dist(0, n_vertices - 1);
  for (int i = 0; i < 100; ++i) {
    const int64_t u = dist(rng);
    OR_DIE(corebfs::owner_rank(global_vertex{u}) == vertex_owner(u));
  }
}

static void init(const int scale) {
  // Show all messages only in rank 0
  log::config.min_severity =
      net::world().rank() == 0 ? log::severity::trace : log::severity::warning;
  log::config.printer = print_log;

  test_rank_mapping(scale);

  if (mpi.rank_2d == 0) {
    std::atexit(time::print_all_scopes);
  }
}

//
// Apply preprocesses of CoreBFS.
// Specifically,
// 1. copy the edges in the 2-core from `input` to `output`, and
// 2. returns parents of vertices that are not in the 2-core.
//
// `input == output` is allowed.
//
static parent_array preprocess(const int scale, edge_storage *input,
                               edge_storage *output) {
  static bool initialized = false;
  if (!initialized) {
    init(scale);
    initialized = true;
  }

  auto ix = corebfs::construct(scale, read_all_edges(input));
  write_all_edges(ix.core, output);
  return std::move(ix.tree_parents);
}

static int64_t find_highest_degree_vertex(const Graph2DCSR &g) {
  // Since vertices are in the descending order of degree, the first reordered
  // vertex should have the greatest degree in this process.
  const vertex root_local = vertex(g.invert_map_[0]);
  const int64_t deg = g.degree_[root_local.t];
  global_vertex root_global = to_global(root_local);

  struct long_int {
    long deg;
    int rank;
  };
  long_int tx = {deg, mpi.rank_2d};
  long_int rx;
  // TODO: Implement `max_by_key`
  MPI_Allreduce(&tx, &rx, 1, MPI_LONG_INT, MPI_MAXLOC, mpi.comm_2d);
  return net::bcast(root_global, rx.rank, net::world()).t;
}

//
// Returns a vertex ID packed in an entry obtained from `BfsOnCPU::get_pred()`.
// See `BfsValidation::get_pred_from_pred_entry()`, which is a private function.
//
static global_vertex get_pred_from_pred_entry(const int64_t pred_entry) {
  return global_vertex{(pred_entry << 16) >> 16};
}

//
// Removes the parents of vertices not in the giant connected component.
// This function assumes that a vertex with the highest degree is in the giant
// connected component.
//
// `typename Bfs` is parameterized because this header is included from
// `bfs.hpp`, and so `BfsOnCPU` is unavailable here.
//
template <typename Bfs>
static void reset_unreachable(Bfs *const bfs, const int edge_factor,
                              const double alpha, const double beta,
                              int64_t *const pred,
                              parent_array *const tree_parents) {
  const int64_t root_global = find_highest_degree_vertex(bfs->graph_);
  LOG_I << "Highest-degree vertex: " << root_global;

  int64_t auto_tuning_data[AUTO_NUM];  // Not used
  bfs->run_bfs_core(root_global, pred, edge_factor, alpha, beta,
                    auto_tuning_data);
  bfs->get_pred(pred);

  for (vertex u{0}; u.t < bfs->graph_.pred_size(); ++u.t) {
    pred[u.t] = get_pred_from_pred_entry(pred[u.t]).t;
  }

  // Unsafe conversion from `int64_t` to `global_vertex`
  static_assert(std::is_same<int64_t, global_vertex_int>::value, "");
  static_assert(sizeof(int64_t) == sizeof(global_vertex), "");
  corebfs::reset_unreachable(reinterpret_cast<global_vertex *>(pred),
                             tree_parents);
}

//
// Traverse the tree part of the graph, correcting parents of tree vertices.
//
// This function returns `(core_root, [(u, parent)])`, where
// - `core_root` is the core vertex closest to `root`,
// - `u` is a local tree vertex that has a parent differs from that in
//   `tree_parents`, and
// - `parent` is the parent of `u`.
//
// `[(u, parent)]` would be empty on most of the ranks because a path from
// `root` to the 2-core is expected to be short (<10 even for SCALE 43).
//
static std::pair<int64_t, std::vector<std::pair<LocalVertex, int64_t>>>
bfs_tree(const parent_array &tree_parents, const global_vertex_int root) {
  INDEXED_BFS_TIMED_SCOPE(bfs_tree);

  // No reservation because it remains empty in most cases
  std::vector<std::pair<LocalVertex, int64_t>> path;

  const global_vertex core_root = corebfs::bfs_tree_with_callback(
      tree_parents, global_vertex{root}, [&](vertex u, global_vertex parent) {
        path.push_back(std::make_pair(u.t, parent.t));
      });

  return std::make_pair(core_root.t, std::move(path));
}

//
// Write out the parents of tree vertices to `pred`.
//
static void write_tree_parents(const parent_array &tree_parents,
                               int64_t *const pred) {
  INDEXED_BFS_TIMED_SCOPE(write_tree_parents);

  // `dump()` writes out each element of `pred` as if it is simple 64-bit parent
  // IDs although it is actually a concatenation of 16-bit depth and 48-bit
  // parent ID. As a result, the depth parts of all the element become zero.
  // The depths need to be computed in the validation phase.
  tree_parents.dump(reinterpret_cast<global_vertex *>(pred));
}

//
// Returns `true` if the parent of `u_global` is in `tree_parents`.
//
static bool contains(const parent_array &tree_parents, const int64_t u_global) {
  const global_vertex u_global_(u_global);
  return owner_rank(u_global_) == net::world().rank() &&
         tree_parents.contains(to_local(u_global_));
}

}  // namespace detail

using detail::bfs_tree;
using detail::contains;
using detail::parent_array;
using detail::preprocess;
using detail::reset_unreachable;
using detail::write_tree_parents;

}  // namespace corebfs_adaptor

#endif  // COREBFS_ADAPTOR_HPP
