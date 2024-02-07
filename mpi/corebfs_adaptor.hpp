//
// Adaptor between Graph500-BFS and CoreBFS
//
// Authored by ARAI Junya <araijn@gmail.com> on 2024-01-09.
//
#ifndef COREBFS_ADAPTOR_HPP
#define COREBFS_ADAPTOR_HPP

#include <mpi.h>

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

// static const indexed_bfs::net::detail::comm
// &indexed_bfs::net::detail::world() {
//   static comm c = comm{mpi.comm_2d, mpi.rank_2d};
//   return c;
// }

namespace corebfs_adaptor {
namespace detail {

using namespace indexed_bfs::bfs;
using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using namespace indexed_bfs;
using namespace indexed_bfs::bfs::corebfs;
using namespace indexed_bfs::bfs::corebfs::distribution;
using namespace indexed_bfs::bfs::corebfs::decomposition;
using namespace indexed_bfs::bfs::corebfs::structures;
using util::memory::make_with_capacity;
using util::types::to_sig;
using util::types::to_unsig;

using edge_storage = EdgeListStorage<UnweightedPackedEdge>;

static void print_log(const log::record &r) {
  const char *const sev = log::severity_string(r.severity);
  print_with_prefix("%-5s [%s@%d] %s", sev, r.function, r.line, r.message);
}

static edge make_edge(const UnweightedPackedEdge &e) {
  return make_edge(global_vertex{e.v0()}, global_vertex{e.v1()});
}

static UnweightedPackedEdge make_unweighted_packed_edge(const global_vertex s,
                                                        const global_vertex t) {
  UnweightedPackedEdge e;
  e.set(s.t, t.t);
  return e;
}

//
// Returns a vertex ID packed in an entry obtained from `BfsOnCPU::get_pred()`.
// See `BfsValidation::get_pred_from_pred_entry()`, which is a private function.
//
static global_vertex get_pred_from_pred_entry(const int64_t pred_entry) {
  return global_vertex{(pred_entry << 16) >> 16};
}

static std::vector<edge_2d> distribute(const yoo &d, edge_storage *const stor) {
  const size_t m_estimate =
      static_cast<size_t>(stor->edge_filled_size() * 2 * 1.2);
  distributor distr(d, m_estimate);

  const int n = stor->beginRead(false);
  for (int i = 0; i < n; ++i) {
    UnweightedPackedEdge *p;
    const int n = stor->read(&p);

    auto edges = make_with_capacity<std::vector<edge>>(n);
    std::transform(p, p + n, std::back_inserter(edges), make_edge);

    distr.feed(edges.begin(), edges.end());
  }
  stor->endRead();

  return distr.drain();
}

//
// Writes all the edges in `edges_2d` to `output`.
// Note that `edges_2d` is assumed to be symmetric, and this function writes
// edge `(s, t)` s.t. `s < t` only.
//
static void write(const yoo &d, const std::vector<edge_2d> &edges_2d,
                  edge_storage *const output) {
  output->beginWrite();

  for (size_t i = 0; i < edges_2d.size(); i += edge_storage::CHUNK_SIZE) {
    const size_t ilast =
        std::min(i + edge_storage::CHUNK_SIZE, edges_2d.size());
    auto chunk =
        make_with_capacity<std::vector<UnweightedPackedEdge>>(ilast - i);
    for (size_t j = i; j < ilast; ++j) {
      const global_vertex u = d.to_global(edges_2d[j].source());
      const global_vertex v = d.to_global(edges_2d[j].target());
      if (u < v) {
        chunk.push_back(make_unweighted_packed_edge(u, v));
      }
    }

    output->write(chunk.data(), chunk.size());
  }

  output->endWrite();
}

static void init(const int scale) {
  // Show all messages only in rank 0
  log::config.min_severity =
      mpi.rank_2d == 0 ? log::severity::trace : log::severity::warning;
  log::config.printer = print_log;

  if (mpi.rank_2d == 0) {
    std::atexit(time::print_all_scopes);
  }
}

static int64_t find_highest_degree_vertex(const Graph2DCSR &g) {
  // Since vertices are in the descending order of degree, the first reordered
  // vertex should have the greatest degree in this process.
  const int64_t root_local = g.invert_map_[0];
  const int64_t deg = g.degree_[root_local];

  struct long_int {
    long deg;
    int rank;
  };
  long_int tx = {deg, mpi.rank_2d};
  long_int rx;
  MPI_Allreduce(&tx, &rx, 1, MPI_LONG_INT, MPI_MAXLOC, mpi.comm_2d);

  const int64_t root_global = root_local * mpi.size_2d + mpi.rank_2d;
  return net::bcast(root_global, rx.rank, net::world());
}

////////////////////////////////////////////////////////////////////////////////
//
// corebfs_index
//
////////////////////////////////////////////////////////////////////////////////

class corebfs_index;
static corebfs_index preprocess(const int scale, edge_storage *const input,
                                edge_storage *const output);

class corebfs_index : types::noncopyable {
 public:
  yoo dist_;
  parent_array tree_parents_;

  corebfs_index(corebfs_index &&) = default;
  corebfs_index &operator=(corebfs_index &&) = default;

  //
  // Removes the parents of vertices not in the giant connected component.
  // This function assumes that a vertex with the highest degree is in the giant
  // connected component.
  //
  // `typename Bfs` is parameterized because this header is included from
  // `bfs.hpp`, and so `BfsOnCPU` is unavailable here.
  //
  template <typename Bfs>
  void reset_unreachable(Bfs *const bfs, const int edge_factor,
                         const double alpha, const double beta,
                         int64_t *const pred) {
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
    postprocess::reset_unreachable(
        dist_, reinterpret_cast<global_vertex *>(pred), &tree_parents_);
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
  std::pair<int64_t, std::vector<std::pair<LocalVertex, int64_t>>> bfs_tree(
      const global_vertex_int root) const {
    INDEXED_BFS_TIMED_SCOPE(bfs_tree);

    // No reservation because it remains empty in most cases
    std::vector<std::pair<LocalVertex, int64_t>> path;

    const global_vertex core_root = corebfs::bfs_tree_with_callback(
        dist_, tree_parents_, global_vertex(root),
        [&](vertex u, global_vertex parent) {
          path.push_back(std::make_pair(u.t, parent.t));
        });

    return std::make_pair(core_root.t, std::move(path));
  }

  //
  // Write out the parents of tree vertices to `pred`.
  //
  void write_tree_parents(
      int64_t *const pred,
      const std::vector<std::pair<LocalVertex, int64_t>> &path_to_core) const {
    INDEXED_BFS_TIMED_SCOPE(write_tree_parents);

    // `dump()` writes out each element of `pred` as if it is simple 64-bit
    // parent IDs although it is actually a concatenation of 16-bit depth and
    // 48-bit parent ID. As a result, the depth parts of all the element become
    // zero. The depths need to be computed in the validation phase.
    tree_parents_.dump(reinterpret_cast<global_vertex *>(pred));

    for (const auto &p : path_to_core) {
      pred[p.first] = p.second;
    }
  }

  //
  // Returns `true` if the parent of `u_global` is in `tree_parents`.
  //
  bool contains_parent(const int64_t u_global) const {
    const global_vertex u_global_(u_global);
    return dist_.owner_rank(u_global_) == dist_.all().rank() &&
           tree_parents_.contains(dist_.to_local(u_global_));
  }

 private:
  corebfs_index(yoo dist, parent_array tree_parents)
      : dist_(std::move(dist)), tree_parents_(std::move(tree_parents)) {}

  friend corebfs_index preprocess(const int scale, edge_storage *const input,
                                  edge_storage *const output);
};

//
// Apply preprocesses of CoreBFS.
// Specifically,
// 1. copy the edges in the 2-core from `input` to `output`, and
// 2. returns parents of vertices that are not in the 2-core.
//
// `input == output` is allowed.
//
static corebfs_index preprocess(const int scale, edge_storage *const input,
                                edge_storage *const output) {
  static bool initialized = false;
  if (!initialized) {
    init(scale);
    initialized = true;
  }

  yoo d(scale, net::comm(mpi.comm_2d), net::comm(mpi.comm_2dr),
        net::comm(mpi.comm_2dc));

  LOG_I << "Distributing edges...";
  auto edges_2d = distribute(d, input);
  LOG_RSS();

  LOG_I << "Finding 2-core...";
  std::vector<global_vertex> parents_local = prune_trees(d, &edges_2d);
  assert(parents_local.size() == d.local_vertex_count());

  LOG_I << "Compressing a parent array...";
  parent_array tree_parents(std::move(parents_local));
  LOG_E << "ret tree size: " << tree_parents.size();
  LOG_RSS();

  write(d, edges_2d, output);

  return corebfs_index(std::move(d), std::move(tree_parents));
}

}  // namespace detail

using detail::corebfs_index;
using detail::parent_array;
using detail::preprocess;

}  // namespace corebfs_adaptor

#endif  // COREBFS_ADAPTOR_HPP
