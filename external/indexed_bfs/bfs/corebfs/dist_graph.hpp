//
// Authored by ARAI Junya <araijn@gmail.com> in 2024-01-14.
//
#pragma once

#include "../../common.hpp"
#include "../../graph.hpp"
#include <array>

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace dist_graph {
namespace detail {

using namespace util;
using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::show::show;
using indexed_bfs::util::sort_parallel::sort_parallel;

using binary_csr = jag_array<global_vertex>;

template <typename Container>
auto five_number_summary(const Container &c)
    -> std::array<types::range_value_t<Container>, 5> {
  const auto first = std::begin(c);
  const auto last = std::end(c);
  const auto n = std::distance(first, last);

  auto vec =
      make_with_capacity<std::vector<types::range_value_t<Container>>>(n);
  std::copy(first, last, std::back_inserter(vec));
  sort_parallel(vec.begin(), vec.end());

  const auto q0 = vec[0];
  const auto q1 = vec[static_cast<size_t>(n * 0.25)];
  const auto q2 = vec[static_cast<size_t>(n * 0.50)];
  const auto q3 = vec[static_cast<size_t>(n * 0.75)];
  const auto q4 = vec[n - 1];

  return {q0, q1, q2, q3, q4};
}

template <typename InputIt, typename Pred>
static size_t count_if_global(InputIt first, InputIt last, Pred predicate) {
  const auto n_local = std::count_if(first, last, predicate);
  return net::reduce(n_local, MPI_SUM, 0, net::world());
}

int owner_rank(const global_vertex u) { return u.t % net::world().size(); }

int owner_rank(const edge &e) { return owner_rank(e.source()); }

//
// Returns a local vertex corresponding to `u`.
// This function is allowed to apply a global vertex owned by another process.
//
vertex to_remote_local(const global_vertex u) {
  // Check overflow
  assert(u.t / net::world().size() <= std::numeric_limits<uint32_t>::max());
  return vertex{static_cast<int32_t>(u.t / net::world().size())};
}

//
// Returns a local vertex corresponding to `u`.
// This function is prohibited to apply a global vertex owned by another
// process.
//
vertex to_local(const global_vertex u) {
  assert(owner_rank(u) == net::world().rank());
  return to_remote_local(u);
}

global_vertex to_remote_global(const vertex p, const int owner_rank) {
  return global_vertex{p.t * net::world().size() + owner_rank};
}

global_vertex to_global(const vertex p) {
  return to_remote_global(p, net::world().rank());
}

size_t local_vertex_count(const size_t scale) {
  const auto &c = net::world();
  const auto n_global = global_vertex_count(scale);

  if (to_unsig(c.rank()) < n_global % c.size()) {
    return n_global / c.size() + 1;
  } else {
    return n_global / c.size();
  }
}

static void normalize(std::vector<edge> *const edges) {
  // (1) Ensure "source ID < target ID" for every edge.
#pragma omp parallel for
  for (size_t i = 0; i < edges->size(); ++i) {
    const auto s = (*edges)[i].source();
    const auto t = (*edges)[i].target();
    if (s.t > t.t) {
      (*edges)[i] = make_edge(t, s);
    }
  }

  // (2) Exchange edges so that each rank has the edges for which it is the
  // owner of the edge's source.
  *edges =
      net::alltoallv_inplace_by(
          edges, [](auto &e) { return owner_rank(e.source()); }, net::world())
          .data;

  // (3) Sort edges.
  LOG_I << "Start sorting edges";
  sort_parallel(edges->begin(), edges->end());
  LOG_I << "Completed sorting edges";

  // (4) Remove loops and duplicated edges.
  auto is_loop = [](auto &e) { return e.source().t == e.target().t; };
  auto last = edges->end();
  last = std::remove_if(edges->begin(), last, is_loop);
  last = std::unique(edges->begin(), last);
  edges->erase(last, edges->end());
  edges->shrink_to_fit();
}

// Count the degree of each vertex.
//
// `edges` must be normalized by `normalize()`.
static std::vector<size_t> count_degree(const std::vector<edge> &edges,
                                        const size_t n_local) {
  // Partially count the degree of each local vertex.
  // For each local vertex u, here we count v in N(u) s.t. u < v.
  std::vector<size_t> degs_local(n_local);
  for (size_t i = 0; i < edges.size(); ++i) {
    degs_local[to_remote_local(edges[i].source()).t] += 1;
  }

  // Count v in N(u) s.t. u > v.
  // Exchange targets after converting to a local ID (32 bits) to reduce memory
  // consumption.
  const std::vector<vertex> targets_local =
      net::alltoallv_by(
          edges, [](auto &e) { return owner_rank(e.target()); },
          [](auto &e) { return to_remote_local(e.target()); }, net::world())
          .data;
  for (const vertex u : targets_local) {
    degs_local[u.t] += 1;
  }

  // Print the distribution
  const auto total_deg_local =
      std::accumulate(degs_local.begin(), degs_local.end(), 0);
  const auto total_degs = net::gather(total_deg_local, 0, net::world());
  if (net::world().rank() == 0) {
    const auto s = five_number_summary(total_degs);
    LOG_I << "Five-number summary of the edge distribution per process: "
          << show(s);
  }

  return degs_local;
}

static binary_csr make_csr(std::vector<edge> &&edges,
                           const size_t n_local_vertices,
                           const std::vector<size_t> &degs_local) {
  const size_t m_local =
      std::accumulate(degs_local.begin(), degs_local.end(), 0);

  const size_t csr_size_local =
      sizeof(size_t) * (n_local_vertices + 1) + sizeof(vertex) * m_local;
  const size_t csr_size_max =
      net::reduce(csr_size_local, MPI_MAX, 0, net::world());
  if (net::world().rank() == 0) {
    LOG_I << "Max CSR size in bytes: " << csr_size_max;
  }

  auto offsets = make_with_capacity<std::vector<size_t>>(n_local_vertices + 1);
  std::vector<global_vertex> values(m_local);

  offsets.push_back(0);
  std::partial_sum(degs_local.begin(), degs_local.end(),
                   std::back_inserter(offsets));

  auto indices = offsets; // copy
  // Write greater neighbors (v in N(u) s.t. u < v)
  for (const auto &e : edges) {
    const auto s = to_remote_local(e.source()).t;
    values[indices[s]] = e.target();
    indices[s] += 1;
  }

  // Until here, `edges` is sorted.
  // From here, `edges` is not sorted.
  edges =
      net::alltoallv_inplace_by(
          &edges, [](auto &e) { return owner_rank(e.target()); }, net::world())
          .data;

  for (const auto &e : edges) {
    const auto t = to_remote_local(e.target()).t;
    values[indices[t]] = e.source();
    indices[t] += 1;
  }

  return {std::move(offsets), std::move(values)};
}

static void sort_neighbors(binary_csr *const g) {
  // This sort can be optimized by distributing and sorting edges by their
  // targets (instead of their sources) in `normalize()`.
  // It lets neighbors be added to `values` in the sorted order in `make_csr()`,
  // and so the number of elements that need to be sorted is reduced by half.

  // `schedule(dynamic)` to mitigate an effect of skewed degrees
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < g->slice_count(); ++i) {
    std::sort((*g)[i].begin(), (*g)[i].end());
  }
}

static binary_csr build(const int scale, std::vector<edge> &&edges) {
  const size_t n_global = global_vertex_count(scale);
  const size_t n_local = local_vertex_count(scale);

  LOG_I << "Normalizing and sorting edges...";
  normalize(&edges);

  LOG_I << "Counting degrees...";
  const auto degs_local = count_degree(edges, n_local);

  const auto n_connected_global = count_if_global(
      degs_local.begin(), degs_local.end(), [](auto d) { return d > 0; });
  LOG_I << "#connected vertices: " << n_connected_global;
  LOG_I << "Proportion of connected vertices to all vertices: "
        << static_cast<double>(n_connected_global) / n_global;

  LOG_I << "Making a CSR data structure...";
  auto g = make_csr(std::move(edges), n_local, degs_local);

  LOG_I << "Sorting neighbors...";
  sort_neighbors(&g);

  return g;
}

} // namespace detail

using detail::binary_csr;
using detail::build;
using detail::five_number_summary;
using detail::local_vertex_count;
using detail::owner_rank;
using detail::to_global;
using detail::to_local;
using detail::to_remote_global;
using detail::to_remote_local;

} // namespace dist_graph
} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
