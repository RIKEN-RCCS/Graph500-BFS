//
// Authored by ARAI Junya <araijn@gmail.com> in 2024-01-14.
//
#pragma once

#include "../../common.hpp"
#include "../../graph.hpp"
#include "distribution.hpp"
#include <array>

namespace indexed_bfs {
namespace bfs {
namespace corebfs {
namespace csr_1d {
namespace detail {

using namespace indexed_bfs::graph;
using namespace indexed_bfs::util;
using indexed_bfs::bfs::corebfs::distribution::yoo;
using indexed_bfs::util::memory::make_with_capacity;
using indexed_bfs::util::show::show;
using indexed_bfs::util::sort_parallel::sort_parallel;
using indexed_bfs::util::types::to_sig;
using indexed_bfs::util::types::to_unsig;

using binary_csr = jag_array<global_vertex>;

static void normalize_dist_1d(const yoo &d, std::vector<edge> *const edges) {
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
          edges, [&](auto &e) { return d.owner_rank(e.source()); }, d.all())
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

//
// Count the degree of each vertex.
//
// `edges` must be normalized by `normalize_dist_1d()`.
//
static std::vector<size_t> count_degree(const yoo &d,
                                        const std::vector<edge> &edges) {
  // Partially count the degree of each local vertex.
  // For each local vertex u, here we count v in N(u) s.t. u < v.
  std::vector<size_t> degs_local(d.local_vertex_count());
  for (size_t i = 0; i < edges.size(); ++i) {
    degs_local[d.to_local_remote(edges[i].source()).t] += 1;
  }

  // Count v in N(u) s.t. u > v.
  // Exchange targets after converting to a local ID (32 bits) to reduce memory
  // consumption.
  const std::vector<vertex> targets_local =
      net::alltoallv_by(
          edges, [&](auto &e) { return d.owner_rank(e.target()); },
          [&](auto &e) { return d.to_local_remote(e.target()); }, d.all())
          .data;
  for (const vertex u : targets_local) {
    if (to_sig(degs_local.size()) <= u.t) {
      LOG_E << "degs_local.size() = " << degs_local.size();
      LOG_E << "u.t = " << u.t;
    }
    degs_local[u.t] += 1;
  }

  // Print the distribution
  const auto total_deg_local =
      std::accumulate(degs_local.begin(), degs_local.end(), 0);
  const auto total_degs = net::gather(total_deg_local, 0, d.all());
  if (d.all().rank() == 0) {
    const auto s = common::five_number_summary(total_degs);
    LOG_I << "Five-number summary of the edge distribution per process: "
          << show(s);
  }

  return degs_local;
}

static binary_csr make_csr(const yoo &d, std::vector<edge> &&edges,
                           const std::vector<size_t> &degs_local) {
  assert(degs_local.size() == d.local_vertex_count());

  const size_t m_local =
      std::accumulate(degs_local.begin(), degs_local.end(), 0);

  const size_t csr_size_local =
      sizeof(size_t) * (d.local_vertex_count() + 1) + sizeof(vertex) * m_local;
  const size_t csr_size_max = net::reduce(csr_size_local, MPI_MAX, 0, d.all());
  if (d.all().rank() == 0) {
    LOG_I << "Max CSR size in bytes: " << csr_size_max;
  }

  auto offsets =
      make_with_capacity<std::vector<size_t>>(d.local_vertex_count() + 1);
  std::vector<global_vertex> values(m_local);

  offsets.push_back(0);
  std::partial_sum(degs_local.begin(), degs_local.end(),
                   std::back_inserter(offsets));

  auto indices = offsets; // copy
  // Write greater neighbors (v in N(u) s.t. u < v)
  for (const auto &e : edges) {
    const auto s = d.to_local_remote(e.source()).t;
    values[indices[s]] = e.target();
    indices[s] += 1;
  }

  // Until here, `edges` is sorted.
  // From here, `edges` is not sorted.
  edges =
      net::alltoallv_inplace_by(
          &edges, [&](auto &e) { return d.owner_rank(e.target()); }, d.all())
          .data;

  for (const auto &e : edges) {
    const auto t = d.to_local_remote(e.target()).t;
    values[indices[t]] = e.source();
    indices[t] += 1;
  }

  return {std::move(offsets), std::move(values)};
}

static void sort_neighbors(binary_csr *const g) {
  // This sort can be optimized by distributing and sorting edges by their
  // targets (instead of their sources) in `normalize_dist_1d()`.
  // It lets neighbors be added to `values` in the sorted order in `make_csr()`,
  // and so the number of elements that need to be sorted is reduced by half.

  // `schedule(dynamic)` to mitigate an effect of skewed degrees
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < g->slice_count(); ++i) {
    std::sort((*g)[i].begin(), (*g)[i].end());
  }
}

static binary_csr build(const yoo &d, std::vector<edge> &&edges) {
  const size_t n_global = d.global_vertex_count();

  LOG_I << "Normalizing and sorting edges...";
  normalize_dist_1d(d, &edges);

  LOG_I << "Counting degrees...";
  const auto degs_local = count_degree(d, edges);

  const auto n_connected_global =
      net::count_if(degs_local.begin(), degs_local.end(), d.all(),
                    [](auto d) { return d > 0; });
  LOG_I << "#connected vertices: " << n_connected_global;
  LOG_I << "Proportion of connected vertices to all vertices: "
        << static_cast<double>(n_connected_global) / n_global;

  LOG_I << "Making a CSR data structure...";
  auto g = make_csr(d, std::move(edges), degs_local);

  LOG_I << "Sorting neighbors...";
  sort_neighbors(&g);

  return g;
}

} // namespace detail

using detail::binary_csr;
using detail::build;

} // namespace csr_1d
} // namespace corebfs
} // namespace bfs
} // namespace indexed_bfs
