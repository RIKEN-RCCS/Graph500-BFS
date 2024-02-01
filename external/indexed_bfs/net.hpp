//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-10-19.
//
#pragma once

#include "common.hpp"

namespace indexed_bfs {
namespace net {
namespace detail {

using namespace indexed_bfs;
using namespace indexed_bfs::util;
using memory::make_with_capacity;
using types::to_sig;

struct overload_datatype_of {};

template <typename T> MPI_Datatype datatype_of() {
  assert(!"MPI_Datatype is undefined for the given type");
  return MPI_DATATYPE_NULL;
}

template <> MPI_Datatype datatype_of<bool>() { return MPI_CXX_BOOL; }

template <> MPI_Datatype datatype_of<char>() { return MPI_CHAR; }

template <> MPI_Datatype datatype_of<short>() { return MPI_SHORT; }

template <> MPI_Datatype datatype_of<int>() { return MPI_INT; }

template <> MPI_Datatype datatype_of<long>() { return MPI_LONG; }

template <> MPI_Datatype datatype_of<long long>() { return MPI_LONG_LONG; }

template <> MPI_Datatype datatype_of<float>() { return MPI_FLOAT; }

template <> MPI_Datatype datatype_of<double>() { return MPI_DOUBLE; }

template <> MPI_Datatype datatype_of<unsigned char>() {
  return MPI_UNSIGNED_CHAR;
}

template <> MPI_Datatype datatype_of<unsigned short>() {
  return MPI_UNSIGNED_SHORT;
}

template <> MPI_Datatype datatype_of<unsigned int>() { return MPI_UNSIGNED; }

template <> MPI_Datatype datatype_of<unsigned long>() {
  return MPI_UNSIGNED_LONG;
}

template <> MPI_Datatype datatype_of<unsigned long long>() {
  return MPI_UNSIGNED_LONG_LONG;
}

#ifndef COREBFS_ADAPTOR_HPP
// Wrapper of `MPI_Comm` with member functions corresponding to `MPI_Comm_*()`.
struct comm {
  MPI_Comm t; // Named after an internal value of BOOST_STRONG_TYPEDEF

  int rank() const {
    int x;
    MPI_Comm_rank(this->t, &x);
    return x;
  }

  int size() const {
    int x;
    MPI_Comm_size(this->t, &x);
    return x;
  }
};

static const comm &world() {
  static comm c = comm{MPI_COMM_WORLD};
  return c;
}
#endif // COREBFS_ADAPTOR_HPP

template <typename T> struct gathered_data {
  std::vector<T> data;
  std::vector<int> counts;
  std::vector<int> displacements;
};

template <typename T> struct scattering_data {
  std::vector<T> data;
  std::vector<int> counts;
  std::vector<int> displacements;
};

static std::vector<int> make_displacements(const std::vector<int> &counts) {
  auto ds = make_with_capacity<std::vector<int>>(counts.size());
  ds.push_back(0);
  std::partial_sum(counts.begin(), counts.end() - 1, std::back_inserter(ds));
  return ds;
}

// TODO: replace the same procedure with this function
template <typename T, typename RankMapper, typename Transformer>
static auto arrange(const std::vector<T> &data, RankMapper m, Transformer t,
                    const int comm_size)
    -> scattering_data<decltype(t(std::declval<const T>()))> {
  using output_type = decltype(t(std::declval<const T>()));

  std::vector<int> counts(comm_size);
  for (const auto &x : data) {
    counts[m(x)] += 1;
  }

  std::vector<int> displs = make_displacements(counts);

  // Cannot use `std::vector` because `output_type` may lack a default
  // constructor.
  std::vector<output_type> reordered_data(data.size());
  auto indices = displs; // copy
  for (const auto &x : data) {
    const int rank = m(x);
    reordered_data[indices[rank]] = t(x);
    indices[rank] += 1;
  }

  return scattering_data<output_type>{std::move(reordered_data),
                                      std::move(counts), std::move(displs)};
}

template <typename T, typename RankMapper>
static scattering_data<T> arrange(const std::vector<T> &data, RankMapper m,
                                  const int comm_size) {
  return arrange(
      data, std::move(m), [](auto &&x) { return x; }, comm_size);
}

template <typename T>
static std::vector<T> allgather(const T &data, const comm &c) {
  std::vector<T> recv_data(c.size());
  MPI_Allgather(&data, 1, datatype_of<T>(), recv_data.data(), 1,
                datatype_of<T>(), c.t);
  return recv_data;
}

template <typename T>
static gathered_data<T> allgatherv(const std::vector<T> &data, const comm &c) {
  const std::vector<int> counts = allgather(static_cast<int>(data.size()), c);
  const std::vector<int> displs = make_displacements(counts);
  std::vector<T> recv_data(displs.back() + counts.back());

  MPI_Allgatherv(data.data(), data.size(), datatype_of<T>(), recv_data.data(),
                 counts.data(), displs.data(), datatype_of<T>(), c.t);

  return gathered_data<T>{std::move(recv_data), std::move(counts),
                          std::move(displs)};
}

template <typename T>
static T allreduce(const T &value, const MPI_Op op, const comm &c) {
  T recv;
  MPI_Allreduce(const_cast<T *>(&value), &recv, 1, datatype_of<T>(), op, c.t);
  return recv;
}

template <typename T>
static gathered_data<T> alltoallv(const T *const data, const int *const counts,
                                  const int *const displs, const comm &c) {
  std::vector<int> recv_counts(c.size());
  MPI_Alltoall(const_cast<int *>(counts), 1, MPI_INT,
               const_cast<int *>(recv_counts.data()), 1, MPI_INT, c.t);

  auto recv_displs = make_displacements(recv_counts);
  std::vector<T> recv_data(recv_displs.back() + recv_counts.back());
  MPI_Alltoallv(const_cast<T *>(data), const_cast<int *>(counts),
                const_cast<int *>(displs), datatype_of<T>(), recv_data.data(),
                recv_counts.data(), recv_displs.data(), datatype_of<T>(), c.t);

  return gathered_data<T>{std::move(recv_data), std::move(recv_counts),
                          std::move(recv_displs)};
}

template <typename T>
static gathered_data<T>
alltoallv(const std::vector<T> &data, const std::vector<int> &counts,
          const std::vector<int> &displs, const comm &c) {
  return alltoallv(data.data(), counts.data(), displs.data(), c);
}

template <typename T>
static gathered_data<T> alltoallv(const std::vector<T> &data,
                                  const std::vector<int> &counts,
                                  const comm &c) {
  return alltoallv(data, counts, make_displacements(counts), c);
}

static void barrier(const comm &c) { MPI_Barrier(c.t); }

template <typename T>
static std::enable_if_t<!std::is_pointer<T>::value, T>
bcast(const T &data, const int root, const comm &c) {
  if (c.rank() == root) {
    MPI_Bcast(const_cast<T *>(&data), 1, datatype_of<T>(), root, c.t);
    return data;
  } else {
    T x;
    MPI_Bcast(&x, 1, datatype_of<T>(), root, c.t);
    return x;
  }
}

template <typename T>
static void bcast(const size_t count, const int root, const comm &c,
                  T *const data) {
  MPI_Bcast(data, count, datatype_of<T>(), root, c.t);
}

static MPI_Datatype commit_contiguous(const int count,
                                      const MPI_Datatype type) {
  MPI_Datatype t;
  MPI_Type_contiguous(count, type, &t);
  MPI_Type_commit(&t);
  return t;
}

static void finalize() { MPI_Finalize(); }

template <typename T>
static std::vector<T> gather(const T &data, const int root, const comm &c) {
  std::vector<T> recv_data;
  if (c.rank() == root) {
    recv_data.resize(c.size());
  }

  MPI_Gather(const_cast<T *>(&data), 1, datatype_of<T>(), recv_data.data(), 1,
             datatype_of<T>(), root, c.t);

  return recv_data;
}

template <typename T>
static std::vector<T> gatherv(const std::vector<T> &data, const int root,
                              const comm &c) {
  const auto recv_counts = gather(static_cast<int>(data.size()), root, c);

  std::vector<T> recv_data;
  std::vector<int> displs;
  if (root == c.rank()) {
    displs = make_displacements(recv_counts);
    recv_data.resize(displs.back() + recv_counts.back());
  }

  MPI_Gatherv(const_cast<T *>(data.data()), data.size(), datatype_of<T>(),
              recv_data.data(), recv_counts.data(), displs.data(),
              datatype_of<T>(), root, c.t);

  return recv_data;
}

template <typename T, typename RankMapper>
static std::vector<T> scatterv_by(const std::vector<T> &data, RankMapper m,
                                  const int root, const comm &c) {
  scattering_data<T> send;
  if (c.rank() == root) {
    send = arrange(data, m, c.size());
  }

  int recv_count;
  MPI_Scatter(send.counts.data(), 1, datatype_of<int>(), &recv_count, 1,
              datatype_of<int>(), root, c.t);

  std::vector<T> recv_data(recv_count);
  MPI_Scatterv(send.data.data(), send.counts.data(), send.displacements.data(),
               datatype_of<T>(), recv_data.data(), recv_count, datatype_of<T>(),
               0, c.t);

  return recv_data;
}

static int init_thread(int *const argc, char ***const argv,
                       const int required) {
  int provided;
  MPI_Init_thread(argc, argv, required, &provided);
  return provided;
}

template <typename T>
static T reduce(const T &value, const MPI_Op op, const int root,
                const comm &c) {
  T recv;
  MPI_Reduce(const_cast<T *>(&value), &recv, 1, datatype_of<T>(), op, root,
             c.t);
  return recv;
}

// Perform Alltoallv by given mapping and transformation.
//
// Note that `map` is assumed to be lightweight; it is applied multiple
// times to each element.
template <typename T, typename RankMapper, typename Transformer>
static auto alltoallv_by(const std::vector<T> &data, RankMapper map,
                         Transformer transform, const comm &c)
    -> gathered_data<decltype(transform(std::declval<const T>()))> {
  using output_type = decltype(transform(std::declval<const T>()));

  std::vector<int> counts(c.size());
  for (const auto &x : data) {
    counts[map(x)] += 1;
  }

  const auto displs = make_displacements(counts);

  // Cannot use `std::vector` because `output_type` may lack a default
  // constructor.
  std::unique_ptr<output_type[]> reordered_data(new output_type[data.size()]);
  auto indices = displs; // copy
  for (const auto &x : data) {
    const int rank = map(x);
    reordered_data[indices[rank]] = transform(x);
    indices[rank] += 1;
    assert(rank + 1 == c.size() || indices[rank] <= displs[rank + 1]);
    assert(rank + 1 < c.size() || indices[rank] <= to_sig(data.size()));
  }

  return alltoallv(reordered_data.get(), counts.data(), displs.data(), c);
}

//
// Returns (received_data, received_counts, received_displacements).
//
// `rank_map` is assumed to be lightweight; it is applied multiple times to
// each element.
//
template <typename T, typename RankMapper>
static gathered_data<T> alltoallv_by(const std::vector<T> &data, RankMapper map,
                                     const comm &c) {
  return alltoallv_by(
      data, map, [](T x) { return x; }, c);
}

// Perform `alltoallv_by` by using `data` as a send buffer.
//
// The order of elements in `data` will be modified.
// Note that this function still needs an additional memory for a receive
// buffer although it is named "inplace".
template <typename T, typename RankMapper>
static gathered_data<T> alltoallv_inplace_by(std::vector<T> *const data,
                                             RankMapper m, const comm &c) {
  std::vector<int> counts(c.size());
  for (const auto &x : *data) {
    counts[m(x)] += 1;
  }

  const auto displs = make_displacements(counts);

  //
  // Sort the elements in `data` by the destination rank
  //
  auto indices = displs; // copy;
  int i = 0;
  while (i < to_sig(data->size())) {
    const int rank = m((*data)[i]);
    if (displs[rank] <= i && i < displs[rank] + counts[rank]) {
      // The i-th element is already in the correct position; go next
      ++i;
    } else {
      // Put the i-th element in the correct position and get the element
      // originally in that position
      std::swap((*data)[i], (*data)[indices[rank]]);
    }
    // One of the elements mapped to `rank` is placed correctly
    indices[rank] += 1;
  }

  return alltoallv(*data, counts, displs, c);
}

} // namespace detail

using detail::allgather;
using detail::allgatherv;
using detail::allreduce;
using detail::alltoallv;
using detail::alltoallv_by;
using detail::alltoallv_inplace_by;
using detail::barrier;
using detail::bcast;
using detail::comm;
using detail::commit_contiguous;
using detail::datatype_of;
using detail::finalize;
using detail::gather;
using detail::gathered_data;
using detail::gatherv;
using detail::init_thread;
using detail::make_displacements;
using detail::reduce;
using detail::scatterv_by;
using detail::world;

} // namespace net
} // namespace indexed_bfs
