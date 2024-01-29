# `valgrind/`

This directory contains error suppression files for Valgrind.

## How to use

Execute `valgrind` via `mpiexec`, specifying a `.supp` file by `--suppressions`.

Note that, if the executable is built without the `DEBUG` macro, Valgrind may
generate errors related to pages marked as `MADV_DONTNEED`.
See comments in `alloc_array_dontneed()` in `mpi/bfs.hpp`.

### Examples

```shell
cd mpi/
mpiexec -n 4 valgrind --suppressions=../valgrind/openmpi.supp ./runnable 10
```

