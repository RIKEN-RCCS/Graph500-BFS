# Graph500-BFS

This is a Graph500 benchmark program developed for the K computer and its successor, the supercomputer Fugaku, in Japan.
Please cite the following paper if you use our code:

- Junya Arai, Masahiro Nakao, Yuto Inoue, Kanto Teranishi, Koji Ueno, Keiichiro Yamamura, Mitsuhisa Sato, and Katsuki Fujisawa, "Doubling Graph Traversal Efficiency to 198 TeraTEPS on the Supercomputer Fugaku," in SC24: International Conference for High Performance Computing, Networking, Storage and Analysis, 2024.

Our previous publications are listed below (from most recent to oldest):

- Masahiro Nakao, Koji Ueno, Katsuki Fujisawa, Yuetsu Kodama, and Mitsuhisa, Sato, "Performance of the Supercomputer Fugaku for Breadth-First Search in Graph500 Benchmark," in High Performance Computing:
36th International Conference, ISC High Performance 2021, 2021, pp. 372-390.
- Koji Ueno, Toyotaro Suzumura, Naoya Maruyama, Katsuki Fujisawa, Satoshi Matsuoka, "Efficient Breadth-First Search on Massively Parallel and Distributed-Memory Machines," Data Science and Engineering, vol. 2, no. 1, pp. 22-35, 2017.
- Koji Ueno, Toyotaro Suzumura, Naoya Maruyama, Katsuki Fujisawa, Satoshi Matsuoka. "Extreme Scale Breadth-First Search on Supercomputers". in 2016 IEEE International Conference on Big Data (Big Data), 2016, pp. 1040-1047.

The code is licensed under the Apache License, Version 2.0.

## How to Use

While this program performs best on the supercomputer Fugaku, you can test it on an x86_64 Linux server.
For technical details, please refer to our papers listed above.

### Tested Environment

- CPU: AMD EPYC 7713 (64 cores) x 2
- Memory: 1 TiB
- Storage: 2 TB SATA SSD x 2
- OS: Ubuntu 22.04.4
- MPI: OpenMPI 4.1.2
- Compiler: GCC 11.4.0
- Libraries: libnuma 2.0.14

### Build

Use the following command to build the executable with `mpicxx`:
```console
$ (cd mpi && make)
```

Note that the group reordering technique is disabled by default.
To enable it, build the executable using the following command:
```console
$ (cd mpi && make SMALL_REORDER_BIT=true)
```

### Execution

For single-node execution, use the following command:
```console
$ mpiexec -n 1 --bind-to none mpi/runnable <scale> <options>
```

Each placeholder is described below:

- `<scale>`: A SCALE value.
  It is recommended to start with SCALE 20 and increase it gradually, as both memory requirement and execution time grow exponentially with a SCALE value.
- `<options>`: A combination of the following options:
    - `-A`: Enable adaptive parameter tuning.
    - `-C`: Enable forest pruning.
    - `-n <N>`: Set the number of search keys to `<N>`. By default, 16 search keys are sampled and processed.

Note that specifying `-n 64` is necessary to comply with the Graph500 specification, and all the results in the paper were obtained using this setting.
For example, to measure performance for SCALE 20 with all the proposed techniques enabled, use the following command:
```console
$ mpiexec -n 1 --bind-to none mpi/runnable 20 -A -C -n 64
```

The measured performance in TEPS will be displayed in the line beginning with `harmonic_mean_TEPS`.
