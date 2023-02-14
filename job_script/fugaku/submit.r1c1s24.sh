#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=0:10:00"
#PJM -L "freq=2200,eco_state=2"
#PJM --rsc-list "retention_state=0"
#PJM --mpi "assign-online-node"
#PJM -g ra000019
#PJM -m b
#PJM --mail-list masahiro.nakao@riken.jp
#PJM -S

export TOFU_6D=xyzac
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=48
export FLIB_BARRIER=HARD
S=24
#############################
DIR=result/r1c1s24
mkdir -p ${DIR}
OPTION="--mca coll_select_alltoallv_algorithm doublespread --mca coll_select_allgather_algorithm 3dtorus_fm --mca coll_select_allgatherv_algorithm gtvbc"
OPTION="${OPTION} -stdout-proc ./${DIR}/%j/%/1000R/stdout -stderr-proc ./${DIR}/%j/%/1000R/stderr"
module list
mpiexec -mca btl_tofu_eager_limit 512000 ${OPTION} -mca mpi_print_stats 3 ./runnable $S -A -R
echo $SECONDS
