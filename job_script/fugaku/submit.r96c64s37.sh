#!/bin/bash
#PJM -L "node=8x48x16:strict"
#PJM -L "rscgrp=large"
#PJM -L "elapse=1:00:00"
#PJM -L "freq=2200,eco_state=2"
#PJM --rsc-list "retention_state=0"
#PJM --llio localtmp-size=2Gi
#PJM --llio sharedtmp-size=2Gi
#PJM --mpi "assign-online-node"
#PJM -g ra000019
#PJM -m b
#PJM --mail-list masahiro.nakao@riken.jp
#PJM -S

export TOFU_6D=ybc
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=48
export FLIB_BARRIER=HARD
S=37
#############################
DIR=result/r96c64s37
mkdir -p ${DIR}
llio_transfer ./runnable
OPTION="--mca coll_select_alltoallv_algorithm doublespread --mca coll_select_allgather_algorithm 3dtorus_fm --mca coll_select_allgatherv_algorithm gtvbc"
OPTION="${OPTION} -stdout-proc ./${DIR}/%j/%/1000R/stdout -stderr-proc ./${DIR}/%j/%/1000R/stderr"
module list
mpiexec -mca btl_tofu_eager_limit 512000 ${OPTION} -mca mpi_print_stats 3 ./runnable $S -A -R
echo $SECONDS
