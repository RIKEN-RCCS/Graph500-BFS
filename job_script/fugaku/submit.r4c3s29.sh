#!/bin/bash
#PJM -L "node=2x3x2:strict"
#PJM -L "rscgrp=small-torus"
#PJM -L "elapse=0:30:00"
#PJM -L "freq=2200,eco_state=2"
#PJM --rsc-list "retention_state=0"
#PJM --mpi "assign-online-node"
#PJM -g rccs-aot
##PJM -m b
##PJM --mail-list masahiro.nakao@riken.jp
#PJM -S

export TOFU_6D=xyzac
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=48
export FLIB_BARRIER=HARD
export XOS_MMM_L_PAGING_POLICY=prepage:demand:demand
S=29
#############################
DIR=result/r4c3s${S}
mkdir -p ${DIR}
OPTION="--mca coll_select_alltoallv_algorithm doublespread --mca coll_select_allgather_algorithm 3dtorus_fm --mca coll_select_allgatherv_algorithm gtvbc"
OPTION="${OPTION} -stdout-proc ./${DIR}/%j/%/1000R/stdout -stderr-proc ./${DIR}/%j/%/1000R/stderr"
module list
mpiexec -mca btl_tofu_eager_limit 512000 ${OPTION} -mca mpi_print_stats 3 ./runnable ${S} -A -R
echo $SECONDS
