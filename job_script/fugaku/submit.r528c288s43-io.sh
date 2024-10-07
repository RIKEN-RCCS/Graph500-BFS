#!/bin/bash
#PJM -L "node=48x66x48:strict"
#PJM -L "rscgrp=huge_2307-4"
#PJM -L "elapse=7:00:00"
#PJM -L "freq=2200,eco_state=2"
#PJM --rsc-list "retention_state=0"
#PJM --llio localtmp-size=80Gi
#PJM --mpi "assign-online-node"
#PJM -g ra000019
#PJM -S

export TOFU_6D=yz
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=48
export FLIB_BARRIER=HARD
export XOS_MMM_L_PAGING_POLICY=prepage:demand:demand
S=43
#############################
DIR=result/r552c288s43
mkdir -p ${DIR}
llio_transfer ./runnable
OPTION="--mca coll_select_alltoallv_algorithm doublespread --mca coll_select_allgather_algorithm 3dtorus_fm --mca coll_select_allgatherv_algorithm gtvbc"
OPTION="${OPTION} -stdout-proc ./${DIR}/%j/%/1000R/stdout -stderr-proc ./${DIR}/%j/%/1000R/stderr"
module list
export TMPFILE=${PJM_LOCALTMP}/tmp
mpiexec -mca btl_tofu_eager_limit 512000 ${OPTION} -mca mpi_print_stats 3 ./runnable $S -A -R
echo $SECONDS