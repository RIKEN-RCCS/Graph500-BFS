SW =GNU3
# SW=FUGAKU
VAMP = false
SCOREP = false

# default values
LDFLAGS := -fopenmp -g # -pg
NVCCFLAGS := -Drestrict=__restrict__ -Xcompiler -fopenmp,-ffast-math,-Wall -arch=compute_20 -code=sm_20

FUGAKU_POWER_MEASUREMENT=false
#FUGAKU_POWER_MEASUREMENT=true
VERBOSE = false
VERTEX_REORDERING = 2
ifeq ($(VERBOSE), false)
VERBOSE_OPT = -DVERVOSE_MODE=0
else
VERBOSE_OPT = -DVERVOSE_MODE=1
endif
VERTEX_REORDERING_OPT = -DVERTEX_REORDERING=$(VERTEX_REORDERING)
ifeq ($(FUGAKU_POWER_MEASUREMENT), true)
FUGAKU_POWER_CFLAGS=-D_FUGAKU_POWER_MEASUREMENT
FUGAKU_POWER_CLINK=-lpwr
endif

# use small bit for reordering instead of whole vertex bit
SMALL_REORDER_BIT ?= false
ifeq ($(SMALL_REORDER_BIT), true)
SMALL_REORDER_BIT_OPT = -DSMALL_REORDER_BIT
else
SMALL_REORDER_BIT_OPT =
endif

# use Edge list predistribution
EDGE_LIST_PREDISTRIBUTION = true
ifeq ($(EDGE_LIST_PREDISTRIBUTION), true)
EDGE_LIST_PREDISTRIBUTION_OPT = -DEDGE_LIST_PREDISTRIBUTION
else
EDGE_LIST_PREDISTRIBUTION_OPT =
endif

GCC_BASE := -fopenmp -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT)
#GCC_BASE := -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math -msse4.2 # -pg
FCC_BASE := -Kopenmp -Xg -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -g #-Nquickdbg=heapchk # -Koptmsg=2
#FCC_BASE := -Xg -g -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS

MPICC := mpicc
MPICPP := mpicxx

CUDALIB := -L$(CUDA_PATH)/lib64 -lcudart -lcuda
NVCC := nvcc

INC := -I../external # -I$(CUDA_PATH)/include
BINS := runnable
OBJS := main.o ../generator/splittable_mrg.o
CUOBJ := cuda_kernel.o
LIBS := -lm

ifeq ($(SW), GNU3)
CFLAGS := $(GCC_BASE) -O3 -DNDEBUG -DPROFILE_REGIONS $(SMALL_REORDER_BIT_OPT) $(EDGE_LIST_PREDISTRIBUTION_OPT)
NVCCFLAGS := $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LIBS := $(LIBS) -lnuma
endif
ifeq ($(SW), FUGAKU)
MPICC := mpifccpx
MPICPP := mpiFCCpx
FCC_BASE := -Nclang -fopenmp -Drestrict=__restrict__  $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT)
CFLAGS := $(FCC_BASE) -Ofast -D__FUJITSU -DNDEBUG -DPROFILE_REGIONS $(FUGAKU_POWER_CFLAGS) $(SMALL_REORDER_BIT_OPT) $(EDGE_LIST_PREDISTRIBUTION_OPT)
LDFLAGS := $(CFLAGS) $(FUGAKU_POWER_CLINK)
endif
ifeq ($(SW), INTEL3)
MPICC := mpiicc
MPICPP := mpiicpc
CFLAGS := -ipo -no-prec-div -qopenmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT) $(SMALL_REORDER_BIT_OPT) $(EDGE_LIST_PREDISTRIBUTION_OPT)
NVCCFLAGS := $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS := -qopenmp -g
LIBS := $(LIBS) -lnuma -lmpi
endif

ifeq ($(VAMP), true)
MPICC := vtcc -vt:cc $(MPICC) -vt:inst manual -vt:hyb -DVTRACE
MPICPP := vtcc -vt:cc $(MPICPP) -vt:inst manual -vt:hyb -DVTRACE
endif
ifeq ($(SCOREP), true)
MPICC := scorep --nocompiler --user --static $(MPICC) -DSCOREP
MPICPP := scorep --nocompiler --user --static $(MPICPP) -DSCOREP
endif

CPPFLAGS := $(CFLAGS)

all: cpu

cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)

gpu: $(OBJS) $(CUOBJ)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(CUOBJ) $(LIBS) $(CUDALIB)

gnu_func.o: gnu_func.cc
	cp ../../$*.o_ $*.o
#	g++ -c -g -O3 -Wall $< -o $*.o

# Always rebuild because header files may be changed
.PHONY: force
$(OBJS) $(CUOBJ): force

.SUFFIXES: .o .c .cpp .cc .cu

.c.o:
	$(MPICC) $(INC) -c $(CFLAGS) $< -o $*.o

.cc.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $< -o $*.o

.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS) $(CUOBJ)

.PHONY: format
format:
	clang-format -i $(shell find .. -type f \( -iname "*.c" -o -iname "*.cc" -o -iname "*.cpp" -o -iname "*.h" -o -iname "*.hpp" \))
