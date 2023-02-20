/*
 * main.cc
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */

// C includes
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#ifdef _FUGAKU_POWER_MEASUREMENT
#include "pwr.h"
#endif

// C++ includes
#include <string>

#include "parameters.h"
#include "utils_core.h"
#include "primitives.hpp"
#include "utils.hpp"
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"
#include "benchmark_helper.hpp"
#include "validate.hpp"
#include "bfs.hpp"
#include "bfs_cpu.hpp"
#if CUDA_ENABLED
#include "bfs_gpu.hpp"
#endif

double calc_TEPS(int root_start, int num_bfs_roots, double *perf)
{
  double tmp = 0;
  for(int i = root_start; i < num_bfs_roots; ++i)
    tmp += ((double)1.0)/perf[i];

  return (num_bfs_roots - root_start) / tmp;
}

bool auto_tuning_each(int param, int root_start, int num_bfs_roots, BfsOnCPU* benchmark,
                      int64_t* bfs_roots, int64_t *pred, int SCALE, int edgefactor,
                      int64_t auto_tuning_data[][AUTO_NUM], double* alpha, double* beta, double *perf)
{
  auto_tuning_t s[num_bfs_roots], l[num_bfs_roots];
  int s_num = 0, l_num = 0;
  double pre_param = 0, pre_TEPS = calc_TEPS(root_start, num_bfs_roots, perf);

  if(mpi.isMaster()){
    print_with_prefix("========= START AUTO TUNING FOR %s =========", (IS_ALPHA)? "ALPHA" : "BETA");
    print_with_prefix("Alpha = %f, Beta = %f, Harmonic Mean = %.0f MTEPS", *alpha, *beta, pre_TEPS/1000000);
    
    if(IS_ALPHA){
      for(int i = root_start; i < num_bfs_roots; ++i)
        print_with_prefix("[%02d] Perf. = %.0f MTEPS, T2B = %d, %" PRId64 " > %" PRId64 "/%f (%" PRId64 ")",
                          i, perf[i]/1000000, (int)auto_tuning_data[i][AUTO_T2B_LEVEL],
                          auto_tuning_data[i][AUTO_GLOBAL_NQ_EDGES], auto_tuning_data[i][AUTO_NUM_GLOBAL_EDGES], *alpha,
                          auto_tuning_data[i][AUTO_PRE_GLOBAL_NQ_EDGES]);
    }
    else{
      for(int i = root_start; i < num_bfs_roots; ++i)
        print_with_prefix("[%02d] Perf. = %.0f MTEPS, B2T = %d, %" PRId64 " < %" PRId64 "/(%f * %d * 2) : (%" PRId64 ")",
                          i, perf[i]/1000000, (int)auto_tuning_data[i][AUTO_B2T_LEVEL],
                          auto_tuning_data[i][AUTO_GLOBAL_NQ_SIZE], auto_tuning_data[i][AUTO_NUM_GLOBAL_VERTS], *beta,
                          edgefactor, auto_tuning_data[i][AUTO_PRE_GLOBAL_NQ_SIZE]);
    }
  }

  for(int i = root_start; i < num_bfs_roots; ++i){
    int64_t tmp = (IS_ALPHA)? auto_tuning_data[i][AUTO_GLOBAL_NQ_EDGES] : auto_tuning_data[i][AUTO_PRE_GLOBAL_NQ_SIZE];
    if(tmp != AUTO_NOT_DEFINED){
      s[s_num].val = tmp;
      s[s_num].idx = i;
      s_num++;
    }
    tmp = (IS_ALPHA)? auto_tuning_data[i][AUTO_PRE_GLOBAL_NQ_EDGES] : auto_tuning_data[i][AUTO_GLOBAL_NQ_SIZE];
    if(tmp != AUTO_NOT_DEFINED){
      l[l_num].val = tmp;
      l[l_num].idx = i;
      l_num++;
    }
  }

  std::sort(s, s + s_num);
  std::sort(l, l + l_num);

  bool value_is_smaller = false;
  if(s_num == 0){
    return false;
  }
  else if(l_num == 0){
    value_is_smaller = true;
  }
  else if(perf[s[0].idx] < perf[l[l_num-1].idx]){
    if(mpi.isMaster()){
      print_with_prefix("Perf. = %.0f MTEPS in [%02d] < Perf. = %.0f MTEPS in [%02d]",
                        perf[s[0].idx]/1000000, s[0].idx, perf[l[l_num-1].idx]/1000000, l[l_num-1].idx);
      print_with_prefix("%s is smaller", (IS_ALPHA)? "Alpha" : "Beta");
    }
    value_is_smaller = true;
  }
  else{
    if(mpi.isMaster()){
      print_with_prefix("Perf. = %.0f MTEPS in [%02d] >= Perf. = %.0f MTEPS in [%02d]",
                        perf[s[0].idx]/1000000, s[0].idx, perf[l[l_num-1].idx]/1000000, l[l_num-1].idx);
      print_with_prefix("%s is larger", (IS_ALPHA)? "Alpha" : "Beta");
    }
  }
  
  if(value_is_smaller){
    // When Alpha is decreased, T2B is increased.
    // When Beta  is decreased, B2T is decreased.
    for(int i = 0; i < s_num; ++i){
      pre_param = (IS_ALPHA)? *alpha : *beta;
      int pre_level = (IS_ALPHA)? auto_tuning_data[s[i].idx][AUTO_T2B_LEVEL] : auto_tuning_data[s[i].idx][AUTO_B2T_LEVEL];
      if((IS_ALPHA && pre_level+1 == auto_tuning_data[s[i].idx][AUTO_LEVEL]) || (IS_BETA && pre_level == 1) ||
         pre_level == AUTO_NOT_DEFINED){
        if(mpi.isMaster()){
          print_with_prefix("[%02d] %s cannot be %s", s[i].idx, (IS_ALPHA)? "T2B" : "B2T", (IS_ALPHA)? "larger" : "smaller");
          print_with_prefix("Auto tuning for %s is stopped", (IS_ALPHA)? "Alpha" : "Beta");
          print_with_prefix("%s is determined to be %f", (IS_ALPHA)? "Alpha" : "Beta", pre_param);
        }
        return false;
      }

      if(IS_ALPHA){
        *alpha = (double)auto_tuning_data[s[i].idx][AUTO_NUM_GLOBAL_EDGES] / s[i].val * 0.99;
        while(s[i].val > int64_t(auto_tuning_data[s[i].idx][AUTO_NUM_GLOBAL_EDGES] / *alpha))
          *alpha *= 0.99;
      }
      else{
        *beta = (double)auto_tuning_data[s[i].idx][AUTO_NUM_GLOBAL_VERTS]/ (s[i].val * edgefactor * 2.0) * 0.99;
        while(s[i].val >= int64_t(auto_tuning_data[s[i].idx][AUTO_NUM_GLOBAL_VERTS] / (*beta * edgefactor * 2.0)))
          *beta *= 0.99;
      }

      if(mpi.isMaster())
        print_with_prefix("[%02d] %s: %f -> %f", s[i].idx, (IS_ALPHA)? "Alpha" : "Beta", pre_param, (IS_ALPHA)? *alpha : *beta);

      for(int j = 0; j < AUTO_NUM; ++j)
        auto_tuning_data[s[i].idx][j] = AUTO_NOT_DEFINED;
      
      MPI_Barrier(mpi.comm_2d);
      double t = MPI_Wtime();
      benchmark->run_bfs(bfs_roots[s[i].idx], pred, edgefactor, *alpha, *beta, auto_tuning_data[s[i].idx]);
      t = MPI_Wtime() - t;
      double new_perf = pf_nedge[SCALE] / t;
      MPI_Bcast(&new_perf, 1, MPI_DOUBLE, 0, mpi.comm_2d);

      int cur_level = (IS_ALPHA)? auto_tuning_data[s[i].idx][AUTO_T2B_LEVEL] : auto_tuning_data[s[i].idx][AUTO_B2T_LEVEL];
      if(pre_level == cur_level){
        print_with_prefix("Something Wrong %d", pre_level);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }      

      double pre_perf = perf[s[i].idx];
      perf[s[i].idx] = new_perf;
      double new_TEPS = calc_TEPS(root_start, num_bfs_roots, perf);
      if(mpi.isMaster()){
        print_with_prefix("[%02d] %s: %d -> %d", s[i].idx, (IS_ALPHA)? "T2B" : "B2T", pre_level, cur_level);
        print_with_prefix("[%02d] Perf.: %.0f -> %.0f MTEPS (Harmonic Mean: %.0f -> %.0f MTEPS)",
                          s[i].idx, pre_perf/1000000, new_perf/1000000, pre_TEPS/1000000, new_TEPS/1000000);
      }

      if(pre_perf > new_perf){
        if(mpi.isMaster()){
          print_with_prefix("Auto tuning for %s is stopped", (IS_ALPHA)? "Alpha" : "Beta");
          print_with_prefix("%s is determined to be %f", (IS_ALPHA)? "Alpha" : "Beta", pre_param);
        }
        perf[s[i].idx] = pre_perf;
        if(IS_ALPHA) *alpha = pre_param;
        else         *beta  = pre_param;
        return false;
      }
      else{
        pre_TEPS = new_TEPS;
      }
    }
  }
  else{
    // When Alpha is increased, T2B is decreased.
    // When Beta  is increased, B2T is increased.
    for(int i = l_num-1; i >= 0; --i){
      pre_param = (IS_ALPHA)? *alpha : *beta;
      int pre_level = (IS_ALPHA)? auto_tuning_data[l[i].idx][AUTO_T2B_LEVEL] : auto_tuning_data[l[i].idx][AUTO_B2T_LEVEL];
      if((IS_ALPHA && pre_level == 1) || (IS_BETA && pre_level+1 == auto_tuning_data[l[i].idx][AUTO_LEVEL]) ||
         pre_level == AUTO_NOT_DEFINED){
        if(mpi.isMaster()){
          print_with_prefix("[%02d] %s cannot be %s", l[i].idx, (IS_ALPHA)? "T2B" : "B2T", (IS_ALPHA)? "smaller" : "larger");
          print_with_prefix("Auto tuning for %s is stopped", (IS_ALPHA)? "Alpha" : "Beta");
          print_with_prefix("%s is determined to be %f", (IS_ALPHA)? "Alpha" : "Beta", pre_param);
        }
        return false;
      }
      
      if(IS_ALPHA){
        *alpha = (double)auto_tuning_data[l[i].idx][AUTO_NUM_GLOBAL_EDGES] / l[i].val * 1.01;
        while(l[i].val <= int64_t(auto_tuning_data[l[i].idx][AUTO_NUM_GLOBAL_EDGES] / *alpha))
          *alpha *= 1.01;
      }
      else{
        *beta = (double)auto_tuning_data[l[i].idx][AUTO_NUM_GLOBAL_VERTS] / (l[i].val * edgefactor * 2.0) * 1.01;
        while(l[i].val < int64_t(auto_tuning_data[l[i].idx][AUTO_NUM_GLOBAL_VERTS] / (*beta * edgefactor * 2.0)))
          *beta *= 1.01;
      }

      if(mpi.isMaster())
        print_with_prefix("[%02d] %s: %f -> %f", l[i].idx, (IS_ALPHA)? "Alpha" : "Beta", pre_param, (IS_ALPHA)? *alpha : *beta);

      for(int j = 0; j < AUTO_NUM; ++j)
        auto_tuning_data[l[i].idx][j] = AUTO_NOT_DEFINED;
      
      MPI_Barrier(mpi.comm_2d);
      double t = MPI_Wtime();
      benchmark->run_bfs(bfs_roots[l[i].idx], pred, edgefactor, *alpha, *beta, auto_tuning_data[l[i].idx]);
      t = MPI_Wtime() - t;
      double new_perf = pf_nedge[SCALE] / t;
      MPI_Bcast(&new_perf, 1, MPI_DOUBLE, 0, mpi.comm_2d);
      
      int cur_level = (IS_ALPHA)? auto_tuning_data[l[i].idx][AUTO_T2B_LEVEL] : auto_tuning_data[l[i].idx][AUTO_B2T_LEVEL];
      if(pre_level == cur_level){
        print_with_prefix("Something Wrong %d", pre_level);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      double pre_perf = perf[l[i].idx];
      perf[l[i].idx] = new_perf;
      double new_TEPS = calc_TEPS(root_start, num_bfs_roots, perf);
      if(mpi.isMaster()){
        print_with_prefix("[%02d] %s: %d -> %d", l[i].idx, (IS_ALPHA)? "T2B" : "B2T", pre_level, cur_level);
        print_with_prefix("[%02d] Perf.: %.0f -> %.0f MTEPS (Harmonic Mean: %.0f -> %.0f MTEPS)",
                          l[i].idx, pre_perf/1000000, new_perf/1000000, pre_TEPS/1000000, new_TEPS/1000000);
      }
      
      if(pre_perf > new_perf){
        if(mpi.isMaster()){
          print_with_prefix("Auto tuning for %s is stopped", (IS_ALPHA)? "Alpha" : "Beta");
          print_with_prefix("%s is determined to be %f", (IS_ALPHA)? "Alpha" : "Beta", pre_param);
        }
        perf[l[i].idx] = pre_perf;
        if(IS_ALPHA) *alpha = pre_param;
        else         *beta  = pre_param;
        return false;
      }
      else{
        pre_TEPS = new_TEPS;
      }
    }
  }

  return true;
}

void measure_performance(int root_start, int num_bfs_roots, BfsOnCPU* benchmark, int64_t* bfs_roots,
                         int64_t *pred, int SCALE, int edgefactor, int64_t auto_tuning_data[][AUTO_NUM],
                         double alpha, double beta, double *perf)
{
  for(int i = 0; i < num_bfs_roots; ++i)
    for(int j = 0; j < AUTO_NUM; ++j)
      auto_tuning_data[i][j] = AUTO_NOT_DEFINED;

  for(int i = root_start; i < num_bfs_roots; ++i){
    MPI_Barrier(mpi.comm_2d);
    double t = MPI_Wtime();
    benchmark->run_bfs(bfs_roots[i], pred, edgefactor, alpha, beta, auto_tuning_data[i]);
    t = MPI_Wtime() - t;
    perf[i] = pf_nedge[SCALE] / t;
  }
  MPI_Bcast(&perf[root_start], num_bfs_roots - root_start, MPI_DOUBLE, 0, mpi.comm_2d);
}

// Re-pickup roots if even one root exists in a small graph
void find_roots_in_large_graph(int root_start, int num_bfs_roots, BfsOnCPU* benchmark, int64_t* bfs_roots,
                               int64_t *pred, int SCALE, int edgefactor, int64_t auto_tuning_data[][AUTO_NUM],
                               double alpha, double beta, double *perf)
{
  int r = 0;
  find_roots(benchmark->graph_, bfs_roots, num_bfs_roots, r++, 0);
  measure_performance(root_start, num_bfs_roots, benchmark, bfs_roots, pred,
                      SCALE, edgefactor, auto_tuning_data, alpha, beta, perf);
  
  if(SCALE > REMOVE_ROOTS_SCALE_THED){
    while(1){
      bool flag = true;
      for(int i = root_start; i < num_bfs_roots; ++i)
        if(auto_tuning_data[i][AUTO_LEVEL] <= REMOVE_ROOTS_LEVEL_THED)
          flag = false;

      if(flag) break;
      if(mpi.isMaster()) print_with_prefix("Pick roots again");
      find_roots(benchmark->graph_, bfs_roots, num_bfs_roots, r++, 0);
      measure_performance(root_start, num_bfs_roots, benchmark, bfs_roots, pred,
                          SCALE, edgefactor, auto_tuning_data, alpha, beta, perf);
    }
  }
}

void auto_tuning(int root_start, int num_bfs_roots, BfsOnCPU* benchmark, int64_t* bfs_roots,
                 int64_t *pred, int SCALE, int edgefactor, double* alpha, double* beta)
{
  int64_t auto_tuning_data[num_bfs_roots][AUTO_NUM];
  double perf[num_bfs_roots];

  find_roots_in_large_graph(root_start, num_bfs_roots, benchmark, bfs_roots, pred, SCALE, edgefactor, auto_tuning_data, *alpha, *beta, perf);
  
  while(auto_tuning_each(AUTO_ALPHA, root_start, num_bfs_roots, benchmark, bfs_roots, pred, SCALE, edgefactor, auto_tuning_data, alpha, beta, perf))
    measure_performance(root_start, num_bfs_roots, benchmark, bfs_roots, pred, SCALE, edgefactor, auto_tuning_data, *alpha, *beta, perf);

  while(auto_tuning_each(AUTO_BETA,  root_start, num_bfs_roots, benchmark, bfs_roots, pred, SCALE, edgefactor, auto_tuning_data, alpha, beta, perf))
    measure_performance(root_start, num_bfs_roots, benchmark, bfs_roots, pred, SCALE, edgefactor, auto_tuning_data, *alpha, *beta, perf);

  if(mpi.isMaster())
    print_with_prefix("Estimated Performance = %.0f MTEPS with Alpha = %f and Beta = %f",
                      calc_TEPS(root_start, num_bfs_roots, perf)/1000000, *alpha, *beta);
}

void graph500_bfs(int SCALE, int edgefactor, double alpha, double beta, int validation_level,
                  bool auto_tuning_enabled, bool pre_exec, bool real_benchmark)
{
	using namespace PRM;
	SET_AFFINITY;

        int num_bfs_roots = (real_benchmark)? REAL_BFS_ROOTS : TEST_BFS_ROOTS;
        int64_t auto_tuning_data[num_bfs_roots][AUTO_NUM];
	double bfs_times[num_bfs_roots], validate_times[num_bfs_roots], edge_counts[num_bfs_roots];
	LogFileFormat log = {0};
	int root_start = read_log_file(&log, SCALE, edgefactor, bfs_times, validate_times, edge_counts);
	if(mpi.isMaster() && root_start != 0)
		print_with_prefix("Resume from %d th run", root_start);

	auto edge_list = new EdgeListStorage<UnweightedPackedEdge, 8*1024*1024>(
			(int64_t(1) << SCALE) * edgefactor / mpi.size_2d, getenv("TMPFILE"));

	BfsOnCPU::printInformation(validation_level, pre_exec, real_benchmark);

	if(mpi.isMaster()) print_with_prefix("Graph generation");
	double generation_time = MPI_Wtime();
	generate_graph_spec2010(edge_list, SCALE, edgefactor);
	generation_time = MPI_Wtime() - generation_time;

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	// Create BFS instance and the *COMMUNICATION THREAD*.
	BfsOnCPU* benchmark = new BfsOnCPU();
	double construction_time = MPI_Wtime();
	benchmark->construct(edge_list);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) print_with_prefix("Save graph in file temporally...");
	// メモリを空けるためグラフデータの一部を一時的にファイルに保存
	benchmark->graph_.save_in_file(getenv("TMPFILE2"));

	if(mpi.isMaster()) print_with_prefix("Redistributing edge list...");
	double redistribution_time = MPI_Wtime();
	redistribute_edge_2d(edge_list);
	redistribution_time = MPI_Wtime() - redistribution_time;

	if(mpi.isMaster()) print_with_prefix("Prepare fast validation...");
	PredGather pred_gather;
	pred_gather.init(edge_list);

	// validatioにはedge_listの代わりにpred_gatherを使用するので、edge_listを解放
	delete edge_list; edge_list = nullptr;
	
	if(mpi.isMaster()) print_with_prefix("Restore graph from file...");
	benchmark->graph_.restore_from_file(getenv("TMPFILE2"));

	int64_t bfs_roots[num_bfs_roots];
	const int64_t max_used_vertex = find_max_used_vertex(benchmark->graph_);
	const int64_t nlocalverts = benchmark->graph_.pred_size();

	int64_t *pred = static_cast<int64_t*>(
		cache_aligned_xmalloc(nlocalverts*sizeof(pred[0])));

#if INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp parallel for
	for(int64_t i = 0; i < nlocalverts; ++i) {
		pred[i] = -1;
	}
#endif

	bool result_ok = true;

	if(root_start == 0)
          init_log(SCALE, edgefactor, generation_time, construction_time, redistribution_time, &log);

	benchmark->prepare_bfs(validation_level, pre_exec, real_benchmark);

#if PERSISTENT_COMM
	// To improve performance, MPI_Request is pre-created for MPI_Send_init() and MPI_Recv_init()
	if(SCALE > PERSISTENT_COMM_PRE_EXE_SCALE_THED){
	  for(int j = 0; j < AUTO_NUM; ++j)
	    auto_tuning_data[0][j] = AUTO_NOT_DEFINED;

	  int k = 0;
	  while(auto_tuning_data[0][AUTO_T2B_LEVEL] == AUTO_NOT_DEFINED){
	    benchmark->run_bfs(k++, pred, edgefactor, alpha, beta, auto_tuning_data[0]);
	  }
	}
#endif
        if(auto_tuning_enabled == false){
          int64_t auto_tuning_data[num_bfs_roots][AUTO_NUM]; // not used
          double perf[num_bfs_roots];                        // not used
          find_roots_in_large_graph(root_start, num_bfs_roots, benchmark, bfs_roots, pred,
                                    SCALE, edgefactor, auto_tuning_data, alpha, beta, perf);
        }
        else{
          if(SCALE > 42){
            if(mpi.isMaster())
              print_with_prefix("Auto-tuning option cannot be supported SCALE > 42"); // Please define pf_nedge[43].
            MPI_Finalize();
            exit(1);
          }
          MPI_Barrier(mpi.comm_2d);
          double elapsed_time = MPI_Wtime();
          
          auto_tuning(root_start, num_bfs_roots, benchmark, bfs_roots, pred,
                      SCALE, edgefactor, &alpha, &beta);

          if(mpi.isMaster())
            print_with_prefix("Elapsed Time for auto tuning = %f sec.", MPI_Wtime() - elapsed_time);
        }

	if(pre_exec){
#ifdef _FUGAKU_POWER_MEASUREMENT
	  PWR_Cntxt cntxt = NULL;
	  PWR_Obj obj = NULL;
	  int rc;
	  double energy1 = 0.0;
	  double energy2 = 0.0;
	  double menergy1 = 0.0;
	  double menergy2 = 0.0;
	  double ave_power[2];
	  double t_power[2];
	  PWR_Time ts1 = 0;
	  PWR_Time ts2 = 0;
	  rc = PWR_CntxtInit(PWR_CNTXT_FX1000, PWR_ROLE_APP, "app", &cntxt);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("CntxtInit Failed\n");
	  }
	  rc = PWR_CntxtGetObjByName(cntxt, "plat.node", &obj);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("CntxtGetObjByName Failed\n");
	  }
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_MEASURED_ENERGY, &menergy1, &ts1);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy1, NULL);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
#endif
      if(mpi.isMaster()){
        time_t t = time(NULL);
        print_with_prefix("Start energy loop : %s", ctime(&t));
      }
	  double time_left = PRE_EXEC_TIME;
	  for(int c = root_start; time_left > 0.0; ++c) {
		if(mpi.isMaster())
		  print_with_prefix("========== Pre Running BFS %d ==========", c);
		MPI_Barrier(mpi.comm_2d);
		double bfs_time = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[c % num_bfs_roots], pred, edgefactor, alpha, beta, auto_tuning_data[c % num_bfs_roots]);
		bfs_time = MPI_Wtime() - bfs_time;
		if(mpi.isMaster()) {
		  print_with_prefix("Time for BFS %d is %f", c, bfs_time);
		  time_left -= bfs_time;
		}
		MPI_Bcast(&time_left, 1, MPI_DOUBLE, 0, mpi.comm_2d);
	  }
      if(mpi.isMaster()){
        time_t t = time(NULL);
        print_with_prefix("End energy loop : %s", ctime(&t));
      }
#ifdef _FUGAKU_POWER_MEASUREMENT
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_MEASURED_ENERGY, &menergy2, &ts2);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy2, NULL);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
	  ave_power[0] = (menergy2 - menergy1) / ((ts2 - ts1) / 1000000000.0);
	  ave_power[1] = (energy2 - energy1) / ((ts2 - ts1) / 1000000000.0);
	  MPI_Reduce(ave_power, t_power, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  if(mpi.isMaster()){
        print_with_prefix("total measured average power : %lf", t_power[0]);
        print_with_prefix("total estimated average power : %lf", t_power[1]);
	  }
#endif
	}

#ifdef PROFILE_REGIONS
	timer_clear();
#endif
	for(int i = root_start; i < num_bfs_roots; ++i) {
		VERVOSE(print_max_memory_usage());

		if(mpi.isMaster())  print_with_prefix("========== Running BFS %02d ==========", i);
#if ENABLE_FUJI_PROF
		fapp_start("bfs", i, 1);
#endif
		MPI_Barrier(mpi.comm_2d);
#if FUGAKU_MPI_PRINT_STATS
		FJMPI_Collection_start();
#endif
		PROF(profiling::g_pis.reset());
		bfs_times[i] = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[i], pred, edgefactor, alpha, beta, auto_tuning_data[i]);
		bfs_times[i] = MPI_Wtime() - bfs_times[i];
#if FUGAKU_MPI_PRINT_STATS
                FJMPI_Collection_stop();
#endif
#if ENABLE_FUJI_PROF
		fapp_stop("bfs", i, 1);
#endif
		PROF(profiling::g_pis.printResult());
		if(mpi.isMaster()) {
			print_with_prefix("Time for BFS %02d is %f", i, bfs_times[i]);
		}

		benchmark->get_pred(pred);

		validate_times[i] = MPI_Wtime();
		int64_t edge_visit_count = 0;
                if(validation_level == 2){
                  result_ok = validate_bfs_result(&pred_gather, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
                }
                else if(validation_level == 1){
                  if(i == 0) {
			result_ok = validate_bfs_result(&pred_gather, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
			pf_nedge[SCALE] = edge_visit_count;
                  }
                  else {
                    edge_visit_count = pf_nedge[SCALE];
                  }
                }
                else{ // validation_level == 0
                  edge_visit_count = pf_nedge[SCALE];
                }

		validate_times[i] = MPI_Wtime() - validate_times[i];
		edge_counts[i] = (double)edge_visit_count;

		if(mpi.isMaster()) {
			print_with_prefix("Validate time for BFS %02d is %f", i, validate_times[i]);
			print_with_prefix("Number of traversed edges is %" PRId64 "", edge_visit_count);
			print_with_prefix("TEPS for BFS %02d is %g", i, edge_visit_count / bfs_times[i]);
		}

		if(result_ok == false) {
			break;
		}

		update_log_file(&log, bfs_times[i], validate_times[i], edge_visit_count);
	}
	benchmark->end_bfs();

	if(mpi.isMaster()) {
	  fprintf(stdout, "============= Result ==============\n");
	  fprintf(stdout, "SCALE:                          %d\n", SCALE);
	  fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
          fprintf(stdout, "alpha:                          %f\n", alpha);
          fprintf(stdout, "beta:                           %f\n", beta);
	  fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
	  fprintf(stdout, "graph_generation:               %g\n", generation_time);
	  fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size_2d);
	  fprintf(stdout, "construction_time:              %g\n", construction_time);
	  fprintf(stdout, "redistribution_time:            %g\n", redistribution_time);
	  print_bfs_result(num_bfs_roots, bfs_times, validate_times, edge_counts, result_ok);
	}
#ifdef PROFILE_REGIONS
	timer_print(bfs_times, num_bfs_roots);
#endif

#if FUGAKU_MPI_PRINT_STATS
        FJMPI_Collection_print(const_cast<char *>("Communication Statistics\n"));
#endif

	delete benchmark;

	free(pred);
}
#if 0
void test02(int SCALE, int edgefactor)
{
	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
			(INT64_C(1) << SCALE) * edgefactor / mpi.size, getenv("TMPFILE"));
	RmatGraphGenerator<UnweightedPackedEdge> graph_generator(
//	RandomGraphGenerator<UnweightedPackedEdge> graph_generator(
				SCALE, edgefactor, 2, 3, InitialEdgeType::NONE);
	Graph2DCSR<Pack40bit, uint32_t> graph;

	double generation_time = MPI_Wtime();
	generate_graph(&edge_list, &graph_generator);
	generation_time = MPI_Wtime() - generation_time;

	double construction_time = MPI_Wtime();
	construct_graph(&edge_list, true, graph);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) {
		print_with_prefix("TEST02");
		fprintf(stdout, "SCALE:                          %d\n", SCALE);
		fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
		fprintf(stdout, "graph_generation:               %g\n", generation_time);
		fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size);
		fprintf(stdout, "construction_time:              %g\n", construction_time);
	}
}
#endif

#define ERROR(...) do{fprintf(stderr, __VA_ARGS__); exit(1);}while(0)
static void print_help(char *argv)
{
  ERROR("%s SCALE [-e edge_factor] [-a alpha] [-b beta] [-v validation level] [-P] [-R]\n", argv);
  // Validation Level: 0: No validation, 1: validate at first time only, 2: validate all results
  // Note: To conform to the specification, you must set 2
}

static void set_args(const int argc, char **argv, int *edge_factor, double *alpha, double *beta,
                     int *validation_level, bool *auto_tuning_enabled, bool *pre_exec, bool *real_benchmark)
{
  int result;
  while((result = getopt(argc,argv,"e:a:b:v:APR"))!=-1){
    switch(result){
    case 'e':
      *edge_factor = atoi(optarg);
      if(*edge_factor <= 0)
        ERROR("-e value > 0\n");
      break;
    case 'a':
      *alpha = atof(optarg);
      if(*alpha <= 0)
        ERROR("-a value > 0\n");
      break;
    case 'b':
      *beta = atof(optarg);
      if(*beta <= 0)
        ERROR("-b value > 0\n");
      break;
    case 'v':
      *validation_level = atoi(optarg);
      if(*validation_level < 0 || *validation_level > 2)
        ERROR("-v value >= 0 && value <= 2\n");
      break;
    case 'A':
      *auto_tuning_enabled = true;
      break;
    case 'P':
      *pre_exec = true;
      break;
    case 'R':
      *real_benchmark = true;
      break;
    default:
      print_help(argv[0]);
    }
  }
}

int main(int argc, char** argv)
{
  if(argc <= 1 || atoi(argv[1]) <= 0)
    print_help(argv[0]);
  
  int scale                = atoi(argv[1]);
  int edge_factor          = DEFAULT_EDGE_FACTOR; // nedges / nvertices, i.e., 2*avg. degree
  double alpha             = DEFAULT_ALPHA;
  double beta              = DEFAULT_BETA;
  int validation_level     = DEFAULT_VALIDATION_LEVEL;
  bool auto_tuning_enabled = false;
  bool real_benchmark      = false;
  bool pre_exec            = false;

  set_args(argc, argv, &edge_factor, &alpha, &beta, &validation_level, &auto_tuning_enabled, &pre_exec, &real_benchmark);
  if(real_benchmark){
    validation_level = 2;
    pre_exec = true;
  }
  
  setup_globals(argc, argv, scale, edge_factor);
  graph500_bfs(scale, edge_factor, alpha, beta, validation_level, auto_tuning_enabled, pre_exec, real_benchmark);
  cleanup_globals();
  
  return 0;
}

double elapsed[NUM_RESIONS], start[NUM_RESIONS];
void timer_clear()
{
  for(int i=0;i<NUM_RESIONS;i++)
    elapsed[i] = 0.0;
}

void timer_start(const int n)
{
  start[n] = MPI_Wtime();
}

void timer_stop(const int n)
{
  double now = MPI_Wtime();
  double t = now - start[n];
  elapsed[n] += t;
}

double timer_read(const int n)
{
  return(elapsed[n]);
}

void timer_print(double *bfs_times, const int num_bfs_roots)
{
  double t[NUM_RESIONS], t_max[NUM_RESIONS], t_min[NUM_RESIONS], t_ave[NUM_RESIONS];
  for(int i=0;i<NUM_RESIONS;i++)
    t[i] = timer_read(i);

  t[TOTAL_TIME] = 0.0;
  for(int i=0;i<num_bfs_roots;i++)
	t[TOTAL_TIME] += bfs_times[i];

  double comm_time = t[TD_EXPAND_TIME] + t[BU_EXPAND_TIME] + t[TD_FOLD_TIME] + t[BU_FOLD_TIME] + t[BU_NBR_TIME];
  t[CALC_TIME]	= (t[TD_TIME] + t[BU_TIME]) - comm_time - t[IMBALANCE_TIME];
  t[OTHER_TIME] = t[TOTAL_TIME] - (t[TD_TIME] + t[BU_TIME]);

  MPI_Reduce(t, t_max, NUM_RESIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(t, t_min, NUM_RESIONS, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(t, t_ave, NUM_RESIONS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for(int i=0;i<NUM_RESIONS;i++)
    t_ave[i] /= size;

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi.isMaster()){
    printf("---\n");
    printf("CATEGORY :                 :   MAX    MIN    AVE   AVE/TIME\n");
    printf("TOTAL                      : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TOTAL_TIME));
    printf(" - TOP_DOWN                : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TD_TIME));
    printf(" - BOTTOM_UP               : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_TIME));
    printf("   - LOCAL_CALC            : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(CALC_TIME));
    printf("   - TD_EXPAND(allgather)  : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TD_EXPAND_TIME));
    printf("   - BU_EXPAND(allgather)  : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_EXPAND_TIME));
    printf("   - TD_FOLD(alltoall)     : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TD_FOLD_TIME));
    printf("   - BU_FOLD(alltoall)     : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_FOLD_TIME));
    printf("   - BU_NEIGHBOR(sendrecv) : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_NBR_TIME));
    printf("   - PROC_IMBALANCE        : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(IMBALANCE_TIME));
    printf(" - OTHER                   : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(OTHER_TIME));
  }
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
}
