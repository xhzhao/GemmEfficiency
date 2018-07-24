// icc -g -O2 -o nested nested.cpp -fopenmp -std=c++11
// OMP_NUM_THREADS=56 KMP_HW_SUBSET=2s,28c,1t KMP_AFFINITY=compact,granularity=fine KMP_BLOCKTIME=infinite ./nested

#include <malloc.h>
#include <cxxabi.h>
#include <chrono>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>

const int C = 40;
const int T = 1024;
const int V = 16;

#define MD(type, array, dims, ptr)                                             \
  auto &array = *reinterpret_cast<type(*) dims>(ptr)

#define _T(x) x
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define __tstart(n) _T(Time::time_point __s##n = Time::now());
#define __tend(n)                                                              \
  _T(Time::time_point __e##n = Time::now());                                   \
  _T(printf("time: %s, th=%d, %.2f ms\n", #n, omp_get_thread_num(),            \
      Duration(__e##n - __s##n).count()));


int compute_all(float *data1, float *data2, float *data3) {
  MD(float, adata1, [C][T][V], data1);
  MD(float, adata2, [C][T][V], data2);
  MD(float, adata3, [C][T][V], data3);
#pragma omp parallel for num_threads(C) proc_bind(close)
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < T; ++i) {
#pragma omp simd
      for (int v = 0; v < V; ++v) {
        adata3[c][i][v] = (adata1[c][i][v] * 1.01) * (adata2[c][i][v] + 0.001);
      }
    }
  }
}

int compute_half(float *data1, float *data2, float *data3) {
  MD(float, adata1, [C/2][T][V], data1);
  MD(float, adata2, [C/2][T][V], data2);
  MD(float, adata3, [C/2][T][V], data3);
//#pragma omp parallel num_threads(C/2) proc_bind(close)
#pragma omp for nowait collapse(1)
  for (int c = 0; c < C/2; ++c) {
    for (int i = 0; i < T; ++i) {
#pragma omp simd
      for (int v = 0; v < V; ++v) {
        adata3[c][i][v] = (adata1[c][i][v] * 1.01) * (adata2[c][i][v] + 0.001);
      }
    }
  }
}

int nested(float *data1, float *data2, float *data3) {
  MD(float, adata1, [C][T][V], data1);
  MD(float, adata2, [C][T][V], data2);
  MD(float, adata3, [C][T][V], data3);
  omp_set_nested(1);
#pragma omp parallel for num_threads(2) proc_bind(spread)
  for (int i = 0; i < 2; ++i) {
    //omp_set_num_threads(C/3);
#pragma omp parallel num_threads(C/2) proc_bind(close)
    {
      compute_half((float *)adata1[i * C/2],
                   (float *)adata2[i * C/2],
                   (float *)adata3[i * C/2]);
    }
  }
  return 0;
}

int plain(float *data1, float *data2, float *data3) {
  compute_all(data1, data2, data3);
  return 0;
}

int main(int argc, char ** argv) {
  float *data1 = (float *)memalign(64, C * T * V * sizeof(float));
  float *data2 = (float *)memalign(64, C * T * V * sizeof(float));
  float *data3 = (float *)memalign(64, C * T * V * sizeof(float));
  MD(float, adata1, [C][T][V], data1);
  MD(float, adata2, [C][T][V], data2);
  MD(float, adata3, [C][T][V], data3);

//#pragma omp parallel for num_threads(C) proc_bind(close)
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < T; ++i) {
      for (int v = 0; v < V; ++v) {
        adata1[c][i][v] = 1.0f;
        adata2[c][i][v] = 1.0f;
      }
    }
  }

  char a = 'p';
  if(argc > 1){
    a = *(argv[1]);
  }

  if(a == 'n'){
  for (int i = 0; i < 10; i++)
    nested(data1, data2, data3);
  __tstart(nested);
  for (int i = 0; i < 1000; i++)
    nested(data1, data2, data3);
  __tend(nested);
  } else {
  for (int i = 0; i < 10; i++)
    plain(data1, data2, data3);
  __tstart(plain);
  for (int i = 0; i < 1000; i++)
    plain(data1, data2, data3);
  __tend(plain);
  }
}


