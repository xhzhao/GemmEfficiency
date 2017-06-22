
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
//#include "mkl.h"
#include "immintrin.h"
#include<math.h>

#define THREAD_NUM  10


void sgemm1_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc)
{

  const int M = *pM;
  const int N = *pN;
  const int K = *pK;
  const int lda = *plda;
  const int ldb = *pldb;
  const int ldc = *pldc;
  float alpha = *pAlpha;
  float beta = *pBeta;
  const int row_per_thread = M/THREAD_NUM;
  printf("sgemm1_opt start, m = %d, n =%d, k = %d, row_per_thread = %d\n",M,N,K,row_per_thread);

#pragma omp parallel num_threads(THREAD_NUM)
  {
    int tid = omp_get_thread_num();
    printf("tid = %d start\n", tid);
    int i=0;
    for(i = tid*row_per_thread ; i < (tid+1)*row_per_thread ; i++) {

      float * bA = pa + i * lda  ;
      int j = 0;
      for(j = 0; j < N; j++){
        float sum = 0;
        float * bB = pb + j * ldb ;
        int l = 0;
        for(l = 0; l < K; l++){
            sum += bA[l]*bB[l];
        }
        if (beta == 0)
          pc[j*ldc+i] = alpha*sum;
        else
          pc[j*ldc+i] = beta*pc[j*ldc+i]+alpha*sum;
      }
    }
  }

}
