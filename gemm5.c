
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
//#include "mkl.h"
#include "immintrin.h"
#include<math.h>

#define THREAD_NUM  10

void trans(int M, int N, float a[M][N])
{
    int m = 0;
    int n = 0;
    float tmp;
    for(m = 0; m < M; m++)
      for(n = 0; n < N; n++)
      {
        tmp = a[m][n];
        a[m][n] = a[n][m];
        a[n][m] = tmp;
      }
}


void sgemm5_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc)
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
  printf("sgemm5_opt start, m = %d, n =%d, k = %d, row_per_thread = %d\n",M,N,K,row_per_thread);

#pragma omp parallel num_threads(THREAD_NUM)
  {
    int tid = omp_get_thread_num();
    printf("tid = %d start\n", tid);
    int i=0;
    for(i = tid*row_per_thread ; i < (tid+1)*row_per_thread ; i++) {

      float * bA = pa + i  ;
      int j = 0;
      for(j = 0; j < N; j++){
        float sum = 0;
        float * bB = pb + j ;
        int l = 0;
        for(l = 0; l < K; l++){
            sum += bA[l*lda]*bB[l*ldb];
        }
        if (beta == 0)
          pc[j*ldc+i] = alpha*sum;
        else
          pc[j*ldc+i] = beta*pc[j*ldc+i]+alpha*sum;
      }
    }
  }

}