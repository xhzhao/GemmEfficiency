
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
//#include "mkl.h"
#include "immintrin.h"
#include<math.h>

#define THREAD_NUM  8

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
  int i,j,l;

    float * a_ = pa;
    for(i = 0; i < M; i++)
      {
        float *b_ = pb;
        for(j = 0; j < N; j++)
        {
          float sum = 0;
          for(l = 0; l < K; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
          if (beta == 0)
            pc[j*ldc+i] = alpha*sum;
          else
            pc[j*ldc+i] = beta*pc[j*ldc+i]+alpha*sum;
        }
        a_++;
      }



}
