
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
//#include "mkl.h"
#include "immintrin.h"
#include<math.h>
#include <mkl.h>
#define THREAD_NUM  90

//float buffer[35820][35820];
//float buffer[35820][512];
float * buffer = NULL;

void transpose( int M, int N, float * a)
{
  //mkl_simatcopy('r', 't', M, N, 1.0, a, N, M);
  mkl_somatcopy('r','t', N, M, 1.0, a, M, buffer, N);
}

/*
void sgemm4_test( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc)
{
    printf("sgemm4_opt start \n");

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
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
          pc[j*ldc+i] = beta*pc[j*ldc+i]+alpha*sum;
        }
        a_++;
      }


}
void sgemm1_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc)
{
  if(buffer == NULL)
  {
     buffer = malloc(64*512*sizeof(float));
  }
  transpose( 64,512, pa);
  //sgemm4_test(pTransA,pTransB,pM,pN,pK,pAlpha,pa,pM,pb,pldb,pBeta,pc,pldc);
  //sgemm4_test(pTransA,pTransB,pM,pN,pK,pAlpha,pa,pM,buffer,pldb,pBeta,pc,pldc);

}
*/
#if 1
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
  //printf("sgemm1_opt start, m = %d, n =%d, k = %d, row_per_thread = %d\n",M,N,K,row_per_thread);
#pragma omp parallel num_threads(THREAD_NUM)
//for(int tid = 0; tid < THREAD_NUM; tid ++)
  {
    int tid = omp_get_thread_num();
    //printf("tid = %d start\n", tid);
    int i=0;
    for(i = tid*row_per_thread ; i < (tid+1)*row_per_thread ; i++) {

      float * bA = pa + i * lda  ;
      int j = 0;
      for(j = 0; j < N; j++){
        float sum = 0;
        float * bB = pb + j * ldb ;
        float * c_base = pc + j * ldc + i;
        __m512 c_0_15  = _mm512_setzero_ps();
        __m512 c_16_31 = _mm512_setzero_ps();
        __m512 c_32_47 = _mm512_setzero_ps();
        __m512 c_48_63 = _mm512_setzero_ps();
        __m512 c_sum_1 = _mm512_setzero_ps();
        __m512 c_sum_2 = _mm512_setzero_ps();
        __m512 c_sum_3 = _mm512_setzero_ps();
        int kk = 0;
        float * a_base = bA;
        float * b_base = bB;
        for(kk = 0; kk < K/64; kk++){


          __m512 a_0_15  = _mm512_load_ps(a_base );
          __m512 b_0_15  = _mm512_load_ps(b_base);
          __m512 a_16_31 = _mm512_load_ps(a_base + 16);
          __m512 b_16_31 = _mm512_load_ps(b_base + 16);
          c_0_15  = _mm512_fmadd_ps(a_0_15,  b_0_15,  c_0_15);
          __m512 a_32_47 = _mm512_load_ps(a_base + 32);
          __m512 b_32_47 = _mm512_load_ps(b_base + 32);
          c_16_31 = _mm512_fmadd_ps(a_16_31, b_16_31, c_16_31);
          __m512 a_48_63 = _mm512_load_ps(a_base + 48);
          __m512 b_48_63 = _mm512_load_ps(b_base + 48);

          c_32_47 = _mm512_fmadd_ps(a_32_47, b_32_47, c_32_47);
          c_48_63 = _mm512_fmadd_ps(a_48_63, b_48_63, c_48_63);
          
          a_base += 64;
          b_base += 64; 
        }

        c_sum_1 = _mm512_add_ps(c_0_15, c_16_31);
        c_sum_2 = _mm512_add_ps(c_32_47, c_48_63);
        c_sum_3 = _mm512_add_ps(c_sum_1, c_sum_2);
        (* c_base) = _mm512_reduce_add_ps(c_sum_3);

/*
        if (beta == 0)
          pc[j*ldc+i] = alpha*sum;
        else
          pc[j*ldc+i] = beta*pc[j*ldc+i]+alpha*sum;
*/
      }
    }
  }


}
#endif
