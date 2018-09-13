#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <string.h>


int my_sgemm_base(char transa, char transb,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int lda,
  const float * B, const int ldb, 
  const float beta, float * C, const int ldc){
  #pragma omp parallel for simd
  for(int n = 0; n < N; n++) {
    for(int m = 0; m < M; m++) {
      C[m * N + n] = 0;
      for(int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }

  return 1;
}

#define MB 5
#define NB 16
#define KB 16
#define KCB 256
//calc [5,16] of C
//get a col of A to vector, get a row of B to vector
//get a col of C ,scatter to memory finally
void gemmblock_5_16_v2(int M, int N, int K,int kmin, int kmax, int m, int n, const float * A, const int lda, const float * B, const int ldb, float * C, const int ldc) {

  float* cbase = C + m * N + n;

  __m512 c0x = _mm512_load_ps(cbase + 0x00 * N);
  __m512 c1x = _mm512_load_ps(cbase + 0x01 * N);
  __m512 c2x = _mm512_load_ps(cbase + 0x02 * N);
  __m512 c3x = _mm512_load_ps(cbase + 0x03 * N);
  __m512 c4x = _mm512_load_ps(cbase + 0x04 * N);


  for(int k = kmin; k < kmax; k++) {
    //load a col of A to a vector
	const float* abase = A + m * lda + k;
	const float* bbase = B + k * N + n;

    //load [16,1] of A to 16  vector
    __m512 a00 = _mm512_set1_ps(*(abase + 0x00 * lda));
    __m512 a10 = _mm512_set1_ps(*(abase + 0x01 * lda));
    __m512 a20 = _mm512_set1_ps(*(abase + 0x02 * lda));
    __m512 a30 = _mm512_set1_ps(*(abase + 0x03 * lda));
    __m512 a40 = _mm512_set1_ps(*(abase + 0x04 * lda));

    
    __m512 b0x = _mm512_load_ps(bbase);
    
    c0x = _mm512_fmadd_ps(a00, b0x, c0x);
    c1x = _mm512_fmadd_ps(a10, b0x, c1x);
    c2x = _mm512_fmadd_ps(a20, b0x, c2x);
    c3x = _mm512_fmadd_ps(a30, b0x, c3x);
    c4x = _mm512_fmadd_ps(a40, b0x, c4x);
  }
  
  //store [16,16] result to C
  _mm512_store_ps(cbase + 0x00 * N, c0x);
  _mm512_store_ps(cbase + 0x01 * N, c1x);
  _mm512_store_ps(cbase + 0x02 * N, c2x);
  _mm512_store_ps(cbase + 0x03 * N, c3x);
  _mm512_store_ps(cbase + 0x04 * N, c4x);
  
}




int my_sgemm_opt4(char transa, char transb,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int lda,
  const float * B, const int ldb, 
  const float beta, float * C, const int ldc){
  //printf("my_sgemm_opt4------------------------------------------------\n");
  //#pragma omp parallel num_threads(20)
  {
    int tid = omp_get_thread_num();
    int n_start = (N/10) * (tid % 10);
    int n_end = n_start + N/10;
    int m_start = (M/2) * (tid/10);
    int m_end = m_start + M/2;

    for(int m = m_start; m < m_end; m++) {
      for(int n = n_start; n < n_end; n ++) {
        C[m*N + n] = 0;
      //memset(C + m * N, 0, 4*N/10);
      }
    }
    //printf("tid = %d, [m = %d:%d, n = %d:%d] \n", tid, m_start, m_end, n_start, n_end);

    //for(int kc = 0; kc < K; kc+=KCB) {
      //int kmax = (K) < (kc+KCB) ? (K) : (kc+KCB);
      for(int m = m_start; m < m_end; m += MB) {
        for(int n = n_start; n < n_end; n += NB) {

          //this block will calc [5,16] of C
          //gemmblock_5_16(M, N, K, kc, kmax, m, n, A, lda, B, ldb, C, ldc);
          //gemmblock_5_16(M, N, K, 0, K, m, n, A, lda, B, ldb, C, ldc);
          gemmblock_5_16_v2(M, N, K, 0, K, m, n, A, lda, B, ldb, C, ldc);
          //gemmblock_5_16_v2(M, N, K, kc, kmax, m, n, A, lda, B, ldb, C, ldc);
        }
      }
    //}
  }

  return 1;
}


int my_sgemm(char transa, char transb,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int lda,
  const float * B, const int ldb, 
  const float beta, float * C, const int ldc) {

  //my_sgemm_base(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  my_sgemm_opt4(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return 1;
}
