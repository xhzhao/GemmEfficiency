#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>



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
int my_sgemm_opt4(char transa, char transb,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int lda,
  const float * B, const int ldb, 
  const float beta, float * C, const int ldc){
  //printf("my_sgemm_opt4------------------------------------------------\n");
  #pragma omp parallel num_threads(20)
  {
    int tid = omp_get_thread_num();
    int n_start = (N/10) * (tid % 10);
    int n_end = n_start + N/10;
    int m_start = (M/2) * (tid/10);
    int m_end = m_start + M/2;
    int BL = N * 4;
    //printf("tid = %d, [m = %d:%d, n = %d:%d] \n", tid, m_start, m_end, n_start, n_end);
    for(int m = m_start; m < m_end; m += MB) {
      for(int n = n_start; n < n_end; n += NB) {
        //this block will calc [5,16] of C
        __m512 cx_16[80];
        for(int ci = 0; ci < 80; ci++) {
          cx_16[ci] = _mm512_setzero_ps();
        }
        //target c address base
        float* cbase = C + m * N + n;

        // will load [16, 16] from B
        __m512 b16_16[16];

        for(int k = 0; k < K; k+=KB) {
          // base address for A and B
          const float* abase = A + m * K + k;
          const float* bbase = B + k * N + n;

          // load [5, 16] from A
          __m512 ax_16[5];
          for(int ai = 0; ai < 5; ai++) {
            ax_16[ai] = _mm512_load_ps(abase + ai * K);
          }

          // load [16, 16] from B
          __m512i bindex = _mm512_set_epi32(15*BL, 14*BL, 13*BL, 12*BL, 11*BL,
            10*BL, 9*BL, 8*BL,7*BL, 6*BL, 5*BL, 4*BL, 3*BL, 2*BL, 1*BL, 0);
          for(int bi = 0; bi < 16; bi++) {
            b16_16[bi] = _mm512_i32gather_ps(bindex, bbase + bi, 1);
          }

          for(int ci = 0; ci < 5; ci++) {
            for(int cj = 0; cj < 16; cj++) {
              cx_16[ci*16 + cj] = _mm512_fmadd_ps(ax_16[ci], b16_16[cj], cx_16[ci*16 + cj]);
            }
          }
        }

        float cx[16];
        for(int ci = 0; ci < 5; ci++) {
          for(int cj = 0; cj < 16; cj++) {
            cx[cj] = _mm512_reduce_add_ps(cx_16[ci * 16 + cj]);
            cbase[ci * N + cj] = cx[cj];
          }
        }

      }
    }
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
