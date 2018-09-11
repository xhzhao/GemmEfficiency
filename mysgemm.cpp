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

int reducesum_and_set(float* dst, __m512 s0, __m512 s1, __m512 s2, __m512 s3,
  __m512 s4, __m512 s5, __m512 s6, __m512 s7, __m512 s8, __m512 s9, __m512 sa,
  __m512 sb, __m512 sc, __m512 sd, __m512 se, __m512 sf) {
  dst[0x0] += _mm512_reduce_add_ps(s0);
  dst[0x1] += _mm512_reduce_add_ps(s1);
  dst[0x2] += _mm512_reduce_add_ps(s2);
  dst[0x3] += _mm512_reduce_add_ps(s3);
  dst[0x4] += _mm512_reduce_add_ps(s4);
  dst[0x5] += _mm512_reduce_add_ps(s5);
  dst[0x6] += _mm512_reduce_add_ps(s6);
  dst[0x7] += _mm512_reduce_add_ps(s7);
  dst[0x8] += _mm512_reduce_add_ps(s8);
  dst[0x9] += _mm512_reduce_add_ps(s9);
  dst[0xa] += _mm512_reduce_add_ps(sa);
  dst[0xb] += _mm512_reduce_add_ps(sb);
  dst[0xc] += _mm512_reduce_add_ps(sc);
  dst[0xd] += _mm512_reduce_add_ps(sd);
  dst[0xe] += _mm512_reduce_add_ps(se);
  dst[0xf] += _mm512_reduce_add_ps(sf);
  //__m512 cout = _mm512_set_ps(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,fa,fb,fc,fd,fe,ff);
  //_mm512_store_ps(dst, cout);
  return 1;
}

#define MB 5
#define NB 16
#define KB 16
#define KCB 256

//calc [5,16] of C
void gemmblock_5_16(int M, int N, int K,int kmin, int kmax, int m, int n, const float * A, const int lda, const float * B, const int ldb, float * C, const int ldc) {

    int BL = N * 4;
        __m512 c00 = _mm512_setzero_ps();
        __m512 c01 = _mm512_setzero_ps();
        __m512 c02 = _mm512_setzero_ps();
        __m512 c03 = _mm512_setzero_ps();
        __m512 c04 = _mm512_setzero_ps();
        __m512 c05 = _mm512_setzero_ps();
        __m512 c06 = _mm512_setzero_ps();
        __m512 c07 = _mm512_setzero_ps();
        __m512 c08 = _mm512_setzero_ps();
        __m512 c09 = _mm512_setzero_ps();
        __m512 c0a = _mm512_setzero_ps();
        __m512 c0b = _mm512_setzero_ps();
        __m512 c0c = _mm512_setzero_ps();
        __m512 c0d = _mm512_setzero_ps();
        __m512 c0e = _mm512_setzero_ps();
        __m512 c0f = _mm512_setzero_ps();

        __m512 c10 = _mm512_setzero_ps();
        __m512 c11 = _mm512_setzero_ps();
        __m512 c12 = _mm512_setzero_ps();
        __m512 c13 = _mm512_setzero_ps();
        __m512 c14 = _mm512_setzero_ps();
        __m512 c15 = _mm512_setzero_ps();
        __m512 c16 = _mm512_setzero_ps();
        __m512 c17 = _mm512_setzero_ps();
        __m512 c18 = _mm512_setzero_ps();
        __m512 c19 = _mm512_setzero_ps();
        __m512 c1a = _mm512_setzero_ps();
        __m512 c1b = _mm512_setzero_ps();
        __m512 c1c = _mm512_setzero_ps();
        __m512 c1d = _mm512_setzero_ps();
        __m512 c1e = _mm512_setzero_ps();
        __m512 c1f = _mm512_setzero_ps();

        __m512 c20 = _mm512_setzero_ps();
        __m512 c21 = _mm512_setzero_ps();
        __m512 c22 = _mm512_setzero_ps();
        __m512 c23 = _mm512_setzero_ps();
        __m512 c24 = _mm512_setzero_ps();
        __m512 c25 = _mm512_setzero_ps();
        __m512 c26 = _mm512_setzero_ps();
        __m512 c27 = _mm512_setzero_ps();
        __m512 c28 = _mm512_setzero_ps();
        __m512 c29 = _mm512_setzero_ps();
        __m512 c2a = _mm512_setzero_ps();
        __m512 c2b = _mm512_setzero_ps();
        __m512 c2c = _mm512_setzero_ps();
        __m512 c2d = _mm512_setzero_ps();
        __m512 c2e = _mm512_setzero_ps();
        __m512 c2f = _mm512_setzero_ps();

        __m512 c30 = _mm512_setzero_ps();
        __m512 c31 = _mm512_setzero_ps();
        __m512 c32 = _mm512_setzero_ps();
        __m512 c33 = _mm512_setzero_ps();
        __m512 c34 = _mm512_setzero_ps();
        __m512 c35 = _mm512_setzero_ps();
        __m512 c36 = _mm512_setzero_ps();
        __m512 c37 = _mm512_setzero_ps();
        __m512 c38 = _mm512_setzero_ps();
        __m512 c39 = _mm512_setzero_ps();
        __m512 c3a = _mm512_setzero_ps();
        __m512 c3b = _mm512_setzero_ps();
        __m512 c3c = _mm512_setzero_ps();
        __m512 c3d = _mm512_setzero_ps();
        __m512 c3e = _mm512_setzero_ps();
        __m512 c3f = _mm512_setzero_ps();

        __m512 c40 = _mm512_setzero_ps();
        __m512 c41 = _mm512_setzero_ps();
        __m512 c42 = _mm512_setzero_ps();
        __m512 c43 = _mm512_setzero_ps();
        __m512 c44 = _mm512_setzero_ps();
        __m512 c45 = _mm512_setzero_ps();
        __m512 c46 = _mm512_setzero_ps();
        __m512 c47 = _mm512_setzero_ps();
        __m512 c48 = _mm512_setzero_ps();
        __m512 c49 = _mm512_setzero_ps();
        __m512 c4a = _mm512_setzero_ps();
        __m512 c4b = _mm512_setzero_ps();
        __m512 c4c = _mm512_setzero_ps();
        __m512 c4d = _mm512_setzero_ps();
        __m512 c4e = _mm512_setzero_ps();
        __m512 c4f = _mm512_setzero_ps();

        // will load [16, 16] from B
        __m512 b00 = _mm512_setzero_ps();
        __m512 b01 = _mm512_setzero_ps();
        __m512 b02 = _mm512_setzero_ps();
        __m512 b03 = _mm512_setzero_ps();
        __m512 b04 = _mm512_setzero_ps();
        __m512 b05 = _mm512_setzero_ps();
        __m512 b06 = _mm512_setzero_ps();
        __m512 b07 = _mm512_setzero_ps();
        __m512 b08 = _mm512_setzero_ps();
        __m512 b09 = _mm512_setzero_ps();
        __m512 b0a = _mm512_setzero_ps();
        __m512 b0b = _mm512_setzero_ps();
        __m512 b0c = _mm512_setzero_ps();
        __m512 b0d = _mm512_setzero_ps();
        __m512 b0e = _mm512_setzero_ps();
        __m512 b0f = _mm512_setzero_ps();

        float* cbase = C + m * N + n;

        for(int k = kmin; k < kmax; k+=KB) {
          // base address for A and B
          const float* abase = A + m * K + k;
          const float* bbase = B + k * N + n;

          // load [5, 16] from A
          __m512 a00 = _mm512_load_ps(abase);
          __m512 a10 = _mm512_load_ps(abase + K);
          __m512 a20 = _mm512_load_ps(abase + K * 2);
          __m512 a30 = _mm512_load_ps(abase + K * 3);
          __m512 a40 = _mm512_load_ps(abase + K * 4);

          // load [16, 16] from B
          __m512i bindex = _mm512_set_epi32(15*BL, 14*BL, 13*BL, 12*BL, 11*BL,
            10*BL, 9*BL, 8*BL,7*BL, 6*BL, 5*BL, 4*BL, 3*BL, 2*BL, 1*BL, 0);
          b00 = _mm512_i32gather_ps(bindex, bbase, 1);
          b01 = _mm512_i32gather_ps(bindex, bbase + 1, 1);
          b02 = _mm512_i32gather_ps(bindex, bbase + 2, 1);
          b03 = _mm512_i32gather_ps(bindex, bbase + 3, 1);
          b04 = _mm512_i32gather_ps(bindex, bbase + 4, 1);
          b05 = _mm512_i32gather_ps(bindex, bbase + 5, 1);
          b06 = _mm512_i32gather_ps(bindex, bbase + 6, 1);
          b07 = _mm512_i32gather_ps(bindex, bbase + 7, 1);
          b08 = _mm512_i32gather_ps(bindex, bbase + 8, 1);
          b09 = _mm512_i32gather_ps(bindex, bbase + 9, 1);
          b0a = _mm512_i32gather_ps(bindex, bbase + 10, 1);
          b0b = _mm512_i32gather_ps(bindex, bbase + 11, 1);
          b0c = _mm512_i32gather_ps(bindex, bbase + 12, 1);
          b0d = _mm512_i32gather_ps(bindex, bbase + 13, 1);
          b0e = _mm512_i32gather_ps(bindex, bbase + 14, 1);
          b0f = _mm512_i32gather_ps(bindex, bbase + 15, 1);

          c00 = _mm512_fmadd_ps(a00, b00, c00);
          c01 = _mm512_fmadd_ps(a00, b01, c01);
          c02 = _mm512_fmadd_ps(a00, b02, c02);
          c03 = _mm512_fmadd_ps(a00, b03, c03);
          c04 = _mm512_fmadd_ps(a00, b04, c04);
          c05 = _mm512_fmadd_ps(a00, b05, c05);
          c06 = _mm512_fmadd_ps(a00, b06, c06);
          c07 = _mm512_fmadd_ps(a00, b07, c07);
          c08 = _mm512_fmadd_ps(a00, b08, c08);
          c09 = _mm512_fmadd_ps(a00, b09, c09);
          c0a = _mm512_fmadd_ps(a00, b0a, c0a);
          c0b = _mm512_fmadd_ps(a00, b0b, c0b);
          c0c = _mm512_fmadd_ps(a00, b0c, c0c);
          c0d = _mm512_fmadd_ps(a00, b0d, c0d);
          c0e = _mm512_fmadd_ps(a00, b0e, c0e);
          c0f = _mm512_fmadd_ps(a00, b0f, c0f);

          c10 = _mm512_fmadd_ps(a10, b00, c10);
          c11 = _mm512_fmadd_ps(a10, b01, c11);
          c12 = _mm512_fmadd_ps(a10, b02, c12);
          c13 = _mm512_fmadd_ps(a10, b03, c13);
          c14 = _mm512_fmadd_ps(a10, b04, c14);
          c15 = _mm512_fmadd_ps(a10, b05, c15);
          c16 = _mm512_fmadd_ps(a10, b06, c16);
          c17 = _mm512_fmadd_ps(a10, b07, c17);
          c18 = _mm512_fmadd_ps(a10, b08, c18);
          c19 = _mm512_fmadd_ps(a10, b09, c19);
          c1a = _mm512_fmadd_ps(a10, b0a, c1a);
          c1b = _mm512_fmadd_ps(a10, b0b, c1b);
          c1c = _mm512_fmadd_ps(a10, b0c, c1c);
          c1d = _mm512_fmadd_ps(a10, b0d, c1d);
          c1e = _mm512_fmadd_ps(a10, b0e, c1e);
          c1f = _mm512_fmadd_ps(a10, b0f, c1f);

          c20 = _mm512_fmadd_ps(a20, b00, c20);
          c21 = _mm512_fmadd_ps(a20, b01, c21);
          c22 = _mm512_fmadd_ps(a20, b02, c22);
          c23 = _mm512_fmadd_ps(a20, b03, c23);
          c24 = _mm512_fmadd_ps(a20, b04, c24);
          c25 = _mm512_fmadd_ps(a20, b05, c25);
          c26 = _mm512_fmadd_ps(a20, b06, c26);
          c27 = _mm512_fmadd_ps(a20, b07, c27);
          c28 = _mm512_fmadd_ps(a20, b08, c28);
          c29 = _mm512_fmadd_ps(a20, b09, c29);
          c2a = _mm512_fmadd_ps(a20, b0a, c2a);
          c2b = _mm512_fmadd_ps(a20, b0b, c2b);
          c2c = _mm512_fmadd_ps(a20, b0c, c2c);
          c2d = _mm512_fmadd_ps(a20, b0d, c2d);
          c2e = _mm512_fmadd_ps(a20, b0e, c2e);
          c2f = _mm512_fmadd_ps(a20, b0f, c2f);

          c30 = _mm512_fmadd_ps(a30, b00, c30);
          c31 = _mm512_fmadd_ps(a30, b01, c31);
          c32 = _mm512_fmadd_ps(a30, b02, c32);
          c33 = _mm512_fmadd_ps(a30, b03, c33);
          c34 = _mm512_fmadd_ps(a30, b04, c34);
          c35 = _mm512_fmadd_ps(a30, b05, c35);
          c36 = _mm512_fmadd_ps(a30, b06, c36);
          c37 = _mm512_fmadd_ps(a30, b07, c37);
          c38 = _mm512_fmadd_ps(a30, b08, c38);
          c39 = _mm512_fmadd_ps(a30, b09, c39);
          c3a = _mm512_fmadd_ps(a30, b0a, c3a);
          c3b = _mm512_fmadd_ps(a30, b0b, c3b);
          c3c = _mm512_fmadd_ps(a30, b0c, c3c);
          c3d = _mm512_fmadd_ps(a30, b0d, c3d);
          c3e = _mm512_fmadd_ps(a30, b0e, c3e);
          c3f = _mm512_fmadd_ps(a30, b0f, c3f);

          c40 = _mm512_fmadd_ps(a40, b00, c40);
          c41 = _mm512_fmadd_ps(a40, b01, c41);
          c42 = _mm512_fmadd_ps(a40, b02, c42);
          c43 = _mm512_fmadd_ps(a40, b03, c43);
          c44 = _mm512_fmadd_ps(a40, b04, c44);
          c45 = _mm512_fmadd_ps(a40, b05, c45);
          c46 = _mm512_fmadd_ps(a40, b06, c46);
          c47 = _mm512_fmadd_ps(a40, b07, c47);
          c48 = _mm512_fmadd_ps(a40, b08, c48);
          c49 = _mm512_fmadd_ps(a40, b09, c49);
          c4a = _mm512_fmadd_ps(a40, b0a, c4a);
          c4b = _mm512_fmadd_ps(a40, b0b, c4b);
          c4c = _mm512_fmadd_ps(a40, b0c, c4c);
          c4d = _mm512_fmadd_ps(a40, b0d, c4d);
          c4e = _mm512_fmadd_ps(a40, b0e, c4e);
          c4f = _mm512_fmadd_ps(a40, b0f, c4f);
        }

        reducesum_and_set(cbase,         c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c0a,c0b,c0c,c0d,c0e,c0f);
        reducesum_and_set(cbase + N,     c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c1a,c1b,c1c,c1d,c1e,c1f);
        reducesum_and_set(cbase + 2 * N, c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c2a,c2b,c2c,c2d,c2e,c2f);
        reducesum_and_set(cbase + 3 * N, c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c3a,c3b,c3c,c3d,c3e,c3f);
        reducesum_and_set(cbase + 4 * N, c40,c41,c42,c43,c44,c45,c46,c47,c48,c49,c4a,c4b,c4c,c4d,c4e,c4f);

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
    //  int kmax = (K) < (kc+KCB) ? (K) : (kc+KCB);
      for(int m = m_start; m < m_end; m += MB) {
        for(int n = n_start; n < n_end; n += NB) {

          //this block will calc [5,16] of C
          //gemmblock_5_16(M, N, K, kc, kmax, m, n, A, lda, B, ldb, C, ldc);
          gemmblock_5_16(M, N, K, 0, K, m, n, A, lda, B, ldb, C, ldc);

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
