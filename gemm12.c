#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include "omp.h"
#include "mkl.h"
#include "immintrin.h"

typedef unsigned long long int uint64_t;

static inline void
sgemm_opt(uint64_t m, uint64_t n, const uint64_t p,
  const float pAlpha, const float * A, const float * B,
  const float pBeta, float * C)
{
  n = 500;

  uint64_t i = 0;
  for (i = 0; i + 4 < m; i += 4) {
    uint64_t j = 0;
    __m512 zero = _mm512_setzero_ps();
    float a0_0_i_k_sc = *(A + i + 0x0);
    float a1_1_i_k_sc = *(A + i + 0x1);
    float a2_2_i_k_sc = *(A + i + 0x2);
    float a3_3_i_k_sc = *(A + i + 0x3);

    __m512 a0_0_i_k = _mm512_set1_ps(a0_0_i_k_sc);
    __m512 a1_1_i_k = _mm512_set1_ps(a1_1_i_k_sc);
    __m512 a2_2_i_k = _mm512_set1_ps(a2_2_i_k_sc);
    __m512 a3_3_i_k = _mm512_set1_ps(a3_3_i_k_sc);

    float *c0_addr = C + (i + 0x0) * n;
    float *c1_addr = C + (i + 0x1) * n;
    float *c2_addr = C + (i + 0x2) * n;
    float *c3_addr = C + (i + 0x3) * n;

    for (j = 0; j + 16 < n; j += 16) {
      __m512 b_k_0_f_j = _mm512_load_ps(B + j);

      __m512 c0_0_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a0_0_i_k);
      __m512 c1_1_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a1_1_i_k);
      __m512 c2_2_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a2_2_i_k);
      __m512 c3_3_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a3_3_i_k);

      _mm512_store_ps(c0_addr, c0_0_i_0_f_j);
      _mm512_store_ps(c1_addr, c1_1_i_0_f_j);
      _mm512_store_ps(c2_addr, c2_2_i_0_f_j);
      _mm512_store_ps(c3_addr, c3_3_i_0_f_j);

      c0_addr += 16;
      c1_addr += 16;
      c2_addr += 16;
      c3_addr += 16;
    }

    if (j < n && (j + 16) >= n) {
      __mmask16 mask = (1u << (n - j)) - 1;

      __m512 b_k_0_f_j = _mm512_mask_load_ps(zero, mask, B + j);

      __m512 c0_0_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a0_0_i_k);
      __m512 c1_1_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a1_1_i_k);
      __m512 c2_2_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a2_2_i_k);
      __m512 c3_3_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a3_3_i_k);

      _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
      _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
      _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
      _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
    }
  }

  for (; i < m; i ++) {
    uint64_t j = 0;
    __m512 zero = _mm512_setzero_ps();
    float a0_0_i_k_sc = *(A + i + 0x0);

    __m512 a0_0_i_k = _mm512_set1_ps(a0_0_i_k_sc);

    float *c0_addr = C + (i + 0x0) * n;

    for (j = 0; j + 16 < n; j += 16) {
      __m512 b_k_0_f_j = _mm512_load_ps(B + j);

      __m512 c0_0_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a0_0_i_k);

      _mm512_store_ps(c0_addr, c0_0_i_0_f_j);

      c0_addr += 16;
    }

    if (j < n && (j + 16) >= n) {
      __mmask16 mask = (1u << (n - j)) - 1;

      __m512 b_k_0_f_j = _mm512_mask_load_ps(zero, mask, B + j);

      __m512 c0_0_i_0_f_j = _mm512_mul_ps(b_k_0_f_j, a0_0_i_k);

      _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
    }
  }
}

void sgemm12_opt(char* pTransA, char* pTransB,
      const int* pM, const int* pN, const int* pK,
      const float *pAlpha, const float *pa, const int*plda,
      const float *pb, const int *pldb, const float *pBeta,
      float *pc, const int*pldc)
{
  const uint64_t m = *pN;
  const uint64_t n = *pM;
  const uint64_t p = *pK;

  const float *A = pb;
  const float *B = pa;
  float *C = pc;

  const float alpha = *pAlpha;
  const float beta  = *pBeta;

  assert(*pTransA == 'n' && *pTransB == 'n');
  assert(m <= 48 && n == 500 && p == 1);
  assert(alpha == 1.0f && beta == 0.0f);
  assert((*pM == *plda) && (*pK == *pldb) && (*pM == *pldc));

  sgemm_opt(m, n, p, alpha, A, B, beta, C);
}
