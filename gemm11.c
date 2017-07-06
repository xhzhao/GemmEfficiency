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

  for (uint64_t j = 0; j < n; j += 16) {
    __mmask16 mask = 0xffff;

    uint64_t k = 0;
    if ((j + 16) >= n) {
    mask = (1u << (n - j)) - 1;
    }

    __m512 zero = _mm512_setzero_ps();

    __m512 c0_0_i_0_f_j_0 = _mm512_setzero_ps();
    __m512 c0_0_i_0_f_j_1 = _mm512_setzero_ps();
    __m512 c0_0_i_0_f_j_2 = _mm512_setzero_ps();
    __m512 c0_0_i_0_f_j_3 = _mm512_setzero_ps();
    __m512 c0_0_i_0_f_j   = _mm512_setzero_ps();

    for (; k + 16 <= p; k += 16) {
      float *a0_f_i_0_f_k_base = (float *)(A + k);
      float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);


      __m512 b0_0_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x0);
      __m512 b1_1_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x1);
      __m512 b2_2_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x2);
      __m512 b3_3_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x3);
      __m512 b4_4_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x4);
      __m512 b5_5_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x5);
      __m512 b6_6_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x6);
      __m512 b7_7_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x7);
      __m512 b8_8_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x8);
      __m512 b9_9_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0x9);
      __m512 ba_a_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0xa);
      __m512 bb_b_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0xb);
      __m512 bc_c_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0xc);
      __m512 bd_d_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0xd);
      __m512 be_e_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0xe);
      __m512 bf_f_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_f_k_0_f_j_base + n * 0xf);

      __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);

      c0_0_i_0_f_j_0 = _mm512_4fmadd_ps(c0_0_i_0_f_j_0, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
      c0_0_i_0_f_j_1 = _mm512_4fmadd_ps(c0_0_i_0_f_j_1, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
      c0_0_i_0_f_j_2 = _mm512_4fmadd_ps(c0_0_i_0_f_j_2, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
      c0_0_i_0_f_j_3 = _mm512_4fmadd_ps(c0_0_i_0_f_j_3, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
    }

    if (k + 8 <= p) {
      float *a0_f_i_0_7_k_base = (float *)(A + k);
      float *b0_7_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);


      __m512 b0_0_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x0);
      __m512 b1_1_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x1);
      __m512 b2_2_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x2);
      __m512 b3_3_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x3);
      __m512 b4_4_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x4);
      __m512 b5_5_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x5);
      __m512 b6_6_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x6);
      __m512 b7_7_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_7_k_0_f_j_base + n * 0x7);

      __m128 *a0_0_i_0_7_k = (__m128 *)(a0_f_i_0_7_k_base + p * 0x0);

      c0_0_i_0_f_j_0 = _mm512_4fmadd_ps(c0_0_i_0_f_j_0, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_7_k + 0x0);
      c0_0_i_0_f_j_1 = _mm512_4fmadd_ps(c0_0_i_0_f_j_1, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_7_k + 0x1);
      
      k += 8;
    }

    if (k + 4 <= p) {
      float *a0_3_i_0_f_k_base = (float *)(A + k);
      float *b0_3_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

      __m512 b0_0_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_3_k_0_f_j_base + n * 0x0);
      __m512 b1_1_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_3_k_0_f_j_base + n * 0x1);
      __m512 b2_2_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_3_k_0_f_j_base + n * 0x2);
      __m512 b3_3_k_0_f_j = _mm512_mask_load_ps(zero, mask, b0_3_k_0_f_j_base + n * 0x3);

      __m128 *a0_0_i_0_3_k = (__m128 *)(a0_3_i_0_f_k_base);

      c0_0_i_0_f_j_0 = _mm512_4fmadd_ps(c0_0_i_0_f_j_0, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_3_k + 0x0);
      
      k += 4;
    }

    for (; k < p; ++k) {
      __m512 b_k_0_f_j = _mm512_mask_load_ps(zero, mask, B + (k + 0x0) * n + j);
      __m512 a_i_0_f_k = _mm512_set1_ps (*(A + k));

      c0_0_i_0_f_j_0 = _mm512_fmadd_ps(b_k_0_f_j, a_i_0_f_k, c0_0_i_0_f_j_0);
    }

    c0_0_i_0_f_j_0 = _mm512_add_ps(c0_0_i_0_f_j_0, c0_0_i_0_f_j_1);
    c0_0_i_0_f_j_2 = _mm512_add_ps(c0_0_i_0_f_j_2, c0_0_i_0_f_j_3);
    c0_0_i_0_f_j = _mm512_add_ps(c0_0_i_0_f_j_2, c0_0_i_0_f_j_0);
    _mm512_mask_store_ps(C + j, mask, c0_0_i_0_f_j);

  }
}

void sgemm11_opt(char* pTransA, char* pTransB,
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
  assert(m == 1 && n == 500 && p <= 48);
  assert(alpha == 1.0f && beta == 0.0f);
  assert((*pM == *plda) && (*pK == *pldb) && (*pM == *pldc));

  sgemm_opt(m, n, p, alpha, A, B, beta, C);
}
