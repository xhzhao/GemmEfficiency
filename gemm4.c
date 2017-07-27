#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <mkl.h>
#include <immintrin.h>

typedef unsigned long long int uint64_t;

static inline void
sgemm_opt(const uint64_t m, const uint64_t n, const uint64_t p,
          const float pAlpha, const float * A, const float * B,
          const float pBeta, float * C)
{

#define LI (32)
#define LJ (16)
#define LK (2000)

#define TNUM (64)
#pragma omp parallel num_threads(TNUM)
  {
    /*m:64 n:500 p:2000*/
    int tid = omp_get_thread_num();

    uint64_t block_id_i = 0;
    uint64_t block_id_j = 0;

    uint64_t thread_size_i = 0;
    uint64_t thread_size_j = 0;

    //assert(m <= 64 && (n == 500 || n == 1000) && (p == 2000 || p == 500 ));
    if (n == 1000) {
       block_id_i = 0;
       block_id_j = tid;

       thread_size_i = m;
       thread_size_j = 32;
    } else {
      /* n == 500 */
      if (m >= 28) {
        if (m > 32) {
          block_id_i = tid >> 5;
          block_id_j = tid & 0x1f;

          thread_size_i = 32;
          thread_size_j = 16;
        } else {
          block_id_i = tid >> 5;
          block_id_j = tid & 0x1f;

          thread_size_i = 16;
          thread_size_j = 16;
        }
      } else {
        if (tid >= 32) {
          thread_size_j = 0;
        } else {
          block_id_i = 0;
          block_id_j = tid & 0x1f;

          thread_size_i = m;
          thread_size_j = 16;
        }
      }
    }

    for (uint64_t tid_j = 0; tid_j < thread_size_j; tid_j += LJ) {
      uint64_t jj = thread_size_j * block_id_j + tid_j;
      if (jj >= n) {
        break;
      }

      for (uint64_t tid_i = 0; tid_i < thread_size_i; tid_i += LI) {
        uint64_t ii = thread_size_i * block_id_i + tid_i;

        if (ii >= m) {
          break;
        }

        for (uint64_t kk = 0; kk < p; kk += LK) {
          __mmask16 mask = 0xffff;
          for (uint64_t j = jj; j < jj + LJ; j += 16) {
            uint64_t i = ii;
            if ((j + 16) >= n) {
              mask = (1u << (n - j)) - 1;
            }

            for (i = ii; i < ii + LI; i += 16) {
              if (i + 16 > m) {
                break;
              }
              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();
              __m512 c4_4_i_0_f_j = _mm512_setzero_ps();
              __m512 c5_5_i_0_f_j = _mm512_setzero_ps();
              __m512 c6_6_i_0_f_j = _mm512_setzero_ps();
              __m512 c7_7_i_0_f_j = _mm512_setzero_ps();
              __m512 c8_8_i_0_f_j = _mm512_setzero_ps();
              __m512 c9_9_i_0_f_j = _mm512_setzero_ps();
              __m512 ca_a_i_0_f_j = _mm512_setzero_ps();
              __m512 cb_b_i_0_f_j = _mm512_setzero_ps();
              __m512 cc_c_i_0_f_j = _mm512_setzero_ps();
              __m512 cd_d_i_0_f_j = _mm512_setzero_ps();
              __m512 ce_e_i_0_f_j = _mm512_setzero_ps();
              __m512 cf_f_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);
                __m128 *aa_a_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xa);
                __m128 *ab_b_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xb);
                __m128 *ac_c_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xc);
                __m128 *ad_d_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xd);
                __m128 *ae_e_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xe);
                __m128 *af_f_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xf);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, aa_a_i_0_f_k + 0x0);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ab_b_i_0_f_k + 0x0);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ac_c_i_0_f_k + 0x0);
                cd_d_i_0_f_j = _mm512_4fmadd_ps(cd_d_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ad_d_i_0_f_k + 0x0);
                ce_e_i_0_f_j = _mm512_4fmadd_ps(ce_e_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ae_e_i_0_f_k + 0x0);
                cf_f_i_0_f_j = _mm512_4fmadd_ps(cf_f_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, af_f_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a4_4_i_0_f_k + 0x1);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a5_5_i_0_f_k + 0x1);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a6_6_i_0_f_k + 0x1);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a7_7_i_0_f_k + 0x1);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a8_8_i_0_f_k + 0x1);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a9_9_i_0_f_k + 0x1);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, aa_a_i_0_f_k + 0x1);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, ab_b_i_0_f_k + 0x1);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, ac_c_i_0_f_k + 0x1);
                cd_d_i_0_f_j = _mm512_4fmadd_ps(cd_d_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, ad_d_i_0_f_k + 0x1);
                ce_e_i_0_f_j = _mm512_4fmadd_ps(ce_e_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, ae_e_i_0_f_k + 0x1);
                cf_f_i_0_f_j = _mm512_4fmadd_ps(cf_f_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, af_f_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a4_4_i_0_f_k + 0x2);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a5_5_i_0_f_k + 0x2);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a6_6_i_0_f_k + 0x2);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a7_7_i_0_f_k + 0x2);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a8_8_i_0_f_k + 0x2);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a9_9_i_0_f_k + 0x2);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, aa_a_i_0_f_k + 0x2);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, ab_b_i_0_f_k + 0x2);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, ac_c_i_0_f_k + 0x2);
                cd_d_i_0_f_j = _mm512_4fmadd_ps(cd_d_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, ad_d_i_0_f_k + 0x2);
                ce_e_i_0_f_j = _mm512_4fmadd_ps(ce_e_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, ae_e_i_0_f_k + 0x2);
                cf_f_i_0_f_j = _mm512_4fmadd_ps(cf_f_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, af_f_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a4_4_i_0_f_k + 0x3);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a5_5_i_0_f_k + 0x3);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a6_6_i_0_f_k + 0x3);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a7_7_i_0_f_k + 0x3);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a8_8_i_0_f_k + 0x3);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a9_9_i_0_f_k + 0x3);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, aa_a_i_0_f_k + 0x3);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, ab_b_i_0_f_k + 0x3);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, ac_c_i_0_f_k + 0x3);
                cd_d_i_0_f_j = _mm512_4fmadd_ps(cd_d_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, ad_d_i_0_f_k + 0x3);
                ce_e_i_0_f_j = _mm512_4fmadd_ps(ce_e_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, ae_e_i_0_f_k + 0x3);
                cf_f_i_0_f_j = _mm512_4fmadd_ps(cf_f_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, af_f_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);
                __m128 *aa_a_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xa);
                __m128 *ab_b_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xb);
                __m128 *ac_c_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xc);
                __m128 *ad_d_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xd);
                __m128 *ae_e_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xe);
                __m128 *af_f_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xf);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, aa_a_i_0_f_k + 0x0);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ab_b_i_0_f_k + 0x0);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ac_c_i_0_f_k + 0x0);
                cd_d_i_0_f_j = _mm512_4fmadd_ps(cd_d_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ad_d_i_0_f_k + 0x0);
                ce_e_i_0_f_j = _mm512_4fmadd_ps(ce_e_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ae_e_i_0_f_k + 0x0);
                cf_f_i_0_f_j = _mm512_4fmadd_ps(cf_f_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, af_f_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x4) * n + j, mask, c4_4_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x5) * n + j, mask, c5_5_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x6) * n + j, mask, c6_6_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x7) * n + j, mask, c7_7_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x8) * n + j, mask, c8_8_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x9) * n + j, mask, c9_9_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xa) * n + j, mask, ca_a_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xb) * n + j, mask, cb_b_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xc) * n + j, mask, cc_c_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xd) * n + j, mask, cd_d_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xe) * n + j, mask, ce_e_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xf) * n + j, mask, cf_f_i_0_f_j);
            }

            if((i + 13) <= (ii + LI) && (i + 13) <= m) {
              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();
              __m512 c4_4_i_0_f_j = _mm512_setzero_ps();
              __m512 c5_5_i_0_f_j = _mm512_setzero_ps();
              __m512 c6_6_i_0_f_j = _mm512_setzero_ps();
              __m512 c7_7_i_0_f_j = _mm512_setzero_ps();
              __m512 c8_8_i_0_f_j = _mm512_setzero_ps();
              __m512 c9_9_i_0_f_j = _mm512_setzero_ps();
              __m512 ca_a_i_0_f_j = _mm512_setzero_ps();
              __m512 cb_b_i_0_f_j = _mm512_setzero_ps();
              __m512 cc_c_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);
                __m128 *aa_a_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xa);
                __m128 *ab_b_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xb);
                __m128 *ac_c_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xc);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, aa_a_i_0_f_k + 0x0);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ab_b_i_0_f_k + 0x0);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ac_c_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a4_4_i_0_f_k + 0x1);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a5_5_i_0_f_k + 0x1);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a6_6_i_0_f_k + 0x1);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a7_7_i_0_f_k + 0x1);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a8_8_i_0_f_k + 0x1);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a9_9_i_0_f_k + 0x1);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, aa_a_i_0_f_k + 0x1);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, ab_b_i_0_f_k + 0x1);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, ac_c_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a4_4_i_0_f_k + 0x2);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a5_5_i_0_f_k + 0x2);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a6_6_i_0_f_k + 0x2);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a7_7_i_0_f_k + 0x2);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a8_8_i_0_f_k + 0x2);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a9_9_i_0_f_k + 0x2);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, aa_a_i_0_f_k + 0x2);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, ab_b_i_0_f_k + 0x2);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, ac_c_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a4_4_i_0_f_k + 0x3);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a5_5_i_0_f_k + 0x3);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a6_6_i_0_f_k + 0x3);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a7_7_i_0_f_k + 0x3);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a8_8_i_0_f_k + 0x3);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a9_9_i_0_f_k + 0x3);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, aa_a_i_0_f_k + 0x3);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, ab_b_i_0_f_k + 0x3);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, ac_c_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);
                __m128 *aa_a_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xa);
                __m128 *ab_b_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xb);
                __m128 *ac_c_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xc);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, aa_a_i_0_f_k + 0x0);
                cb_b_i_0_f_j = _mm512_4fmadd_ps(cb_b_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ab_b_i_0_f_k + 0x0);
                cc_c_i_0_f_j = _mm512_4fmadd_ps(cc_c_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, ac_c_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x4) * n + j, mask, c4_4_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x5) * n + j, mask, c5_5_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x6) * n + j, mask, c6_6_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x7) * n + j, mask, c7_7_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x8) * n + j, mask, c8_8_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x9) * n + j, mask, c9_9_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xa) * n + j, mask, ca_a_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xb) * n + j, mask, cb_b_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xc) * n + j, mask, cc_c_i_0_f_j);
              i += 13;
            } else if((i + 11) <= (ii + LI) && (i + 11) <= m) {

              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();
              __m512 c4_4_i_0_f_j = _mm512_setzero_ps();
              __m512 c5_5_i_0_f_j = _mm512_setzero_ps();
              __m512 c6_6_i_0_f_j = _mm512_setzero_ps();
              __m512 c7_7_i_0_f_j = _mm512_setzero_ps();
              __m512 c8_8_i_0_f_j = _mm512_setzero_ps();
              __m512 c9_9_i_0_f_j = _mm512_setzero_ps();
              __m512 ca_a_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);
                __m128 *aa_a_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xa);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, aa_a_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a4_4_i_0_f_k + 0x1);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a5_5_i_0_f_k + 0x1);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a6_6_i_0_f_k + 0x1);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a7_7_i_0_f_k + 0x1);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a8_8_i_0_f_k + 0x1);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a9_9_i_0_f_k + 0x1);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, aa_a_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a4_4_i_0_f_k + 0x2);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a5_5_i_0_f_k + 0x2);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a6_6_i_0_f_k + 0x2);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a7_7_i_0_f_k + 0x2);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a8_8_i_0_f_k + 0x2);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a9_9_i_0_f_k + 0x2);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, aa_a_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a4_4_i_0_f_k + 0x3);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a5_5_i_0_f_k + 0x3);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a6_6_i_0_f_k + 0x3);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a7_7_i_0_f_k + 0x3);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a8_8_i_0_f_k + 0x3);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a9_9_i_0_f_k + 0x3);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, aa_a_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);
                __m128 *aa_a_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0xa);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);
                ca_a_i_0_f_j = _mm512_4fmadd_ps(ca_a_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, aa_a_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x4) * n + j, mask, c4_4_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x5) * n + j, mask, c5_5_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x6) * n + j, mask, c6_6_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x7) * n + j, mask, c7_7_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x8) * n + j, mask, c8_8_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x9) * n + j, mask, c9_9_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0xa) * n + j, mask, ca_a_i_0_f_j);
              i += 11;
            } else if((i + 10) <= (ii + LI) && (i + 10) <= m) {

              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();
              __m512 c4_4_i_0_f_j = _mm512_setzero_ps();
              __m512 c5_5_i_0_f_j = _mm512_setzero_ps();
              __m512 c6_6_i_0_f_j = _mm512_setzero_ps();
              __m512 c7_7_i_0_f_j = _mm512_setzero_ps();
              __m512 c8_8_i_0_f_j = _mm512_setzero_ps();
              __m512 c9_9_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a4_4_i_0_f_k + 0x1);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a5_5_i_0_f_k + 0x1);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a6_6_i_0_f_k + 0x1);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a7_7_i_0_f_k + 0x1);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a8_8_i_0_f_k + 0x1);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a9_9_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a4_4_i_0_f_k + 0x2);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a5_5_i_0_f_k + 0x2);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a6_6_i_0_f_k + 0x2);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a7_7_i_0_f_k + 0x2);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a8_8_i_0_f_k + 0x2);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a9_9_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a4_4_i_0_f_k + 0x3);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a5_5_i_0_f_k + 0x3);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a6_6_i_0_f_k + 0x3);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a7_7_i_0_f_k + 0x3);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a8_8_i_0_f_k + 0x3);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a9_9_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);
                __m128 *a9_9_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x9);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);
                c9_9_i_0_f_j = _mm512_4fmadd_ps(c9_9_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a9_9_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x4) * n + j, mask, c4_4_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x5) * n + j, mask, c5_5_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x6) * n + j, mask, c6_6_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x7) * n + j, mask, c7_7_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x8) * n + j, mask, c8_8_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x9) * n + j, mask, c9_9_i_0_f_j);
              i += 10;
            } else if((i + 9) <= (ii + LI) && (i + 9) <= m) {

              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();
              __m512 c4_4_i_0_f_j = _mm512_setzero_ps();
              __m512 c5_5_i_0_f_j = _mm512_setzero_ps();
              __m512 c6_6_i_0_f_j = _mm512_setzero_ps();
              __m512 c7_7_i_0_f_j = _mm512_setzero_ps();
              __m512 c8_8_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a4_4_i_0_f_k + 0x1);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a5_5_i_0_f_k + 0x1);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a6_6_i_0_f_k + 0x1);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a7_7_i_0_f_k + 0x1);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a8_8_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a4_4_i_0_f_k + 0x2);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a5_5_i_0_f_k + 0x2);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a6_6_i_0_f_k + 0x2);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a7_7_i_0_f_k + 0x2);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a8_8_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a4_4_i_0_f_k + 0x3);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a5_5_i_0_f_k + 0x3);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a6_6_i_0_f_k + 0x3);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a7_7_i_0_f_k + 0x3);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a8_8_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);
                __m128 *a8_8_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x8);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);
                c8_8_i_0_f_j = _mm512_4fmadd_ps(c8_8_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a8_8_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x4) * n + j, mask, c4_4_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x5) * n + j, mask, c5_5_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x6) * n + j, mask, c6_6_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x7) * n + j, mask, c7_7_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x8) * n + j, mask, c8_8_i_0_f_j);
              i += 9;
            } else if((i + 8) <= (ii + LI) && (i + 8) <= m) {
              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();
              __m512 c4_4_i_0_f_j = _mm512_setzero_ps();
              __m512 c5_5_i_0_f_j = _mm512_setzero_ps();
              __m512 c6_6_i_0_f_j = _mm512_setzero_ps();
              __m512 c7_7_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a4_4_i_0_f_k + 0x1);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a5_5_i_0_f_k + 0x1);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a6_6_i_0_f_k + 0x1);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a7_7_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a4_4_i_0_f_k + 0x2);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a5_5_i_0_f_k + 0x2);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a6_6_i_0_f_k + 0x2);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a7_7_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a4_4_i_0_f_k + 0x3);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a5_5_i_0_f_k + 0x3);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a6_6_i_0_f_k + 0x3);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a7_7_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);
                __m128 *a4_4_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x4);
                __m128 *a5_5_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x5);
                __m128 *a6_6_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x6);
                __m128 *a7_7_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x7);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);
                c4_4_i_0_f_j = _mm512_4fmadd_ps(c4_4_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a4_4_i_0_f_k + 0x0);
                c5_5_i_0_f_j = _mm512_4fmadd_ps(c5_5_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a5_5_i_0_f_k + 0x0);
                c6_6_i_0_f_j = _mm512_4fmadd_ps(c6_6_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a6_6_i_0_f_k + 0x0);
                c7_7_i_0_f_j = _mm512_4fmadd_ps(c7_7_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a7_7_i_0_f_k + 0x0);

                k += 4;
              }
              
              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x4) * n + j, mask, c4_4_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x5) * n + j, mask, c5_5_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x6) * n + j, mask, c6_6_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x7) * n + j, mask, c7_7_i_0_f_j);
              i += 8;
            } else if((i + 4) <= (ii + LI) && (i + 4) <= m) {
              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();
              __m512 c1_1_i_0_f_j = _mm512_setzero_ps();
              __m512 c2_2_i_0_f_j = _mm512_setzero_ps();
              __m512 c3_3_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a1_1_i_0_f_k + 0x1);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a2_2_i_0_f_k + 0x1);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a3_3_i_0_f_k + 0x1);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a1_1_i_0_f_k + 0x2);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a2_2_i_0_f_k + 0x2);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a3_3_i_0_f_k + 0x2);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a1_1_i_0_f_k + 0x3);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a2_2_i_0_f_k + 0x3);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a3_3_i_0_f_k + 0x3);
              }
              
              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);
                __m128 *a1_1_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x1);
                __m128 *a2_2_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x2);
                __m128 *a3_3_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x3);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c1_1_i_0_f_j = _mm512_4fmadd_ps(c1_1_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a1_1_i_0_f_k + 0x0);
                c2_2_i_0_f_j = _mm512_4fmadd_ps(c2_2_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a2_2_i_0_f_k + 0x0);
                c3_3_i_0_f_j = _mm512_4fmadd_ps(c3_3_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a3_3_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x1) * n + j, mask, c1_1_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x2) * n + j, mask, c2_2_i_0_f_j);
              _mm512_mask_store_ps(C + (i + 0x3) * n + j, mask, c3_3_i_0_f_j);
              i += 4;
            }

            for (; i < ii + LI && i < m; ++i) {
              uint64_t k = kk;
              __m512 c0_0_i_0_f_j = _mm512_setzero_ps();

              for (k = kk; k < kk + LK; k += 16) {
                if (k + 16 > p) {
                  break;
                }
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0xf);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);
                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b4_4_k_0_f_j, b5_5_k_0_f_j, b6_6_k_0_f_j, b7_7_k_0_f_j, a0_0_i_0_f_k + 0x1);
                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b8_8_k_0_f_j, b9_9_k_0_f_j, ba_a_k_0_f_j, bb_b_k_0_f_j, a0_0_i_0_f_k + 0x2);
                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, bc_c_k_0_f_j, bd_d_k_0_f_j, be_e_k_0_f_j, bf_f_k_0_f_j, a0_0_i_0_f_k + 0x3);
              }

              if((k + 4) <= (kk + LK) && (k + 4) <= p) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_maskz_load_ps(mask, b0_f_k_0_f_j_base + n * 0x3);

                __m128 *a0_0_i_0_f_k = (__m128 *)(a0_f_i_0_f_k_base + p * 0x0);

                c0_0_i_0_f_j = _mm512_4fmadd_ps(c0_0_i_0_f_j, b0_0_k_0_f_j, b1_1_k_0_f_j, b2_2_k_0_f_j, b3_3_k_0_f_j, a0_0_i_0_f_k + 0x0);

                k += 4;
              }

              _mm512_mask_store_ps(C + (i + 0x0) * n + j, mask, c0_0_i_0_f_j);
            }
          }
        }
      }
    }
  }

#undef LI
#undef LJ
#undef LK

#undef TNUM
}

void sgemm4_opt(char* pTransA, char* pTransB,
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
  assert((m == 64 || m == 58 || m == 53 || m == 49 ||
          m == 43 || m == 40 || m == 29 || m == 9) &&
         (n == 500 || n == 1000) &&
         (p == 2000 || p == 500 ));

  assert(alpha == 1.0f && beta == 0.0f);
  assert((*pM == *plda) && (*pK == *pldb) && (*pM == *pldc));

  sgemm_opt(m, n, p, alpha, A, B, beta, C);
}
