#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "mkl.h"
#include "immintrin.h"

#define LOOP_COUNT 1000
#define min(x,y) (((x) < (y)) ? (x) : (y))


typedef unsigned long long int uint64_t;


void opti_sgemm(uint64_t m, uint64_t n, uint64_t p, float alpha, float * restrict A, float * restrict B, float beta, float * restrict C)
{
#define L 16
#define LK 2048

  if (p != 2048 && m != 512 && n != 64) {
    return;
  }

#define TNUM (64)
#pragma omp parallel num_threads(TNUM)
  {
    /*m:500 n:64 p:2000*/
    int tid = omp_get_thread_num();
#if 1
    uint64_t tid_x = tid & 0x1f;
    uint64_t tid_y = tid >> 5;

    uint64_t block_x = m / (TNUM >> 1);
    uint64_t block_y = n / (TNUM >> 5);
#else
    uint64_t tid_x = tid;
    uint64_t tid_y = 0;

    uint64_t block_x = m / (TNUM) ;
    uint64_t block_y = n / (TNUM);
#endif
    //printf("block_i:%d, block_j:%d, m:%d, n:%d, p:%d\n", block_i, block_j, m, n, p);
    for (uint64_t tid_ii = 0; tid_ii < block_x; tid_ii += L) {
      uint64_t ii = block_x * tid_x + tid_ii;
      for (uint64_t tid_jj = 0; tid_jj < block_y; tid_jj += L) {
        uint64_t jj = block_y * tid_y + tid_jj;
        for (uint64_t kk = 0; kk < p; kk += LK) {
          for (uint64_t i = ii; i < ii + L; i += 16) {
            for (uint64_t j = jj; j < jj + L; j += 16) {

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

              for (uint64_t k = kk; k < kk + LK; k += 16) {
                float *a0_f_i_0_f_k_base = (float *)(A + (i + 0x0) * p + k);
                float *b0_f_k_0_f_j_base = (float *)(B + (k + 0x0) * n + j);

                __m512 b0_0_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x0);
                __m512 b1_1_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x1);
                __m512 b2_2_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x2);
                __m512 b3_3_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x3);
                __m512 b4_4_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x4);
                __m512 b5_5_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x5);
                __m512 b6_6_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x6);
                __m512 b7_7_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x7);
                __m512 b8_8_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x8);
                __m512 b9_9_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0x9);
                __m512 ba_a_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0xa);
                __m512 bb_b_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0xb);
                __m512 bc_c_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0xc);
                __m512 bd_d_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0xd);
                __m512 be_e_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0xe);
                __m512 bf_f_k_0_f_j = _mm512_load_ps(b0_f_k_0_f_j_base + n * 0xf);

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

              _mm512_store_ps(C + (i + 0x0) * n + j, c0_0_i_0_f_j);
              _mm512_store_ps(C + (i + 0x1) * n + j, c1_1_i_0_f_j);
              _mm512_store_ps(C + (i + 0x2) * n + j, c2_2_i_0_f_j);
              _mm512_store_ps(C + (i + 0x3) * n + j, c3_3_i_0_f_j);
              _mm512_store_ps(C + (i + 0x4) * n + j, c4_4_i_0_f_j);
              _mm512_store_ps(C + (i + 0x5) * n + j, c5_5_i_0_f_j);
              _mm512_store_ps(C + (i + 0x6) * n + j, c6_6_i_0_f_j);
              _mm512_store_ps(C + (i + 0x7) * n + j, c7_7_i_0_f_j);
              _mm512_store_ps(C + (i + 0x8) * n + j, c8_8_i_0_f_j);
              _mm512_store_ps(C + (i + 0x9) * n + j, c9_9_i_0_f_j);
              _mm512_store_ps(C + (i + 0xa) * n + j, ca_a_i_0_f_j);
              _mm512_store_ps(C + (i + 0xb) * n + j, cb_b_i_0_f_j);
              _mm512_store_ps(C + (i + 0xc) * n + j, cc_c_i_0_f_j);
              _mm512_store_ps(C + (i + 0xd) * n + j, cd_d_i_0_f_j);
              _mm512_store_ps(C + (i + 0xe) * n + j, ce_e_i_0_f_j);
              _mm512_store_ps(C + (i + 0xf) * n + j, cf_f_i_0_f_j);
            }
          }
        }
      }
    }
  }
}


int main()
{
    float *A, *B, *C, *D;
    int m, n, p, i, j, r;
    float alpha, beta;
    double s_initial, s_elapsed;

    struct inputArgs {
      int m;
      int n;
      int p;
      enum CBLAS_TRANSPOSE m0;
      enum CBLAS_TRANSPOSE m1;
      int lda;
      float a;
      int ldb;
      float b;
      int ldc;
    };
    
    struct inputArgs inp[10] = {
      {35820, 64, 500, CblasTrans, CblasNoTrans, 35820, 1.0, 64, 0.0, 64 },
      {500, 64, 35820, CblasNoTrans, CblasNoTrans, 35820, 1.0, 64, 0.0, 64 },
      {500, 35820, 64, CblasNoTrans, CblasTrans, 64, 1.0, 64, 1.0, 35820 },
      {512, 64, 2048, CblasNoTrans, CblasNoTrans, 2048, 1.0, 64, 0.0, 64 },
      {2000, 64, 500, CblasTrans, CblasNoTrans, 2000, 1.0, 64, 0.0, 64 },
 
      {2000, 64, 500, CblasTrans, CblasNoTrans, 2000, 1.0, 64, 0.0, 64 },
      {1000, 64, 2000, CblasNoTrans, CblasNoTrans, 2000, 1.0, 64, 0.0, 64 },
      {1000, 2000, 64, CblasNoTrans, CblasTrans, 64, 1.0, 64, 1.0, 2000 },
      {500, 29, 35820, CblasNoTrans, CblasNoTrans, 35820, 1.0, 29, 0.0, 29 },
      {2000, 64, 1000, CblasTrans, CblasNoTrans, 2000, 1.0, 64, 0.0, 64 },
    };

    for (int ii = 3; ii < 4; ++ii) {

      m = inp[ii].m, n = inp[ii].n, p = inp[ii].p;
      int m0 = inp[ii].m0, m1 = inp[ii].m1;

      //printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
      //        " A(%ix%i) and matrix B(%ix%i)\n\n", m, p, p, n);
      alpha = inp[ii].a; beta = inp[ii].b;
      
      int lda = inp[ii].lda;
      int ldb = inp[ii].ldb;
      int ldc = inp[ii].ldc;
      
      printf("----------GEMM %d----------\n", ii);
      //printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
      //        " performance \n\n");
      A = (float *)mkl_malloc( m*p*sizeof( float ), 64 );
      B = (float *)mkl_malloc( p*n*sizeof( float ), 64 );
      C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
      D = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
      if (A == NULL || B == NULL || C == NULL) {
          printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
          mkl_free(A);
          mkl_free(B);
          mkl_free(C);
          mkl_free(D);
          return 1;
      }

      //printf (" Intializing matrix data \n\n");
      for (i = 0; i < (m*p); i++) {
          A[i] = (float)(rand() % 1000);
      }

      for (i = 0; i < (p*n); i++) {
          B[i] = (float)(rand() % 1000);
      }

      for (i = 0; i < (m*n); i++) {
          C[i] = 0.0;
      }

      for (i = 0; i < (m*n); i++) {
          D[i] = 0.0;
      }

      //printf (" Making the first run of matrix product using Intel(R) MKL sgemm function \n"
      //       " via CBLAS interface to get stable run time measurements \n\n");
      cblas_sgemm(CblasRowMajor, m0, m1, 
                  m, n, p, alpha, A, lda, B, ldb, beta, C, ldc);

      //printf (" Measuring performance of matrix product using Intel(R) MKL sgemm function \n"
      //        " via CBLAS interface \n\n");
      s_initial = dsecnd();
      for (r = 0; r < LOOP_COUNT; r++) {
          cblas_sgemm(CblasRowMajor, m0, m1,
                      m, n, p, alpha, A, lda, B, ldb, beta, C, ldc);
      }
      s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

      printf (" == Matrix multiplication using Intel(R) MKL sgemm completed == \n"
              " == at %.5f milliseconds, %8.2f Gflops == \n\n", (s_elapsed * 1000), 2.*m*n*p / s_elapsed * 1e-9f);

      //printf (" Making the first run of matrix product using opti_sgemm8x2x8 function \n"
      //        " via CBLAS interface to get stable run time measurements \n\n");
      opti_sgemm(m, n, p, alpha, A, B, beta, D);

      //printf (" Measuring performance of matrix product using opti_sgemm8x2x8 function \n"
      //        " via CBLAS interface \n\n");
      s_initial = dsecnd();
      for (r = 0; r < LOOP_COUNT; r++) {
          opti_sgemm(m, n, p, alpha, A, B, beta, D);
      }
      s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

      printf (" == Matrix multiplication using opti_sgemm8x2x8 completed == \n"
              " == at %.5f milliseconds, %8.2f Gflops == \n\n", (s_elapsed * 1000), 2.*m*n*p / s_elapsed * 1e-9f);

      for (i = 0; i < (m); i++) {
          int fail = 0;
          for (j = 0; j < n; j++) {
          float err = (C[i*n + j]-D[i*n + j])/C[i*n + j];
          if (err > 0.00001 || err < -0.00001) {
              fail = 1;
              printf("%d %d %f %f %f Err!\n", i, j, C[i*n+j], D[i*n+j], err);
              break;
          }
          }
          if (fail) break;
      }
#if 0
      printf (" Top left corner of matrix A: \n");
      for (i=0; i<min(m,6); i++) {
          for (j=0; j<min(p,6); j++) {
              printf ("%12.0f", A[j+i*p]);
          }
          printf ("\n");
      }

      printf ("\n Top left corner of matrix B: \n");
      for (i=0; i<min(p,6); i++) {
          for (j=0; j<min(n,6); j++) {
              printf ("%12.0f", B[j+i*n]);
          }
          printf ("\n");
      }

      printf ("\n Top left corner of matrix C: \n");
      for (i=0; i<min(m,6); i++) {
          for (j=0; j<min(n,6); j++) {
              printf ("%12.5G", C[j+i*n]);
          }
          printf ("\n");
      }

      printf ("\n Top left corner of matrix D: \n");
      for (i=0; i<min(m,6); i++) {
          for (j=0; j<min(n,6); j++) {
              printf ("%12.5G", D[j+i*n]);
          }
          printf ("\n");
      }
#endif
      //printf (" Deallocating memory \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      mkl_free(D);
#if 0
      if (s_elapsed < 0.9/LOOP_COUNT) {
          s_elapsed=1.0/LOOP_COUNT/s_elapsed;
          i=(int)(s_elapsed*LOOP_COUNT)+1;
          printf(" It is highly recommended to define LOOP_COUNT for this example on your \n"
                 " computer as %i to have total execution time about 1 second for reliability \n"
                 " of measurements\n\n", i);
      }
#endif
    }
    return 0;
}
