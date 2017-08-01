#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <mkl.h>
#include <immintrin.h>

typedef unsigned long long int uint64_t;

#define matrix_transpose_16x16(                                           \
r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf)           \
{                                                                         \
    __m512 t0, t1, t2, t3, t4, t5, t6, t7;                                \
    __m512 t8, t9, ta, tb, tc, td, te, tf;                                \
                                                                          \
    t0 = _mm512_unpacklo_ps(r0,r1);                                       \
    t1 = _mm512_unpackhi_ps(r0,r1);                                       \
    t2 = _mm512_unpacklo_ps(r2,r3);                                       \
    t3 = _mm512_unpackhi_ps(r2,r3);                                       \
    t4 = _mm512_unpacklo_ps(r4,r5);                                       \
    t5 = _mm512_unpackhi_ps(r4,r5);                                       \
    t6 = _mm512_unpacklo_ps(r6,r7);                                       \
    t7 = _mm512_unpackhi_ps(r6,r7);                                       \
    t8 = _mm512_unpacklo_ps(r8,r9);                                       \
    t9 = _mm512_unpackhi_ps(r8,r9);                                       \
    ta = _mm512_unpacklo_ps(ra,rb);                                       \
    tb = _mm512_unpackhi_ps(ra,rb);                                       \
    tc = _mm512_unpacklo_ps(rc,rd);                                       \
    td = _mm512_unpackhi_ps(rc,rd);                                       \
    te = _mm512_unpacklo_ps(re,rf);                                       \
    tf = _mm512_unpackhi_ps(re,rf);                                       \
                                                                          \
    r0 = (__m512)_mm512_unpacklo_pd((__m512d)t0,(__m512d)t2);             \
    r1 = (__m512)_mm512_unpackhi_pd((__m512d)t0,(__m512d)t2);             \
    r2 = (__m512)_mm512_unpacklo_pd((__m512d)t1,(__m512d)t3);             \
    r3 = (__m512)_mm512_unpackhi_pd((__m512d)t1,(__m512d)t3);             \
    r4 = (__m512)_mm512_unpacklo_pd((__m512d)t4,(__m512d)t6);             \
    r5 = (__m512)_mm512_unpackhi_pd((__m512d)t4,(__m512d)t6);             \
    r6 = (__m512)_mm512_unpacklo_pd((__m512d)t5,(__m512d)t7);             \
    r7 = (__m512)_mm512_unpackhi_pd((__m512d)t5,(__m512d)t7);             \
    r8 = (__m512)_mm512_unpacklo_pd((__m512d)t8,(__m512d)ta);             \
    r9 = (__m512)_mm512_unpackhi_pd((__m512d)t8,(__m512d)ta);             \
    ra = (__m512)_mm512_unpacklo_pd((__m512d)t9,(__m512d)tb);             \
    rb = (__m512)_mm512_unpackhi_pd((__m512d)t9,(__m512d)tb);             \
    rc = (__m512)_mm512_unpacklo_pd((__m512d)tc,(__m512d)te);             \
    rd = (__m512)_mm512_unpackhi_pd((__m512d)tc,(__m512d)te);             \
    re = (__m512)_mm512_unpacklo_pd((__m512d)td,(__m512d)tf);             \
    rf = (__m512)_mm512_unpackhi_pd((__m512d)td,(__m512d)tf);             \
                                                                          \
    t0 = _mm512_shuffle_f32x4(r0, r4, 0x88);                              \
    t1 = _mm512_shuffle_f32x4(r1, r5, 0x88);                              \
    t2 = _mm512_shuffle_f32x4(r2, r6, 0x88);                              \
    t3 = _mm512_shuffle_f32x4(r3, r7, 0x88);                              \
    t4 = _mm512_shuffle_f32x4(r0, r4, 0xdd);                              \
    t5 = _mm512_shuffle_f32x4(r1, r5, 0xdd);                              \
    t6 = _mm512_shuffle_f32x4(r2, r6, 0xdd);                              \
    t7 = _mm512_shuffle_f32x4(r3, r7, 0xdd);                              \
    t8 = _mm512_shuffle_f32x4(r8, rc, 0x88);                              \
    t9 = _mm512_shuffle_f32x4(r9, rd, 0x88);                              \
    ta = _mm512_shuffle_f32x4(ra, re, 0x88);                              \
    tb = _mm512_shuffle_f32x4(rb, rf, 0x88);                              \
    tc = _mm512_shuffle_f32x4(r8, rc, 0xdd);                              \
    td = _mm512_shuffle_f32x4(r9, rd, 0xdd);                              \
    te = _mm512_shuffle_f32x4(ra, re, 0xdd);                              \
    tf = _mm512_shuffle_f32x4(rb, rf, 0xdd);                              \
                                                                          \
    r0 = _mm512_shuffle_f32x4(t0, t8, 0x88);                              \
    r1 = _mm512_shuffle_f32x4(t1, t9, 0x88);                              \
    r2 = _mm512_shuffle_f32x4(t2, ta, 0x88);                              \
    r3 = _mm512_shuffle_f32x4(t3, tb, 0x88);                              \
    r4 = _mm512_shuffle_f32x4(t4, tc, 0x88);                              \
    r5 = _mm512_shuffle_f32x4(t5, td, 0x88);                              \
    r6 = _mm512_shuffle_f32x4(t6, te, 0x88);                              \
    r7 = _mm512_shuffle_f32x4(t7, tf, 0x88);                              \
    r8 = _mm512_shuffle_f32x4(t0, t8, 0xdd);                              \
    r9 = _mm512_shuffle_f32x4(t1, t9, 0xdd);                              \
    ra = _mm512_shuffle_f32x4(t2, ta, 0xdd);                              \
    rb = _mm512_shuffle_f32x4(t3, tb, 0xdd);                              \
    rc = _mm512_shuffle_f32x4(t4, tc, 0xdd);                              \
    rd = _mm512_shuffle_f32x4(t5, td, 0xdd);                              \
    re = _mm512_shuffle_f32x4(t6, te, 0xdd);                              \
    rf = _mm512_shuffle_f32x4(t7, tf, 0xdd);                              \
}

#define matrix_transpose_8x16(r0, r1, r2, r3, r4, r5, r6, r7)             \
{                                                                         \
    __m512 t0, t1, t2, t3, t4, t5, t6, t7;                                \
                                                                          \
    t0 = _mm512_unpacklo_ps(r0,r1);                                       \
    t1 = _mm512_unpackhi_ps(r0,r1);                                       \
    t2 = _mm512_unpacklo_ps(r2,r3);                                       \
    t3 = _mm512_unpackhi_ps(r2,r3);                                       \
    t4 = _mm512_unpacklo_ps(r4,r5);                                       \
    t5 = _mm512_unpackhi_ps(r4,r5);                                       \
    t6 = _mm512_unpacklo_ps(r6,r7);                                       \
    t7 = _mm512_unpackhi_ps(r6,r7);                                       \
                                                                          \
    r0 = (__m512)_mm512_unpacklo_pd((__m512d)t0,(__m512d)t2);             \
    r1 = (__m512)_mm512_unpackhi_pd((__m512d)t0,(__m512d)t2);             \
    r2 = (__m512)_mm512_unpacklo_pd((__m512d)t1,(__m512d)t3);             \
    r3 = (__m512)_mm512_unpackhi_pd((__m512d)t1,(__m512d)t3);             \
    r4 = (__m512)_mm512_unpacklo_pd((__m512d)t4,(__m512d)t6);             \
    r5 = (__m512)_mm512_unpackhi_pd((__m512d)t4,(__m512d)t6);             \
    r6 = (__m512)_mm512_unpacklo_pd((__m512d)t5,(__m512d)t7);             \
    r7 = (__m512)_mm512_unpackhi_pd((__m512d)t5,(__m512d)t7);             \
                                                                          \
    t0 = _mm512_shuffle_f32x4(r0, r4, 0x88);                              \
    t1 = _mm512_shuffle_f32x4(r1, r5, 0x88);                              \
    t2 = _mm512_shuffle_f32x4(r2, r6, 0x88);                              \
    t3 = _mm512_shuffle_f32x4(r3, r7, 0x88);                              \
    t4 = _mm512_shuffle_f32x4(r0, r4, 0xdd);                              \
    t5 = _mm512_shuffle_f32x4(r1, r5, 0xdd);                              \
    t6 = _mm512_shuffle_f32x4(r2, r6, 0xdd);                              \
    t7 = _mm512_shuffle_f32x4(r3, r7, 0xdd);                              \
                                                                          \
    r0 = _mm512_shuffle_f32x4(t0, t4, 0x88);                              \
    r1 = _mm512_shuffle_f32x4(t1, t5, 0x88);                              \
    r2 = _mm512_shuffle_f32x4(t2, t6, 0x88);                              \
    r3 = _mm512_shuffle_f32x4(t3, t7, 0x88);                              \
    r4 = _mm512_shuffle_f32x4(t0, t4, 0xdd);                              \
    r5 = _mm512_shuffle_f32x4(t1, t5, 0xdd);                              \
    r6 = _mm512_shuffle_f32x4(t2, t6, 0xdd);                              \
    r7 = _mm512_shuffle_f32x4(t3, t7, 0xdd);                              \
}

#define matrix_transpose_4x16(r0, r1, r2, r3)                             \
{                                                                         \
    __m512 t0, t1, t2, t3;                                                \
                                                                          \
    t0 = _mm512_unpacklo_ps(r0,r1);                                       \
    t1 = _mm512_unpackhi_ps(r0,r1);                                       \
    t2 = _mm512_unpacklo_ps(r2,r3);                                       \
    t3 = _mm512_unpackhi_ps(r2,r3);                                       \
                                                                          \
    r0 = (__m512)_mm512_unpacklo_pd((__m512d)t0,(__m512d)t2);             \
    r1 = (__m512)_mm512_unpackhi_pd((__m512d)t0,(__m512d)t2);             \
    r2 = (__m512)_mm512_unpacklo_pd((__m512d)t1,(__m512d)t3);             \
    r3 = (__m512)_mm512_unpackhi_pd((__m512d)t1,(__m512d)t3);             \
                                                                          \
    t0 = _mm512_shuffle_f32x4(r0, r1, 0x88);                              \
    t1 = _mm512_shuffle_f32x4(r2, r3, 0x88);                              \
    t2 = _mm512_shuffle_f32x4(r0, r1, 0xdd);                              \
    t3 = _mm512_shuffle_f32x4(r2, r3, 0xdd);                              \
                                                                          \
    r0 = _mm512_shuffle_f32x4(t0, t1, 0x88);                              \
    r1 = _mm512_shuffle_f32x4(t2, t3, 0x88);                              \
    r2 = _mm512_shuffle_f32x4(t0, t1, 0xdd);                              \
    r3 = _mm512_shuffle_f32x4(t2, t3, 0xdd);                              \
                                                                          \
}

static inline void
copy_matrix_a(float * at, float * a,
              uint64_t m, uint64_t p)
{
  __mmask16 mask = 0xffff;
  __m512 zero = _mm512_setzero_ps();
  float *at_t = NULL;
  uint64_t i = 0;

  for (; i < m; i += 16) {
    uint64_t k = 0;
    if ((i + 16) >= m) {
      mask = (1u << (m - i)) - 1;
    }
    at_t = at;
    for (k = 0; k < p; k += 4) {
      if (k + 4 > p) { break; }
      __m512 a_0_f_i_0_k = _mm512_maskz_load_ps(mask, a + (k + 0x0) * m);
      __m512 a_0_f_i_1_k = _mm512_maskz_load_ps(mask, a + (k + 0x1) * m);
      __m512 a_0_f_i_2_k = _mm512_maskz_load_ps(mask, a + (k + 0x2) * m);
      __m512 a_0_f_i_3_k = _mm512_maskz_load_ps(mask, a + (k + 0x3) * m);

      matrix_transpose_4x16(a_0_f_i_0_k, a_0_f_i_1_k, a_0_f_i_2_k, a_0_f_i_3_k);

      if (mask == 0xffff) {
        _mm512_store_ps(at + 0x00, a_0_f_i_0_k);
        _mm512_store_ps(at + 0x10, a_0_f_i_1_k);
        _mm512_store_ps(at + 0x20, a_0_f_i_2_k);
        _mm512_store_ps(at + 0x30, a_0_f_i_3_k);
        at += 0x40;
      } else {
        if((mask & 0x000f) == 0x000f) { _mm512_store_ps(at + 0x00, a_0_f_i_0_k); at_t += 0x10; }
        if((mask & 0x00f0) == 0x00f0) { _mm512_store_ps(at + 0x10, a_0_f_i_1_k); at_t += 0x10; }
        if((mask & 0x0f00) == 0x0f00) { _mm512_store_ps(at + 0x20, a_0_f_i_2_k); at_t += 0x10; }
        if((mask & 0xf000) == 0xf000) { _mm512_store_ps(at + 0x30, a_0_f_i_3_k); at_t += 0x10; }
        at = at_t;
      }
    }

    if (k < p) {
      __m512 a_0_f_i_0_k = _mm512_setzero_ps();
      __m512 a_0_f_i_1_k = _mm512_setzero_ps();
      __m512 a_0_f_i_2_k = _mm512_setzero_ps();
      __m512 a_0_f_i_3_k = _mm512_setzero_ps();

      a_0_f_i_0_k = _mm512_maskz_load_ps(mask, a + (k + 0x0) * m);
      if ((k + 1) < p) { a_0_f_i_1_k = _mm512_maskz_load_ps(mask, a + (k + 0x1) * m); }
      if ((k + 2) < p) { a_0_f_i_2_k = _mm512_maskz_load_ps(mask, a + (k + 0x2) * m); }
      if ((k + 3) < p) { a_0_f_i_3_k = _mm512_maskz_load_ps(mask, a + (k + 0x3) * m); }

      matrix_transpose_4x16(a_0_f_i_0_k, a_0_f_i_1_k, a_0_f_i_2_k, a_0_f_i_3_k);

      if (mask == 0xffff) {
        _mm512_store_ps(at + 0x00, a_0_f_i_0_k);
        _mm512_store_ps(at + 0x10, a_0_f_i_1_k);
        _mm512_store_ps(at + 0x20, a_0_f_i_2_k);
        _mm512_store_ps(at + 0x30, a_0_f_i_3_k);
        at += 0x40;
      } else {
        if((mask & 0x000f) == 0x000f) { _mm512_store_ps(at + 0x00, a_0_f_i_0_k); at_t += 0x10; }
        if((mask & 0x00f0) == 0x00f0) { _mm512_store_ps(at + 0x10, a_0_f_i_1_k); at_t += 0x10; }
        if((mask & 0x0f00) == 0x0f00) { _mm512_store_ps(at + 0x20, a_0_f_i_2_k); at_t += 0x10; }
        if((mask & 0xf000) == 0xf000) { _mm512_store_ps(at + 0x30, a_0_f_i_3_k); at_t += 0x10; }
        at = at_t;
      }
    }

    a += 0x10;
  }
}

static inline void
sgemm_opt(uint64_t m, uint64_t n, uint64_t p,
  const float pAlpha, const float * A, const float * B,
  const float pBeta, float * C)
{

  uint64_t ap = (p + 3) & 0xfffffffc;
  float *AT = (float *)mkl_malloc(m*ap*sizeof(float), 2048*1024);
  copy_matrix_a(AT, A, m, p);

  uint64_t LII = 0;
  uint64_t LJJ = 0;

  uint64_t LI = 0;
  uint64_t LJ = 0;

#define TNUM (72)
#pragma omp parallel num_threads(TNUM)
  {
    uint64_t tid = omp_get_thread_num();
    uint64_t block_id_j = 0;
    uint64_t thread_size_j = 0;
    uint64_t block_id_i = 0;
    uint64_t thread_size_i = 0;

    if (n == 50004) {
      thread_size_j = 704;
      block_id_j = tid;
      thread_size_i = m;
      block_id_i = 0;

      LII = 512;
      LJJ = 704;
      LI = 64;
      LJ = 64;
    } else if (n == 35820) {
      thread_size_j = 512;
      block_id_j = tid;
      thread_size_i = m;
      block_id_i = 0;

      LII = 512;
      LJJ = 512;
      LI = 64;
      LJ = 64;
    }

    /* m=500; n=50004; p=64; */
    for (uint64_t tid_j = 0; tid_j < thread_size_j; tid_j += LJJ) {
      uint64_t jjj = thread_size_j * block_id_j + tid_j;
      if (jjj >= n) {
        break;
      }

      for (uint64_t tid_i = 0; tid_i < thread_size_i; tid_i += LII) {
        uint64_t iii = thread_size_i * block_id_i + tid_i;
        if (iii >= m) {
          break;
        }

        for (uint64_t jj = jjj; jj < jjj + LJJ; jj += LJ) {
          if (jj >= n) {
            break;
          }
          for (uint64_t ii = iii; ii < iii + LII; ii += LI) {
            __mmask16 mask = 0xffff;
            if (ii >= m) {
                break;
            }
            for (uint64_t j = jj; j < jj + LJ; j += 16) {
              uint64_t i = ii;
              if (j >= n) { break; }
              if ((j + 16) >= n) {
                mask = (1u << (n - j)) - 1;
              }

              for (; i < ii + LI; i += 16) {
                if ((i + 16) > m) { break; }
                uint64_t k = 0;
                __m128 *a_base = (__m128 *)(AT + i * ap);
                __m512 c0 = _mm512_setzero_ps();
                __m512 c1 = _mm512_setzero_ps();
                __m512 c2 = _mm512_setzero_ps();
                __m512 c3 = _mm512_setzero_ps();
                __m512 c4 = _mm512_setzero_ps();
                __m512 c5 = _mm512_setzero_ps();
                __m512 c6 = _mm512_setzero_ps();
                __m512 c7 = _mm512_setzero_ps();
                __m512 c8 = _mm512_setzero_ps();
                __m512 c9 = _mm512_setzero_ps();
                __m512 ca = _mm512_setzero_ps();
                __m512 cb = _mm512_setzero_ps();
                __m512 cc = _mm512_setzero_ps();
                __m512 cd = _mm512_setzero_ps();
                __m512 ce = _mm512_setzero_ps();
                __m512 cf = _mm512_setzero_ps();

                for (k = 0; k < p; k = k + 16) {
                  if (k + 16 > p) break;

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);
                  __m512 b40 = _mm512_maskz_load_ps(mask, b_base + n*0x4);
                  __m512 b50 = _mm512_maskz_load_ps(mask, b_base + n*0x5);
                  __m512 b60 = _mm512_maskz_load_ps(mask, b_base + n*0x6);
                  __m512 b70 = _mm512_maskz_load_ps(mask, b_base + n*0x7);
                  __m512 b80 = _mm512_maskz_load_ps(mask, b_base + n*0x8);
                  __m512 b90 = _mm512_maskz_load_ps(mask, b_base + n*0x9);
                  __m512 ba0 = _mm512_maskz_load_ps(mask, b_base + n*0xa);
                  __m512 bb0 = _mm512_maskz_load_ps(mask, b_base + n*0xb);
                  __m512 bc0 = _mm512_maskz_load_ps(mask, b_base + n*0xc);
                  __m512 bd0 = _mm512_maskz_load_ps(mask, b_base + n*0xd);
                  __m512 be0 = _mm512_maskz_load_ps(mask, b_base + n*0xe);
                  __m512 bf0 = _mm512_maskz_load_ps(mask, b_base + n*0xf);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);
                  c8 = _mm512_4fmadd_ps(c8, b00, b10, b20, b30, a_base+0x08);
                  c9 = _mm512_4fmadd_ps(c9, b00, b10, b20, b30, a_base+0x09);
                  ca = _mm512_4fmadd_ps(ca, b00, b10, b20, b30, a_base+0x0a);
                  cb = _mm512_4fmadd_ps(cb, b00, b10, b20, b30, a_base+0x0b);
                  cc = _mm512_4fmadd_ps(cc, b00, b10, b20, b30, a_base+0x0c);
                  cd = _mm512_4fmadd_ps(cd, b00, b10, b20, b30, a_base+0x0d);
                  ce = _mm512_4fmadd_ps(ce, b00, b10, b20, b30, a_base+0x0e);
                  cf = _mm512_4fmadd_ps(cf, b00, b10, b20, b30, a_base+0x0f);

                  c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a_base+0x10);
                  c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a_base+0x11);
                  c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a_base+0x12);
                  c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a_base+0x13);
                  c4 = _mm512_4fmadd_ps(c4, b40, b50, b60, b70, a_base+0x14);
                  c5 = _mm512_4fmadd_ps(c5, b40, b50, b60, b70, a_base+0x15);
                  c6 = _mm512_4fmadd_ps(c6, b40, b50, b60, b70, a_base+0x16);
                  c7 = _mm512_4fmadd_ps(c7, b40, b50, b60, b70, a_base+0x17);
                  c8 = _mm512_4fmadd_ps(c8, b40, b50, b60, b70, a_base+0x18);
                  c9 = _mm512_4fmadd_ps(c9, b40, b50, b60, b70, a_base+0x19);
                  ca = _mm512_4fmadd_ps(ca, b40, b50, b60, b70, a_base+0x1a);
                  cb = _mm512_4fmadd_ps(cb, b40, b50, b60, b70, a_base+0x1b);
                  cc = _mm512_4fmadd_ps(cc, b40, b50, b60, b70, a_base+0x1c);
                  cd = _mm512_4fmadd_ps(cd, b40, b50, b60, b70, a_base+0x1d);
                  ce = _mm512_4fmadd_ps(ce, b40, b50, b60, b70, a_base+0x1e);
                  cf = _mm512_4fmadd_ps(cf, b40, b50, b60, b70, a_base+0x1f);

                  c0 = _mm512_4fmadd_ps(c0, b80, b90, ba0, bb0, a_base+0x20);
                  c1 = _mm512_4fmadd_ps(c1, b80, b90, ba0, bb0, a_base+0x21);
                  c2 = _mm512_4fmadd_ps(c2, b80, b90, ba0, bb0, a_base+0x22);
                  c3 = _mm512_4fmadd_ps(c3, b80, b90, ba0, bb0, a_base+0x23);
                  c4 = _mm512_4fmadd_ps(c4, b80, b90, ba0, bb0, a_base+0x24);
                  c5 = _mm512_4fmadd_ps(c5, b80, b90, ba0, bb0, a_base+0x25);
                  c6 = _mm512_4fmadd_ps(c6, b80, b90, ba0, bb0, a_base+0x26);
                  c7 = _mm512_4fmadd_ps(c7, b80, b90, ba0, bb0, a_base+0x27);
                  c8 = _mm512_4fmadd_ps(c8, b80, b90, ba0, bb0, a_base+0x28);
                  c9 = _mm512_4fmadd_ps(c9, b80, b90, ba0, bb0, a_base+0x29);
                  ca = _mm512_4fmadd_ps(ca, b80, b90, ba0, bb0, a_base+0x2a);
                  cb = _mm512_4fmadd_ps(cb, b80, b90, ba0, bb0, a_base+0x2b);
                  cc = _mm512_4fmadd_ps(cc, b80, b90, ba0, bb0, a_base+0x2c);
                  cd = _mm512_4fmadd_ps(cd, b80, b90, ba0, bb0, a_base+0x2d);
                  ce = _mm512_4fmadd_ps(ce, b80, b90, ba0, bb0, a_base+0x2e);
                  cf = _mm512_4fmadd_ps(cf, b80, b90, ba0, bb0, a_base+0x2f);

                  c0 = _mm512_4fmadd_ps(c0, bc0, bd0, be0, bf0, a_base+0x30);
                  c1 = _mm512_4fmadd_ps(c1, bc0, bd0, be0, bf0, a_base+0x31);
                  c2 = _mm512_4fmadd_ps(c2, bc0, bd0, be0, bf0, a_base+0x32);
                  c3 = _mm512_4fmadd_ps(c3, bc0, bd0, be0, bf0, a_base+0x33);
                  c4 = _mm512_4fmadd_ps(c4, bc0, bd0, be0, bf0, a_base+0x34);
                  c5 = _mm512_4fmadd_ps(c5, bc0, bd0, be0, bf0, a_base+0x35);
                  c6 = _mm512_4fmadd_ps(c6, bc0, bd0, be0, bf0, a_base+0x36);
                  c7 = _mm512_4fmadd_ps(c7, bc0, bd0, be0, bf0, a_base+0x37);
                  c8 = _mm512_4fmadd_ps(c8, bc0, bd0, be0, bf0, a_base+0x38);
                  c9 = _mm512_4fmadd_ps(c9, bc0, bd0, be0, bf0, a_base+0x39);
                  ca = _mm512_4fmadd_ps(ca, bc0, bd0, be0, bf0, a_base+0x3a);
                  cb = _mm512_4fmadd_ps(cb, bc0, bd0, be0, bf0, a_base+0x3b);
                  cc = _mm512_4fmadd_ps(cc, bc0, bd0, be0, bf0, a_base+0x3c);
                  cd = _mm512_4fmadd_ps(cd, bc0, bd0, be0, bf0, a_base+0x3d);
                  ce = _mm512_4fmadd_ps(ce, bc0, bd0, be0, bf0, a_base+0x3e);
                  cf = _mm512_4fmadd_ps(cf, bc0, bd0, be0, bf0, a_base+0x3f);

                  a_base += 0x40;
                }

                if (k + 8 <= p) {
                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);
                  __m512 b40 = _mm512_maskz_load_ps(mask, b_base + n*0x4);
                  __m512 b50 = _mm512_maskz_load_ps(mask, b_base + n*0x5);
                  __m512 b60 = _mm512_maskz_load_ps(mask, b_base + n*0x6);
                  __m512 b70 = _mm512_maskz_load_ps(mask, b_base + n*0x7);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);
                  c8 = _mm512_4fmadd_ps(c8, b00, b10, b20, b30, a_base+0x08);
                  c9 = _mm512_4fmadd_ps(c9, b00, b10, b20, b30, a_base+0x09);
                  ca = _mm512_4fmadd_ps(ca, b00, b10, b20, b30, a_base+0x0a);
                  cb = _mm512_4fmadd_ps(cb, b00, b10, b20, b30, a_base+0x0b);
                  cc = _mm512_4fmadd_ps(cc, b00, b10, b20, b30, a_base+0x0c);
                  cd = _mm512_4fmadd_ps(cd, b00, b10, b20, b30, a_base+0x0d);
                  ce = _mm512_4fmadd_ps(ce, b00, b10, b20, b30, a_base+0x0e);
                  cf = _mm512_4fmadd_ps(cf, b00, b10, b20, b30, a_base+0x0f);

                  c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a_base+0x10);
                  c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a_base+0x11);
                  c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a_base+0x12);
                  c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a_base+0x13);
                  c4 = _mm512_4fmadd_ps(c4, b40, b50, b60, b70, a_base+0x14);
                  c5 = _mm512_4fmadd_ps(c5, b40, b50, b60, b70, a_base+0x15);
                  c6 = _mm512_4fmadd_ps(c6, b40, b50, b60, b70, a_base+0x16);
                  c7 = _mm512_4fmadd_ps(c7, b40, b50, b60, b70, a_base+0x17);
                  c8 = _mm512_4fmadd_ps(c8, b40, b50, b60, b70, a_base+0x18);
                  c9 = _mm512_4fmadd_ps(c9, b40, b50, b60, b70, a_base+0x19);
                  ca = _mm512_4fmadd_ps(ca, b40, b50, b60, b70, a_base+0x1a);
                  cb = _mm512_4fmadd_ps(cb, b40, b50, b60, b70, a_base+0x1b);
                  cc = _mm512_4fmadd_ps(cc, b40, b50, b60, b70, a_base+0x1c);
                  cd = _mm512_4fmadd_ps(cd, b40, b50, b60, b70, a_base+0x1d);
                  ce = _mm512_4fmadd_ps(ce, b40, b50, b60, b70, a_base+0x1e);
                  cf = _mm512_4fmadd_ps(cf, b40, b50, b60, b70, a_base+0x1f);

                  a_base += 0x20;
                  k += 8;
                }

                if (k + 4 <= p) {

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);
                  c8 = _mm512_4fmadd_ps(c8, b00, b10, b20, b30, a_base+0x08);
                  c9 = _mm512_4fmadd_ps(c9, b00, b10, b20, b30, a_base+0x09);
                  ca = _mm512_4fmadd_ps(ca, b00, b10, b20, b30, a_base+0x0a);
                  cb = _mm512_4fmadd_ps(cb, b00, b10, b20, b30, a_base+0x0b);
                  cc = _mm512_4fmadd_ps(cc, b00, b10, b20, b30, a_base+0x0c);
                  cd = _mm512_4fmadd_ps(cd, b00, b10, b20, b30, a_base+0x0d);
                  ce = _mm512_4fmadd_ps(ce, b00, b10, b20, b30, a_base+0x0e);
                  cf = _mm512_4fmadd_ps(cf, b00, b10, b20, b30, a_base+0x0f);

                  a_base += 0x10;
                  k += 4;
                }

                if (k < p) {

                  float *b_base = (float *)(B + k * n + j);
                  __m512 b00 = _mm512_setzero_ps();
                  __m512 b10 = _mm512_setzero_ps();
                  __m512 b20 = _mm512_setzero_ps();
                  __m512 b30 = _mm512_setzero_ps();

                  b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  if (k + 1 < p) b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  if (k + 2 < p) b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  if (k + 3 < p) b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);
                  c8 = _mm512_4fmadd_ps(c8, b00, b10, b20, b30, a_base+0x08);
                  c9 = _mm512_4fmadd_ps(c9, b00, b10, b20, b30, a_base+0x09);
                  ca = _mm512_4fmadd_ps(ca, b00, b10, b20, b30, a_base+0x0a);
                  cb = _mm512_4fmadd_ps(cb, b00, b10, b20, b30, a_base+0x0b);
                  cc = _mm512_4fmadd_ps(cc, b00, b10, b20, b30, a_base+0x0c);
                  cd = _mm512_4fmadd_ps(cd, b00, b10, b20, b30, a_base+0x0d);
                  ce = _mm512_4fmadd_ps(ce, b00, b10, b20, b30, a_base+0x0e);
                  cf = _mm512_4fmadd_ps(cf, b00, b10, b20, b30, a_base+0x0f);

                  a_base += 0x10;
                  k += 4;
                }

                matrix_transpose_16x16(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf);

                float *c_base = C + j*m + i;

                if (mask == 0xffff) {
                  __m512 p0 = _mm512_load_ps(c_base + 0x0 * m);
                  __m512 p1 = _mm512_load_ps(c_base + 0x1 * m);
                  __m512 p2 = _mm512_load_ps(c_base + 0x2 * m);
                  __m512 p3 = _mm512_load_ps(c_base + 0x3 * m);
                  __m512 p4 = _mm512_load_ps(c_base + 0x4 * m);
                  __m512 p5 = _mm512_load_ps(c_base + 0x5 * m);
                  __m512 p6 = _mm512_load_ps(c_base + 0x6 * m);
                  __m512 p7 = _mm512_load_ps(c_base + 0x7 * m);
                  __m512 p8 = _mm512_load_ps(c_base + 0x8 * m);
                  __m512 p9 = _mm512_load_ps(c_base + 0x9 * m);
                  __m512 pa = _mm512_load_ps(c_base + 0xa * m);
                  __m512 pb = _mm512_load_ps(c_base + 0xb * m);
                  __m512 pc = _mm512_load_ps(c_base + 0xc * m);
                  __m512 pd = _mm512_load_ps(c_base + 0xd * m);
                  __m512 pe = _mm512_load_ps(c_base + 0xe * m);
                  __m512 pf = _mm512_load_ps(c_base + 0xf * m);

                  c0 = _mm512_add_ps(c0, p0);
                  c1 = _mm512_add_ps(c1, p1);
                  c2 = _mm512_add_ps(c2, p2);
                  c3 = _mm512_add_ps(c3, p3);
                  c4 = _mm512_add_ps(c4, p4);
                  c5 = _mm512_add_ps(c5, p5);
                  c6 = _mm512_add_ps(c6, p6);
                  c7 = _mm512_add_ps(c7, p7);
                  c8 = _mm512_add_ps(c8, p8);
                  c9 = _mm512_add_ps(c9, p9);
                  ca = _mm512_add_ps(ca, pa);
                  cb = _mm512_add_ps(cb, pb);
                  cc = _mm512_add_ps(cc, pc);
                  cd = _mm512_add_ps(cd, pd);
                  ce = _mm512_add_ps(ce, pe);
                  cf = _mm512_add_ps(cf, pf);

                  _mm512_store_ps(c_base + 0x0 * m, c0);
                  _mm512_store_ps(c_base + 0x1 * m, c1);
                  _mm512_store_ps(c_base + 0x2 * m, c2);
                  _mm512_store_ps(c_base + 0x3 * m, c3);
                  _mm512_store_ps(c_base + 0x4 * m, c4);
                  _mm512_store_ps(c_base + 0x5 * m, c5);
                  _mm512_store_ps(c_base + 0x6 * m, c6);
                  _mm512_store_ps(c_base + 0x7 * m, c7);
                  _mm512_store_ps(c_base + 0x8 * m, c8);
                  _mm512_store_ps(c_base + 0x9 * m, c9);
                  _mm512_store_ps(c_base + 0xa * m, ca);
                  _mm512_store_ps(c_base + 0xb * m, cb);
                  _mm512_store_ps(c_base + 0xc * m, cc);
                  _mm512_store_ps(c_base + 0xd * m, cd);
                  _mm512_store_ps(c_base + 0xe * m, ce);
                  _mm512_store_ps(c_base + 0xf * m, cf);

                } else if (mask == 0xfff) {
                  __m512 p0 = _mm512_load_ps(c_base + 0x0 * m);
                  __m512 p1 = _mm512_load_ps(c_base + 0x1 * m);
                  __m512 p2 = _mm512_load_ps(c_base + 0x2 * m);
                  __m512 p3 = _mm512_load_ps(c_base + 0x3 * m);
                  __m512 p4 = _mm512_load_ps(c_base + 0x4 * m);
                  __m512 p5 = _mm512_load_ps(c_base + 0x5 * m);
                  __m512 p6 = _mm512_load_ps(c_base + 0x6 * m);
                  __m512 p7 = _mm512_load_ps(c_base + 0x7 * m);
                  __m512 p8 = _mm512_load_ps(c_base + 0x8 * m);
                  __m512 p9 = _mm512_load_ps(c_base + 0x9 * m);
                  __m512 pa = _mm512_load_ps(c_base + 0xa * m);
                  __m512 pb = _mm512_load_ps(c_base + 0xb * m);

                  c0 = _mm512_add_ps(c0, p0);
                  c1 = _mm512_add_ps(c1, p1);
                  c2 = _mm512_add_ps(c2, p2);
                  c3 = _mm512_add_ps(c3, p3);
                  c4 = _mm512_add_ps(c4, p4);
                  c5 = _mm512_add_ps(c5, p5);
                  c6 = _mm512_add_ps(c6, p6);
                  c7 = _mm512_add_ps(c7, p7);
                  c8 = _mm512_add_ps(c8, p8);
                  c9 = _mm512_add_ps(c9, p9);
                  ca = _mm512_add_ps(ca, pa);
                  cb = _mm512_add_ps(cb, pb);

                  _mm512_store_ps(c_base + 0x0 * m, c0);
                  _mm512_store_ps(c_base + 0x1 * m, c1);
                  _mm512_store_ps(c_base + 0x2 * m, c2);
                  _mm512_store_ps(c_base + 0x3 * m, c3);
                  _mm512_store_ps(c_base + 0x4 * m, c4);
                  _mm512_store_ps(c_base + 0x5 * m, c5);
                  _mm512_store_ps(c_base + 0x6 * m, c6);
                  _mm512_store_ps(c_base + 0x7 * m, c7);
                  _mm512_store_ps(c_base + 0x8 * m, c8);
                  _mm512_store_ps(c_base + 0x9 * m, c9);
                  _mm512_store_ps(c_base + 0xa * m, ca);
                  _mm512_store_ps(c_base + 0xb * m, cb);

                } else if (mask == 0xf) {
                  __m512 p0, p1, p2, p3;

                  p0 = _mm512_load_ps(c_base + 0x0 * m);
                  p1 = _mm512_load_ps(c_base + 0x1 * m);
                  p2 = _mm512_load_ps(c_base + 0x2 * m);
                  p3 = _mm512_load_ps(c_base + 0x3 * m);

                  c0 = _mm512_add_ps(c0, p0);
                  c1 = _mm512_add_ps(c1, p1);
                  c2 = _mm512_add_ps(c2, p2);
                  c3 = _mm512_add_ps(c3, p3);

                  _mm512_store_ps(c_base + 0x0 * m, c0);
                  _mm512_store_ps(c_base + 0x1 * m, c1);
                  _mm512_store_ps(c_base + 0x2 * m, c2);
                  _mm512_store_ps(c_base + 0x3 * m, c3);
                } else {
                  assert(0);
                }
              }


              if ((m-i) == 4 && ((i + 4) <= (ii + LI))) {
                __m128 *a_base = (__m128 *)(AT + i * ap);

                uint64_t k = 0;
                __m512 c0 = _mm512_setzero_ps();
                __m512 c1 = _mm512_setzero_ps();
                __m512 c2 = _mm512_setzero_ps();
                __m512 c3 = _mm512_setzero_ps();

                for (; k < p; k = k + 16) {
                  if (k + 16 > p) break;

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);
                  __m512 b40 = _mm512_maskz_load_ps(mask, b_base + n*0x4);
                  __m512 b50 = _mm512_maskz_load_ps(mask, b_base + n*0x5);
                  __m512 b60 = _mm512_maskz_load_ps(mask, b_base + n*0x6);
                  __m512 b70 = _mm512_maskz_load_ps(mask, b_base + n*0x7);
                  __m512 b80 = _mm512_maskz_load_ps(mask, b_base + n*0x8);
                  __m512 b90 = _mm512_maskz_load_ps(mask, b_base + n*0x9);
                  __m512 ba0 = _mm512_maskz_load_ps(mask, b_base + n*0xa);
                  __m512 bb0 = _mm512_maskz_load_ps(mask, b_base + n*0xb);
                  __m512 bc0 = _mm512_maskz_load_ps(mask, b_base + n*0xc);
                  __m512 bd0 = _mm512_maskz_load_ps(mask, b_base + n*0xd);
                  __m512 be0 = _mm512_maskz_load_ps(mask, b_base + n*0xe);
                  __m512 bf0 = _mm512_maskz_load_ps(mask, b_base + n*0xf);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);

                  c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a_base+0x04);
                  c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a_base+0x05);
                  c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a_base+0x06);
                  c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a_base+0x07);

                  c0 = _mm512_4fmadd_ps(c0, b80, b90, ba0, bb0, a_base+0x08);
                  c1 = _mm512_4fmadd_ps(c1, b80, b90, ba0, bb0, a_base+0x09);
                  c2 = _mm512_4fmadd_ps(c2, b80, b90, ba0, bb0, a_base+0x0a);
                  c3 = _mm512_4fmadd_ps(c3, b80, b90, ba0, bb0, a_base+0x0b);

                  c0 = _mm512_4fmadd_ps(c0, bc0, bd0, be0, bf0, a_base+0x0c);
                  c1 = _mm512_4fmadd_ps(c1, bc0, bd0, be0, bf0, a_base+0x0d);
                  c2 = _mm512_4fmadd_ps(c2, bc0, bd0, be0, bf0, a_base+0x0e);
                  c3 = _mm512_4fmadd_ps(c3, bc0, bd0, be0, bf0, a_base+0x0f);

                  a_base += 0x10;
                }

                if (k + 8 <= p) {
                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);
                  __m512 b40 = _mm512_maskz_load_ps(mask, b_base + n*0x4);
                  __m512 b50 = _mm512_maskz_load_ps(mask, b_base + n*0x5);
                  __m512 b60 = _mm512_maskz_load_ps(mask, b_base + n*0x6);
                  __m512 b70 = _mm512_maskz_load_ps(mask, b_base + n*0x7);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);

                  c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a_base+0x04);
                  c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a_base+0x05);
                  c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a_base+0x06);
                  c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a_base+0x07);

                  a_base += 0x08;
                  k += 8;
                }

                if (k + 4 <= p) {

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);

                  a_base += 0x04;
                  k += 4;
                }

                if (k < p) {

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_setzero_ps();
                  __m512 b10 = _mm512_setzero_ps();
                  __m512 b20 = _mm512_setzero_ps();
                  __m512 b30 = _mm512_setzero_ps();

                  if (k + 0 < p) b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  if (k + 1 < p) b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  if (k + 2 < p) b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  if (k + 3 < p) b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);

                  a_base += 0x04;
                  k += 4;

                }

                matrix_transpose_4x16(c0, c1, c2, c3);

                float *c_base = C + j*m + i;

                if (mask == 0xffff) {
                  __m128 p0_0 = _mm_load_ps(c_base + 0x0 * m);
                  __m128 p0_1 = _mm_load_ps(c_base + 0x1 * m);
                  __m128 p0_2 = _mm_load_ps(c_base + 0x2 * m);
                  __m128 p0_3 = _mm_load_ps(c_base + 0x3 * m);
                  __m128 p1_0 = _mm_load_ps(c_base + 0x4 * m);
                  __m128 p1_1 = _mm_load_ps(c_base + 0x5 * m);
                  __m128 p1_2 = _mm_load_ps(c_base + 0x6 * m);
                  __m128 p1_3 = _mm_load_ps(c_base + 0x7 * m);
                  __m128 p2_0 = _mm_load_ps(c_base + 0x8 * m);
                  __m128 p2_1 = _mm_load_ps(c_base + 0x9 * m);
                  __m128 p2_2 = _mm_load_ps(c_base + 0xa * m);
                  __m128 p2_3 = _mm_load_ps(c_base + 0xb * m);
                  __m128 p3_0 = _mm_load_ps(c_base + 0xc * m);
                  __m128 p3_1 = _mm_load_ps(c_base + 0xd * m);
                  __m128 p3_2 = _mm_load_ps(c_base + 0xe * m);
                  __m128 p3_3 = _mm_load_ps(c_base + 0xf * m);

                  __m128 c0_0 = _mm512_extractf32x4_ps(c0, 0);
                  __m128 c0_1 = _mm512_extractf32x4_ps(c0, 1);
                  __m128 c0_2 = _mm512_extractf32x4_ps(c0, 2);
                  __m128 c0_3 = _mm512_extractf32x4_ps(c0, 3);
                  __m128 c1_0 = _mm512_extractf32x4_ps(c1, 0);
                  __m128 c1_1 = _mm512_extractf32x4_ps(c1, 1);
                  __m128 c1_2 = _mm512_extractf32x4_ps(c1, 2);
                  __m128 c1_3 = _mm512_extractf32x4_ps(c1, 3);
                  __m128 c2_0 = _mm512_extractf32x4_ps(c2, 0);
                  __m128 c2_1 = _mm512_extractf32x4_ps(c2, 1);
                  __m128 c2_2 = _mm512_extractf32x4_ps(c2, 2);
                  __m128 c2_3 = _mm512_extractf32x4_ps(c2, 3);
                  __m128 c3_0 = _mm512_extractf32x4_ps(c3, 0);
                  __m128 c3_1 = _mm512_extractf32x4_ps(c3, 1);
                  __m128 c3_2 = _mm512_extractf32x4_ps(c3, 2);
                  __m128 c3_3 = _mm512_extractf32x4_ps(c3, 3);

                  c0_0 = _mm_add_ps(c0_0, p0_0);
                  c0_1 = _mm_add_ps(c0_1, p0_1);
                  c0_2 = _mm_add_ps(c0_2, p0_2);
                  c0_3 = _mm_add_ps(c0_3, p0_3);
                  c1_0 = _mm_add_ps(c1_0, p1_0);
                  c1_1 = _mm_add_ps(c1_1, p1_1);
                  c1_2 = _mm_add_ps(c1_2, p1_2);
                  c1_3 = _mm_add_ps(c1_3, p1_3);
                  c2_0 = _mm_add_ps(c2_0, p2_0);
                  c2_1 = _mm_add_ps(c2_1, p2_1);
                  c2_2 = _mm_add_ps(c2_2, p2_2);
                  c2_3 = _mm_add_ps(c2_3, p2_3);
                  c3_0 = _mm_add_ps(c3_0, p3_0);
                  c3_1 = _mm_add_ps(c3_1, p3_1);
                  c3_2 = _mm_add_ps(c3_2, p3_2);
                  c3_3 = _mm_add_ps(c3_3, p3_3);

                  _mm_store_ps(c_base + 0x0 * m, c0_0);
                  _mm_store_ps(c_base + 0x1 * m, c0_1);
                  _mm_store_ps(c_base + 0x2 * m, c0_2);
                  _mm_store_ps(c_base + 0x3 * m, c0_3);
                  _mm_store_ps(c_base + 0x4 * m, c1_0);
                  _mm_store_ps(c_base + 0x5 * m, c1_1);
                  _mm_store_ps(c_base + 0x6 * m, c1_2);
                  _mm_store_ps(c_base + 0x7 * m, c1_3);
                  _mm_store_ps(c_base + 0x8 * m, c2_0);
                  _mm_store_ps(c_base + 0x9 * m, c2_1);
                  _mm_store_ps(c_base + 0xa * m, c2_2);
                  _mm_store_ps(c_base + 0xb * m, c2_3);
                  _mm_store_ps(c_base + 0xc * m, c3_0);
                  _mm_store_ps(c_base + 0xd * m, c3_1);
                  _mm_store_ps(c_base + 0xe * m, c3_2);
                  _mm_store_ps(c_base + 0xf * m, c3_3);

                } else if (mask == 0xfff) {
                  __m128 p0_0 = _mm_load_ps(c_base + 0x0 * m);
                  __m128 p0_1 = _mm_load_ps(c_base + 0x1 * m);
                  __m128 p0_2 = _mm_load_ps(c_base + 0x2 * m);
                  __m128 p0_3 = _mm_load_ps(c_base + 0x3 * m);
                  __m128 p1_0 = _mm_load_ps(c_base + 0x4 * m);
                  __m128 p1_1 = _mm_load_ps(c_base + 0x5 * m);
                  __m128 p1_2 = _mm_load_ps(c_base + 0x6 * m);
                  __m128 p1_3 = _mm_load_ps(c_base + 0x7 * m);
                  __m128 p2_0 = _mm_load_ps(c_base + 0x8 * m);
                  __m128 p2_1 = _mm_load_ps(c_base + 0x9 * m);
                  __m128 p2_2 = _mm_load_ps(c_base + 0xa * m);
                  __m128 p2_3 = _mm_load_ps(c_base + 0xb * m);

                  __m128 c0_0 = _mm512_extractf32x4_ps(c0, 0);
                  __m128 c0_1 = _mm512_extractf32x4_ps(c0, 1);
                  __m128 c0_2 = _mm512_extractf32x4_ps(c0, 2);
                  __m128 c0_3 = _mm512_extractf32x4_ps(c0, 3);
                  __m128 c1_0 = _mm512_extractf32x4_ps(c1, 0);
                  __m128 c1_1 = _mm512_extractf32x4_ps(c1, 1);
                  __m128 c1_2 = _mm512_extractf32x4_ps(c1, 2);
                  __m128 c1_3 = _mm512_extractf32x4_ps(c1, 3);
                  __m128 c2_0 = _mm512_extractf32x4_ps(c2, 0);
                  __m128 c2_1 = _mm512_extractf32x4_ps(c2, 1);
                  __m128 c2_2 = _mm512_extractf32x4_ps(c2, 2);
                  __m128 c2_3 = _mm512_extractf32x4_ps(c2, 3);

                  c0_0 = _mm_add_ps(c0_0, p0_0);
                  c0_1 = _mm_add_ps(c0_1, p0_1);
                  c0_2 = _mm_add_ps(c0_2, p0_2);
                  c0_3 = _mm_add_ps(c0_3, p0_3);
                  c1_0 = _mm_add_ps(c1_0, p1_0);
                  c1_1 = _mm_add_ps(c1_1, p1_1);
                  c1_2 = _mm_add_ps(c1_2, p1_2);
                  c1_3 = _mm_add_ps(c1_3, p1_3);
                  c2_0 = _mm_add_ps(c2_0, p2_0);
                  c2_1 = _mm_add_ps(c2_1, p2_1);
                  c2_2 = _mm_add_ps(c2_2, p2_2);
                  c2_3 = _mm_add_ps(c2_3, p2_3);

                  _mm_store_ps(c_base + 0x0 * m, c0_0);
                  _mm_store_ps(c_base + 0x1 * m, c0_1);
                  _mm_store_ps(c_base + 0x2 * m, c0_2);
                  _mm_store_ps(c_base + 0x3 * m, c0_3);
                  _mm_store_ps(c_base + 0x4 * m, c1_0);
                  _mm_store_ps(c_base + 0x5 * m, c1_1);
                  _mm_store_ps(c_base + 0x6 * m, c1_2);
                  _mm_store_ps(c_base + 0x7 * m, c1_3);
                  _mm_store_ps(c_base + 0x8 * m, c2_0);
                  _mm_store_ps(c_base + 0x9 * m, c2_1);
                  _mm_store_ps(c_base + 0xa * m, c2_2);
                  _mm_store_ps(c_base + 0xb * m, c2_3);

                } else if (mask == 0xf) {
                  __m128 c0_0, c0_1, c0_2, c0_3;
                  __m128 p0_0, p0_1, p0_2, p0_3;

                  p0_0 = _mm_load_ps(c_base + 0x0 * m);
                  p0_1 = _mm_load_ps(c_base + 0x1 * m);
                  p0_2 = _mm_load_ps(c_base + 0x2 * m);
                  p0_3 = _mm_load_ps(c_base + 0x3 * m);

                  c0_0 = _mm512_extractf32x4_ps(c0, 0);
                  c0_1 = _mm512_extractf32x4_ps(c0, 1);
                  c0_2 = _mm512_extractf32x4_ps(c0, 2);
                  c0_3 = _mm512_extractf32x4_ps(c0, 3);

                  c0_0 = _mm_add_ps(c0_0, p0_0);
                  c0_1 = _mm_add_ps(c0_1, p0_1);
                  c0_2 = _mm_add_ps(c0_2, p0_2);
                  c0_3 = _mm_add_ps(c0_3, p0_3);


                  _mm_store_ps(c_base + 0x0 * m, c0_0);
                  _mm_store_ps(c_base + 0x1 * m, c0_1);
                  _mm_store_ps(c_base + 0x2 * m, c0_2);
                  _mm_store_ps(c_base + 0x3 * m, c0_3);

                } else {
                  assert(0);
                }
                i += 4;
              } else if ((m - i) == 8 && ((i + 8) <= (ii + LI))) {
                uint64_t k = 0;

                __m128 *a_base = (__m128 *)(AT + i * ap);
                __m512 c0 = _mm512_setzero_ps();
                __m512 c1 = _mm512_setzero_ps();
                __m512 c2 = _mm512_setzero_ps();
                __m512 c3 = _mm512_setzero_ps();
                __m512 c4 = _mm512_setzero_ps();
                __m512 c5 = _mm512_setzero_ps();
                __m512 c6 = _mm512_setzero_ps();
                __m512 c7 = _mm512_setzero_ps();

                for (k = 0; k < p; k = k + 16) {
                  if (k + 16 > p) break;
                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);
                  __m512 b40 = _mm512_maskz_load_ps(mask, b_base + n*0x4);
                  __m512 b50 = _mm512_maskz_load_ps(mask, b_base + n*0x5);
                  __m512 b60 = _mm512_maskz_load_ps(mask, b_base + n*0x6);
                  __m512 b70 = _mm512_maskz_load_ps(mask, b_base + n*0x7);
                  __m512 b80 = _mm512_maskz_load_ps(mask, b_base + n*0x8);
                  __m512 b90 = _mm512_maskz_load_ps(mask, b_base + n*0x9);
                  __m512 ba0 = _mm512_maskz_load_ps(mask, b_base + n*0xa);
                  __m512 bb0 = _mm512_maskz_load_ps(mask, b_base + n*0xb);
                  __m512 bc0 = _mm512_maskz_load_ps(mask, b_base + n*0xc);
                  __m512 bd0 = _mm512_maskz_load_ps(mask, b_base + n*0xd);
                  __m512 be0 = _mm512_maskz_load_ps(mask, b_base + n*0xe);
                  __m512 bf0 = _mm512_maskz_load_ps(mask, b_base + n*0xf);
                
                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);

                  c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a_base+0x08);
                  c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a_base+0x09);
                  c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a_base+0x0a);
                  c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a_base+0x0b);
                  c4 = _mm512_4fmadd_ps(c4, b40, b50, b60, b70, a_base+0x0c);
                  c5 = _mm512_4fmadd_ps(c5, b40, b50, b60, b70, a_base+0x0d);
                  c6 = _mm512_4fmadd_ps(c6, b40, b50, b60, b70, a_base+0x0e);
                  c7 = _mm512_4fmadd_ps(c7, b40, b50, b60, b70, a_base+0x0f);

                  c0 = _mm512_4fmadd_ps(c0, b80, b90, ba0, bb0, a_base+0x10);
                  c1 = _mm512_4fmadd_ps(c1, b80, b90, ba0, bb0, a_base+0x11);
                  c2 = _mm512_4fmadd_ps(c2, b80, b90, ba0, bb0, a_base+0x12);
                  c3 = _mm512_4fmadd_ps(c3, b80, b90, ba0, bb0, a_base+0x13);
                  c4 = _mm512_4fmadd_ps(c4, b80, b90, ba0, bb0, a_base+0x14);
                  c5 = _mm512_4fmadd_ps(c5, b80, b90, ba0, bb0, a_base+0x15);
                  c6 = _mm512_4fmadd_ps(c6, b80, b90, ba0, bb0, a_base+0x16);
                  c7 = _mm512_4fmadd_ps(c7, b80, b90, ba0, bb0, a_base+0x17);

                  c0 = _mm512_4fmadd_ps(c0, bc0, bd0, be0, bf0, a_base+0x18);
                  c1 = _mm512_4fmadd_ps(c1, bc0, bd0, be0, bf0, a_base+0x19);
                  c2 = _mm512_4fmadd_ps(c2, bc0, bd0, be0, bf0, a_base+0x1a);
                  c3 = _mm512_4fmadd_ps(c3, bc0, bd0, be0, bf0, a_base+0x1b);
                  c4 = _mm512_4fmadd_ps(c4, bc0, bd0, be0, bf0, a_base+0x1c);
                  c5 = _mm512_4fmadd_ps(c5, bc0, bd0, be0, bf0, a_base+0x1d);
                  c6 = _mm512_4fmadd_ps(c6, bc0, bd0, be0, bf0, a_base+0x1e);
                  c7 = _mm512_4fmadd_ps(c7, bc0, bd0, be0, bf0, a_base+0x1f);

                  a_base += 0x20;
                }


                if (k + 8 <= p) {
                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);
                  __m512 b40 = _mm512_maskz_load_ps(mask, b_base + n*0x4);
                  __m512 b50 = _mm512_maskz_load_ps(mask, b_base + n*0x5);
                  __m512 b60 = _mm512_maskz_load_ps(mask, b_base + n*0x6);
                  __m512 b70 = _mm512_maskz_load_ps(mask, b_base + n*0x7);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);

                  c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a_base+0x08);
                  c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a_base+0x09);
                  c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a_base+0x0a);
                  c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a_base+0x0b);
                  c4 = _mm512_4fmadd_ps(c4, b40, b50, b60, b70, a_base+0x0c);
                  c5 = _mm512_4fmadd_ps(c5, b40, b50, b60, b70, a_base+0x0d);
                  c6 = _mm512_4fmadd_ps(c6, b40, b50, b60, b70, a_base+0x0e);
                  c7 = _mm512_4fmadd_ps(c7, b40, b50, b60, b70, a_base+0x0f);

                  a_base += 0x10;
                  k += 8;
                }

                if (k + 4 <= p) {

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  __m512 b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  __m512 b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  __m512 b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);

                  a_base += 0x08;
                  k += 4;
                }

                if (k < p) {

                  float *b_base = (float *)(B + k * n + j);

                  __m512 b00 = _mm512_setzero_ps();
                  __m512 b10 = _mm512_setzero_ps();
                  __m512 b20 = _mm512_setzero_ps();
                  __m512 b30 = _mm512_setzero_ps();

                  if (k + 0 < p) b00 = _mm512_maskz_load_ps(mask, b_base + n*0x0);
                  if (k + 1 < p) b10 = _mm512_maskz_load_ps(mask, b_base + n*0x1);
                  if (k + 2 < p) b20 = _mm512_maskz_load_ps(mask, b_base + n*0x2);
                  if (k + 3 < p) b30 = _mm512_maskz_load_ps(mask, b_base + n*0x3);

                  c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a_base+0x00);
                  c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a_base+0x01);
                  c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a_base+0x02);
                  c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a_base+0x03);
                  c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a_base+0x04);
                  c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a_base+0x05);
                  c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a_base+0x06);
                  c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a_base+0x07);

                  a_base += 0x08;
                  k += 4;
                }

                matrix_transpose_8x16(c0, c1, c2, c3, c4, c5, c6, c7);

                float *c_base = C + j*m + i;

                if (mask == 0xffff) {
                  __m512 p0_0 = _mm512_maskz_load_ps(0xff, c_base + 0x0 * m);
                  __m512 p0_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x4 * m - 0x08);
                  __m512 p1_0 = _mm512_maskz_load_ps(0xff, c_base + 0x1 * m);
                  __m512 p1_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x5 * m - 0x08);
                  __m512 p2_0 = _mm512_maskz_load_ps(0xff, c_base + 0x2 * m);
                  __m512 p2_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x6 * m - 0x08);
                  __m512 p3_0 = _mm512_maskz_load_ps(0xff, c_base + 0x3 * m);
                  __m512 p3_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x7 * m - 0x08);
                  __m512 p4_0 = _mm512_maskz_load_ps(0xff, c_base + 0x8 * m);
                  __m512 p4_1 = _mm512_maskz_load_ps(0xff00, c_base + 0xc * m - 0x08);
                  __m512 p5_0 = _mm512_maskz_load_ps(0xff, c_base + 0x9 * m);
                  __m512 p5_1 = _mm512_maskz_load_ps(0xff00, c_base + 0xd * m - 0x08);
                  __m512 p6_0 = _mm512_maskz_load_ps(0xff, c_base + 0xa * m);
                  __m512 p6_1 = _mm512_maskz_load_ps(0xff00, c_base + 0xe * m - 0x08);
                  __m512 p7_0 = _mm512_maskz_load_ps(0xff, c_base + 0xb * m);
                  __m512 p7_1 = _mm512_maskz_load_ps(0xff00, c_base + 0xf * m - 0x08);

                  __m512 c0_0 = _mm512_maskz_mov_ps(0x00ff, c0);
                  __m512 c0_1 = _mm512_maskz_mov_ps(0xff00, c0);
                  __m512 c1_0 = _mm512_maskz_mov_ps(0x00ff, c1);
                  __m512 c1_1 = _mm512_maskz_mov_ps(0xff00, c1);
                  __m512 c2_0 = _mm512_maskz_mov_ps(0x00ff, c2);
                  __m512 c2_1 = _mm512_maskz_mov_ps(0xff00, c2);
                  __m512 c3_0 = _mm512_maskz_mov_ps(0x00ff, c3);
                  __m512 c3_1 = _mm512_maskz_mov_ps(0xff00, c3);
                  __m512 c4_0 = _mm512_maskz_mov_ps(0x00ff, c4);
                  __m512 c4_1 = _mm512_maskz_mov_ps(0xff00, c4);
                  __m512 c5_0 = _mm512_maskz_mov_ps(0x00ff, c5);
                  __m512 c5_1 = _mm512_maskz_mov_ps(0xff00, c5);
                  __m512 c6_0 = _mm512_maskz_mov_ps(0x00ff, c6);
                  __m512 c6_1 = _mm512_maskz_mov_ps(0xff00, c6);
                  __m512 c7_0 = _mm512_maskz_mov_ps(0x00ff, c7);
                  __m512 c7_1 = _mm512_maskz_mov_ps(0xff00, c7);

                  c0_0 = _mm512_add_ps(c0_0, p0_0);
                  c0_1 = _mm512_add_ps(c0_1, p0_1);
                  c1_0 = _mm512_add_ps(c1_0, p1_0);
                  c1_1 = _mm512_add_ps(c1_1, p1_1);
                  c2_0 = _mm512_add_ps(c2_0, p2_0);
                  c2_1 = _mm512_add_ps(c2_1, p2_1);
                  c3_0 = _mm512_add_ps(c3_0, p3_0);
                  c3_1 = _mm512_add_ps(c3_1, p3_1);
                  c4_0 = _mm512_add_ps(c4_0, p4_0);
                  c4_1 = _mm512_add_ps(c4_1, p4_1);
                  c5_0 = _mm512_add_ps(c5_0, p5_0);
                  c5_1 = _mm512_add_ps(c5_1, p5_1);
                  c6_0 = _mm512_add_ps(c6_0, p6_0);
                  c6_1 = _mm512_add_ps(c6_1, p6_1);
                  c7_0 = _mm512_add_ps(c7_0, p7_0);
                  c7_1 = _mm512_add_ps(c7_1, p7_1);

                  _mm512_mask_store_ps(c_base + 0x0 * m, 0x00ff, c0_0);
                  _mm512_mask_store_ps(c_base + 0x1 * m, 0x00ff, c1_0);
                  _mm512_mask_store_ps(c_base + 0x2 * m, 0x00ff, c2_0);
                  _mm512_mask_store_ps(c_base + 0x3 * m, 0x00ff, c3_0);
                  _mm512_mask_store_ps(c_base + 0x4 * m - 0x08, 0xff00, c0_1);
                  _mm512_mask_store_ps(c_base + 0x5 * m - 0x08, 0xff00, c1_1);
                  _mm512_mask_store_ps(c_base + 0x6 * m - 0x08, 0xff00, c2_1);
                  _mm512_mask_store_ps(c_base + 0x7 * m - 0x08, 0xff00, c3_1);

                  _mm512_mask_store_ps(c_base + 0x8 * m, 0x00ff, c4_0);
                  _mm512_mask_store_ps(c_base + 0x9 * m, 0x00ff, c5_0);
                  _mm512_mask_store_ps(c_base + 0xa * m, 0x00ff, c6_0);
                  _mm512_mask_store_ps(c_base + 0xb * m, 0x00ff, c7_0);
                  _mm512_mask_store_ps(c_base + 0xc * m - 0x08, 0xff00, c4_1);
                  _mm512_mask_store_ps(c_base + 0xd * m - 0x08, 0xff00, c5_1);
                  _mm512_mask_store_ps(c_base + 0xe * m - 0x08, 0xff00, c6_1);
                  _mm512_mask_store_ps(c_base + 0xf * m - 0x08, 0xff00, c7_1);
                } else if (mask == 0xfff) {
                  __m512 p0_0 = _mm512_maskz_load_ps(0xff, c_base + 0x0 * m);
                  __m512 p0_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x4 * m - 0x08);
                  __m512 p1_0 = _mm512_maskz_load_ps(0xff, c_base + 0x1 * m);
                  __m512 p1_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x5 * m - 0x08);
                  __m512 p2_0 = _mm512_maskz_load_ps(0xff, c_base + 0x2 * m);
                  __m512 p2_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x6 * m - 0x08);
                  __m512 p3_0 = _mm512_maskz_load_ps(0xff, c_base + 0x3 * m);
                  __m512 p3_1 = _mm512_maskz_load_ps(0xff00, c_base + 0x7 * m - 0x08);
                  __m512 p4_0 = _mm512_maskz_load_ps(0xff, c_base + 0x8 * m);
                  __m512 p5_0 = _mm512_maskz_load_ps(0xff, c_base + 0x9 * m);
                  __m512 p6_0 = _mm512_maskz_load_ps(0xff, c_base + 0xa * m);
                  __m512 p7_0 = _mm512_maskz_load_ps(0xff, c_base + 0xb * m);

                  __m512 c0_0 = _mm512_maskz_mov_ps(0x00ff, c0);
                  __m512 c0_1 = _mm512_maskz_mov_ps(0xff00, c0);
                  __m512 c1_0 = _mm512_maskz_mov_ps(0x00ff, c1);
                  __m512 c1_1 = _mm512_maskz_mov_ps(0xff00, c1);
                  __m512 c2_0 = _mm512_maskz_mov_ps(0x00ff, c2);
                  __m512 c2_1 = _mm512_maskz_mov_ps(0xff00, c2);
                  __m512 c3_0 = _mm512_maskz_mov_ps(0x00ff, c3);
                  __m512 c3_1 = _mm512_maskz_mov_ps(0xff00, c3);
                  __m512 c4_0 = _mm512_maskz_mov_ps(0x00ff, c4);
                  __m512 c5_0 = _mm512_maskz_mov_ps(0x00ff, c5);
                  __m512 c6_0 = _mm512_maskz_mov_ps(0x00ff, c6);
                  __m512 c7_0 = _mm512_maskz_mov_ps(0x00ff, c7);

                  c0_0 = _mm512_add_ps(c0_0, p0_0);
                  c0_1 = _mm512_add_ps(c0_1, p0_1);
                  c1_0 = _mm512_add_ps(c1_0, p1_0);
                  c1_1 = _mm512_add_ps(c1_1, p1_1);
                  c2_0 = _mm512_add_ps(c2_0, p2_0);
                  c2_1 = _mm512_add_ps(c2_1, p2_1);
                  c3_0 = _mm512_add_ps(c3_0, p3_0);
                  c3_1 = _mm512_add_ps(c3_1, p3_1);
                  c4_0 = _mm512_add_ps(c4_0, p4_0);
                  c5_0 = _mm512_add_ps(c5_0, p5_0);
                  c6_0 = _mm512_add_ps(c6_0, p6_0);
                  c7_0 = _mm512_add_ps(c7_0, p7_0);

                  _mm512_mask_store_ps(c_base + 0x0 * m, 0x00ff, c0_0);
                  _mm512_mask_store_ps(c_base + 0x1 * m, 0x00ff, c1_0);
                  _mm512_mask_store_ps(c_base + 0x2 * m, 0x00ff, c2_0);
                  _mm512_mask_store_ps(c_base + 0x3 * m, 0x00ff, c3_0);
                  _mm512_mask_store_ps(c_base + 0x4 * m - 0x08, 0xff00, c0_1);
                  _mm512_mask_store_ps(c_base + 0x5 * m - 0x08, 0xff00, c1_1);
                  _mm512_mask_store_ps(c_base + 0x6 * m - 0x08, 0xff00, c2_1);
                  _mm512_mask_store_ps(c_base + 0x7 * m - 0x08, 0xff00, c3_1);

                  _mm512_mask_store_ps(c_base + 0x8 * m, 0x00ff, c4_0);
                  _mm512_mask_store_ps(c_base + 0x9 * m, 0x00ff, c5_0);
                  _mm512_mask_store_ps(c_base + 0xa * m, 0x00ff, c6_0);
                  _mm512_mask_store_ps(c_base + 0xb * m, 0x00ff, c7_0);

                } else if (mask == 0xf) {

                  __m512 p0_0 = _mm512_maskz_load_ps(0xff, c_base + 0x0 * m);
                  __m512 p1_0 = _mm512_maskz_load_ps(0xff, c_base + 0x1 * m);
                  __m512 p2_0 = _mm512_maskz_load_ps(0xff, c_base + 0x2 * m);
                  __m512 p3_0 = _mm512_maskz_load_ps(0xff, c_base + 0x3 * m);

                  __m512 c0_0 = _mm512_maskz_mov_ps(0x00ff, c0);
                  __m512 c1_0 = _mm512_maskz_mov_ps(0x00ff, c1);
                  __m512 c2_0 = _mm512_maskz_mov_ps(0x00ff, c2);
                  __m512 c3_0 = _mm512_maskz_mov_ps(0x00ff, c3);

                  c0_0 = _mm512_add_ps(c0_0, p0_0);
                  c1_0 = _mm512_add_ps(c1_0, p1_0);
                  c2_0 = _mm512_add_ps(c2_0, p2_0);
                  c3_0 = _mm512_add_ps(c3_0, p3_0);

                  _mm512_mask_store_ps(c_base + 0x0 * m, 0x00ff, c0_0);
                  _mm512_mask_store_ps(c_base + 0x1 * m, 0x00ff, c1_0);
                  _mm512_mask_store_ps(c_base + 0x2 * m, 0x00ff, c2_0);
                  _mm512_mask_store_ps(c_base + 0x3 * m, 0x00ff, c3_0);

                } else {
                  assert(0);
                }
                i += 8;
              }

            }
          }
        }
      }
    }
  }
  mkl_free(AT);
}

void sgemm3_opt(char* pTransA, char* pTransB,
      const int* pM, const int* pN, const int* pK,
      const float *pAlpha, const float *pa, const int*plda,
      const float *pb, const int *pldb, const float *pBeta,
      float *pc, const int*pldc)
{
  const uint64_t m = *pM;
  const uint64_t n = *pN;
  const uint64_t p = *pK;

  const float *A = pa;
  const float *B = pb;
  float *C = pc;

  const float alpha = *pAlpha;
  const float beta  = *pBeta;

  assert(*pTransA == 'n' && *pTransB == 't');
  assert((m == 500 || m == 1000) && (n == 35820 || n == 50004) && p <= 64);
  assert(alpha == 1.0f && beta == 1.0f);
  assert((*pM == *plda) && (*pN == *pldb) && (*pM == *pldc));

  sgemm_opt(m, n, p, alpha, A, B, beta, C);
}


