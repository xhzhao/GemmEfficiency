
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
#include "mkl.h"
#include "immintrin.h"
#include<math.h>

#define THREAD_NUM  8

void print_mm512(__m512 data) {
  float mem[16];
  _mm512_store_ps(mem, data);

  int i = 0;
  printf("\n");
  for (i = 0; i < 16; ++i) {
    printf("%f, ", mem[i]);
  }
  printf("\n");
}

void sgemm3_ref( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc);

void sgemm3_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *A, const int*plda, const float *B, const int *pldb, const float *pBeta, float *C, const int*pldc)
{
    int m = *pM;
    int n = *pN;
    int k = *pK;
    float *AT = (float *)mkl_malloc(m*k*sizeof(float), 2048*1024);
    mkl_somatcopy('r','t', k, m, 1.0, A, m, AT, k);
#define CB_ITER 16
#define RA_ITER 16 
#define CA_ITER 16
#define AL 512*4 //(length of A)
#define RA_BLOCK 64
#define CB_BLOCK 64
#define TNUM (64)
#pragma omp parallel num_threads(TNUM)
  {
    int tid = omp_get_thread_num();
    int block_num = n/TNUM;
//    int AL = m*4; 
    //for(int ra_block = 0; ra_block < m/RA_BLOCK; ra_block = ra_block+1 ){    
    //printf("tid, %d, block_num %d   ", tid, block_num);
    for(int cb = tid * block_num; cb < (tid +1)*block_num; cb = cb + CB_ITER){
//    for(int cb_t = tid * block_num; cb_t < (tid +1)*block_num; cb_t = cb_t + CB_BLOCK){
//    for(int cb = cb_t; cb < (cb_t + CB_BLOCK); cb = cb + CB_ITER){
//    for(int cb = 0; cb < n; cb = cb + CB_ITER) { //loop for column of b, 35820, 4 AVX512 (64 FP) per iter
/*        _mm_prefetch((B + 0 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 1 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 2 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 3 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 4 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 5 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 6 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 7 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 8 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 9 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 10 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 11 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 12 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 13 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 14 * n + cb), _MM_HINT_T0);
        _mm_prefetch((B + 15 * n + cb), _MM_HINT_T0);
*/
        //for(int ra = ra_block *RA_BLOCK; ra < (ra_block + 1)*RA_BLOCK; ra = ra + RA_ITER) {//loop for row of a, 500, 4 line 4 FP per iter
        for(int ra = 0; ra < m; ra = ra + RA_ITER) {//loop for row of a, 500, 4 line 4 FP per iter
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
             __m512 c10 = _mm512_setzero_ps();
             __m512 c11 = _mm512_setzero_ps();

             __m512 c12 = _mm512_setzero_ps();
             __m512 c13 = _mm512_setzero_ps();
             __m512 c14 = _mm512_setzero_ps();
             __m512 c15 = _mm512_setzero_ps();
            
            for(int ca = 0; ca < k; ca = ca + CA_ITER) {      //ca = rb = 64,   k
                 float *abase = (float *)(AT + ra * k + ca);  //fetch ?
                 float *bbase = (float *)(B + ca * n + cb);            //fetch 4 AVX512

                 __m512 b00 = _mm512_load_ps(bbase);
                 __m512 b10 = _mm512_load_ps(bbase + n);
                 __m512 b20 = _mm512_load_ps(bbase + n*2);
                 __m512 b30 = _mm512_load_ps(bbase + n*3);

                 __m512 b40 = _mm512_load_ps(bbase + n*4);
                 __m512 b50 = _mm512_load_ps(bbase + n*5);
                 __m512 b60 = _mm512_load_ps(bbase + n*6);
                 __m512 b70 = _mm512_load_ps(bbase + n*7);

                 __m512 b80 = _mm512_load_ps(bbase + n*8);
                 __m512 b90 = _mm512_load_ps(bbase + n*9);
                 __m512 ba0 = _mm512_load_ps(bbase + n*10);
                 __m512 bb0 = _mm512_load_ps(bbase + n*11);

                 __m512 bc0 = _mm512_load_ps(bbase + n*12);
                 __m512 bd0 = _mm512_load_ps(bbase + n*13);
                 __m512 be0 = _mm512_load_ps(bbase + n*14);
                 __m512 bf0 = _mm512_load_ps(bbase + n*15);

                __m128 *a00 = (__m128 *)(abase);
                __m128 *a10 = (__m128 *)(abase + k);
                __m128 *a20 = (__m128 *)(abase + k*2);
                __m128 *a30 = (__m128 *)(abase + k*3);

                __m128 *a40 = (__m128 *)(abase + k*4);
                __m128 *a50 = (__m128 *)(abase + k*5);
                __m128 *a60 = (__m128 *)(abase + k*6);
                __m128 *a70 = (__m128 *)(abase + k*7);

                __m128 *a80 = (__m128 *)(abase + k*8);
                __m128 *a90 = (__m128 *)(abase + k*9);
                __m128 *aa0 = (__m128 *)(abase + k*10);
                __m128 *ab0 = (__m128 *)(abase + k*11);

                __m128 *ac0 = (__m128 *)(abase + k*12);
                __m128 *ad0 = (__m128 *)(abase + k*13);
                __m128 *ae0 = (__m128 *)(abase + k*14);
                __m128 *af0 = (__m128 *)(abase + k*15);

                c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a00);
                c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a10);
                c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a20);
                c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a30);


                c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, a40);
                c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, a50);
                c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, a60);
                c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, a70);

                c8 = _mm512_4fmadd_ps(c8, b00, b10, b20, b30, a80);
                c9 = _mm512_4fmadd_ps(c9, b00, b10, b20, b30, a90);
                c10 = _mm512_4fmadd_ps(c10, b00, b10, b20, b30, aa0);
                c11 = _mm512_4fmadd_ps(c11, b00, b10, b20, b30, ab0);

                c12 = _mm512_4fmadd_ps(c12, b00, b10, b20, b30, ac0);
                c13 = _mm512_4fmadd_ps(c13, b00, b10, b20, b30, ad0);
                c14 = _mm512_4fmadd_ps(c14, b00, b10, b20, b30, ae0);
                c15 = _mm512_4fmadd_ps(c15, b00, b10, b20, b30, af0);


                c0 = _mm512_4fmadd_ps(c0, b40, b50, b60, b70, a00+1);
                c1 = _mm512_4fmadd_ps(c1, b40, b50, b60, b70, a10+1);
                c2 = _mm512_4fmadd_ps(c2, b40, b50, b60, b70, a20+1);
                c3 = _mm512_4fmadd_ps(c3, b40, b50, b60, b70, a30+1);
                c4 = _mm512_4fmadd_ps(c4, b40, b50, b60, b70, a40+1);
                c5 = _mm512_4fmadd_ps(c5, b40, b50, b60, b70, a50+1);
                c6 = _mm512_4fmadd_ps(c6, b40, b50, b60, b70, a60+1);
                c7 = _mm512_4fmadd_ps(c7, b40, b50, b60, b70, a70+1);
                c8 = _mm512_4fmadd_ps(c8, b40, b50, b60, b70, a80+1);
                c9 = _mm512_4fmadd_ps(c9, b40, b50, b60, b70, a90+1);
                c10 = _mm512_4fmadd_ps(c10, b40, b50, b60, b70, aa0+1);
                c11 = _mm512_4fmadd_ps(c11, b40, b50, b60, b70, ab0+1);
                c12 = _mm512_4fmadd_ps(c12, b40, b50, b60, b70, ac0+1);
                c13 = _mm512_4fmadd_ps(c13, b40, b50, b60, b70, ad0+1);
                c14 = _mm512_4fmadd_ps(c14, b40, b50, b60, b70, ae0+1);
                c15 = _mm512_4fmadd_ps(c15, b40, b50, b60, b70, af0+1);


                c0 = _mm512_4fmadd_ps(c0, b80, b90, ba0, bb0, a00+2);
                c1 = _mm512_4fmadd_ps(c1, b80, b90, ba0, bb0, a10+2);
                c2 = _mm512_4fmadd_ps(c2, b80, b90, ba0, bb0, a20+2);
                c3 = _mm512_4fmadd_ps(c3, b80, b90, ba0, bb0, a30+2);
                c4 = _mm512_4fmadd_ps(c4, b80, b90, ba0, bb0, a40+2);
                c5 = _mm512_4fmadd_ps(c5, b80, b90, ba0, bb0, a50+2);
                c6 = _mm512_4fmadd_ps(c6, b80, b90, ba0, bb0, a60+2);
                c7 = _mm512_4fmadd_ps(c7, b80, b90, ba0, bb0, a70+2);
                c8 = _mm512_4fmadd_ps(c8, b80, b90, ba0, bb0, a80+2);
                c9 = _mm512_4fmadd_ps(c9, b80, b90, ba0, bb0, a90+2);
                c10 = _mm512_4fmadd_ps(c10, b80, b90, ba0, bb0, aa0+2);
                c11 = _mm512_4fmadd_ps(c11, b80, b90, ba0, bb0, ab0+2);
                c12 = _mm512_4fmadd_ps(c12, b80, b90, ba0, bb0, ac0+2);
                c13 = _mm512_4fmadd_ps(c13, b80, b90, ba0, bb0, ad0+2);
                c14 = _mm512_4fmadd_ps(c14, b80, b90, ba0, bb0, ae0+2);
                c15 = _mm512_4fmadd_ps(c15, b80, b90, ba0, bb0, af0+2);



                c0 = _mm512_4fmadd_ps(c0, bc0, bd0, be0, bf0, a00+3);
                c1 = _mm512_4fmadd_ps(c1, bc0, bd0, be0, bf0, a10+3);
                c2 = _mm512_4fmadd_ps(c2, bc0, bd0, be0, bf0, a20+3);
                c3 = _mm512_4fmadd_ps(c3, bc0, bd0, be0, bf0, a30+3);
                c4 = _mm512_4fmadd_ps(c4, bc0, bd0, be0, bf0, a40+3);
                c5 = _mm512_4fmadd_ps(c5, bc0, bd0, be0, bf0, a50+3);
                c6 = _mm512_4fmadd_ps(c6, bc0, bd0, be0, bf0, a60+3);
                c7 = _mm512_4fmadd_ps(c7, bc0, bd0, be0, bf0, a70+3);
                c8 = _mm512_4fmadd_ps(c8, bc0, bd0, be0, bf0, a80+3);
                c9 = _mm512_4fmadd_ps(c9, bc0, bd0, be0, bf0, a90+3);
                c10 = _mm512_4fmadd_ps(c10, bc0, bd0, be0, bf0, aa0+3);
                c11 = _mm512_4fmadd_ps(c11, bc0, bd0, be0, bf0, ab0+3);
                c12 = _mm512_4fmadd_ps(c12, bc0, bd0, be0, bf0, ac0+3);
                c13 = _mm512_4fmadd_ps(c13, bc0, bd0, be0, bf0, ad0+3);
                c14 = _mm512_4fmadd_ps(c14, bc0, bd0, be0, bf0, ae0+3);
                c15 = _mm512_4fmadd_ps(c15, bc0, bd0, be0, bf0, af0+3);
            }
           
            float *cbase = C + cb*m + ra;
             __m512i index_c0 = _mm512_set_epi32(15*AL,14*AL, 13*AL, 12*AL, 11*AL,10*AL,9*AL,8*AL,7*AL
                                                 ,6*AL, 5*AL,4*AL,3*AL,2*AL,1*AL,0);
#if 1 
            _mm512_i32scatter_ps((void*)cbase,index_c0,c0,1);
            _mm512_i32scatter_ps((void*)(cbase + 1),index_c0,c1,1);
            _mm512_i32scatter_ps((void*)(cbase + 2),index_c0,c2,1);
            _mm512_i32scatter_ps((void*)(cbase + 3),index_c0,c3,1);

            _mm512_i32scatter_ps((void*)(cbase + 4),index_c0,c4,1);
            _mm512_i32scatter_ps((void*)(cbase + 5),index_c0,c5,1);
            _mm512_i32scatter_ps((void*)(cbase + 6),index_c0,c6,1);
            _mm512_i32scatter_ps((void*)(cbase + 7),index_c0,c7,1);


            _mm512_i32scatter_ps((void*)(cbase + 8),index_c0,c8,1);
            _mm512_i32scatter_ps((void*)(cbase + 9),index_c0,c9,1);
            _mm512_i32scatter_ps((void*)(cbase + 10),index_c0,c10,1);
            _mm512_i32scatter_ps((void*)(cbase + 11),index_c0,c11,1);

             _mm512_i32scatter_ps((void*)(cbase + 12),index_c0,c12,1);
            _mm512_i32scatter_ps((void*)(cbase + 13),index_c0,c13,1);
            _mm512_i32scatter_ps((void*)(cbase + 14),index_c0,c14,1);
            _mm512_i32scatter_ps((void*)(cbase + 15),index_c0,c15,1);
#endif

           // if(cb == 0 && ca == 0)
               // print_mm512(c0);
        }
    }
  
}
   mkl_free(AT);
//    sgemm3_ref(pTransA, pTransB, pM, pN, pK, pAlpha, A, plda, B, pldb, pBeta, C, pldc);
}
#if 1
void sgemm3_ref( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc);
void sgemm3_opt_no_transpose( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *A, const int*plda, const float *B, const int *pldb, const float *pBeta, float *C, const int*pldc)
{
    int m = *pM;
    int n = *pN;
    int k = *pK;
#define CB_ITER 32
#define CA_ITER 16
#define RA_ITER 4
#define AL 512*4 //(length of A)

#define TNUM (64)
#pragma omp parallel num_threads(TNUM)
  {
    int tid = omp_get_thread_num();
    __m128 *buffer = (__m128 *)mkl_malloc( 4*64, 64 );
    float *buffer_fp = (float *)buffer;
    int block_num = n/TNUM;
    //printf("tid, %d, block_num %d   ", tid, block_num);
    for(int cb = tid * block_num; cb < (tid +1)*block_num; cb = cb + CB_ITER){
//    for(int cb = 0; cb < n; cb = cb + CB_ITER) { //loop for row of b, 35820, 4 AVX512 (64 FP) per iter

        for(int ca = 0; ca < m; ca = ca + CA_ITER) {//loop for row of a, 500, 4 FP per iter
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
             __m512 c30 = _mm512_setzero_ps();
             __m512 c31 = _mm512_setzero_ps();
             
            for(int ra = 0; ra < k; ra = ra + RA_ITER) {      //ra = rb = 64,   k
                 float *abase = (float *)(A + ra * m + ca);  //fetch ?
                 float *bbase = (float *)(B + ra * n + cb);            //fetch 4 AVX512
                 float *bbase2 = (float *)(B + ra * n + cb + 16);            //fetch 4 AVX512
                 __m512 a0 = _mm512_load_ps(abase);
                 __m512 a1 = _mm512_load_ps(abase + m);
                 __m512 a2 = _mm512_load_ps(abase + m*2);
                 __m512 a3 = _mm512_load_ps(abase + m*3);
                 
                 __m512i index_a0 = _mm512_set_epi32(60,56,52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0);
                
                 _mm512_i32scatter_ps((void*)buffer_fp,index_a0,a0,4);
                 _mm512_i32scatter_ps((void*)(buffer_fp +1),index_a0,a1,4);
                 _mm512_i32scatter_ps((void*)(buffer_fp + 2),index_a0,a2,4);
                 _mm512_i32scatter_ps((void*)(buffer_fp + 3),index_a0,a3,4);

                 __m512 b00 = _mm512_load_ps(bbase);
                 __m512 b10 = _mm512_load_ps(bbase + n);
                 __m512 b20 = _mm512_load_ps(bbase + n*2);
                 __m512 b30 = _mm512_load_ps(bbase + n*3);

                 __m512 b01 = _mm512_load_ps(bbase2);
                 __m512 b11 = _mm512_load_ps(bbase2 + n);
                 __m512 b21 = _mm512_load_ps(bbase2 + n*2);
                 __m512 b31 = _mm512_load_ps(bbase2 + n*3);
                 
                c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, buffer); 
                c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, buffer+1); 
                c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, buffer+2); 
                c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, buffer+3); 

                c4 = _mm512_4fmadd_ps(c4, b00, b10, b20, b30, buffer + 4); 
                c5 = _mm512_4fmadd_ps(c5, b00, b10, b20, b30, buffer+5); 
                c6 = _mm512_4fmadd_ps(c6, b00, b10, b20, b30, buffer+6); 
                c7 = _mm512_4fmadd_ps(c7, b00, b10, b20, b30, buffer+7); 

                c8 = _mm512_4fmadd_ps(c8, b00, b10, b20, b30, buffer + 8); 
                c9 = _mm512_4fmadd_ps(c9, b00, b10, b20, b30, buffer+9); 
                c10 = _mm512_4fmadd_ps(c10, b00, b10, b20, b30, buffer+10); 
                c11 = _mm512_4fmadd_ps(c11, b00, b10, b20, b30, buffer+11); 

                c12 = _mm512_4fmadd_ps(c12, b00, b10, b20, b30, buffer +12); 
                c13 = _mm512_4fmadd_ps(c13, b00, b10, b20, b30, buffer+13); 
                c14 = _mm512_4fmadd_ps(c14, b00, b10, b20, b30, buffer+14); 
                c15 = _mm512_4fmadd_ps(c15, b00, b10, b20, b30, buffer+15); 

                c16 = _mm512_4fmadd_ps(c16, b01, b11, b21, b31, buffer); 
                c17 = _mm512_4fmadd_ps(c17, b01, b11, b21, b31, buffer+1); 
                c18 = _mm512_4fmadd_ps(c18, b01, b11, b21, b31, buffer+2); 
                c19 = _mm512_4fmadd_ps(c19, b01, b11, b21, b31, buffer + 3);
                
                c20 = _mm512_4fmadd_ps(c20, b01, b11, b21, b31, buffer+4); 
                c21 = _mm512_4fmadd_ps(c21, b01, b11, b21, b31, buffer+5); 
                c22 = _mm512_4fmadd_ps(c22, b01, b11, b21, b31, buffer+6); 
                c23 = _mm512_4fmadd_ps(c23, b01, b11, b21, b31, buffer + 7);
                
                c24 = _mm512_4fmadd_ps(c24, b01, b11, b21, b31, buffer+8); 
                c25 = _mm512_4fmadd_ps(c25, b01, b11, b21, b31, buffer+9); 
                c26 = _mm512_4fmadd_ps(c26, b01, b11, b21, b31, buffer+10); 
                c27 = _mm512_4fmadd_ps(c27, b01, b11, b21, b31, buffer +11);
                
                c28 = _mm512_4fmadd_ps(c28, b01, b11, b21, b31, buffer+12); 
                c29 = _mm512_4fmadd_ps(c29, b01, b11, b21, b31, buffer+13); 
                c30 = _mm512_4fmadd_ps(c30, b01, b11, b21, b31, buffer+14); 
                c31 = _mm512_4fmadd_ps(c31, b01, b11, b21, b31, buffer+15); 
            }
           
            float *cbase = C + cb*m + ca;
             __m512i index_c0 = _mm512_set_epi32(15*AL,14*AL, 13*AL, 12*AL, 11*AL,10*AL,9*AL,8*AL,7*AL
                                                                             ,6*AL, 5*AL,4*AL,3*AL,2*AL,1*AL,0);

            _mm512_i32scatter_ps((void*)cbase,index_c0,c0,1);
            _mm512_i32scatter_ps((void*)(cbase + 1),index_c0,c1,1);
            _mm512_i32scatter_ps((void*)(cbase + 2),index_c0,c2,1);
            _mm512_i32scatter_ps((void*)(cbase + 3),index_c0,c3,1);

             _mm512_i32scatter_ps((void*)(cbase + 4),index_c0,c4,1);
            _mm512_i32scatter_ps((void*)(cbase + 5),index_c0,c5,1);
            _mm512_i32scatter_ps((void*)(cbase + 6),index_c0,c6,1);
            _mm512_i32scatter_ps((void*)(cbase + 7),index_c0,c7,1);

            _mm512_i32scatter_ps((void*)(cbase + 8),index_c0,c8,1);
            _mm512_i32scatter_ps((void*)(cbase + 9),index_c0,c9,1);
            _mm512_i32scatter_ps((void*)(cbase + 10),index_c0,c10,1);
            _mm512_i32scatter_ps((void*)(cbase + 11),index_c0,c11,1);

             _mm512_i32scatter_ps((void*)(cbase + 12),index_c0,c12,1);
            _mm512_i32scatter_ps((void*)(cbase + 13),index_c0,c13,1);
            _mm512_i32scatter_ps((void*)(cbase + 14),index_c0,c14,1);
            _mm512_i32scatter_ps((void*)(cbase + 15),index_c0,c15,1);

            float *cbase2 = C + (cb + 16)*m + ca;
             _mm512_i32scatter_ps((void*)cbase2,index_c0,c16,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 1),index_c0,c17,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 2),index_c0,c18,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 3),index_c0,c19,1);

             _mm512_i32scatter_ps((void*)(cbase2 + 4),index_c0,c20,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 5),index_c0,c21,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 6),index_c0,c22,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 7),index_c0,c23,1);

            _mm512_i32scatter_ps((void*)(cbase2 + 8),index_c0,c24,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 9),index_c0,c25,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 10),index_c0,c26,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 11),index_c0,c27,1);

            _mm512_i32scatter_ps((void*)(cbase2 + 12),index_c0,c28,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 13),index_c0,c29,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 14),index_c0,c30,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 15),index_c0,c31,1);

           // if(cb == 0 && ca == 0)
               // print_mm512(c0);
        }

    }
    mkl_free(buffer_fp);
}
//    sgemm3_ref(pTransA, pTransB, pM, pN, pK, pAlpha, A, plda, B, pldb, pBeta, C, pldc);
}
#endif

#if 0
void sgemm3_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *A, const int*plda, const float *B, const int *pldb, const float *pBeta, float *C, const int*pldc)
{
    int m = *pM;
    int n = *pN;
    int k = *pK;
    for(int cb = 0; cb < n; cb = cb + 64) { //loop for row of b, 35820, 4 AVX512 (64 FP) per iter

        for(int ca = 0; ca < m; ca = ca + 64) {//loop for row of a, 500, 4 FP per iter
             __m512 c0 = _mm512_setzero_ps();
             __m512 c1 = _mm512_setzero_ps();
             __m512 c2 = _mm512_setzero_ps();
             __m512 c3 = _mm512_setzero_ps();
            for(int ra = 0; ra < k; ra = ra++) {      //64,   k
                 float *abase = (float *)(A + ra * m + ca);  //fetch 4 AVX512
                 float *bbase = (float *)(B + ra * n + cb);            //fetch 4 AVX512

                 __m512 a00 = _mm512_load_ps(abase);
                 __m512 b00 = _mm512_load_ps(bbase);
                 c0 = _mm512_fmadd_ps(a00, b00, c0);           
                 
                 __m512 a01 = _mm512_load_ps(abase + 16);
                 __m512 b01 = _mm512_load_ps(bbase + 16);
                 c1 = _mm512_fmadd_ps(a01, b01, c1);    
                 
                 __m512 a02 = _mm512_load_ps(abase + 32);
                 __m512 b02 = _mm512_load_ps(bbase + 32);
                 c2 = _mm512_fmadd_ps(a02, b02, c2);   
                 
                 __m512 a03 = _mm512_load_ps(abase + 48);
                 __m512 b03 = _mm512_load_ps(bbase + 48);
                 c3 = _mm512_fmadd_ps(a03, b03, c3);
    
            }
            float *cbase1 = C + cb*m + ca;
            float *cbase2 = C + (cb + 16)*m + ca + 16;
            float *cbase3 = C + (cb + 32)*m + ca + 32;
            float *cbase4 = C + (cb + 48)*m + ca + 48;
            __m512i index = _mm512_set1_ps(m);
            #if 0
            _mm512_i32scatter_ps((void*)cbase1,index,c0,1);
            _mm512_i32scatter_ps((void*)cbase2,index,c1,1);
            _mm512_i32scatter_ps((void*)cbase3,index,c2,1);
            _mm512_i32scatter_ps((void*)cbase4,index,c3,1);
            #endif
            if(cb == 0 && ca == 0)
                print_mm512(c0);
            //__mm512_store_ps(C + ram* n + cbn, c0_15);
        }

    }

    sgemm3_ref(pTransA, pTransB, pM, pN, pK, pAlpha, A, plda, B, pldb, pBeta, C, pldc);
}
#endif


#if 1
void sgemm3_ref( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc)
{
    printf("sgemm3_opt start \n");

  const int M = *pM;
  const int N = *pN;
  const int K = *pK;
  const int lda = *plda;
  const int ldb = *pldb;
  const int ldc = *pldc;
  float alpha = *pAlpha;
  float beta = *pBeta;
  const int row_per_thread = M/THREAD_NUM;
  int i,j,l;

    float * a_ = pa;
    for(i = 0; i < M; i++)
      {
        float *b_ = pb;
        for(j = 0; j < N; j++)
        {
          float sum = 0;
          for(l = 0; l < K; l =l +4) {
                 sum += a_[l*lda]*b_[l*ldb];
                 sum += a_[(l +1)*lda]*b_[(l+1)*ldb];
                 sum += a_[(l + 2)*lda]*b_[(l+2)*ldb];
                 sum += a_[(l + 3)*lda]*b_[(l+3)*ldb];
                 //if(i=0 && j==0)
                  //  printf("sum , %f , " , sum);
            }
          b_++;
          if (beta == 0)
            pc[j*ldc+i] = alpha*sum;
          else
            pc[j*ldc+i] = beta*pc[j*ldc+i]+alpha*sum;
         // if(i==0 && j ==0)
          //    printf("[%d, %d]: %f,  ", i, j, pc[j*ldc+i]);               
        }
        a_++;
      }
}
#endif
