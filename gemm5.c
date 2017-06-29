
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
#include "mkl.h"
#include "immintrin.h"
#include<math.h>



extern float * buffer;

void sgemm5_test( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *A, const int*plda, const float *B, const int *pldb, const float *pBeta, float *C, const int*pldc)
{
    int m = *pM;
    int n = *pN;
    int k = *pK;

#define CB_ITER 32
#define RA_ITER 4
#define CA_ITER 4
//#define AL 32*4 //(length of A)
    int AL = m*4;

#define TNUM (64)
#pragma omp parallel num_threads(TNUM)
  {
    int tid = omp_get_thread_num();
    int block_num = n/TNUM;
    //printf("tid, %d, block_num %d   ", tid, block_num);
    for(int cb = tid * block_num; cb < (tid +1)*block_num; cb = cb + CB_ITER){
//    for(int cb = 0; cb < n; cb = cb + CB_ITER) { //loop for column of b, 35820, 4 AVX512 (64 FP) per iter

        for(int ra = 0; ra < m; ra = ra + RA_ITER) {//loop for row of a, 500, 4 line 4 FP per iter
             __m512 c0 = _mm512_setzero_ps();
             __m512 c1 = _mm512_setzero_ps();
             __m512 c2 = _mm512_setzero_ps();
             __m512 c3 = _mm512_setzero_ps();

             __m512 c4 = _mm512_setzero_ps();
             __m512 c5 = _mm512_setzero_ps();
             __m512 c6 = _mm512_setzero_ps();
             __m512 c7 = _mm512_setzero_ps();
#if 0
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
#endif            
            for(int ca = 0; ca < k; ca = ca + CA_ITER) {      //ca = rb = 64,   k
                 float *abase = (float *)(A + ra * k + ca);  //fetch ?
                 float *bbase = (float *)(B + ca * n + cb);            //fetch 4 AVX512
                 float *bbase2 = (float *)(B + ca * n + cb + 16);            //fetch 4 AVX512

                 __m512 b00 = _mm512_load_ps(bbase);
                 __m512 b10 = _mm512_load_ps(bbase + n);
                 __m512 b20 = _mm512_load_ps(bbase + n*2);
                 __m512 b30 = _mm512_load_ps(bbase + n*3);

                 __m512 b01 = _mm512_load_ps(bbase2);
                 __m512 b11 = _mm512_load_ps(bbase2 + n);
                 __m512 b21 = _mm512_load_ps(bbase2 + n*2);
                 __m512 b31 = _mm512_load_ps(bbase2 + n*3);
                
                 __m128 *a00 = (__m128 *)(abase); 
		         __m128 *a10 = (__m128 *)(abase + k);
	             __m128 *a20 = (__m128 *)(abase + k*2);
		         __m128 *a30 = (__m128 *)(abase + k*3);

                c0 = _mm512_4fmadd_ps(c0, b00, b10, b20, b30, a00); 
                c1 = _mm512_4fmadd_ps(c1, b00, b10, b20, b30, a10); 
                c2 = _mm512_4fmadd_ps(c2, b00, b10, b20, b30, a20); 
                c3 = _mm512_4fmadd_ps(c3, b00, b10, b20, b30, a30); 

                c4 = _mm512_4fmadd_ps(c4, b01, b11, b21, b31, a00); 
                c5 = _mm512_4fmadd_ps(c5, b01, b11, b21, b31, a10); 
                c6 = _mm512_4fmadd_ps(c6, b01, b11, b21, b31, a20); 
                c7 = _mm512_4fmadd_ps(c7, b01, b11, b21, b31, a30); 
#if 0
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
#endif
            }
           
            float *cbase = C + cb*m + ra;
            float *cbase2 = C + (cb + 16)*m + ra;
             __m512i index_c0 = _mm512_set_epi32(15*AL,14*AL, 13*AL, 12*AL, 11*AL,10*AL,9*AL,8*AL,7*AL
                                                 ,6*AL, 5*AL,4*AL,3*AL,2*AL,1*AL,0);

            _mm512_i32scatter_ps((void*)cbase,index_c0,c0,1);
            _mm512_i32scatter_ps((void*)(cbase + 1),index_c0,c1,1);
            _mm512_i32scatter_ps((void*)(cbase + 2),index_c0,c2,1);
            _mm512_i32scatter_ps((void*)(cbase + 3),index_c0,c3,1);

            _mm512_i32scatter_ps((void*)(cbase2),index_c0,c4,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 1),index_c0,c5,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 2),index_c0,c6,1);
            _mm512_i32scatter_ps((void*)(cbase2 + 3),index_c0,c7,1);
#if 0
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
#endif
        }

    }
  }
}


void sgemm5_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc)
{
  if(buffer == NULL)
  {
     buffer = malloc((*pK)*(*pM)*sizeof(float));
  }
  mkl_somatcopy('r','t', *pK, *pM, 1.0, pa, *pM, buffer, *pK);
  sgemm5_test(pTransA,pTransB,pM,pN,pK,pAlpha,buffer,pK,pb,pldb,pBeta,pc,pldc);
}
