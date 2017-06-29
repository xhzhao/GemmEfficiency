
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

#if 1
void sgemm3_ref( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc);
void sgemm3_opt( char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *A, const int*plda, const float *B, const int *pldb, const float *pBeta, float *C, const int*pldc)
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
