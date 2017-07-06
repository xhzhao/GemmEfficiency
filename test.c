
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
#include "mkl.h"
#include "immintrin.h"
#include<math.h>

#define SGEMM_COUNT  (1)		    // every sgemm iteration numbers
#define BUFFER_COUNT 100		    // cause cache miss manaully
#define HW_GFLOPS   3097 

//get the system time in ms
double get_time(void)
{
    struct timeval start;
    gettimeofday(&start,NULL);
    double time = start.tv_sec * 1000 + start.tv_usec /1000;
    return time; 
}

extern void sgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm1_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm3_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm4_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm5_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm11_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm12_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);

// profile for one type of sgemm, 50 iterations
void sgemm_opt(int index, char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    //define function pointer
    void (* sgemm_pcall)(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
   
    switch(index){
        case 1:
            sgemm_pcall = sgemm1_opt;break;
        case 3:
            sgemm_pcall = sgemm3_opt;break;
        case 4:
            sgemm_pcall = sgemm4_opt;break;
        case 5:
            sgemm_pcall = sgemm5_opt;break;
        case 11:
            sgemm_pcall = sgemm11_opt;break;
        case 12:
            sgemm_pcall = sgemm12_opt;
        default:
            sgemm_pcall = sgemm_;break;
    } 

    double t0 = get_time();
    for(i=0; i < SGEMM_COUNT; i++)
    {
        sgemm_pcall(pTransA, pTransB, pM, pN, pK, pAlpha, pa, plda, pb, pldb, pBeta, pc, pldc);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_profile end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
}

// profile for one type of sgemm, 50 iterations
void sgemm_mkl(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    double t0 = get_time();
    for(i=0; i < SGEMM_COUNT; i++)
    {
        sgemm_(pTransA, pTransB, pM, pN, pK, pAlpha, pa, plda, pb, pldb, pBeta, pc, pldc);
        //cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_mkl end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
}

float* matrix_init(int A, int B)
{
    float * p = mkl_malloc(A*B*sizeof(float), 1024*1024*2);
    int a,b;
#pragma omp parallel for
    for(a=0; a < A; a++)
        for(b=0; b < B;b++)
            p[a*B+b] = (float)(rand() % 1000)/100; 
    return p;
}

void verify_result(float *c, float *c_mkl, int M, int N)
{
    int i, j;

    for(i=0; i < M; i++)
        for(j=0; j < N;j++){
            if(fabs((c[i*N+j] -c_mkl[i*N + j])) > 0.01) {
                if(fabs(c[i*N+j] -c_mkl[i*N + j])/ fabs(c_mkl[i*N + j]) > 0.01 ){
                    printf("result mismatch, i = %d, j = %d, a = %.4f, b = %.4f\n", i,j,c[i*N+j],c_mkl[i*N + j]);
                    return;
                }
            }

        }
    printf("verify result OK.\n");
}


void sgemm_main_cahceMiss(int index, char transa, char transb, int M, int N, int K, int lda, float alpha, int ldb, float beta, int ldc)
{

    float * a[BUFFER_COUNT] = 0;
    float * b[BUFFER_COUNT] = 0;
    float * c[BUFFER_COUNT] = 0;
    int i = 0;
    int j = 0;
    for(i = 0; i < BUFFER_COUNT; i++)
    {
        a[i] = matrix_init(M,K);
        b[i] = matrix_init(K,N);
        c[i] = matrix_init(M,N);
    }
    printf("----------GEMM %d----------\n", index);
    float f_M = M;
    float f_N = N;
    float f_K = K;
    double gflops = (f_M*f_N*f_K*2 + 2*f_M*f_N ) * (1e-6);
    double t0 = get_time();
    for(i=0; i < SGEMM_COUNT; i++)
    {
        j = i%BUFFER_COUNT;
        sgemm_(&transa, &transb, &M, &N, &K, &alpha, a[j], &lda, b[j], &ldb, &beta, c[j], &ldc);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_mkl cacheMiss, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
    for(i = 0; i < BUFFER_COUNT; i++)
    {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }

 

}

void sgemm_main(int index, char transa, char transb, int M, int N, int K, int lda, float alpha, int ldb, float beta, int ldc)
{
    float * a = matrix_init(M,K);
    float * b = matrix_init(K,N);
    float * c = matrix_init(M,N);
    float *a_mkl = malloc(M*K*sizeof(float));
    memcpy(a_mkl, a, M*K*sizeof(float));

     float *b_mkl = malloc(K*N*sizeof(float));
    memcpy(b_mkl, b, K*N*sizeof(float));
    
    float *c_mkl = malloc(M*N*sizeof(float));
    memcpy(c_mkl, c, M*N*sizeof(float));
    
    printf("----------GEMM %d----------\n", index);
    sgemm_mkl(&transa, &transb, &M, &N, &K, &alpha, a_mkl, &lda, b_mkl, &ldb, &beta, c_mkl, &ldc);
    sgemm_opt(index,&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);




   verify_result(c, c_mkl, M, N);

        
    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
}


int main(void)
{
    printf("main start \n");
    char transa, transb;
    int m,n,k,lda,ldb,ldc;
    float alpha,beta;






#if 0
    transa='t'; transb='n'; m=35820; n=64; k=500; lda=500; alpha=1.0000; ldb=500; beta=0.0000; ldc=35820;
    sgemm_main(1, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=500; n=64; k=35820; lda=500; alpha=1.0000; ldb=35820; beta=0.0000; ldc=500;
    sgemm_main(2, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='t'; m=500; n=35820; k=64; lda=500; alpha=1.0000; ldb=35820; beta=1.0000; ldc=500;
    sgemm_main(3, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=500; n=64; k=2000; lda=500; alpha=1.0000; ldb=2000; beta=0.0000; ldc=500;
    sgemm_main(4, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='t'; m=500; n=2000; k=64; lda=500; alpha=1.0000; ldb=2000; beta=1.0000; ldc=500;
    sgemm_main(5, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='t'; transb='n'; m=2000; n=64; k=500; lda=500; alpha=1.0000; ldb=500; beta=0.0000; ldc=2000;
    sgemm_main(6, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=1000; n=64; k=2000; lda=1000; alpha=1.0000; ldb=2000; beta=0.0000; ldc=1000;
    sgemm_main(7, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='t'; m=1000; n=2000; k=64; lda=1000; alpha=1.0000; ldb=2000; beta=1.0000; ldc=1000;
    sgemm_main(8, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=500; n=29; k=35820; lda=500; alpha=1.0000; ldb=35820; beta=0.0000; ldc=500;
    sgemm_main(9, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='t'; transb='n'; m=2000; n=64; k=1000; lda=1000; alpha=1.0000; ldb=1000; beta=0.0000; ldc=2000;
    sgemm_main(10, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
#endif

#if 0
    transa='t'; transb='n'; m=35820; n=64; k=500; lda=500; alpha=1.0000; ldb=500; beta=0.0000; ldc=35820;
    sgemm_main_cahceMiss(1, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=500; n=64; k=35820; lda=500; alpha=1.0000; ldb=35820; beta=0.0000; ldc=500;
    sgemm_main_cahceMiss(2, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='t'; m=500; n=35820; k=64; lda=500; alpha=1.0000; ldb=35820; beta=1.0000; ldc=500;
    sgemm_main_cahceMiss(3, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=500; n=64; k=2000; lda=500; alpha=1.0000; ldb=2000; beta=0.0000; ldc=500;
    sgemm_main_cahceMiss(4, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='t'; m=500; n=2000; k=64; lda=500; alpha=1.0000; ldb=2000; beta=1.0000; ldc=500;
    sgemm_main_cahceMiss(5, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='t'; transb='n'; m=2000; n=64; k=500; lda=500; alpha=1.0000; ldb=500; beta=0.0000; ldc=2000;
    sgemm_main_cahceMiss(6, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=1000; n=64; k=2000; lda=1000; alpha=1.0000; ldb=2000; beta=0.0000; ldc=1000;
    sgemm_main_cahceMiss(7, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='t'; m=1000; n=2000; k=64; lda=1000; alpha=1.0000; ldb=2000; beta=1.0000; ldc=1000;
    sgemm_main_cahceMiss(8, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n'; transb='n'; m=500; n=29; k=35820; lda=500; alpha=1.0000; ldb=35820; beta=0.0000; ldc=500;
    sgemm_main_cahceMiss(9, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='t'; transb='n'; m=2000; n=64; k=1000; lda=1000; alpha=1.0000; ldb=1000; beta=0.0000; ldc=2000;
    sgemm_main_cahceMiss(10, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);

#endif

#if 0

    transa='n'; transb='n';

    m=500; n=1; k=25; lda=500; alpha=1.0000; ldb=25; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=20; lda=500; alpha=1.0000; ldb=20; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=21; lda=500; alpha=1.0000; ldb=21; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=18; lda=500; alpha=1.0000; ldb=18; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=30; lda=500; alpha=1.0000; ldb=30; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=11; lda=500; alpha=1.0000; ldb=11; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=15; lda=500; alpha=1.0000; ldb=15; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=24; lda=500; alpha=1.0000; ldb=24; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=31; lda=500; alpha=1.0000; ldb=31; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=29; lda=500; alpha=1.0000; ldb=29; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=41; lda=500; alpha=1.0000; ldb=41; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=33; lda=500; alpha=1.0000; ldb=33; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=38; lda=500; alpha=1.0000; ldb=38; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=39; lda=500; alpha=1.0000; ldb=39; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=28; lda=500; alpha=1.0000; ldb=28; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=34; lda=500; alpha=1.0000; ldb=34; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=27; lda=500; alpha=1.0000; ldb=27; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=26; lda=500; alpha=1.0000; ldb=26; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=23; lda=500; alpha=1.0000; ldb=23; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=9;  lda=500; alpha=1.0000; ldb=9;  beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=17; lda=500; alpha=1.0000; ldb=17; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=40; lda=500; alpha=1.0000; ldb=40; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=14; lda=500; alpha=1.0000; ldb=14; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=12; lda=500; alpha=1.0000; ldb=12; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    m=500; n=1; k=16; lda=500; alpha=1.0000; ldb=16; beta=0.0000; ldc=500;
    sgemm_main(11, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);

#endif

#if 0
    transa='n', transb='n', m=500, n=25, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=20, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=21, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=30, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=11, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=41, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=18, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=15, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=31, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=24, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=29, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=38, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=33, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=39, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=34, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=28, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=40, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=9, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=27, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=26, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=23, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=17, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=14, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=12, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=16, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    transa='n', transb='n', m=500, n=10, k=1, lda=500, alpha=1.0000, ldb=1, beta=0.0000, ldc=500;
    sgemm_main(12, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
#endif
    return 1;
}
