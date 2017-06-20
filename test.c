
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include "omp.h"
//#include "mkl.h"
#include "immintrin.h"
#include<math.h>

#define SGEMM_COUNT 500		    // every sgemm iteration numbers
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
extern void sgemm3_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm4_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
extern void sgemm5_opt(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);


// profile for one type of sgemm, 50 iterations
void sgemm_opt(int index, char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    double t0 = get_time();
    //define function pointer
    void (* sgemm_pcall)(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
   
    switch(index){
        case 3:
            sgemm_pcall = sgemm3_opt;break;
        case 4:
            sgemm_pcall = sgemm4_opt;break;
        case 5:
            sgemm_pcall = sgemm5_opt;break;
    } 

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
        
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_mkl end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
}

float* matrix_init(int A, int B)
{
    float * p = malloc(A*B*sizeof(float));
    int a,b;
    for(a=0; a < A; a++)
        for(b=0; b < B;b++)
            p[a*B+b] = rand() % 1000; 
    return p;
}

void verify_result(float *c, float *c_mkl, int M, int N)
{
    int i, j;
    for(i=0; i < M; i++)
        for(j=0; j <=N;j++){
            if(fabs((c[i*N+j] -c_mkl[i*N + j])) > 0.01) {
                printf("result mismatch\n");
                return;
            }

        }
    return;
}
void sgemm_main(int index, char transa, char transb, int M, int N, int K, int lda, float alpha, int ldb, float beta, int ldc)
{
    float * a = matrix_init(M,K);
    float * b = matrix_init(K,N);
    float * c = matrix_init(M,N);
    float *a_mkl = malloc(M*N*sizeof(float));
    memcpy(a_mkl, a, M*N*sizeof(float));

     float *b_mkl = malloc(K*N*sizeof(float));
    memcpy(b_mkl, b, K*N*sizeof(float));
    
    float *c_mkl = malloc(M*N*sizeof(float));
    memcpy(c_mkl, c, M*N*sizeof(float));
    
    printf("----------GEMM %d----------\n", index);
    sgemm_opt(index,&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    sgemm_mkl(&transa, &transb, &M, &N, &K, &alpha, a_mkl, &lda, b_mkl, &ldb, &beta, c_mkl, &ldc);

   verify_result(c, c_mkl, M, N);

        
    free(a);
    free(b);
    free(c);
}


int main(void)
{
    printf("main start \n");
    char transa, transb;
    int m,n,k,lda,ldb,ldc;
    float alpha,beta;
    transa='n'; transb='n'; m=500; n=64; k=35820; lda=500; alpha=1.0000; ldb=35820; beta=0.0000; ldc=500;
    sgemm_main(2, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);

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


    return 1;
}
