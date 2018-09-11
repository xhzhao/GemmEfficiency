
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include <pthread.h>
#include <omp.h>
#include<math.h>
#include <iostream>
#include <cstdint>
#include <cstring>

#define WARM_UP     10
#define SGEMM_COUNT 10000   // every sgemm iteration numbers
#define USE_VAR     1


extern int my_sgemm(char transa, char transb,
  const int M, const int N, const int K,
  const float alpha, const float* A, const int lda,
  const float * B, const int ldb, 
  const float beta, float * C, const int ldc);

float* matrix_init(int A, int B);
//get the system time in ms
double get_time(void)
{
#if 1
    struct timeval start;
    gettimeofday(&start,NULL);
    double time = start.tv_sec * 1000 + start.tv_usec /1000;
    return time; 
#else
    double time = dsecnd() * 1000;
    return time;
#endif
}

extern void sgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);


inline void variance(int M, int N, float * a, float * b, float *c){
    int i, j;
    #pragma omp for nowait collapse(1)
    for(i = 0; i < M; i++){
        #pragma omp simd
        for(j =0; j < N; j++){
            c[i * N + j] = (a[i * N + j] * 1.01) * (b[i * N + j] + 0.001);
        }
    }
}


void bench_var(int M, int N, float * a, float * b, float *c){
    
    double t0 = 0;
    int t;
    for(t=0; t < SGEMM_COUNT + WARM_UP; t++)
    {
        if (t == WARM_UP){
            t0 = get_time();
        }
        #pragma omp parallel num_threads(40)
        variance(M, N, a, b, c);

    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("bench_var end, avg time = %.2f\n", avg_time);
}

void bench_var_2omp(int M, int N, float * a, float * b, float *c){
    
    double t0 = 0;
    int t,d;
    for(t=0; t < SGEMM_COUNT + WARM_UP; t++)
    {
        if (t == WARM_UP){
            t0 = get_time();
        }
        omp_set_nested(1);
        #pragma omp parallel for num_threads(2) proc_bind(spread)
        for(d = 0; d < 2; d++)
        {

            //printf("d = %d, tid = %d \n", d, omp_get_thread_num());
            if(d == 0){
              #pragma omp parallel num_threads(20) proc_bind(close)
              omp_set_num_threads(20);
              variance(M/2, N, a + d * M * N /2, b + d * M * N /2, c + d * M * N /2);
            } 

        }
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("\n bench_var_2omp end, avg time = %.2f\n", avg_time);
}

/*
void bench_var_2omp2(int M, int N, float * a1, float * b1, float *c1,
        float * a2, float * b2, float * c2){
    
    double t0 = 0;
    int t,d;
    for(t=0; t < SGEMM_COUNT + WARM_UP; t++)
    {
        if (t == WARM_UP){
            t0 = get_time();
        }
        omp_set_nested(1);
        #pragma omp parallel for num_threads(2) proc_bind(spread)
        for(d = 0; d < 2; d++){

            omp_set_num_threads(20);
            if(d == 0){
                variance(M, N, a1, b1, c1);
            }else{
                variance(M, N, a2, b2, c2);
            }
        }
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("bench_var_2omp2 end, avg time = %.2f\n", avg_time);
}
*/

// profile for one type of sgemm, 50 iterations
void sgemm_profile(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    CBLAS_TRANSPOSE  transa = CblasNoTrans;
    CBLAS_TRANSPOSE  transb = CblasNoTrans;

    if( *pTransB == 't'){
        transb = CblasTrans;
    }
    double t0 = 0;
    for(i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        //sgemm_(pTransA, pTransB, pM, pN, pK, pAlpha, pa, plda, pb, pldb, pBeta, pc, pldc);
        cblas_sgemm(CblasRowMajor, transa, transb, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_profile end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
}

void sgemm_profile_pack(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    float * Ap = cblas_sgemm_alloc(CblasAMatrix, M, N, K);
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, M, N, K, 1.0, pa, *plda, Ap);

    double t0 = 0;
    for(i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        //sgemm_(pTransA, pTransB, pM, pN, pK, pAlpha, pa, plda, pb, pldb, pBeta, pc, pldc);
        //cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
        if (i == WARM_UP){
            t0 = get_time();
        }
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, M, N, K, Ap, K, pb, N, 1.0, pc, N);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_profile_pack end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
    cblas_sgemm_free(Ap);
}

void sgemm_profile_batch(int batchsize, char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const int*plda, const int *pldb, const float *pBeta, const int*pldc) {
    int m[1] = {*pM};
    int n[1] = {*pN};
    int k[1] = {*pK};
        
    int lda[1] = {*plda};
    int ldb[1] = {*pldb};
    int ldc[1] = {*pldc};
        
    CBLAS_TRANSPOSE transA[1];
    CBLAS_TRANSPOSE transB[1];
    transA[0] = (*pTransA == 'n')?CblasNoTrans:CblasTrans;
    transB[0] = (*pTransB == 'n')?CblasNoTrans:CblasTrans;
        
    float alpha[1] = {*pAlpha}; 
    float beta[1]  = {*pBeta};
    int size_per_grp[1] = {batchsize};

    //init a,b,c
    float ** A = (float **) malloc(batchsize * sizeof(float *));
    float ** B = (float **) malloc(batchsize * sizeof(float *));
    float ** C = (float **) malloc(batchsize * sizeof(float *));
    int i = 0;
    for(i=0; i<batchsize; i++){
        A[i] = matrix_init(*pM, *pK);
        B[i] = matrix_init(*pK, *pN);
        C[i] = matrix_init(*pM, *pN);
    }
    float M = *pM;                                                              
    float N = *pN;                                                              
    float K = *pK;
    double gflops = batchsize * (M*N*K*2 + 2*M*N ) * (1e-6);

    double t0 = 0;                                                     
    for(i=0; i < SGEMM_COUNT + WARM_UP; i++)                                              
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, (const float**)A, lda, (const float **)B, ldb, beta, C, ldc, 1, size_per_grp);
    }                                                                           
    double t1 = get_time() - t0;                                                
    double avg_time = t1/SGEMM_COUNT; 
    printf("sgemm_profile_batch end, avg time = %.2f, GFLOPS = %.2f \n", avg_time, gflops/avg_time);

    
    for(i=0; i<batchsize; i++){
       mkl_free(A[i]); 
       mkl_free(B[i]); 
       mkl_free(C[i]); 
    }
}

typedef struct gemm_para{
    char* pTransA;
    char* pTransB;
    const int* pM;
    const int* pN;
    const int* pK;
    const float* pAlpha;
    const float* pa;
    const int* plda;
    const float* pb;
    const int* pldb;
    const float* pBeta;
    float *pc;
    const int* pldc;
}GemmPara;

#define HALF_OMP_THREADS 20

void* thread_gemm(void * para){

    //printf("para = 0x%x \n", para);

    GemmPara p = *((GemmPara*)para);

    char* pTransA = p.pTransA;
    char* pTransB = p.pTransB;
    const int* pM = p.pM;
    const int* pN = p.pN;
    const int* pK = p.pK;
    const float* pAlpha = p.pAlpha;
    const float* pa = p.pa;
    const int* plda = p.plda;
    const float* pb = p.pb;
    const int* pldb = p.pldb;
    const float* pBeta = p.pBeta;
    float *pc = p.pc;
    const int* pldc = p.pldc;

    //printf("pTransA = 0x%x, pTransB = 0x%x, pM = 0x%x, pN = 0x%x, pK = 0x%x, pAlpha = 0x%x, pa = 0x%x, plda = 0x%x, \
    //    pb = 0x%x, pldb = 0x%x, pBeta = 0x%x, pc = 0x%x, pldc = 0x%x\n",pTransA,pTransB,pM,pN,pK,pAlpha,pa,plda,pb,pldb,pBeta,pc,pldc);

    int i = 0;
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    CBLAS_TRANSPOSE  transa = CblasNoTrans;
    CBLAS_TRANSPOSE  transb = CblasNoTrans;
    mkl_set_num_threads_local(HALF_OMP_THREADS);

    if( *pTransB == 't'){
        transb = CblasTrans;
    }
    double t0 = 0;
    for(i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        cblas_sgemm(CblasRowMajor, transa, transb, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_profile_2pthread end, avg time = %.2f, GFLOPS = %.2f, start = %.4lf, dura = %.4lf\n", avg_time, gflops/avg_time, t0,t1);
    return 0;
}

void sgemm_profile_2pthread(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    // create GemmPara

    GemmPara* para1 = (GemmPara*)mkl_malloc(sizeof(GemmPara), 64);
    //GemmPara para = {pTransA, pTransB, pM, pN, pK, pAlpha, pa, plda, pb, pldb, pBeta, pc, pldc};
    para1->pTransA = pTransA;
    para1->pTransB = pTransB;
    para1->pM = pM;
    para1->pN = pN;
    para1->pK = pK;
    para1->pAlpha = pAlpha;
    para1->pa = pa;
    para1->plda = plda;
    para1->pb = pb;
    para1->pldb = pldb;
    para1->pBeta = pBeta;
    para1->pc = pc;
    para1->pldc = pldc;
    pthread_t thread1, thread2;

    //create memory for the 2nd sgemm
    float * a = matrix_init(*pM,*pK);
    float * b = matrix_init(*pK,*pN);
    float * c = matrix_init(*pM,*pN);

    GemmPara* para2 = (GemmPara*)mkl_malloc(sizeof(GemmPara), 64);
    std::memcpy(para2, para1, sizeof(GemmPara));
    para2->pa = a;
    para2->pb = b;
    para2->pc = c;

    //printf("size of pthread = %d \n", sizeof(pthread_t));
    int err1 = pthread_create(&thread1, NULL, thread_gemm, para1);
    int err2 = pthread_create(&thread2, NULL, thread_gemm, para2);
    if (err1 | err2){
        printf("error in creating thread \n");
    }
    err1 = pthread_join(thread1, NULL);
    err2 = pthread_join(thread2, NULL);
    if (err1 | err2){
        printf("error in joining thread \n");
    }
    
}

void sgemm_profile_2ompthread(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0, d = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;

    //create memory for the 2nd sgemm
    float * a = matrix_init(*pM,*pK);
    float * b = matrix_init(*pK,*pN);
    float * c = matrix_init(*pM,*pN);

    double gflops = 2 * (M*N*K*2 + 2*M*N ) * (1e-6);
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;

    if( *pTransB == 't'){
        transb = CblasTrans;
    }
    double t0 = 0;

    for(i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        omp_set_nested(1);
        #pragma omp parallel for num_threads(2) proc_bind(spread)
        for(d = 0; d < 2; ++d)
        //#pragma omp parallel num_threads(20) proc_bind(close)
        {
            //printf("d = %d, tid = %d \n", d, omp_get_thread_num());
            if (0 == d){
                #pragma omp parallel num_threads(20) proc_bind(close)
                my_sgemm(transa, transb, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
            } else {
                #pragma omp parallel num_threads(20) proc_bind(close)
                my_sgemm(transa, transb, *pM, *pN, *pK, *pAlpha, a, *plda, b, *pldb, *pBeta, c, *pldc);
            }
        }
        omp_set_nested(0);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_profile_2ompthread end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
}

void sgemm_profile_2ompthread_ideal(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0, d = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;

    //create memory for the 2nd sgemm
    float * a = matrix_init(*pM,*pK);
    float * b = matrix_init(*pK,*pN);
    float * c = matrix_init(*pM,*pN);

    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;

    if( *pTransB == 't'){
        transb = CblasTrans;
    }
    omp_set_nested(1);
    double t0 = 0;
    #pragma omp parallel num_threads(2)
    {
        for(i=0; i < SGEMM_COUNT + WARM_UP; i++){
            if (i == WARM_UP){
                t0 = get_time();
            }
            mkl_set_num_threads_local(20);
            mkl_set_dynamic(0);
            if (omp_get_thread_num() == 0){
                #pragma omp parallel num_threads(20)
                my_sgemm(transa, transb, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
            } else {
                #pragma omp parallel num_threads(20)
                my_sgemm(transa, transb, *pM, *pN, *pK, *pAlpha, a, *plda, b, *pldb, *pBeta, c, *pldc);
            }
        }
        double t1 = get_time() - t0;                                               
        double avg_time = t1/SGEMM_COUNT;                                          
        printf("sgemm_profile_2ompthread_ideal end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
    }
}

int mysgemm_checkresult(char stransa, char stransb, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float * B, const int ldb, const float beta, float * C, const int ldc){
    float * c = matrix_init(M, N);
    float * c_mkl = matrix_init(M, N);
    
    CBLAS_TRANSPOSE  transa = CblasNoTrans;
    CBLAS_TRANSPOSE  transb = CblasNoTrans;
    cblas_sgemm(CblasRowMajor, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, c_mkl, ldc);

    #pragma omp parallel num_threads(20)
    {
      my_sgemm(stransa, stransb, M, N, K, alpha, A, lda, B, ldb, beta, c, ldc);
    }

    int i, j;
    for(i=0; i < M; i++)
        for(j=0; j < N;j++){
            if(fabs((c[i*N+j] -c_mkl[i*N + j])) > 0.01) {
                if(fabs(c[i*N+j] -c_mkl[i*N + j])/ fabs(c_mkl[i*N + j]) > 0.01 ){
                    printf("result mismatch, i = %d, j = %d, a = %.4f, b = %.4f\n", i,j,c[i*N+j],c_mkl[i*N + j]);
                    return 0;
                }
            }

        }
    printf("my_sgemm test pass \n");
    return 1;


}

// profile for one type of sgemm, 50 iterations
void sgemm_profile_mysgemm(char* pTransA, char* pTransB, const int* pM, const int* pN, const int* pK, const float *pAlpha, const float *pa, const int*plda, const float *pb, const int *pldb, const float *pBeta, float *pc, const int*pldc) {
    int i = 0;          // iteratoin in every sgemm test
    float M = *pM;
    float N = *pN;
    float K = *pK;
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    CBLAS_TRANSPOSE  transa = CblasNoTrans;
    CBLAS_TRANSPOSE  transb = CblasNoTrans;

    if( *pTransB == 't'){
        transb = CblasTrans;
    }
    mysgemm_checkresult(transa, transb, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
    double t0 = 0;
    for(i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        #pragma omp parallel num_threads(20)
        my_sgemm(transa, transb, *pM, *pN, *pK, *pAlpha, pa, *plda, pb, *pldb, *pBeta, pc, *pldc);
    }
    double t1 = get_time() - t0;
    double avg_time = t1/SGEMM_COUNT;
    printf("sgemm_profile_mysgemm end, avg time = %.2f, GFLOPS = %.2f\n", avg_time, gflops/avg_time);
}

float* matrix_init(int A, int B)
{
    float * p = (float*)mkl_malloc(A*B*sizeof(float), 64);
    int a,b;
    #pragma omp parallel for collapse(2)
    for(a=0; a < A; a++)
        for(b=0; b < B;b++)
            p[a*B+b] = (rand() % 1000 - 500)/100;
            //p[a*B+b] = 1.0f; 
    return p;
}


void sgemm_main(int index, char transa, char transb, int M, int N, int K, int lda, float alpha, int ldb, float beta, int ldc)
{
    float * a = matrix_init(M,K);
    float * b = matrix_init(K,N);
    float * c = matrix_init(M,N);
    printf("----------GEMM %d----------\n", index);
    //sgemm_profile(&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    //sgemm_profile_pack(&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    //sgemm_profile_batch(4, &transa, &transb, &M, &N, &K, &alpha, &lda, &ldb, &beta, &ldc);
    //sgemm_profile_2pthread(&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    //sgemm_profile_2ompthread(&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    //sgemm_profile_2ompthread_ideal(&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    sgemm_profile_mysgemm(&transa, &transb, &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
}


void var_main(){

    int M = 40;
    int N = 102400*16;

    float * a1 = matrix_init(M,N);
    float * b1 = matrix_init(M,N);
    float * c1 = (float *)mkl_malloc(M*N*sizeof(float), 64);

    bench_var(M, N, a1, b1, c1);
    bench_var_2omp(M, N, a1, b1, c1);



#if 0
    float * a2 = matrix_init(M,N);
    float * b2 = matrix_init(M,N);
    float * c2 = matrix_init(M,N);
    bench_var_2omp2(M, N, a1, b1, c1, a2, b2, c2);
    mkl_free(a2);
    mkl_free(b2);
    mkl_free(c2);
#endif

    mkl_free(a1);
    mkl_free(b1);
    mkl_free(c1);

    

}


int main()
{
    printf("main start \n");
    char transa, transb;
    int m,n,k,lda,ldb,ldc;
    float alpha,beta;

#if 0

    var_main();

#else
    // small sgemm
    transa='n'; transb='n'; m=20; n=2400; k=800; lda=k; alpha=1.0000; ldb=n; beta=0.0000; ldc=n;
    sgemm_main(1, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
    //transa='n'; transb='n'; m=2000; n=2400; k=800; lda=k; alpha=1.0000; ldb=n; beta=0.0000; ldc=n;
    //sgemm_main(2, transa, transb, m, n, k, lda, alpha, ldb, beta, ldc);
#endif

    return 1;
}
