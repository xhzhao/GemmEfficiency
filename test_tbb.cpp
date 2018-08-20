#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include <pthread.h>
#include <omp.h>

#include <iostream>
#include <cstdint>
#include <cstring>


#include <tbb.h>

#define WARM_UP     100
#define SGEMM_COUNT 10000   // every sgemm iteration numbers
#define USE_VAR     1
const int M = 20;
const int N = 2400;
const int K = 800;
const float alpha = 1.0;
const float beta = 0.0;

using namespace tbb;


#include <cxxabi.h>                                                             
#include <chrono>                                                               
#include <unistd.h>                                                             
#define _T(x) x
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define __tstart(n) _T(Time::time_point __s##n = Time::now());
#define __tend(n)                                                              \
  _T(Time::time_point __e##n = Time::now());                                   \
  _T(printf("time: %s, th=%d, %.2f ms\n", #n, omp_get_thread_num(),            \
      Duration(__e##n - __s##n).count()));

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


float* matrix_init(int A, int B)
{
    float * p = (float*)mkl_malloc(A*B*sizeof(float), 64);
    int a,b;
#if 1
    //#pragma omp parallel for collapse(2)
    for(a=0; a < A; a++)
        for(b=0; b < B;b++)
            p[a*B+b] = 1.0f; 
#endif
    return p;
}

void Foo(float& a, float& b, float& c){
    c = (a * 1.01)*(b + 0.001);
}

class ApplyFoo{
    float * const my_a;
    float * const my_b;
    float * const my_c;
public:
    void operator()(const blocked_range<size_t>& r) const {
        float * a = my_a;
        float * b = my_b;
        float * c = my_c;
        for(size_t i = r.begin(); i != r.end(); i++){
            Foo(a[i], b[i], c[i]);
        }
    }
    ApplyFoo(float a[], float b[], float c[]): my_a(a), my_b(b), my_c(c)
    {}
};

class ApplyGemm{
    float * const my_a;
    float * const my_b;
    float * const my_c;
public:
    void operator()(const blocked_range<size_t>& r) const {
        float * a = my_a;
        float * b = my_b;
        float * c = my_c;
        mkl_set_num_threads_local(20);
        for(size_t i = r.begin(); i != r.end(); i++){
            if (i == 0){
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    alpha, a, K, b, N, beta, c, N);
            } else {
                a = my_a + M * K;
                b = my_b + K * N;
                c = my_c + M * N;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    alpha, a, K, b, N, beta, c, N);
            }
        }
    }
    ApplyGemm(float a[], float b[], float c[]): my_a(a), my_b(b), my_c(c)
    {}
};

void FooMain(){
    tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);
    int m = 40;                                                                 
    int n = 102400*16;    
    float * a = matrix_init(m,n);
    float * b = matrix_init(m,n);
    float * c = (float *)mkl_malloc(m*n*sizeof(float), 64);
    for(int i = 0; i < 10; i++)
        parallel_for(blocked_range<size_t>(0,m*n), ApplyFoo(a,b,c));

    __tstart(plain);
    for(int i = 0; i < 1000; i++)
        parallel_for(blocked_range<size_t>(0,m*n), ApplyFoo(a,b,c));
    __tend(plain);

}

void simple_omp_parallel(){
    int dummy[40];
    #pragma omp parallel for num_threads(40)
    for(int i = 0; i < 40; i++){
        dummy[i] = 0;
    }
    printf("simple_omp_parallel done\n");
}

void SgemmMain(){
    tbb::task_scheduler_init init(2);
    //tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

    float * a = matrix_init(2*M,K);
    float * b = matrix_init(2*K,N);
    float * c = (float *)mkl_malloc(2*M*N*sizeof(float), 64);
    
    double t0 = 0;
    static  affinity_partitioner ap;
    for(int i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        parallel_for(blocked_range<size_t>(0,2), ApplyGemm(a,b,c), ap);
    }
    double gflops = 2*(M*N*K*2 + 2*M*N ) * (1e-6);
    double t1 = get_time() - t0;                                                
    double avg_time = t1/SGEMM_COUNT; 
    printf("sgemm_tbb, avg time = %.2f, GFLOPS = %.2f \n", avg_time, gflops/avg_time);
}

void* thread_tbb_gemm(void* p){
    tbb::task_scheduler_init init(20);

    float * a = matrix_init(M,K);
    float * b = matrix_init(K,N);
    float * c = (float *)mkl_malloc(M*N*sizeof(float), 64);
    
    double t0 = 0;
    static  affinity_partitioner ap;
    for(int i=0; i < SGEMM_COUNT + WARM_UP; i++)
    {
        if (i == WARM_UP){
            t0 = get_time();
        }
        //parallel_for(blocked_range<size_t>(0,2), ApplyGemm(a,b,c), ap);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    alpha, a, K, b, N, beta, c, N);
    }
    double gflops = (M*N*K*2 + 2*M*N ) * (1e-6);
    double t1 = get_time() - t0;                                                
    double avg_time = t1/SGEMM_COUNT; 
    printf("sgemm_tbb, avg time = %.2f, GFLOPS = %.2f \n", avg_time, gflops/avg_time);
    return 0;
}
void PthreadMain(){

    pthread_t thread1, thread2;
    int err1 = pthread_create(&thread1, NULL, thread_tbb_gemm, NULL);
    int err2 = pthread_create(&thread2, NULL, thread_tbb_gemm, NULL);
    if (err1 | err2){
        printf("error in creating thread \n");
    }
    err1 = pthread_join(thread1, NULL);
    err2 = pthread_join(thread2, NULL);
    if (err1 | err2){
        printf("error in joining thread \n");
    }
}

int main(){
    printf("main start \n");

    //FooMain();
    //simple_omp_parallel();
    SgemmMain();
    //PthreadMain();

    return 1;
}
