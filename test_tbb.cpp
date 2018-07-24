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
#define SGEMM_COUNT 100000   // every sgemm iteration numbers
#define USE_VAR     1

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


void main(){
    printf("main start \n");
    int M = 40;                                                                 
    int N = 102400*16;    

    //tbb::task_scheduler_init init(40);
    tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

    float * a = matrix_init(M,N);
    float * b = matrix_init(M,N);
    float * c = (float *)mkl_malloc(M*N*sizeof(float), 64);
    for(int i = 0; i < 10; i++)
        parallel_for(blocked_range<size_t>(0,M*N), ApplyFoo(a,b,c));

    __tstart(plain);
    for(int i = 0; i < 1000; i++)
        parallel_for(blocked_range<size_t>(0,M*N), ApplyFoo(a,b,c));
    __tend(plain);





















}
