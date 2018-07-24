
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)

set(TBB_INC "/opt/intel/tbb/include/tbb/")
#set(TBB_LIB "/home/zhaoxiao/tbb/lib")

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2 ")

include_directories(${MKLINC}) 
link_directories(${MKLLIBPATH})

include_directories(${TBB_INC}) 
link_directories(${TBB_LIB})

add_executable(test_omp test_omp.cpp)
TARGET_LINK_LIBRARIES(test_omp mkl_rt)
SET_TARGET_PROPERTIES(test_omp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

add_executable(test_tbb test_tbb.cpp)
TARGET_LINK_LIBRARIES(test_tbb mkl_rt tbb)
SET_TARGET_PROPERTIES(test_tbb PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
