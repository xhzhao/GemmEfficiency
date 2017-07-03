export KMP_AFFINITY=compact,1,0,granularity=fine,verbose
export MKL_ENABLE_INSTRUCTIONS=AVX512_MIC_E1
export OMP_NUM_THREADS=72

#MKL IGEMM
numactl -m 1 ./test.x 50000 50000 8640 NN 1 1 0 0 0 0 10
#MKL SGEMM
numactl -m 1 ./stest.x 50000 50000 4320 NN 1 1 10

