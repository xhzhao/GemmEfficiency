#!/bin/sh
t=$1
lscpu

export OMP_NUM_THREADS=56
export KMP_AFFINITY=compact,1,0,granularity=fine

./test.bin
