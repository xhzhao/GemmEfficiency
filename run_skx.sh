#!/bin/sh
export KMP_AFFINITY=compact,1,0,granularity=fine
export OMP_NUM_THREADS=56

./test.bin
