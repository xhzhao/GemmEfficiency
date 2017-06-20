#!/bin/sh

source ~/.bashrc

KMP_AFFINITY=scatter,granularity=fine OMP_NUM_THREADS=72 ./test.bin
