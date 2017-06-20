#!/bin/sh

source ~/.bashrc
#source /opt/intel/mkl/bin/mklvars.sh  intel64
#OMP_NUM_THREADS=52 ./test.bin


#KMP_AFFINITY=scatter,granularity=fine OMP_NUM_THREADS=52 th train.lua -data data/demo-train.t7 -save_model model -profiler true
/home/zhaoxiao/intel/vtune_amplifier_xe/bin64/amplxe-cl -collect hotspots -knob analyze-openmp=true -knob sampling-interval=30 --resume-after 1 -d 50 ./run_skx.sh
