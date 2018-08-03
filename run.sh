#!/bin/sh

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

#KMP_SETTING="KMP_HW_SUBSET=2s,20c,1t KMP_AFFINITY=compact,1,0,granularity=fine"
#KMP_SETTING="KMP_AFFINITY=compact,1,0,granularity=fine,verbose"
KMP_SETTING="KMP_AFFINITY=compact,1,0,granularity=fine"

export MKL_NUM_THREADS=20
export $KMP_SETTING
export MKL_DYNAMIC=FALSE
#export MKL_VERBOSE=1

echo -e "### using $KMP_SETTING\n"

if [ $1 == "omp" ] ; then
    ./build/test_omp
else
    ./build/test_tbb
fi
