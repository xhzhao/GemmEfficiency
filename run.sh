#!/bin/sh

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

if [ $1 == "half" ] ; then
    export OMP_NUM_THREADS=20
    echo -e "### using OMP_NUM_THREADS=20"
else
    export OMP_NUM_THREADS=$TOTAL_CORES
    echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
fi

export $KMP_SETTING


echo -e "### using $KMP_SETTING\n"

./test.bin
