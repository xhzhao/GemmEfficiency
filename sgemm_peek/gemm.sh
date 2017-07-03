##arguments=targetRunDir + " "+ name+" "+str(args) + " " + str(thread_num) + " " + str(memory_type) + " "+ resultfile
targetRunDir=$1
cd $targetRunDir

export OMP_NUM_THREADS=$4
export KMP_AFFINITY=compact,1,0,granularity=fine,verbose
export MKL_ENABLE_INSTRUCTIONS=AVX512_MIC_E1

if [ $2 == "sgemm" ]
then
	if [ $5 == "MCDRAM" ]
	then
		numactl -m 1 ./stest.x $3 2>&1 >> $6
		echo "numactl -m 1 ./stest.x $3 2>&1" >> $6
	else
		numactl -m 0 ./stest.x $3 2>&1 >> $6
		echo "numactl -m 0 ./stest.x $3 2>&1" >>$6
	fi
fi

if [ $2 == "igemm" ]
then
	if [ $5 == "MCDRAM" ]
	then
		numactl -m 1 ./test.x $3 2>&1 >>$6
		echo "numactl -m 1 ./test.x $3 2>&1" >>$6
	else
		numactl -m 1 ./test.x $3 2>&1 >>$6
		echo "numactl -m 0 ./test.x $3 2>&1" >>$6
	fi
fi

