CC=/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc
MKL_DNN_ROOT=/opt/intel/mkl

#COPT=-xMIC-AVX512 -std=c99 -qopenmp
COPT= -xMIC-AVX512 -std=c99 -qopenmp -O3
#INC=
LOPT = -L$(MKL_DNN_ROOT)/lib/intel64  -mkl -static-intel -static-libgcc -O3 -par-affinity=compact,1,0
#LOPT=


%.o: %.c
	$(CC) -c -o $@ $< $(COPT)
test.bin: test.o  gemm1.o gemm3.o gemm4.o gemm5.o 
	$(CC) -o test.bin test.o  gemm1.o gemm3.o gemm4.o gemm5.o $(LOPT) 

clean:
	rm -rf *.bin
	rm -rf *.o
