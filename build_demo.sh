/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc -xMIC-AVX512 -std=c99 -qopenmp -mkl -static-intel -static-libgcc -O3 -par-affinity=compact,1,0 -o demo.bin  demo.c
