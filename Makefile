CC=/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc
MKL_DNN_ROOT=/opt/intel/mkl

COPT=-xMIC-AVX512 -std=c99 -qopenmp 
#INC= 
LOPT = -L$(MKL_DNN_ROOT)/lib/intel64  -mkl -static-intel -static-libgcc -O3 -par-affinity=compact,1,0
#LOPT= 

samples=test
run=$(addsuffix .bin, $(samples))

all: $(run)

#%.out.run: %.out
#	@echo -ne "$< \t" && \
        LD_LIBRARY_PATH=$(MKL_DNN_ROOT)/lib:$(LD_LIBRARY_PATH) ./$<

%.bin: %.o
	$(CC) $< -o $@ $(LOPT)

%.o: %.c
	$(CC) $(COPT) $(INC) $< -c -o $@

clean:
	rm -rf *.bin
