CC=gcc
MKL_DNN_ROOT=/home/zhaoxiao/MKL/EnvCheck/MKL/mklml_lnx_2018.0.20170425


COPT = -Wall -fopenmp
INC  = -I$(MKL_DNN_ROOT)/include
LOPT = -L$(MKL_DNN_ROOT)/lib -lmkl_rt  -liomp5 -lpthread
#LOPT = -L$(MKL_DNN_ROOT)/lib -lmkl_intel_ilp64 -lmkl_gnu_thread  -lmkl_core -lgomp -lpthread -lm -ldl

samples=test
run=$(addsuffix .bin, $(samples))

all: $(run)

#%.out.run: %.out
#	@echo -ne "$< \t" && \
        LD_LIBRARY_PATH=$(MKL_DNN_ROOT)/lib:$(LD_LIBRARY_PATH) ./$<

%.bin: %.o
	$(CC) $< -o $@  $(LOPT)

%.o: %.c
	$(CC) $(COPT) $(INC) $< -c -o $@

clean:
	rm -rf *.bin
