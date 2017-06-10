CC=gcc
MKL_DNN_ROOT=/home/xiaohui/MKLs/mklml_lnx_2017.0.2.20170209

COPT = -Wall
INC  = -I$(MKL_DNN_ROOT)/include
LOPT = -L$(MKL_DNN_ROOT)/lib/intel64 -lmklml_gnu  -liomp5

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
