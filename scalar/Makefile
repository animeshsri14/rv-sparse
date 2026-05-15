CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS =

# default data directory (relative to where you run the binary)
DATA_DIR ?= ../../rv-sparse/matrix_data

all: spmv_csr

spmv_csr: spmv_csr.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: spmv_csr
	./spmv_csr $(DATA_DIR)

clean:
	rm -f spmv_csr

.PHONY: all run clean
