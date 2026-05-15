# rv-sparse-contrib

Preliminary work for the rv-sparse LFX mentorship application.

**Author:** Animesh Srivastava ([github.com/animeshsri14](https://github.com/animeshsri14))

---

## What This Contains

### `audit/` — Codebase audit of rv-sparse
A concise review of the existing `AxBRowIP.c` implementation, identifying structural issues and proposing an architectural direction for the library.

### `scalar/` — Working scalar SpMV baseline
A clean C implementation of CSR Sparse Matrix-Vector Multiply (`y = A * x`) that:
- Reads the existing `matrix_data/` text files from the rv-sparse repository
- Validates the CSR structure (monotonic rowptr, column index bounds)
- Computes SpMV and reports timing

```bash
# Compiled and verified inside riscv64/ubuntu:24.04 Docker container
$ gcc -O2 -Wall -std=c11 -o spmv_csr spmv_csr.c
$ file spmv_csr
spmv_csr: ELF 64-bit LSB pie executable, UCB RISC-V, RVC, double-float ABI, version 1 (SYSV), dynamically linked, interpreter /lib/ld-linux-riscv64-lp64d.so.1, for GNU/Linux 4.15.0, not stripped

$ ./spmv_csr /workspace/matrix_data
rv-sparse scalar SpMV baseline
loading CSR from: /workspace/matrix_data
matrix: 4 x 4, nnz = 9, density = 0.5625
CSR validation: OK

--- SpMV result (y = A * x, x = all-ones) ---
  y[0] = 3.0000
  y[1] = 1.0000
  y[2] = 3.0000
  y[3] = 2.0000

--- timing ---
  100 iterations in 0.000119 s
  avg: 1.193 us/iter
  throughput: 15.09 Mflop/s (2*nnz per SpMV)
```

### `docs/` — RVV vectorization strategy
Approach to vectorizing SpMV and SpMM using RVV intrinsics. Includes notes on why QEMU is insufficient for performance measurement of gather-heavy sparse workloads.

### `benchmarks/` — Benchmarking methodology notes
Guidelines on what QEMU can and cannot tell you about sparse kernel performance, and what tools should be used instead.

---

## Build

```bash
cd scalar
make        # builds spmv_csr
make run    # runs against default data directory
```

Requires: `gcc` (any version supporting C11).

---

## Related

- [rv-sparse](https://github.com/merledu/rv-sparse) — the upstream project
- [riscv-porting](https://github.com/animeshsri14/riscv-porting) — my RISC-V HPC porting work (5 applications ported natively)
