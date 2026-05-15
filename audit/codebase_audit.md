# rv-sparse Codebase Audit

**Author:** Animesh Srivastava  
**Date:** May 2026  
**Scope:** `src/matmul/AxBRowIP.c`, `scripts/cooTOcsr.py`, `matrix_data/`

---

## Summary

The existing rv-sparse repository is a proof-of-concept for sparse matrix-matrix multiplication (SpGEMM) using a row-wise inner product algorithm on CSR data. The Python converter (`cooTOcsr.py`) is solid — it correctly handles Matrix Market files including symmetric and pattern variants.

The C implementation has structural issues that prevent it from serving as a library foundation. These are not minor bugs — they reflect the absence of a library architecture (no public header, no build system, no API separation).

---

## Key Findings

### 1. Per-row heap allocation in the hot loop (line 8)

```c
float* cD_ = (float *)calloc(K, sizeof(float));  // inside for(row=0; row<M; ...)
```

This calls `calloc` and `free` once per row of A. For a matrix with M rows, that is M allocator round-trips in the critical section. The fix is to allocate a single workspace before the loop and selectively zero only the entries that were touched in each iteration.

### 2. Accumulator sized incorrectly for general A×B

```c
float* cD_ = (float *)calloc(K, sizeof(float));  // size K
...
int idx = bC[b_col];       // idx ranges 0..N-1
cD_[idx] += valA * valB;   // heap overflow when N > K
```

Column indices of B range from 0 to N-1, but the accumulator is only K floats wide. The code only works by accident in the A×A case where K == N.

### 3. O(K) dense scan over sparse accumulator (line 18)

```c
for (int i = 0; i < K; i++) {
    if (cD_[i] != 0) { ... }
}
```

After accumulating results, the code scans all K entries to find non-zeros. For a sparse output row with 5 non-zeros and K = 100,000, this is 20,000x more work than necessary. Fix: maintain a `touched[]` index list during accumulation.

### 4. NNZ estimation formula is incorrect

```c
float density = (float)(aNNZ)/MxK + (float)(bNNZ)/MxK;
return density * M * N;
```

Adding densities has no probabilistic basis. The standard estimate for SpGEMM output NNZ is `aNNZ * bNNZ / K` (used by Intel MKL and SciPy).

### 5. No build system, no public header, no API

All functions are in a single `.c` file. There is no `CMakeLists.txt`, no `Makefile`, and no header defining a public interface. This makes the code unusable as a library.

---

## What Works Well

- **`cooTOcsr.py`** correctly parses Matrix Market files including symmetric and pattern variants.
- The `CSR_redColIdx` (reduced column index) output in the Python script is an interesting feature for RVV optimization — local indices enable shorter index vectors and potentially register-resident x sub-vectors.
- The basic row-wise inner product algorithm is a valid starting point for SpGEMM.

---

## Proposed Direction

Rather than patching individual bugs, the codebase needs a clean architectural layer:

1. **Public header** (`rv_sparse.h`) defining CSR/CSC types and function prototypes
2. **Scalar baseline kernels** (SpMV, SpMM) that serve as correctness oracles
3. **RVV kernels** layered on top with compile-time feature guards
4. **CMake build system** supporting native and cross-compilation targets
5. **Test harness** comparing scalar and RVV outputs
