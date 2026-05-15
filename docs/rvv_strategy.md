# RVV Vectorization Strategy for rv-sparse

**Author:** Animesh Srivastava  
**Date:** May 2026

---

## Overview

Sparse kernels are memory-bound, not compute-bound. Writing better intrinsics without addressing memory access patterns yields minimal gains. The approach here follows a strict order: correct scalar baseline → memory layout analysis → targeted RVV acceleration.

---

## Why SpMV and SpMM Are Different Problems for RVV

### SpMV: y = A * x (A sparse CSR, x dense)

The inner loop of CSR SpMV is:
```c
for (int j = rowptr[i]; j < rowptr[i+1]; j++)
    y[i] += val[j] * x[colidx[j]];
```

- `val[j]` is **contiguous** — a simple vector load (`vle32.v`)
- `x[colidx[j]]` is a **gather** — requires indexed load (`vluxei32.v`)

The gather is the bottleneck. `vluxei32.v` takes a base pointer and a vector of byte offsets, fetching scattered elements of `x` in a single vector instruction. But the actual throughput depends heavily on whether those `x` entries are in cache.

**RVV approach for SpMV:**
1. Use `vsetvl_e32m4` for vector-length-agnostic strip mining
2. Load `val` contiguously with `vle32.v`
3. Load `colidx`, multiply by `sizeof(float)` to get byte offsets
4. Gather `x` values with `vluxei32.v`
5. Multiply with `vfmul.vv`, reduce with `vfredusum.vs`
6. For very short rows (< 4 nnz), fall back to scalar — the `vsetvl` + gather setup cost exceeds the compute savings

### SpMM: C = A * B (A sparse CSR, B dense)

```c
for (int i = 0; i < M; i++)
    for (int j = rowptr[i]; j < rowptr[i+1]; j++)
        for (int n = 0; n < N; n++)
            C[i*N + n] += val[j] * B[colidx[j]*N + n];
```

The inner `n`-loop is over a **contiguous row of B** — no gather needed. This is the highest-throughput kernel because:
- `B[k*N + n]` is a stride-1 load (`vle32.v`)
- `C[i*N + n]` is a stride-1 load/store
- `val[j]` is a scalar broadcast (`vfmacc.vf`)

**RVV approach for SpMM:**
- Use `m8` LMUL to maximize register width (no gather pressure)
- Vectorize across columns of B with `vsetvl_e32m8`
- Use `vfmacc.vf` for fused multiply-accumulate with scalar `val[j]`
- Tail handling via `vsetvl` — no scalar epilogue needed

---

## The QEMU Limitation (Critical for Benchmarking)

QEMU user-mode emulation is adequate for **functional correctness** but unreliable for **performance measurement** of sparse workloads where the dominant cost is memory access latency:

- QEMU does not model cache hierarchy. L1/L2 miss behavior — the primary bottleneck for SpMV gathers — is invisible.
- QEMU does not model gather instruction pipeline behavior. On real silicon, `vluxei32` throughput depends on how many unique cache lines the index vector touches.
- `perf stat` under QEMU reports emulated instruction counts, not actual hardware cycle costs.

**Approach:**
- Use QEMU for correctness verification (scalar == RVV output)
- For performance analysis, clearly label QEMU numbers as "instruction-count proxies" rather than timing data
- If cycle-accurate simulation is needed, Spike (the RISC-V ISA simulator) with timing extensions or gem5 with the RVV model provides more meaningful data
- RISC-V HPM counters (`mcycle`, `minstret`, and the programmable `mhpmcounter` registers) should be used on real hardware when available

---

## Dispatch Strategy

Not all rows benefit from RVV. The crossover point where RVV setup cost (`vsetvl` + index multiplication + gather) exceeds scalar compute depends on:
- VLEN of the target implementation
- Actual gather throughput (hardware-dependent)
- Row length (nnz per row)

Rather than hardcoding a threshold, the dispatch should be calibrated:
```c
static int rvv_threshold = 8;  // conservative default

// Per-row dispatch in SpMV
for (int i = 0; i < M; i++) {
    int row_len = rowptr[i+1] - rowptr[i];
    if (row_len >= rvv_threshold)
        spmv_row_rvv(A, x, y, i);
    else
        spmv_row_scalar(A, x, y, i);
}
```

The default threshold of 8 is conservative. On wide VLEN implementations (512-bit), it could drop to 3-4. On narrow implementations or under emulation, it should stay at 8+.

