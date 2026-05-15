# Benchmarking Notes

## QEMU vs Real Hardware

QEMU user-mode (`qemu-riscv64 -cpu rv64,v=true`) is appropriate for:
- Verifying that RVV code compiles and executes correctly
- Comparing scalar vs RVV output for functional equivalence
- Running the test harness

QEMU is **not** appropriate for:
- Measuring SpMV gather throughput (no cache model)
- Comparing kernel variants by wall-clock time
- Reporting absolute GFLOP/s or bandwidth numbers
- Tuning LMUL or dispatch thresholds

## What I Would Use Instead

1. **Spike** (RISC-V ISA simulator) with `--log-commits` for instruction-level tracing
2. **gem5** with the RVV vector extension model for cycle-accurate simulation
3. **Hardware performance counters** (`mcycle`, `minstret`, `mhpmcounter3..31`) on real RISC-V hardware when available
4. **Relative comparisons only** when using QEMU — report instruction counts, not timing

Any performance numbers from QEMU in this repository will be clearly labeled as functional-validation data, not hardware-representative benchmarks.
