/*
 * spmv_csr.c — Scalar CSR SpMV baseline for rv-sparse
 *
 * Computes y = A * x where A is in CSR format.
 * Reads CSR arrays from the existing matrix_data/ text files
 * in the rv-sparse repository.
 *
 * Build:  gcc -O2 -o spmv_csr spmv_csr.c
 * Run:    ./spmv_csr ../rv-sparse/matrix_data
 *
 * Author: Animesh Srivastava
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---------- CSR structure ---------- */
typedef struct {
    int    M;        /* number of rows    */
    int    N;        /* number of columns */
    int    nnz;      /* number of non-zeros */
    int   *rowptr;   /* length M+1 */
    int   *colidx;   /* length nnz */
    float *val;      /* length nnz */
} csr_t;

/* ---------- helpers ---------- */

/* read an array of ints from a text file (one value per line) */
static int *read_int_array(const char *path, int *out_count) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return NULL; }

    int cap = 64, n = 0;
    int *arr = malloc(cap * sizeof(int));
    int v;
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) { cap *= 2; arr = realloc(arr, cap * sizeof(int)); }
        arr[n++] = v;
    }
    fclose(f);
    *out_count = n;
    return arr;
}

/* read an array of floats from a text file (one value per line) */
static float *read_float_array(const char *path, int *out_count) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return NULL; }

    int cap = 64, n = 0;
    float *arr = malloc(cap * sizeof(float));
    float v;
    while (fscanf(f, "%f", &v) == 1) {
        if (n == cap) { cap *= 2; arr = realloc(arr, cap * sizeof(float)); }
        arr[n++] = v;
    }
    fclose(f);
    *out_count = n;
    return arr;
}

/* ---------- CSR loader ---------- */
static csr_t *csr_load(const char *data_dir) {
    char path[512];
    csr_t *A = calloc(1, sizeof(csr_t));

    /* row pointers */
    snprintf(path, sizeof(path), "%s/CSR_rowPtr.txt", data_dir);
    int rp_count;
    A->rowptr = read_int_array(path, &rp_count);
    if (!A->rowptr) { free(A); return NULL; }
    A->M = rp_count - 1;

    /* column indices */
    snprintf(path, sizeof(path), "%s/CSR_colIdx.txt", data_dir);
    A->colidx = read_int_array(path, &A->nnz);
    if (!A->colidx) { free(A->rowptr); free(A); return NULL; }

    /* values */
    snprintf(path, sizeof(path), "%s/CSR_values.txt", data_dir);
    int val_count = 0;
    A->val = read_float_array(path, &val_count);
    if (!A->val || val_count != A->nnz) {
        fprintf(stderr, "error: value count (%d) != nnz (%d)\n", val_count, A->nnz);
        free(A->rowptr); free(A->colidx); free(A->val); free(A);
        return NULL;
    }

    /* infer N from max column index */
    A->N = 0;
    for (int i = 0; i < A->nnz; i++) {
        if (A->colidx[i] + 1 > A->N) A->N = A->colidx[i] + 1;
    }

    return A;
}

static void csr_free(csr_t *A) {
    if (!A) return;
    free(A->rowptr);
    free(A->colidx);
    free(A->val);
    free(A);
}

/* ---------- Scalar SpMV: y = A * x ---------- */
static void spmv_csr_scalar(const csr_t *A, const float *x, float *y) {
    for (int i = 0; i < A->M; i++) {
        float acc = 0.0f;
        for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
            acc += A->val[j] * x[A->colidx[j]];
        }
        y[i] = acc;
    }
}

/* ---------- CSR validation ---------- */
static int csr_validate(const csr_t *A) {
    if (!A || !A->rowptr || !A->colidx || !A->val) return -1;
    if (A->M <= 0 || A->N <= 0 || A->nnz < 0) return -1;

    /* rowptr must be monotonically non-decreasing */
    for (int i = 0; i < A->M; i++) {
        if (A->rowptr[i] > A->rowptr[i + 1]) {
            fprintf(stderr, "validate: rowptr not monotonic at row %d\n", i);
            return -1;
        }
    }
    if (A->rowptr[0] != 0 || A->rowptr[A->M] != A->nnz) {
        fprintf(stderr, "validate: rowptr bounds mismatch\n");
        return -1;
    }

    /* column indices must be in range [0, N) */
    for (int j = 0; j < A->nnz; j++) {
        if (A->colidx[j] < 0 || A->colidx[j] >= A->N) {
            fprintf(stderr, "validate: colidx[%d]=%d out of range [0,%d)\n",
                    j, A->colidx[j], A->N);
            return -1;
        }
    }
    return 0;
}

/* ---------- timing ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    const char *data_dir = (argc > 1) ? argv[1] : "../rv-sparse/matrix_data";

    printf("rv-sparse scalar SpMV baseline\n");
    printf("loading CSR from: %s\n", data_dir);

    csr_t *A = csr_load(data_dir);
    if (!A) { fprintf(stderr, "failed to load CSR data\n"); return 1; }

    printf("matrix: %d x %d, nnz = %d, density = %.4f\n",
           A->M, A->N, A->nnz, (float)A->nnz / ((float)A->M * A->N));

    if (csr_validate(A) != 0) {
        fprintf(stderr, "CSR validation failed\n");
        csr_free(A);
        return 1;
    }
    printf("CSR validation: OK\n");

    /* allocate x (all ones) and y */
    float *x = malloc(A->N * sizeof(float));
    float *y = calloc(A->M, sizeof(float));
    for (int i = 0; i < A->N; i++) x[i] = 1.0f;

    /* warm-up */
    spmv_csr_scalar(A, x, y);

    /* timed run */
    int iters = 100;
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        spmv_csr_scalar(A, x, y);
    }
    double elapsed = now_sec() - t0;

    printf("\n--- SpMV result (y = A * x, x = all-ones) ---\n");
    for (int i = 0; i < A->M; i++) {
        printf("  y[%d] = %.4f\n", i, y[i]);
    }

    printf("\n--- timing ---\n");
    printf("  %d iterations in %.6f s\n", iters, elapsed);
    printf("  avg: %.3f us/iter\n", (elapsed / iters) * 1e6);
    printf("  throughput: %.2f Mflop/s (2*nnz per SpMV)\n",
           (2.0 * A->nnz * iters) / elapsed / 1e6);

    free(x);
    free(y);
    csr_free(A);
    return 0;
}
