// vllm_kernels.h — host-optimized C kernels for vllm-os.
//
// Scalar fallbacks always compile (gcc, clang, tcc, MSVC, freestanding:
// no libc, no libm, no OS calls). On gcc/clang x86 the hot functions
// use function-target AVX2+FMA codegen, so plain portable flags already
// run full speed — no -march=native needed, the binary still runs
// anywhere (runtime dispatch inside each entry point).
//
// Freestanding-safe: no libc/libm calls, no OS interaction, no
// allocation. Bit tricks only (fast exp uses integer reinterpretation).
// The bare-metal kernel port may link this file as-is.
#ifndef VLLM_KERNELS_H
#define VLLM_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// 1 when this TU was built with AVX2+FMA *and* the CPU supports it.
int vllm_has_avx2(void);

// out = W * x, W row-major [rows x cols]. All pointers unaligned-safe.
void vllm_matvec_f32(const float *w, const float *x, float *out, int rows, int cols);

// dot(q_row_int8, x_f32) * scale over n elems (Q8_0 block rows use n=32).
float vllm_q8_dot_f32(const signed char *qs, const float *x, float scale, int n);

// dot(dequant(q_row_4bit), x_f32) * scale over n elems (Q4_0 block rows
// use n=32, qs holds n/2 packed bytes, lo nibble first). Scalar C;
// AVX2 nibble-unpack is follow-up (the fused win is the missing store,
// not the SIMD width).
float vllm_q4_0_dot_f32(const unsigned char *qs, const float *x, float scale, int n);

// x[i] = x[i] * w[i] / sqrt(mean(x^2) + eps), in place.
void vllm_rmsnorm_f32(float *x, const float *w, int n, float eps);

// Plain dot product over n elems.
float vllm_dot_f32(const float *a, const float *b, int n);

// Fast softmax with temperature scale: xs[i] = softmax(xs * scale).
// Uses branchless approximate exp (~0.5% rel err worst case, no libm).
// For sampling only — never for scoring/perplexity. The exact softmax
// stays pure V (sampler.softmax_scaled_in_place): one implementation,
// no numeric drift between toolchains.
void vllm_softmax_scaled_fast_f32(float *xs, int n, float scale);

#ifdef __cplusplus
}
#endif

#endif
