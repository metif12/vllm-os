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
// use n=32, qs holds n/2 packed bytes, SPLIT-HALF layout: byte i carries
// q[i] in the low nibble and q[i+n/2] in the high nibble — the same
// convention as infer.q4_0_dot_row / GGML vec_dot_q4_0).
float vllm_q4_0_dot_f32(const unsigned char *qs, const float *x, float scale, int n);

// out = W * x, W row-major Q4_0 [rows x cols] (cols must divide 32).
// data holds rows*cols/32 packed 18-byte blocks (f16 scale + 16 split-half
// nibble bytes), the exact layout infer.q4_0_matvec consumes.
void vllm_q4_0_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols);

// dot(dequant(q_row_4bit_unsigned), x_f32): Q4_1 block rows use n=32,
// qs holds 16 packed bytes (SPLIT-HALF: byte i carries q[i] low and
// q[i+16] high, values 0..15 unsigned), result = dot(q,x)*scale +
// sum(x)*min — mirrors infer.q4_1_dot_row / GGML vec_dot_q4_1.
float vllm_q4_1_dot_f32(const unsigned char *qs, const float *x, float scale, float min);

// out = W * x, W row-major Q4_1 [rows x cols] (cols must divide 32).
// data holds rows*cols/32 packed 20-byte blocks (f16 scale + f16 min +
// 16 split-half nibble bytes), the exact layout infer.q4_1_matvec
// consumes.
void vllm_q4_1_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols);

// out = W * x, W row-major Q6_K [rows x cols] (cols must divide 256).
// data holds rows*cols/256 packed 210-byte super-blocks
// (ql[128], qh[64], scales[16], f16 d), matching infer.q6_k_dot_block.
void vllm_q6_k_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols);

// IQ4_NL (non-linear 4-bit): dot(dequant(q_row), x_f32) * scale over n elems.
// n=32, qs holds 16 split-half nibble codes, result = dot(grid[code],x)*d
// with the GGML kvalues_iq4nl codebook — mirrors infer.iq4_nl_dot_row.
float vllm_iq4_nl_dot_f32(const unsigned char *qs, const float *x, float scale, int n);

// out = W * x, W row-major IQ4_NL [rows x cols] (cols must divide 32).
// data holds rows*cols/32 packed 18-byte blocks (f16 d + 16 split-half
// nibble bytes), matching infer.iq4_nl_matvec.
void vllm_iq4_nl_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols);

// Q2_0 (signed-scale 2-bit): dot(dequant(q_row), x_f32) * scale over n elems.
// n=64, qs holds 16 bytes (4 codes/byte, element j = (qs[j/4] >> 2*(j%4))&3,
// value = (code-1)*scale) — mirrors infer.q2_0_dot_row.
float vllm_q2_0_dot_f32(const unsigned char *qs, const float *x, float scale, int n);

// out = W * x, W row-major Q2_0 [rows x cols] (cols must divide 64).
// data holds rows*cols/64 packed 18-byte blocks (f16 d + 16 two-bit bytes).
void vllm_q2_0_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols);

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
