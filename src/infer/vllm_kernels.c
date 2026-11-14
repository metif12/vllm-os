// vllm_kernels.c — host-optimized C kernels for vllm-os.
//
// Scalar C paths always compile (gcc, clang, tcc, MSVC, freestanding:
// no libc, no libm, no OS calls). On gcc/clang x86 the hot functions
// carry __attribute__((target("avx2,fma"))) so they get AVX2+FMA codegen
// with PLAIN portable build flags — no -march=native needed, the binary
// still runs anywhere (runtime dispatch via vllm_has_avx2()).
// The bare-metal port links this file as-is.

#include "vllm_kernels.h"

#include <stddef.h>

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#define VLLM_HAVE_SIMD 1
#define VLLM_AVX2_FN __attribute__((target("avx2,fma")))
#include <immintrin.h>
#else
#define VLLM_HAVE_SIMD 0
#define VLLM_AVX2_FN
#endif

int vllm_has_avx2(void) {
#if VLLM_HAVE_SIMD
	// Compile-time target is generic, so confirm at runtime.
	return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
	return 0;
#endif
}

#if VLLM_HAVE_SIMD
// AVX2+FMA fast paths (only ever called when vllm_has_avx2() is true).
static VLLM_AVX2_FN void matvec_avx2(const float *w, const float *x, float *out, int rows,
	int cols) {
	for (int r = 0; r < rows; r++) {
		const float *row = w + (size_t)r * (size_t)cols;
		__m256 s0 = _mm256_setzero_ps();
		__m256 s1 = _mm256_setzero_ps();
		__m256 s2 = _mm256_setzero_ps();
		__m256 s3 = _mm256_setzero_ps();
		int c = 0;
		int lim = cols & ~31;
		for (; c < lim; c += 32) {
			__m256 x0 = _mm256_loadu_ps(x + c);
			__m256 x1 = _mm256_loadu_ps(x + c + 8);
			__m256 x2 = _mm256_loadu_ps(x + c + 16);
			__m256 x3 = _mm256_loadu_ps(x + c + 24);
			s0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), x0, s0);
			s1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8), x1, s1);
			s2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), x2, s2);
			s3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), x3, s3);
		}
		__m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
		__m128 hi = _mm256_extractf128_ps(s, 1);
		__m128 lo = _mm256_castps256_ps128(s);
		__m128 v = _mm_add_ps(lo, hi);
		v = _mm_add_ps(v, _mm_movehl_ps(v, v));
		v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
		float total = _mm_cvtss_f32(v);
		for (; c < cols; c++) {
			total += row[c] * x[c];
		}
		out[r] = total;
	}
}

static VLLM_AVX2_FN float q8_dot_avx2(const signed char *qs, const float *x, float scale,
	int n) {
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	int i = 0;
	int lim = n & ~15;
	for (; i < lim; i += 16) {
		__m128i b0 = _mm_loadl_epi64((const __m128i *)(qs + i));
		__m128i b1 = _mm_loadl_epi64((const __m128i *)(qs + i + 8));
		__m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0));
		__m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
		acc0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + i), acc0);
		acc1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + i + 8), acc1);
	}
	__m256 s = _mm256_add_ps(acc0, acc1);
	__m128 hi = _mm256_extractf128_ps(s, 1);
	__m128 lo = _mm256_castps256_ps128(s);
	__m128 v = _mm_add_ps(lo, hi);
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	float total = _mm_cvtss_f32(v);
	for (; i < n; i++) {
		total += (float)qs[i] * x[i];
	}
	return total * scale;
}

static VLLM_AVX2_FN void rmsnorm_scale_avx2(float *x, const float *w, int n, float scale) {
	__m256 sc = _mm256_set1_ps(scale);
	int i = 0;
	int lim = n & ~7;
	for (; i < lim; i += 8) {
		__m256 xv = _mm256_loadu_ps(x + i);
		__m256 wv = _mm256_loadu_ps(w + i);
		_mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(xv, sc), wv));
	}
	for (; i < n; i++) {
		x[i] = x[i] * scale * w[i];
	}
}

static VLLM_AVX2_FN float dot_avx2(const float *a, const float *b, int n) {
	__m256 s0 = _mm256_setzero_ps();
	__m256 s1 = _mm256_setzero_ps();
	int i = 0;
	int lim = n & ~15;
	for (; i < lim; i += 16) {
		s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
		s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
	}
	__m256 s = _mm256_add_ps(s0, s1);
	__m128 hi = _mm256_extractf128_ps(s, 1);
	__m128 lo = _mm256_castps256_ps128(s);
	__m128 v = _mm_add_ps(lo, hi);
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	float total = _mm_cvtss_f32(v);
	for (; i < n; i++) {
		total += a[i] * b[i];
	}
	return total;
}
#endif

void vllm_matvec_f32(const float *w, const float *x, float *out, int rows, int cols) {
#if VLLM_HAVE_SIMD
	// Fast path selected per call: portable binary, full speed on AVX2.
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		matvec_avx2(w, x, out, rows, cols);
		return;
	}
#endif
	// Scalar 4-accumulator mirror of infer.matvec (same numerics order).
	for (int r = 0; r < rows; r++) {
		const float *row = w + (size_t)r * (size_t)cols;
		float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
		int c = 0;
		for (; c + 4 <= cols; c += 4) {
			s0 += row[c] * x[c];
			s1 += row[c + 1] * x[c + 1];
			s2 += row[c + 2] * x[c + 2];
			s3 += row[c + 3] * x[c + 3];
		}
		for (; c < cols; c++) {
			s0 += row[c] * x[c];
		}
		out[r] = s0 + s1 + s2 + s3;
	}
}

float vllm_q8_dot_f32(const signed char *qs, const float *x, float scale, int n) {
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		return q8_dot_avx2(qs, x, scale, n);
	}
#endif
	float s = 0.0f;
	for (int i = 0; i < n; i++) {
		s += (float)qs[i] * x[i];
	}
	return s * scale;
}

float vllm_q4_0_dot_f32(const unsigned char *qs, const float *x, float scale, int n) {
	float s = 0.0f;
	for (int i = 0; i < n; i += 2) {
		unsigned char b = qs[i >> 1];
		s += (float)((int)(b & 0x0f) - 8) * x[i];
		s += (float)((int)((b >> 4) & 0x0f) - 8) * x[i + 1];
	}
	return s * scale;
}

void vllm_rmsnorm_f32(float *x, const float *w, int n, float eps) {
	// Sum of squares in double (matches infer.rmsnorm_in_place exactly).
	double ss = 0.0;
	for (int i = 0; i < n; i++) {
		double v = (double)x[i];
		ss += v * v;
	}
	float scale = (float)(1.0 / __builtin_sqrt(ss / (double)n + (double)eps));
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		rmsnorm_scale_avx2(x, w, n, scale);
		return;
	}
#endif
	for (int i = 0; i < n; i++) {
		x[i] = x[i] * scale * w[i];
	}
}

float vllm_dot_f32(const float *a, const float *b, int n) {
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		return dot_avx2(a, b, n);
	}
#endif
	float s = 0.0f;
	for (int i = 0; i < n; i++) {
		s += a[i] * b[i];
	}
	return s;
}

// Branchless Schraudolph exp, no libm: reinterpret float bits.
// u.i = x * 2^23/log(2) + 127*2^23. Worst-case rel err ~0.5% — sampling
// only, never scoring. Clamped to avoid inf/nan on wild logits.
static float vllm_expf_fast(float x) {
	union {
		float f;
		unsigned int u;
	} z;
	if (x < -88.0f) {
		return 0.0f;
	}
	if (x > 88.0f) {
		return 3.4028235e+38f;
	}
	z.u = (unsigned int)(x * 12102203.0f + 1065353216.0f);
	return z.f;
}

void vllm_softmax_scaled_fast_f32(float *xs, int n, float scale) {
	float m = xs[0] * scale;
	for (int i = 1; i < n; i++) {
		float v = xs[i] * scale;
		if (v > m) {
			m = v;
		}
	}
	float s = 0.0f;
	for (int i = 0; i < n; i++) {
		float e = vllm_expf_fast(xs[i] * scale - m);
		xs[i] = e;
		s += e;
	}
	float inv = 1.0f / s;
	for (int i = 0; i < n; i++) {
		xs[i] *= inv;
	}
}
