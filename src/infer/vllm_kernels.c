// vllm_kernels.c -- host-optimized C kernels for vllm-os.
//
// Scalar C paths always compile (gcc, clang, tcc, MSVC, freestanding:
// no libc, no libm, no OS calls). On gcc/clang x86 the hot functions
// carry __attribute__((target("avx2,fma"))) so they get AVX2+FMA codegen
// with PLAIN portable build flags -- no -march=native needed, the binary
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

// One 128-position Q6_K half-block dot (no super-scale applied): 4 groups
// of 16 lanes (offsets +0/+32/+64/+96), one int8 scale per group. Nibble
// unpack reuses the epi64->own-byte trick from the Q4_0 path (epi16
// would cross-contaminate odd bytes); the 2-bit qh fields use epi16
// shifts, whose low-2-bit results always stay in their own byte, and the
// <<4 rebuild is lane-safe because the shifted values are <= 3.
static VLLM_AVX2_FN float q6_k_half_dot_avx2(const unsigned char *ql, const unsigned char *qh,
	const signed char *scales, const float *x) {
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 bias = _mm256_set1_ps(32.0f);
	__m128i m15 = _mm_set1_epi8(0x0f);
	__m128i m3 = _mm_set1_epi8(0x03);
	for (int c = 0; c < 2; c++) {
		const unsigned char *qla = ql + c * 16;
		const unsigned char *qlb = ql + 32 + c * 16;
		__m128i a = _mm_loadu_si128((const __m128i *)qla);
		__m128i b = _mm_loadu_si128((const __m128i *)qlb);
		__m128i h = _mm_loadu_si128((const __m128i *)(qh + c * 16));
		__m128i lo_a = _mm_and_si128(a, m15);
		__m128i hi_a = _mm_and_si128(_mm_srli_epi64(a, 4), m15);
		__m128i lo_b = _mm_and_si128(b, m15);
		__m128i hi_b = _mm_and_si128(_mm_srli_epi64(b, 4), m15);
		__m128i f0 = _mm_slli_epi16(_mm_and_si128(h, m3), 4);
		__m128i f1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), m3), 4);
		__m128i f2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m3), 4);
		__m128i f3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), m3), 4);
		__m128i v0 = _mm_or_si128(lo_a, f0);
		__m128i v1 = _mm_or_si128(lo_b, f1);
		__m128i v2 = _mm_or_si128(hi_a, f2);
		__m128i v3 = _mm_or_si128(hi_b, f3);
		__m256 s0 = _mm256_set1_ps((float)scales[c]);
		__m256 s1 = _mm256_set1_ps((float)scales[c + 2]);
		__m256 s2 = _mm256_set1_ps((float)scales[c + 4]);
		__m256 s3 = _mm256_set1_ps((float)scales[c + 6]);
		const float *xb = x + c * 16;
		__m256 g;
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v0)), bias), s0);
		acc0 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb), acc0);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v0, 8))), bias), s0);
		acc1 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 8), acc1);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v1)), bias), s1);
		acc0 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 32), acc0);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v1, 8))), bias), s1);
		acc1 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 32 + 8), acc1);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v2)), bias), s2);
		acc0 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 64), acc0);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v2, 8))), bias), s2);
		acc1 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 64 + 8), acc1);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v3)), bias), s3);
		acc0 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 96), acc0);
		g = _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v3, 8))), bias), s3);
		acc1 = _mm256_fmadd_ps(g, _mm256_loadu_ps(xb + 96 + 8), acc1);
	}
	__m256 s = _mm256_add_ps(acc0, acc1);
	__m128 hi = _mm256_extractf128_ps(s, 1);
	__m128 lo = _mm256_castps256_ps128(s);
	__m128 v = _mm_add_ps(lo, hi);
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	return _mm_cvtss_f32(v);
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

// f16 (LE bytes) -> f32, freestanding bit trick (mirrors
// infer.f16bits_to_f32 exactly, including subnormal normalization).
static float vllm_f16_le_to_f32(const unsigned char *p) {
	unsigned int bits = (unsigned int)p[0] | ((unsigned int)p[1] << 8);
	unsigned int sign = (bits >> 15) << 31;
	unsigned int exp = (bits >> 10) & 0x1f;
	unsigned int mant = bits & 0x3ff;
	unsigned int f;
	if (exp == 0) {
		if (mant == 0) {
			f = sign;
		} else {
			int e = -14;
			unsigned int m = mant;
			while ((m & 0x400) == 0) {
				m <<= 1;
				e--;
			}
			m &= 0x3ff;
			f = sign | ((unsigned int)(e + 127) << 23) | (m << 13);
		}
	} else if (exp == 31) {
		f = sign | 0x7f800000 | (mant << 13);
	} else {
		f = sign | ((unsigned int)(exp - 15 + 127) << 23) | (mant << 13);
	}
	union {
		unsigned int u;
		float f;
	} z;
	z.u = f;
	return z.f;
}

#if VLLM_HAVE_SIMD
// f16 (LE) -> f32 lookup: exact for all 65536 inputs (built from the bit
// trick above on first use), so per-block scale conversion is branch-free
// with zero numeric change. 256 KB of BSS; init is idempotent (every
// entry always gets the same value), so concurrent first calls from
// worker threads are benign. No libc/libm, freestanding-safe.
static float vllm_f16_lut[65536];
static int vllm_f16_lut_ready = 0;

static void vllm_f16_lut_init(void) {
	if (vllm_f16_lut_ready) {
		return;
	}
	for (unsigned i = 0; i < 65536; i++) {
		unsigned char p[2];
		p[0] = (unsigned char)(i & 0xff);
		p[1] = (unsigned char)(i >> 8);
		vllm_f16_lut[i] = vllm_f16_le_to_f32(p);
	}
	vllm_f16_lut_ready = 1;
}

// vllm_f16_lut_get converts two LE bytes via the table (init first).
static float vllm_f16_lut_get(const unsigned char *p) {
	unsigned i = (unsigned)p[0] | ((unsigned)p[1] << 8);
	return vllm_f16_lut[i];
}

// Horizontal sum of one 8-lane vector (same tree as the per-block dots).
static VLLM_AVX2_FN float vllm_hsum_avx2(__m256 s) {
	__m128 hi = _mm256_extractf128_ps(s, 1);
	__m128 lo = _mm256_castps256_ps128(s);
	__m128 v = _mm_add_ps(lo, hi);
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	return _mm_cvtss_f32(v);
}

// Row-vector Q4_0 accumulation: the four dequant lanes stream across
// blocks into accumulators with ONE horizontal sum per row (not per
// block), and no scalar dependency chain through s. Summation order
// changes vs the scalar mirror (lane-wise across blocks, then reduce)
// -- last-ulp drift, gated by simd_test's 1e-3 agreement band.
static VLLM_AVX2_FN void q4_0_matvec_rows_avx2(const unsigned char *data, const float *x,
	float *out, int rows, int bpr) {
	__m128i m15 = _mm_set1_epi8(0x0f);
	__m256 eight = _mm256_set1_ps(8.0f);
	for (int r = 0; r < rows; r++) {
		__m256 acc0 = _mm256_setzero_ps();
		__m256 acc1 = _mm256_setzero_ps();
		__m256 acc2 = _mm256_setzero_ps();
		__m256 acc3 = _mm256_setzero_ps();
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 18;
			__m256 sc = _mm256_set1_ps(vllm_f16_lut_get(blk));
			__m128i q = _mm_loadu_si128((const __m128i *)(blk + 2));
			__m128i lo = _mm_and_si128(q, m15);
			__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
			__m256 f0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), eight);
			__m256 f1 = _mm256_sub_ps(
				_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), eight);
			__m256 f2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), eight);
			__m256 f3 = _mm256_sub_ps(
				_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), eight);
			const float *xb = x + (size_t)b * 32;
			acc0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, sc), _mm256_loadu_ps(xb), acc0);
			acc1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, sc), _mm256_loadu_ps(xb + 8), acc1);
			acc2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, sc), _mm256_loadu_ps(xb + 16), acc2);
			acc3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, sc), _mm256_loadu_ps(xb + 24), acc3);
		}
		__m256 s = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
		out[r] = vllm_hsum_avx2(s);
	}
}

// Row-vector Q4_1 accumulation: same discipline as the Q4_0 loop, with a
// fifth accumulator for the sum(x)*min offset lane.
static VLLM_AVX2_FN void q4_1_matvec_rows_avx2(const unsigned char *data, const float *x,
	float *out, int rows, int bpr) {
	__m128i m15 = _mm_set1_epi8(0x0f);
	for (int r = 0; r < rows; r++) {
		__m256 acc0 = _mm256_setzero_ps();
		__m256 acc1 = _mm256_setzero_ps();
		__m256 acc2 = _mm256_setzero_ps();
		__m256 acc3 = _mm256_setzero_ps();
		__m256 xacc = _mm256_setzero_ps();
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 20;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 20;
			__m256 sc = _mm256_set1_ps(vllm_f16_lut_get(blk));
			__m256 mn = _mm256_set1_ps(vllm_f16_lut_get(blk + 2));
			__m128i q = _mm_loadu_si128((const __m128i *)(blk + 4));
			__m128i lo = _mm_and_si128(q, m15);
			__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
			__m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
			__m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
			__m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
			__m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
			const float *xb = x + (size_t)b * 32;
			__m256 x0 = _mm256_loadu_ps(xb);
			__m256 x1 = _mm256_loadu_ps(xb + 8);
			__m256 x2 = _mm256_loadu_ps(xb + 16);
			__m256 x3 = _mm256_loadu_ps(xb + 24);
			acc0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, sc), x0, acc0);
			acc1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, sc), x1, acc1);
			acc2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, sc), x2, acc2);
			acc3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, sc), x3, acc3);
			__m256 xs = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
			xacc = _mm256_fmadd_ps(xs, mn, xacc);
		}
		__m256 s = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
		out[r] = vllm_hsum_avx2(s) + vllm_hsum_avx2(xacc);
	}
}
#endif

#if VLLM_HAVE_SIMD
// One 32-elem split-half Q4_0 block dot (no scale applied): byte i holds
// q[i] (low nibble) and q[i+16] (high nibble), values are nibble - 8.
// Unpack trick: epi64 >> 4 moves every byte's HIGH nibble into its own
// low nibble (epi16 would cross-contaminate odd bytes from the neighbor's
// low nibble -- verify with byte pattern 0xAB at odd offsets).
static VLLM_AVX2_FN float q4_0_block_dot_avx2(const unsigned char *qs, const float *x) {
	__m128i q = _mm_loadu_si128((const __m128i *)qs);
	__m128i m15 = _mm_set1_epi8(0x0f);
	__m128i lo = _mm_and_si128(q, m15);
	__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
	__m256 eight = _mm256_set1_ps(8.0f);
	// lo[0..16) dots x[0..16), hi[0..16) dots x[16..32).
	__m256i lo0 = _mm256_cvtepi8_epi32(lo);
	__m256i lo1 = _mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8));
	__m256i hi0 = _mm256_cvtepi8_epi32(hi);
	__m256i hi1 = _mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8));
	__m256 f0 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo0), eight);
	__m256 f1 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo1), eight);
	__m256 f2 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi0), eight);
	__m256 f3 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi1), eight);
	__m256 a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x), _mm256_mul_ps(f1, _mm256_loadu_ps(x + 8)));
	__m256 a1 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + 16), _mm256_mul_ps(f3, _mm256_loadu_ps(x + 24)));
	__m256 s = _mm256_add_ps(a0, a1);
	__m128 hi128 = _mm256_extractf128_ps(s, 1);
	__m128 lo128 = _mm256_castps256_ps128(s);
	__m128 v = _mm_add_ps(lo128, hi128);
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	return _mm_cvtss_f32(v);
}
#endif

float vllm_q4_0_dot_f32(const unsigned char *qs, const float *x, float scale, int n) {
#if VLLM_HAVE_SIMD
	if (n == 32 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		return q4_0_block_dot_avx2(qs, x) * scale;
	}
#endif
	// Scalar mirror of infer.q4_0_dot_row (same order, same rounding).
	float s = 0.0f;
	for (int i = 0; i < n / 2; i++) {
		unsigned char b = qs[i];
		s += (float)((int)(b & 0x0f) - 8) * x[i];
		s += (float)((int)((b >> 4) & 0x0f) - 8) * x[i + n / 2];
	}
	return s * scale;
}

void vllm_q4_0_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols) {
	int bpr = cols / 32;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		q4_0_matvec_rows_avx2(data, x, out, rows, bpr);
		return;
	}
#endif
	// Scalar mirror of infer.q4_0_matvec: per-block dot, scaled per block
	// (same order and rounding as the V kernel).
	for (int r = 0; r < rows; r++) {
		float s = 0.0f;
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 18;
			float scale = vllm_f16_le_to_f32(blk);
			const float *xb = x + (size_t)b * 32;
			float d = 0.0f;
			for (int i = 0; i < 16; i++) {
				unsigned char q = blk[2 + i];
				d += (float)((int)(q & 0x0f) - 8) * xb[i];
				d += (float)((int)((q >> 4) & 0x0f) - 8) * xb[i + 16];
			}
			s += d * scale;
		}
		out[r] = s;
	}
}

// --- IQ4_NL (non-linear 4-bit) -------------------------------------------
// The 16-entry int8 codebook from ggml-common.h `kvalues_iq4nl`. Published
// constants only (no code copied); V mirror is infer.iq4_nl_grid.
static const signed char vllm_iq4nl_lut[16] = {
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

#if VLLM_HAVE_SIMD
// One 32-elem IQ4_NL block dot (no scale). Split-half nibbles like Q4_0,
// but each nibble indexes the non-linear int8 grid: use pshufb to map the
// 16 codes to their grid bytes in one instruction, then widen + FMA.
static VLLM_AVX2_FN float iq4_nl_block_dot_avx2(const unsigned char *qs, const float *x) {
	__m128i q = _mm_loadu_si128((const __m128i *)qs);
	__m128i lut = _mm_loadu_si128((const __m128i *)vllm_iq4nl_lut);
	__m128i m15 = _mm_set1_epi8(0x0f);
	__m128i lo = _mm_and_si128(q, m15);
	__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
	__m128i lo_v = _mm_shuffle_epi8(lut, lo);
	__m128i hi_v = _mm_shuffle_epi8(lut, hi);
	__m256i lo0 = _mm256_cvtepi8_epi32(lo_v);
	__m256i lo1 = _mm256_cvtepi8_epi32(_mm_srli_si128(lo_v, 8));
	__m256i hi0 = _mm256_cvtepi8_epi32(hi_v);
	__m256i hi1 = _mm256_cvtepi8_epi32(_mm_srli_si128(hi_v, 8));
	__m256 f0 = _mm256_cvtepi32_ps(lo0);
	__m256 f1 = _mm256_cvtepi32_ps(lo1);
	__m256 f2 = _mm256_cvtepi32_ps(hi0);
	__m256 f3 = _mm256_cvtepi32_ps(hi1);
	__m256 a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x), _mm256_mul_ps(f1, _mm256_loadu_ps(x + 8)));
	__m256 a1 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + 16), _mm256_mul_ps(f3, _mm256_loadu_ps(x + 24)));
	return vllm_hsum_avx2(_mm256_add_ps(a0, a1));
}
#endif

float vllm_iq4_nl_dot_f32(const unsigned char *qs, const float *x, float scale, int n) {
#if VLLM_HAVE_SIMD
	if (n == 32 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		return iq4_nl_block_dot_avx2(qs, x) * scale;
	}
#endif
	float s = 0.0f;
	for (int i = 0; i < n / 2; i++) {
		unsigned char b = qs[i];
		s += (float)vllm_iq4nl_lut[b & 0x0f] * x[i];
		s += (float)vllm_iq4nl_lut[b >> 4] * x[i + n / 2];
	}
	return s * scale;
}

void vllm_iq4_nl_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols) {
	int bpr = cols / 32;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		for (int r = 0; r < rows; r++) {
			float s = 0.0f;
			const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
			for (int b = 0; b < bpr; b++) {
				const unsigned char *blk = row + (size_t)b * 18;
				float scale = vllm_f16_le_to_f32(blk);
				s += iq4_nl_block_dot_avx2(blk + 2, x + (size_t)b * 32) * scale;
			}
			out[r] = s;
		}
		return;
	}
#endif
	for (int r = 0; r < rows; r++) {
		float s = 0.0f;
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 18;
			s += vllm_iq4_nl_dot_f32(blk + 2, x + (size_t)b * 32, vllm_f16_le_to_f32(blk), 32);
		}
		out[r] = s;
	}
}

// --- Q2_0 (signed-scale 2-bit) -------------------------------------------
// One 64-elem block dot. The 2-bit codes are expanded into a small local
// stack buffer (one f32 per element, no caller-visible materialization),
// then the dot runs as eight AVX2 FMAs. The scalar mirror stays bit-close
// to infer.q2_0_dot_row.
#if VLLM_HAVE_SIMD
static VLLM_AVX2_FN float q2_0_block_dot_avx2(const unsigned char *qs, const float *x) {
	float vals[64];
	for (int b = 0; b < 16; b++) {
		unsigned char q = qs[b];
		vals[4*b + 0] = (float)((int)((q >> 0) & 3) - 1);
		vals[4*b + 1] = (float)((int)((q >> 2) & 3) - 1);
		vals[4*b + 2] = (float)((int)((q >> 4) & 3) - 1);
		vals[4*b + 3] = (float)((int)((q >> 6) & 3) - 1);
	}
	__m256 acc = _mm256_setzero_ps();
	for (int i = 0; i < 64; i += 8) {
		acc = _mm256_fmadd_ps(_mm256_loadu_ps(vals + i), _mm256_loadu_ps(x + i), acc);
	}
	return vllm_hsum_avx2(acc);
}
#endif

float vllm_q2_0_dot_f32(const unsigned char *qs, const float *x, float scale, int n) {
#if VLLM_HAVE_SIMD
	if (n == 64 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		return q2_0_block_dot_avx2(qs, x) * scale;
	}
#endif
	float s = 0.0f;
	for (int j = 0; j < n; j++) {
		int code = (int)((qs[j / 4] >> ((j % 4) * 2)) & 3);
		s += (float)(code - 1) * x[j];
	}
	return s * scale;
}

void vllm_q2_0_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols) {
	int bpr = cols / 64;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		for (int r = 0; r < rows; r++) {
			float s = 0.0f;
			const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
			for (int b = 0; b < bpr; b++) {
				const unsigned char *blk = row + (size_t)b * 18;
				s += q2_0_block_dot_avx2(blk + 2, x + (size_t)b * 64) * vllm_f16_le_to_f32(blk);
			}
			out[r] = s;
		}
		return;
	}
#endif
	for (int r = 0; r < rows; r++) {
		float s = 0.0f;
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 18;
			s += vllm_q2_0_dot_f32(blk + 2, x + (size_t)b * 64, vllm_f16_le_to_f32(blk), 64);
		}
		out[r] = s;
	}
}

#if VLLM_HAVE_SIMD
// One 32-elem split-half Q4_1 block dot (no scale/min applied by caller
// confusion -- applied here): byte i holds q[i] (low nibble, 0..15) and
// q[i+16] (high nibble). Same epi64 unpack trick as the Q4_0 path.
// Returns dot(q,x)*scale + sum(x)*min, mirroring infer.q4_1_dot_row.
static VLLM_AVX2_FN float q4_1_block_dot_avx2(const unsigned char *qs, const float *x,
	float scale, float min) {
	__m128i q = _mm_loadu_si128((const __m128i *)qs);
	__m128i m15 = _mm_set1_epi8(0x0f);
	__m128i lo = _mm_and_si128(q, m15);
	__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
	__m256i lo0 = _mm256_cvtepi8_epi32(lo);
	__m256i lo1 = _mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8));
	__m256i hi0 = _mm256_cvtepi8_epi32(hi);
	__m256i hi1 = _mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8));
	__m256 f0 = _mm256_cvtepi32_ps(lo0);
	__m256 f1 = _mm256_cvtepi32_ps(lo1);
	__m256 f2 = _mm256_cvtepi32_ps(hi0);
	__m256 f3 = _mm256_cvtepi32_ps(hi1);
	__m256 x0 = _mm256_loadu_ps(x);
	__m256 x1 = _mm256_loadu_ps(x + 8);
	__m256 x2 = _mm256_loadu_ps(x + 16);
	__m256 x3 = _mm256_loadu_ps(x + 24);
	__m256 a0 = _mm256_fmadd_ps(f0, x0, _mm256_mul_ps(f1, x1));
	__m256 a1 = _mm256_fmadd_ps(f2, x2, _mm256_mul_ps(f3, x3));
	__m256 s = _mm256_add_ps(a0, a1);
	__m128 hi128 = _mm256_extractf128_ps(s, 1);
	__m128 lo128 = _mm256_castps256_ps128(s);
	__m128 v = _mm_add_ps(lo128, hi128);
	v = _mm_add_ps(v, _mm_movehl_ps(v, v));
	v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
	float dot = _mm_cvtss_f32(v);
	// sum(x) over the 32 lanes for the offset term.
	__m256 xs = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
	__m128 xhi = _mm256_extractf128_ps(xs, 1);
	__m128 xlo = _mm256_castps256_ps128(xs);
	__m128 xv = _mm_add_ps(xlo, xhi);
	xv = _mm_add_ps(xv, _mm_movehl_ps(xv, xv));
	xv = _mm_add_ss(xv, _mm_shuffle_ps(xv, xv, 1));
	float xsum = _mm_cvtss_f32(xv);
	return dot * scale + xsum * min;
}
#endif

float vllm_q4_1_dot_f32(const unsigned char *qs, const float *x, float scale, float min) {
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		return q4_1_block_dot_avx2(qs, x, scale, min);
	}
#endif
	// Scalar mirror of infer.q4_1_dot_row (same order, same rounding).
	float s = 0.0f;
	float xsum = 0.0f;
	for (int i = 0; i < 16; i++) {
		unsigned char b = qs[i];
		float lo = (float)(int)(b & 0x0f);
		float hi = (float)(int)((b >> 4) & 0x0f);
		s += lo * x[i] + hi * x[i + 16];
		xsum += x[i] + x[i + 16];
	}
	return s * scale + xsum * min;
}

void vllm_q4_1_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols) {
	int bpr = cols / 32;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		q4_1_matvec_rows_avx2(data, x, out, rows, bpr);
		return;
	}
#endif
	// Scalar mirror of infer.q4_1_matvec: per-block fused dot (same order
	// and rounding as the V kernel).
	for (int r = 0; r < rows; r++) {
		float s = 0.0f;
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 20;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 20;
			float scale = vllm_f16_le_to_f32(blk);
			float min = vllm_f16_le_to_f32(blk + 2);
			const float *xb = x + (size_t)b * 32;
			float d = 0.0f;
			float xsum = 0.0f;
			for (int i = 0; i < 16; i++) {
				unsigned char q = blk[4 + i];
				float lo = (float)(int)(q & 0x0f);
				float hi = (float)(int)((q >> 4) & 0x0f);
				d += lo * xb[i] + hi * xb[i + 16];
				xsum += xb[i] + xb[i + 16];
			}
			s += d * scale + xsum * min;
		}
		out[r] = s;
	}
}

void vllm_q6_k_matvec_f32(const unsigned char *data, const float *x, float *out, int rows,
	int cols) {
	int bpr = cols / 256;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		for (int r = 0; r < rows; r++) {
			float s = 0.0f;
			const unsigned char *row = data + (size_t)r * (size_t)bpr * 210;
			for (int b = 0; b < bpr; b++) {
				const unsigned char *blk = row + (size_t)b * 210;
				const unsigned char *ql = blk;
				const unsigned char *qh = blk + 128;
				const signed char *scales = (const signed char *)(blk + 192);
				float d = vllm_f16_lut_get(blk + 208);
				const float *xb = x + (size_t)b * 256;
				float t = q6_k_half_dot_avx2(ql, qh, scales, xb) +
					q6_k_half_dot_avx2(ql + 64, qh + 32, scales + 8, xb + 128);
				s += t * d;
			}
			out[r] = s;
		}
		return;
	}
#endif
	// Scalar mirror of infer.q6_k_matvec/q6_k_dot_block (same j/l order
	// and per-term rounding, so this path is bit-identical to the V
	// kernel; the AVX2 path above only reassociates the per-half hsum).
	// Scale indices need no division: p0=j*128+l gives
	// p0/16 = j*8+(l>=16), and the +32/+64/+96 lanes add exactly 2/4/6.
	for (int r = 0; r < rows; r++) {
		float s = 0.0f;
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 210;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 210;
			const unsigned char *ql = blk;
			const unsigned char *qh = blk + 128;
			const signed char *scales = (const signed char *)(blk + 192);
			float d = vllm_f16_le_to_f32(blk + 208);
			const float *xb = x + (size_t)b * 256;
			float t = 0.0f;
			for (int j = 0; j < 2; j++) {
				int sbase = j * 8;
				for (int l = 0; l < 32; l++) {
					unsigned char qh_b = qh[j * 32 + l];
					unsigned char q0 = ql[j * 64 + l];
					unsigned char q1 = ql[j * 64 + l + 32];
					int f0 = (qh_b & 3) << 4;
					int f1 = ((qh_b >> 2) & 3) << 4;
					int f2 = ((qh_b >> 4) & 3) << 4;
					int f3 = ((qh_b >> 6) & 3) << 4;
					int v0 = (int)((q0 & 0x0f) | (unsigned char)f0) - 32;
					int v1 = (int)((q1 & 0x0f) | (unsigned char)f1) - 32;
					int v2 = (int)((q0 >> 4) | (unsigned char)f2) - 32;
					int v3 = (int)((q1 >> 4) | (unsigned char)f3) - 32;
					int p0 = j * 128 + l;
					int s0 = sbase + (l >= 16);
					t += (float)v0 * (float)scales[s0] * xb[p0] +
						(float)v1 * (float)scales[s0 + 2] * xb[p0 + 32] +
						(float)v2 * (float)scales[s0 + 4] * xb[p0 + 64] +
						(float)v3 * (float)scales[s0 + 6] * xb[p0 + 96];
				}
			}
			s += t * d;
		}
		out[r] = s;
	}
}

// Freestanding 1/sqrt (no libm anywhere in this file, so no sqrtf):
// bit-level seed + two Newton iterations, ~1e-7 relative error.
static float vllm_rsqrt(float x) {
	union {
		float f;
		unsigned int u;
	} z = { x };
	z.u = 0x5f3759df - (z.u >> 1);
	z.f *= 1.5f - 0.5f * x * z.f * z.f;
	z.f *= 1.5f - 0.5f * x * z.f * z.f;
	return z.f;
}

void vllm_rmsnorm_f32(float *x, const float *w, int n, float eps) {
	// Sum of squares in double (matches infer.rmsnorm_in_place exactly).
	double ss = 0.0;
	for (int i = 0; i < n; i++) {
		double v = (double)x[i];
		ss += v * v;
	}
	float scale = vllm_rsqrt((float)(ss / (double)n + (double)eps));
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
// u.i = x * 2^23/log(2) + 127*2^23. Worst-case rel err ~0.5% -- sampling
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

// --- Batched matmul for speculative verification -------------------------
// Weight blocks are dequantized ONCE and dotted against all k activation
// vectors, so weight traffic is amortized across the draft batch. outs and
// xs are row-major [k][rows] / [k][cols]. The per-token accumulation order
// matches the single-token kernels block-by-block; tiny f32 reassociation
// drift is possible (the same class already documented for row-vector Q4).

static void vllm_q4_0_unpack(const unsigned char *qs, float *vals) {
	for (int i = 0; i < 16; i++) {
		unsigned char b = qs[i];
		vals[i] = (float)((int)(b & 0x0f) - 8);
		vals[i + 16] = (float)((int)((b >> 4) & 0x0f) - 8);
	}
}

VLLM_AVX2_FN void vllm_q4_0_matmul_f32(const unsigned char *data, const float *xs, float *outs, int k,
	int rows, int cols) {
	int bpr = cols / 32;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		__m128i m15 = _mm_set1_epi8(0x0f);
		__m256 eight = _mm256_set1_ps(8.0f);
		// One accumulator per token, chunked at 8 to stay in YMM registers.
		// Weights are dequantized once per block and shared across the chunk;
		// one horizontal sum per row per token (not per block).
		for (int t0 = 0; t0 < k; t0 += 8) {
			int kk = k - t0 < 8 ? k - t0 : 8;
			for (int r = 0; r < rows; r++) {
				__m256 acc[8];
				for (int i = 0; i < kk; i++) acc[i] = _mm256_setzero_ps();
				const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
				for (int b = 0; b < bpr; b++) {
					const unsigned char *blk = row + (size_t)b * 18;
					__m256 sc = _mm256_set1_ps(vllm_f16_lut_get(blk));
					__m128i q = _mm_loadu_si128((const __m128i *)(blk + 2));
					__m128i lo = _mm_and_si128(q, m15);
					__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
					__m256 f0 = _mm256_mul_ps(
						_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), eight), sc);
					__m256 f1 = _mm256_mul_ps(
						_mm256_sub_ps(_mm256_cvtepi32_ps(
							_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), eight), sc);
					__m256 f2 = _mm256_mul_ps(
						_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), eight), sc);
					__m256 f3 = _mm256_mul_ps(
						_mm256_sub_ps(_mm256_cvtepi32_ps(
							_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), eight), sc);
					for (int i = 0; i < kk; i++) {
						const float *x = xs + (size_t)(t0 + i) * cols + (size_t)b * 32;
						__m256 p = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x), _mm256_setzero_ps());
						p = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + 8), p);
						p = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + 16), p);
						p = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x + 24), p);
						acc[i] = _mm256_add_ps(acc[i], p);
					}
				}
				for (int i = 0; i < kk; i++) {
					outs[(size_t)(t0 + i) * rows + r] = vllm_hsum_avx2(acc[i]);
				}
			}
		}
		return;
	}
#endif
	for (int r = 0; r < rows; r++) {
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 18;
		for (int t = 0; t < k; t++) outs[(size_t)t * rows + r] = 0.0f;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 18;
			float scale = vllm_f16_le_to_f32(blk);
			for (int t = 0; t < k; t++) {
				const float *x = xs + (size_t)t * cols + (size_t)b * 32;
				float vals[32];
				vllm_q4_0_unpack(blk + 2, vals);
				outs[(size_t)t * rows + r] += vllm_dot_f32(vals, x, 32) * scale;
			}
		}
	}
}

VLLM_AVX2_FN void vllm_q4_1_matmul_f32(const unsigned char *data, const float *xs, float *outs, int k,
	int rows, int cols) {
	int bpr = cols / 32;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		__m128i m15 = _mm_set1_epi8(0x0f);
		for (int t0 = 0; t0 < k; t0 += 8) {
			int kk = k - t0 < 8 ? k - t0 : 8;
			for (int r = 0; r < rows; r++) {
				__m256 acc[8];
				__m256 xacc[8];
				for (int i = 0; i < kk; i++) {
					acc[i] = _mm256_setzero_ps();
					xacc[i] = _mm256_setzero_ps();
				}
				const unsigned char *row = data + (size_t)r * (size_t)bpr * 20;
				for (int b = 0; b < bpr; b++) {
					const unsigned char *blk = row + (size_t)b * 20;
					__m256 sc = _mm256_set1_ps(vllm_f16_lut_get(blk));
					__m256 mn = _mm256_set1_ps(vllm_f16_lut_get(blk + 2));
					__m128i q = _mm_loadu_si128((const __m128i *)(blk + 4));
					__m128i lo = _mm_and_si128(q, m15);
					__m128i hi = _mm_and_si128(_mm_srli_epi64(q, 4), m15);
					__m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), sc);
					__m256 f1 = _mm256_mul_ps(
						_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), sc);
					__m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), sc);
					__m256 f3 = _mm256_mul_ps(
						_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), sc);
					for (int i = 0; i < kk; i++) {
						const float *x = xs + (size_t)(t0 + i) * cols + (size_t)b * 32;
						__m256 x0 = _mm256_loadu_ps(x);
						__m256 x1 = _mm256_loadu_ps(x + 8);
						__m256 x2 = _mm256_loadu_ps(x + 16);
						__m256 x3 = _mm256_loadu_ps(x + 24);
						__m256 p = _mm256_fmadd_ps(f0, x0, _mm256_setzero_ps());
						p = _mm256_fmadd_ps(f1, x1, p);
						p = _mm256_fmadd_ps(f2, x2, p);
						p = _mm256_fmadd_ps(f3, x3, p);
						acc[i] = _mm256_add_ps(acc[i], p);
						__m256 xs4 = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
						xacc[i] = _mm256_fmadd_ps(xs4, mn, xacc[i]);
					}
				}
				for (int i = 0; i < kk; i++) {
					outs[(size_t)(t0 + i) * rows + r] =
						vllm_hsum_avx2(acc[i]) + vllm_hsum_avx2(xacc[i]);
				}
			}
		}
		return;
	}
#endif
	for (int r = 0; r < rows; r++) {
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 20;
		for (int t = 0; t < k; t++) outs[(size_t)t * rows + r] = 0.0f;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 20;
			float scale = vllm_f16_le_to_f32(blk);
			float min = vllm_f16_le_to_f32(blk + 2);
			for (int t = 0; t < k; t++) {
				const float *x = xs + (size_t)t * cols + (size_t)b * 32;
				float vals[32];
				for (int i = 0; i < 16; i++) {
					unsigned char q = blk[4 + i];
					vals[i] = (float)(int)(q & 0x0f);
					vals[i + 16] = (float)(int)((q >> 4) & 0x0f);
				}
				float dot = vllm_dot_f32(vals, x, 32);
				float xsum = 0.0f;
				for (int i = 0; i < 32; i++) xsum += x[i];
				outs[(size_t)t * rows + r] += dot * scale + xsum * min;
			}
		}
	}
}

VLLM_AVX2_FN void vllm_q6_k_matmul_f32(const unsigned char *data, const float *xs, float *outs, int k,
	int rows, int cols) {
	int bpr = cols / 256;
#if VLLM_HAVE_SIMD
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		vllm_f16_lut_init();
		for (int r = 0; r < rows; r++) {
			const unsigned char *row = data + (size_t)r * (size_t)bpr * 210;
			for (int t = 0; t < k; t++) outs[(size_t)t * rows + r] = 0.0f;
			for (int b = 0; b < bpr; b++) {
				const unsigned char *blk = row + (size_t)b * 210;
				const signed char *sc = (const signed char *)(blk + 192);
				float d = vllm_f16_lut_get(blk + 208);
				for (int t = 0; t < k; t++) {
					const float *x = xs + (size_t)t * cols + (size_t)b * 256;
					float v = q6_k_half_dot_avx2(blk, blk + 128, sc, x) +
						q6_k_half_dot_avx2(blk + 64, blk + 160, sc + 8, x + 128);
					outs[(size_t)t * rows + r] += v * d;
				}
			}
		}
		return;
	}
#endif
	for (int r = 0; r < rows; r++) {
		const unsigned char *row = data + (size_t)r * (size_t)bpr * 210;
		for (int t = 0; t < k; t++) outs[(size_t)t * rows + r] = 0.0f;
		for (int b = 0; b < bpr; b++) {
			const unsigned char *blk = row + (size_t)b * 210;
			const unsigned char *ql = blk;
			const unsigned char *qh = blk + 128;
			const signed char *sc = (const signed char *)(blk + 192);
			float d = vllm_f16_le_to_f32(blk + 208);
			for (int t = 0; t < k; t++) {
				const float *x = xs + (size_t)t * cols + (size_t)b * 256;
				float vals[256];
				for (int j = 0; j < 2; j++) {
					for (int l = 0; l < 32; l++) {
						unsigned char h = qh[j * 32 + l];
						int f0 = (h & 3) << 4;
						int f1 = ((h >> 2) & 3) << 4;
						int f2 = ((h >> 4) & 3) << 4;
						int f3 = ((h >> 6) & 3) << 4;
						int p = j * 128 + l;
						vals[p] = (float)(int)((ql[j * 64 + l] & 0x0f) | f0) - 32.0f;
						vals[p + 32] = (float)(int)((ql[j * 64 + l + 32] & 0x0f) | f1) - 32.0f;
						vals[p + 64] = (float)(int)((ql[j * 64 + l] >> 4) | f2) - 32.0f;
						vals[p + 96] = (float)(int)((ql[j * 64 + l + 32] >> 4) | f3) - 32.0f;
						for (int q = 0; q < 4; q++) {
							int idx = p + q * 32;
							vals[idx] *= (float)sc[idx / 16];
						}
					}
				}
				outs[(size_t)t * rows + r] += vllm_dot_f32(vals, x, 256) * d;
			}
		}
	}
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

