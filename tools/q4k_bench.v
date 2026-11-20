module main

// tools/q4k_bench.v — host-only microbenchmark for the Phase 11 Q4_K
// row-vector kernel (src/infer/quant.v q4_k_matvec vs simd_c.v
// q4_k_matvec_c). Feeds both a synthetic Q4_K blob in the exact
// 142-byte / 256-element GGML super-block layout and reports median
// us/call for two realistic shapes plus the max absolute output
// difference between the two paths. Supports Phase 12 perf work and
// the docs/perf.md rows.
//
// No string interpolation anywhere: the pinned V 0.5.2 release has a
// cgen quirk in tiny module-main binaries that mangles interpolated
// print/println strings, so all values are emitted via .str() calls.
//
// Usage:
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o q4kbench tools/q4k_bench.v
//   ./q4kbench
//
// Note: outside -prod (or without AVX2 / with VLLM_NO_SIMD=1),
// q4_k_matvec_c falls back to the pure-V kernel, so the two median
// numbers converge and max_abs_diff is 0. Build with -prod -cc gcc
// to exercise the dispatched AVX2 kernel.
import infer
import time

fn q4abs(x f32) f32 {
	return if x < 0 { -x } else { x }
}

// q4_k_make_data builds a row-major [rows x cols] Q4_K weight blob using
// the exact block layout from src/infer/quant.v: per 256-element
// super-block, f16 d (2B) + f16 dmin (2B) + 12 packed scale/min bytes +
// 128 nibble bytes = infer.q4_k_block_bytes (142). d and dmin are the
// finite f16 value 0.5 (bits 0x3800); all other bytes use a fixed finite
// pattern, so every dequantized value stays finite (no NaN/Inf).
fn q4_k_make_data(rows int, cols int) []u8 {
	blocks_per_row := cols / infer.qk_k
	mut data := []u8{len: rows * blocks_per_row * infer.q4_k_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. blocks_per_row {
			off := (r * blocks_per_row + b) * infer.q4_k_block_bytes
			data[off] = 0x00
			data[off + 1] = 0x38 // d = 0.5
			data[off + 2] = 0x00
			data[off + 3] = 0x38 // dmin = 0.5
		}
	}
	return data
}

// median returns the middle element after an in-place insertion sort.
// Hand-rolled to avoid relying on array.sort() semantics; reps is small.
fn median(mut xs []f64) f64 {
	for i in 1 .. xs.len {
		key := xs[i]
		mut j := i - 1
		for j >= 0 && xs[j] > key {
			xs[j + 1] = xs[j]
			j--
		}
		xs[j + 1] = key
	}
	return xs[xs.len / 2]
}

fn bench_one(rows int, cols int, reps int) {
	data := q4_k_make_data(rows, cols)
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut out_v := []f32{len: rows}
	mut out_c := []f32{len: rows}

	// Warm up both paths (and let CPU frequency settle).
	infer.q4_k_matvec(data, rows, cols, x, mut out_v)
	infer.q4_k_matvec_c(data, rows, cols, x, mut out_c)

	// Accuracy: max abs difference over the two output vectors.
	mut max_diff := f32(0)
	for i in 0 .. rows {
		d := q4abs(out_v[i] - out_c[i])
		if d > max_diff {
			max_diff = d
		}
	}

	// Time each path per call; collect samples for a median-ish result.
	mut v_samples := []f64{len: reps}
	for i in 0 .. reps {
		sw := time.new_stopwatch()
		infer.q4_k_matvec(data, rows, cols, x, mut out_v)
		v_samples[i] = f64(sw.elapsed().microseconds())
	}
	mut c_samples := []f64{len: reps}
	for i in 0 .. reps {
		sw := time.new_stopwatch()
		infer.q4_k_matvec_c(data, rows, cols, x, mut out_c)
		c_samples[i] = f64(sw.elapsed().microseconds())
	}
	v_med := median(mut v_samples)
	c_med := median(mut c_samples)

	print('q4k rows=')
	print(rows.str())
	print(' cols=')
	print(cols.str())
	print(' avx2=')
	println(infer.has_avx2().str())
	print('  pure-V median_us=')
	println(v_med.str())
	print('  C/AVX2 median_us=')
	println(c_med.str())
	print('  max_abs_diff=')
	println(max_diff.str())
}

fn main() {
	reps := 21
	// Vocab-head-like: many output rows over a 1024-wide hidden state.
	bench_one(2048, 1024, reps)
	// Layer-like: 1024 rows over a 2048-wide hidden state.
	bench_one(1024, 2048, reps)
	println('Q4K-BENCH-OK')
}
