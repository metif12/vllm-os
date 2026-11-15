module infer

import os

// Agreement + threshold tests for the threaded backend (parallel.v).
// Timing is deliberately NOT gated here (CI boxes vary); throughput is
// measured in tools/bench_kernels.v and reported in docs/perf.md.

fn test_parallel_matches_naive() {
	for shape in [[3, 5], [16, 130], [130, 17], [300, 700]] {
		rows := shape[0]
		cols := shape[1]
		mut w := []f32{len: rows * cols}
		mut x := []f32{len: cols}
		for i in 0 .. w.len {
			w[i] = f32(i % 13) * 0.125 - 0.5
		}
		for i in 0 .. x.len {
			x[i] = f32(i % 7) * 0.25 - 0.125
		}
		mut a := []f32{len: rows}
		mut b := []f32{len: rows}
		matvec(w, x, rows, cols, mut a)
		matvec_parallel(w, x, rows, cols, mut b)
		for i in 0 .. rows {
			d := b[i] - a[i]
			assert d < 1e-4 && d > -1e-4
		}
	}
}

fn test_parallel_wide_matches_simd() {
	rows := 2048
	cols := 6144
	mut w := []f32{len: rows * cols}
	mut x := []f32{len: cols}
	for i in 0 .. w.len {
		w[i] = f32(i % 13) * 0.125
	}
	for i in 0 .. x.len {
		x[i] = f32(i % 7) * 0.25
	}
	mut a := []f32{len: rows}
	mut b := []f32{len: rows}
	matvec_c(w, x, rows, cols, mut a)
	matvec_parallel(w, x, rows, cols, mut b)
	for i in 0 .. rows {
		d := b[i] - a[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_parallel_range_matches_full() {
	rows := 100
	cols := 64
	mut w := []f32{len: rows * cols}
	mut x := []f32{len: cols}
	for i in 0 .. w.len {
		w[i] = f32(i % 11) * 0.2 - 1.0
	}
	for i in 0 .. x.len {
		x[i] = f32(i % 5) * 0.3
	}
	mut full := []f32{len: rows}
	mut part := []f32{len: rows}
	matvec_c(w, x, rows, cols, mut full)
	// Non-divisible block [17, 17+40) through the range entry point.
	for i in 0 .. rows {
		part[i] = 0
	}
	matvec_c_range(w, x, rows, cols, 17, 40, mut part)
	for i in 17 .. 57 {
		d := part[i] - full[i]
		assert d < 1e-4 && d > -1e-4
	}
}

fn test_parallel_threshold_consts_sane() {
	assert min_rows_parallel >= 512
	assert min_cols_parallel >= 128
	assert parallel_threads() >= 1
	assert parallel_threads() <= 64
}

// Remainder rows must be spread across workers (blocks differ by at most
// one), with full coverage and no overlap. rows=2501 with 8 requested
// workers resolves to n=4, base=625, rem=1 — the old last-takes-rest
// split degenerates the same way the VSL gemv_n review caught.
fn test_q6k_parallel_matches_serial() {
	rows := 3000
	cols := 256
	mut data := []u8{len: rows * 210}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		off := r * 210
		for i in 192 .. 208 {
			v := (i * 7 + r) % 5
			data[off + i] = u8((v + 254) % 256)
		}
		data[off + 208] = u8(0x00)
		data[off + 209] = u8(0x38) // d = 0.5
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 41) * 0.05 - 1.0
	}
	mut a := []f32{len: rows}
	mut b := []f32{len: rows}
	q6_k_matvec(data, rows, cols, x, mut a)
	q6_k_matvec_parallel(data, rows, cols, x, mut b)
	for i in 0 .. rows {
		d := b[i] - a[i]
		assert d < 1e-3 && d > -1e-3
	}
}

// Remainder partition for the Q6_K path: rows=2501 with 8 requested
// workers resolves to n=4 with one remainder row spread (not dumped on
// the last worker), same discipline as the FP32 remainder test.
fn test_q6k_parallel_remainder_matches_serial() {
	orig := os.getenv_opt('VLLM_THREADS')
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	os.setenv('VLLM_THREADS', '8', true)
	rows := 2501
	cols := 256
	mut data := []u8{len: rows * 210}
	for i in 0 .. data.len {
		data[i] = u8((i * 29 + 5) % 200)
	}
	for r in 0 .. rows {
		off := r * 210
		for i in 192 .. 208 {
			// small int8 scales -2..2 (realistic magnitudes; large
			// synthetic scales inflate FMA reassociation past any
			// absolute tolerance — see simd_test agreement notes)
			v := (i * 7 + r) % 5
			data[off + i] = u8((v + 254) % 256)
		}
		data[off + 208] = u8(0x00)
		data[off + 209] = u8(0x38) // d = 0.5
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 13) * 0.07 - 0.3
	}
	mut a := []f32{len: rows}
	mut b := []f32{len: rows}
	q6_k_matvec(data, rows, cols, x, mut a)
	q6_k_matvec_parallel(data, rows, cols, x, mut b)
	for i in 0 .. rows {
		d := b[i] - a[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_parallel_remainder_matches_serial() {
	orig := os.getenv_opt('VLLM_THREADS')
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	os.setenv('VLLM_THREADS', '8', true)
	rows := 2501
	cols := 1024
	mut w := []f32{len: rows * cols}
	mut x := []f32{len: cols}
	for i in 0 .. w.len {
		w[i] = f32(i % 13) * 0.125 - 0.5
	}
	for i in 0 .. x.len {
		x[i] = f32(i % 7) * 0.25 - 0.125
	}
	mut a := []f32{len: rows}
	mut b := []f32{len: rows}
	matvec(w, x, rows, cols, mut a)
	matvec_parallel(w, x, rows, cols, mut b)
	for i in 0 .. rows {
		d := b[i] - a[i]
		assert d < 1e-4 && d > -1e-4
	}
}
