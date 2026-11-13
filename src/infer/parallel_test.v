module infer

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
