module infer

// Batched matmul gates: the batch kernel (weight block dequantized once,
// dotted against k vectors) must agree with k calls to the single-vector
// kernel within f32 reassociation tolerance. Proves the shared-weight path
// is numerically safe before the batched decode uses it.

fn batch_abs(x f32) f32 {
	return if x < 0 { -x } else { x }
}

fn assert_batch_agree(got []f32, want []f32) {
	assert got.len == want.len
	for i, v in got {
		d := batch_abs(v - want[i])
		m := batch_abs(want[i])
		assert d < 1e-3 * m + 1e-2
	}
}

fn test_q4_0_batch_matches_single() {
	rows := 48
	cols := 128
	k := 4
	mut data := []u8{len: rows * (cols / qk4_0) * 18}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk4_0 {
			off := (r * (cols / qk4_0) + b) * 18
			data[off] = 0x00
			data[off + 1] = 0x38
		}
	}
	mut xs := []f32{len: k * cols}
	for i in 0 .. xs.len {
		xs[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut batched := []f32{len: k * rows}
	q4_0_matmul_c(data, xs, mut batched, k, rows, cols)
	mut want := []f32{len: k * rows}
	for t in 0 .. k {
		q4_0_matvec(data, rows, cols, xs[t * cols..(t + 1) * cols], mut
			want[t * rows..(t + 1) * rows])
	}
	assert_batch_agree(batched, want)
}

fn test_q4_1_batch_matches_single() {
	rows := 48
	cols := 128
	k := 4
	mut data := []u8{len: rows * (cols / qk4_1) * 20}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk4_1 {
			off := (r * (cols / qk4_1) + b) * 20
			data[off] = 0x00
			data[off + 1] = 0x38
			data[off + 2] = 0x00
			data[off + 3] = 0x38
		}
	}
	mut xs := []f32{len: k * cols}
	for i in 0 .. xs.len {
		xs[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut batched := []f32{len: k * rows}
	q4_1_matmul_c(data, xs, mut batched, k, rows, cols)
	mut want := []f32{len: k * rows}
	for t in 0 .. k {
		q4_1_matvec(data, rows, cols, xs[t * cols..(t + 1) * cols], mut
			want[t * rows..(t + 1) * rows])
	}
	assert_batch_agree(batched, want)
}

fn test_q6_k_batch_matches_single() {
	rows := 16
	cols := 256
	k := 4
	mut data := []u8{len: rows * (cols / 256) * 210}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / 256 {
			off := (r * (cols / 256) + b) * 210
			data[off + 208] = 0x00
			data[off + 209] = 0x38
		}
	}
	mut xs := []f32{len: k * cols}
	for i in 0 .. xs.len {
		xs[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut batched := []f32{len: k * rows}
	q6_k_matmul_c(data, xs, mut batched, k, rows, cols)
	mut want := []f32{len: k * rows}
	for t in 0 .. k {
		q6_k_matvec(data, rows, cols, xs[t * cols..(t + 1) * cols], mut
			want[t * rows..(t + 1) * rows])
	}
	assert_batch_agree(batched, want)
}
