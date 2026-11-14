module infer

// Agreement + parity tests for the C kernel bridge (simd_c.v) and the
// fast sampler. Exact paths are the reference; C/fast paths must stay
// within tolerance on every toolchain (tcc scalar-C, gcc, gcc-native).

fn test_matvec_c_matches_naive() {
	for shape in [[2, 3], [16, 17], [64, 130], [256, 512]] {
		rows := shape[0]
		cols := shape[1]
		mut w := []f32{len: rows * cols}
		mut x := []f32{len: cols}
		for i in 0 .. w.len {
			w[i] = f32(i % 13) * 0.125
		}
		for i in 0 .. x.len {
			x[i] = f32(i % 7) * 0.25 - 0.5
		}
		mut a := []f32{len: rows}
		mut b := []f32{len: rows}
		matvec(w, x, rows, cols, mut a)
		matvec_c(w, x, rows, cols, mut b)
		for i in 0 .. rows {
			d := b[i] - a[i]
			assert d < 1e-4 && d > -1e-4
		}
	}
}

fn test_q8_dot_c_matches() {
	mut qs := [32]i8{}
	for i in 0 .. 32 {
		qs[i] = i8(i * 3 % 17) - 8
	}
	b := BlockQ80{
		d:  u16(0x3c00) // 1.0 in f16
		qs: qs
	}
	mut x := []f32{len: 32}
	for i in 0 .. 32 {
		x[i] = f32(i) * 0.1 - 1.0
	}
	a := q8_0_dot_row(b, x, 0)
	c := q8_0_dot_row_c(b, x, 0)
	d := c - a
	assert d < 1e-4 && d > -1e-4
}

fn test_q4_dot_matches_dequant_then_dot() {
	mut qs := [16]u8{}
	for i in 0 .. 16 {
		lo := u8((i * 5 + 1) % 16)
		hi := u8((i * 7 + 3) % 16)
		qs[i] = lo | (hi << 4)
	}
	b := BlockQ40{
		d:  u16(0x3c00) // 1.0 in f16
		qs: qs
	}
	mut x := []f32{len: 32}
	for i in 0 .. 32 {
		x[i] = f32(i) * 0.1 - 1.0
	}
	// Reference: materialize then dot.
	mut row := []f32{len: 32}
	dequant_q4_0_row(b, mut row)
	mut want := f32(0)
	for i in 0 .. 32 {
		want += row[i] * x[i]
	}
	got := q4_0_dot_row(b, x, 0)
	d := got - want
	assert d < 1e-4 && d > -1e-4
	c := q4_0_dot_row_c(b, x, 0)
	dc := c - want
	assert dc < 1e-4 && dc > -1e-4
}

fn test_rmsnorm_c_matches() {
	mut x := [f32(0.5), -1.5, 2.0, -0.25, 1.0, 0.75, -2.5, 3.0]
	w := [f32(1.0), 0.5, 1.5, 1.0, 0.25, 2.0, 1.0, 0.75]
	mut y := x.clone()
	rmsnorm_in_place(mut x, w, 1e-6)
	rmsnorm_c(mut y, w, 1e-6)
	for i in 0 .. x.len {
		d := y[i] - x[i]
		assert d < 1e-5 && d > -1e-5
	}
}

fn test_heap_topk_finds_winners() {
	mut work := []f32{len: 1024}
	for i in 0 .. 1024 {
		work[i] = f32((i * 7919) % 1024) * 0.01
	}
	work[7] = 100.0
	work[999] = 99.0
	work[513] = 98.0
	mut heap := []int{len: 1024}
	heap_topk(work, 3, mut heap)
	mut got := heap[..3].clone()
	got.sort()
	assert got == [7, 513, 999]
}

fn test_sample_fast_close_to_exact() {
	// Peaked vocab-scale distribution: same argmax, close probs.
	// Outside -prod the fast path IS the exact path, so this holds
	// trivially there and meaningfully under -prod.
	mut logits := []f32{len: 4096}
	for i in 0 .. 4096 {
		logits[i] = -f32(i % 64) * 0.1
	}
	logits[11] = 5.0
	mut w1 := logits.clone()
	mut w2 := logits.clone()
	mut i1 := []int{len: 4096}
	mut i2 := []int{len: 4096}
	p := SampleParams{
		temp:  0.8
		top_k: 40
		top_p: 0.9
		seed:  7
	}
	r1 := sample_into(mut w1, mut i1, p)
	r2 := sample_into_fast(mut w2, mut i2, p)
	assert r1 == 11
	assert r2 == 11
	// Probability mass on the winner agrees within fast-exp tolerance.
	mut s1 := f32(0)
	mut s2 := f32(0)
	for v in w1 {
		s1 += v
	}
	for v in w2 {
		s2 += v
	}
	assert s1 > 0.999 && s1 < 1.001
	assert s2 > 0.99 && s2 < 1.01
}
