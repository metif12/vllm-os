module infer

import time

// Perf gates: fail if the token-loop primitives regress beyond the floor.
// Small sizes keep the suite fast; they guard complexity, not wall clock.

// sample_into over 64 logits must sustain 20k iters well under the gate.
fn test_bench_sample_64() {
	mut logits := []f32{len: 64}
	for i in 0 .. 64 {
		logits[i] = -f32(i)
	}
	mut idx := []int{len: 64}
	p := SampleParams{
		temp:  0.8
		top_k: 40
		top_p: 0.9
		seed:  1234
	}
	sw := time.new_stopwatch()
	for i in 0 .. 20000 {
		logits[i % 64] = -f32((i + 7) % 64)
		sample_into(mut logits, mut idx, SampleParams{
			temp:  p.temp
			top_k: p.top_k
			top_p: p.top_p
			seed:  p.seed + u64(i)
		})
	}
	ms := sw.elapsed().milliseconds()
	assert ms < 8000, 'sample_into regressed: 20k iters took ${ms} ms'
}

// Blocked matvec must match naive output and stay within 3x of it.
fn test_bench_matvec_blocked_matches() {
	rows := 256
	cols := 512
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
	sw1 := time.new_stopwatch()
	for _ in 0 .. 20 {
		matvec(w, x, rows, cols, mut a)
	}
	t1 := sw1.elapsed().microseconds()
	sw2 := time.new_stopwatch()
	for _ in 0 .. 20 {
		matvec_blocked(w, x, rows, cols, mut b)
	}
	t2 := sw2.elapsed().microseconds()
	for i in 0 .. rows {
		d := a[i] - b[i]
		assert d < 0.01 && d > -0.01
	}
	assert t2 < t1 * 3 + 1000, 'matvec_blocked regressed: ${t2}us vs naive ${t1}us'
}
