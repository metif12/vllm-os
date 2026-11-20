module infer

import math

// Sliding-window + sinks gates: exact fallback when the window covers the
// whole cache, and agreement with a naive index-list reference otherwise.

fn wabs(x f32) f32 {
	return if x < 0 { -x } else { x }
}

fn new_test_cache(dim int, n int, seed int) !KVCache {
	mut c := new_kv_cache(dim, n)!
	for i in 0 .. n {
		mut row := []f32{len: dim}
		for d in 0 .. dim {
			row[d] = f32(((i * 31 + d * 7 + seed) % 23) - 11) * 0.05
		}
		c.append(row)!
	}
	return c
}

fn naive_window(q []f32, k &KVCache, v &KVCache, scale f32, window int, sinks int, mut out []f32) {
	n := k.len()
	mut idx := []int{}
	ns := if sinks < n { sinks } else { n }
	for i in 0 .. ns {
		idx << i
	}
	mut start := n - window
	if start < ns {
		start = ns
	}
	for i in start .. n {
		idx << i
	}
	mut scores := []f64{len: idx.len}
	mut m := f64(-1e30)
	for j, i in idx {
		mut s := f64(0)
		for d in 0 .. q.len {
			s += f64(q[d]) * f64(k.data[i * k.dim + d])
		}
		scores[j] = s * f64(scale)
		if scores[j] > m {
			m = scores[j]
		}
	}
	mut den := f64(0)
	for j in 0 .. scores.len {
		scores[j] = math.exp(scores[j] - m)
		den += scores[j]
	}
	for j in 0 .. out.len {
		out[j] = 0
	}
	for j, i in idx {
		p := f32(scores[j] / den)
		for d in 0 .. out.len {
			out[d] += p * v.data[i * v.dim + d]
		}
	}
}

fn test_attention_window_exact_fallback() {
	n := 40
	dim := 8
	mut k := new_test_cache(dim, n, 1)!
	mut v := new_test_cache(dim, n, 2)!
	mut q := []f32{len: dim}
	for i in 0 .. dim {
		q[i] = f32((i % 5) - 2) * 0.1
	}
	mut full := []f32{len: dim}
	mut win := []f32{len: dim}
	mut probs := []f32{len: n}
	attention_step(q, &k, &v, 0.25, mut probs, mut full)!
	attention_step_window(q, &k, &v, 0.25, n, 4, mut probs, mut win)!
	for i in 0 .. dim {
		assert wabs(full[i] - win[i]) < 1e-6
	}
}

fn test_attention_window_matches_naive() {
	n := 64
	dim := 8
	mut k := new_test_cache(dim, n, 3)!
	mut v := new_test_cache(dim, n, 4)!
	mut q := []f32{len: dim}
	for i in 0 .. dim {
		q[i] = f32((i % 7) - 3) * 0.1
	}
	for cfg in [[16, 4], [8, 0], [32, 8]] {
		window := cfg[0]
		sinks := cfg[1]
		mut got := []f32{len: dim}
		mut want := []f32{len: dim}
		mut probs := []f32{len: n}
		attention_step_window(q, &k, &v, 0.25, window, sinks, mut probs, mut got)!
		naive_window(q, &k, &v, 0.25, window, sinks, mut want)
		for i in 0 .. dim {
			assert wabs(got[i] - want[i]) < 1e-4
		}
	}
}
