module infer

import math

pub struct SampleParams {
pub:
	temp  f32 = 0.7
	top_k int = 40
	top_p f32 = 0.9
	seed  u64
}

// Xorshift64* — deterministic, no OS entropy needed (bare-metal safe).
pub struct Rng {
mut:
	state u64
}

pub fn new_rng(seed u64) Rng {
	mut s := seed
	if s == 0 {
		s = 0x9e3779b97f4a7c15
	}
	return Rng{
		state: s
	}
}

pub fn (mut r Rng) next_u64() u64 {
	mut x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * 0x2545f4914f6cdd1d
}

pub fn (mut r Rng) next_f32() f32 {
	return f32(r.next_u64() >> 11) * (1.0 / 9007199254740992.0)
}

// sample_greedy returns the argmax index.
@[direct_array_access]
pub fn sample_greedy(logits []f32) int {
	mut best := 0
	mut best_v := logits[0]
	for i, v in logits {
		if v > best_v {
			best_v = v
			best = i
		}
	}
	return best
}

// sample is the allocating convenience wrapper. The token loop must use
// sample_into with reused scratch buffers (zero alloc per token).
pub fn sample(logits []f32, p SampleParams) int {
	mut work := logits.clone()
	mut idx := []int{len: logits.len}
	return sample_into(mut work, mut idx, p)
}

// sample_into runs temp / top-k / top-p then multinomial sampling.
// work is logits on entry, probabilities on exit; idx is scratch space.
// No allocation: safe for the bare-metal token loop.
@[direct_array_access]
pub fn sample_into(mut work []f32, mut idx []int, p SampleParams) int {
	assert work.len == idx.len
	assert work.len > 0
	mut rng := new_rng(p.seed)
	if p.temp <= 0.0 {
		return sample_greedy(work)
	}
	inv_temp := 1.0 / p.temp
	softmax_scaled_in_place(mut work, inv_temp)
	for i in 0 .. work.len {
		idx[i] = i
	}
	// Partial selection sort for top-k (vocab 151k: full sort is wasteful).
	k := if p.top_k > 0 && p.top_k < work.len { p.top_k } else { work.len }
	for i in 0 .. k {
		mut m := i
		mut wm := work[idx[m]]
		for j in i + 1 .. work.len {
			if work[idx[j]] > wm {
				m = j
				wm = work[idx[m]]
			}
		}
		t := idx[i]
		idx[i] = idx[m]
		idx[m] = t
	}
	mut n_cands := k
	if p.top_p < 1.0 {
		mut cum := f32(0)
		for i in 0 .. k {
			cum += work[idx[i]]
			if cum >= p.top_p {
				n_cands = i + 1
				break
			}
		}
	}
	mut mass := f32(0)
	for i in 0 .. n_cands {
		mass += work[idx[i]]
	}
	mut roll := rng.next_f32() * mass
	for i in 0 .. n_cands {
		roll -= work[idx[i]]
		if roll <= 0 {
			return idx[i]
		}
	}
	return idx[n_cands - 1]
}

pub fn sample_top_k(logits []f32, k int) int {
	return sample(logits, SampleParams{
		temp:  0
		top_k: k
		top_p: 1.0
		seed:  0
	})
}

// softmax_scaled_in_place applies softmax(x * scale) in two passes with no
// intermediate store: one full vocabulary-sized memory write saved per token.
@[direct_array_access]
pub fn softmax_scaled_in_place(mut xs []f32, scale f32) {
	mut m := xs[0] * scale
	for i in 1 .. xs.len {
		v := xs[i] * scale
		if v > m {
			m = v
		}
	}
	mut s := f32(0)
	for i in 0 .. xs.len {
		e := f32(math.exp(f64(xs[i] * scale - m)))
		xs[i] = e
		s += e
	}
	inv := 1.0 / s
	for i in 0 .. xs.len {
		xs[i] *= inv
	}
}

@[direct_array_access]
pub fn softmax_in_place(mut xs []f32) {
	mut m := xs[0]
	for v in xs {
		if v > m {
			m = v
		}
	}
	mut s := f32(0)
	for i in 0 .. xs.len {
		xs[i] = f32(math.exp(f64(xs[i] - m)))
		s += xs[i]
	}
	inv := 1.0 / s
	for i in 0 .. xs.len {
		xs[i] *= inv
	}
}
