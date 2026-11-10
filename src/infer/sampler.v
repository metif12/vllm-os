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

// sample applies temp / top-k / top-p then multinomial sampling.
pub fn sample(logits []f32, p SampleParams) int {
	mut rng := new_rng(p.seed)
	mut work := logits.clone()
	if p.temp <= 0.0 {
		return sample_greedy(work)
	}
	for i in 0 .. work.len {
		work[i] /= p.temp
	}
	softmax_in_place(mut work)
	mut idx := []int{cap: work.len}
	for i in 0 .. work.len {
		idx << i
	}
	idx.sort_with_compare(fn (a &int, b &int) int {
		return 0
	})
	// Partial selection sort for top-k (vocab 151k: full sort is wasteful).
	k := if p.top_k > 0 && p.top_k < work.len { p.top_k } else { work.len }
	for i in 0 .. k {
		mut m := i
		for j in i + 1 .. work.len {
			if work[idx[j]] > work[idx[m]] {
				m = j
			}
		}
		t := idx[i]
		idx[i] = idx[m]
		idx[m] = t
	}
	mut cands := idx[..k].clone()
	if p.top_p < 1.0 {
		mut cum := f32(0)
		mut cut := cands.len
		for i, ci in cands {
			cum += work[ci]
			if cum >= p.top_p {
				cut = i + 1
				break
			}
		}
		cands = cands[..cut]
	}
	mut mass := f32(0)
	for ci in cands {
		mass += work[ci]
	}
	mut roll := rng.next_f32() * mass
	for ci in cands {
		roll -= work[ci]
		if roll <= 0 {
			return ci
		}
	}
	return cands[cands.len - 1]
}

pub fn sample_top_k(logits []f32, k int) int {
	return sample(logits, SampleParams{
		temp:  0
		top_k: k
		top_p: 1.0
		seed:  0
	})
}

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
	for i in 0 .. xs.len {
		xs[i] /= s
	}
}
