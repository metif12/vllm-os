module infer

import math

pub struct SampleParams {
pub mut:
	temp  f32 = 0.7
	top_k int = 40
	top_p f32 = 0.9
	// min_p drops candidates below min_p * top-1 probability (0 = off).
	// Applied after top-k, before top-p; the top-1 token always survives.
	min_p f32 = 0.0
	seed  u64
	// fast selects the approximate-softmax path (sampling only).
	fast bool
}

// sampler_presets lists the named SampleParams presets. The registry is
// the single place new presets are added; config + CLI resolve through
// preset_params, so a new preset needs no other code changes.
pub fn sampler_presets() []string {
	return ['deterministic', 'precise', 'balanced', 'creative']
}

// preset_params returns the base SampleParams for a preset name.
// Callers layer explicit overrides on top (see config sampler_params).
pub fn preset_params(name string) !SampleParams {
	return match name {
		'deterministic' {
			SampleParams{
				temp:  0.0
				top_k: 1
				top_p: 1.0
			}
		}
		'precise' {
			SampleParams{
				temp:  0.3
				top_k: 20
				top_p: 0.9
			}
		}
		'balanced' {
			SampleParams{}
		}
		'creative' {
			SampleParams{
				temp:  1.1
				top_k: 100
				top_p: 0.95
			}
		}
		else {
			error('unknown sampler preset: ${name} (want ${sampler_presets().join('|')})')
		}
	}
}

// Xorshift64* — deterministic, no OS entropy needed (bare-metal safe).
pub struct Rng {
mut:
	state u64
}

// new_rng seeds the generator (zero seed maps to a fixed nonzero default).
pub fn new_rng(seed u64) Rng {
	mut s := seed
	if s == 0 {
		s = 0x9e3779b97f4a7c15
	}
	return Rng{
		state: s
	}
}

// next_u64 advances the generator and returns the raw output word.
pub fn (mut r Rng) next_u64() u64 {
	mut x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * 0x2545f4914f6cdd1d
}

// next_f32 returns a uniform sample in [0, 1).
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
	if p.temp <= 0.0 {
		return sample_greedy(work)
	}
	softmax_scaled_in_place(mut work, 1.0 / p.temp)
	return sample_from_probs(mut work, mut idx, p)
}

// sample_into_fast is the opt-in speed path: approximate softmax
// (branchless exp, ~0.5% worst-case error) + heap top-k. Same signature
// and scratch contract as sample_into; distributions match to sampling
// tolerance, never bit-identical. Outside -prod it calls the exact
// sample_into (no C layer linked); the server's generate path always
// uses the exact sampler.
pub fn sample_into_fast(mut work []f32, mut idx []int, p SampleParams) int {
	$if prod {
		assert work.len == idx.len
		assert work.len > 0
		if p.temp <= 0.0 {
			return sample_greedy(work)
		}
		softmax_scaled_fast_in_place(mut work, 1.0 / p.temp)
		return sample_from_probs(mut work, mut idx, p)
	} $else {
		return sample_into(mut work, mut idx, p)
	}
}

// sample_from_probs runs top-k / top-p truncation + multinomial draw over
// work (already probabilities). Shared by the exact and fast paths.
@[direct_array_access]
fn sample_from_probs(mut work []f32, mut idx []int, p SampleParams) int {
	mut rng := new_rng(p.seed)
	// Partial selection for top-k (vocab 151k: full sort is wasteful).
	// Wide vocabs use a min-heap (O(n log k)); narrow ones keep the
	// selection sort (identical order to the historic path).
	k := if p.top_k > 0 && p.top_k < work.len { p.top_k } else { work.len }
	if work.len > 256 {
		heap_topk(work, k, mut idx)
		// Descending insertion sort of the k winners (k is tiny).
		for i in 1 .. k {
			t := idx[i]
			tv := work[t]
			mut j := i - 1
			for j >= 0 && work[idx[j]] < tv {
				idx[j + 1] = idx[j]
				j--
			}
			idx[j + 1] = t
		}
	} else {
		for i in 0 .. work.len {
			idx[i] = i
		}
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
	}
	mut n_cands := k
	// min-p: truncate the descending top-k list at min_p * top-1 mass.
	// Runs before top-p so both truncations compose; idx[0] always survives.
	if p.min_p > 0.0 && k > 1 {
		thresh := work[idx[0]] * p.min_p
		for i in 1 .. k {
			if work[idx[i]] < thresh {
				n_cands = i
				break
			}
		}
	}
	if p.top_p < 1.0 {
		mut cum := f32(0)
		for i in 0 .. n_cands {
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

// sample_top_k draws with temperature 0 semantics restricted to top-k
// (allocating convenience wrapper; the loop uses sample_into).
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

// softmax_in_place normalizes xs to probabilities (unscaled entry point;
// the token loop uses softmax_scaled_in_place to fuse the temperature).
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

// heap_topk fills heap[0..k] with the indices of the k largest work
// values (unordered min-heap internals). O(n log k): the vocab-scale
// path where selection sort's O(n*k) dominates. No allocation.
@[direct_array_access]
fn heap_topk(work []f32, k int, mut heap []int) {
	for i in 0 .. k {
		heap[i] = i
	}
	mut i := k / 2 - 1
	for i >= 0 {
		sift_down(work, mut heap, i, k)
		i--
	}
	for j in k .. work.len {
		if work[j] > work[heap[0]] {
			heap[0] = j
			sift_down(work, mut heap, 0, k)
		}
	}
}

@[direct_array_access]
fn sift_down(work []f32, mut heap []int, root int, n int) {
	mut r := root
	for {
		l := 2 * r + 1
		rr := 2 * r + 2
		mut m := r
		if l < n && work[heap[l]] < work[heap[m]] {
			m = l
		}
		if rr < n && work[heap[rr]] < work[heap[m]] {
			m = rr
		}
		if m == r {
			break
		}
		t := heap[r]
		heap[r] = heap[m]
		heap[m] = t
		r = m
	}
}
