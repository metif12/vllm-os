module main

// tools/bench_kernels.v — reproducible V-side kernel benchmark.
//
// Prints one line per bench in the same shapes as
// scripts/perf_baseline.py so docs/perf.md can quote both sides from
// back-to-back runs on one machine:
//
//   v -prod -cc gcc              -o /tmp/bench-rel  tools/bench_kernels.v && /tmp/bench-rel
//   v -prod -cc gcc -cflags "-O3 -march=native -flto" -o /tmp/bench-nat tools/bench_kernels.v && /tmp/bench-nat
//   python scripts/perf_baseline.py --kernels
//
// What runs here:
//  - `v-*` variants embed verbatim copies of the pure-V kernels
//    (SYNC: src/infer/nn.v, src/infer/sampler.v). Linked agreement of
//    the real kernels is gated in-tree (src/infer/*_test.v), so these
//    numbers describe the shipped code, not a fork.
//  - `c-*` variants call the REAL shipped C file
//    (src/infer/vllm_kernels.c) via #flag — no copy, always in sync.
//    Without -march=native the scalar fallbacks run; with it, AVX2+FMA.

import math
import sync
import time

// ---------- shipped C layer (no copy) ----------
#flag -I @VMODROOT/src/infer
#flag @VMODROOT/src/infer/vllm_kernels.c
#include "vllm_kernels.h"

fn C.vllm_has_avx2() int
fn C.vllm_matvec_f32(w &f32, x &f32, out &f32, rows int, cols int)
fn C.vllm_softmax_scaled_fast_f32(xs &f32, n int, scale f32)

// ---------- pure-V copies (SYNC with src/infer) ----------
@[direct_array_access]
fn v_matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	for r in 0 .. rows {
		mut s0 := f32(0)
		mut s1 := f32(0)
		mut s2 := f32(0)
		mut s3 := f32(0)
		base := r * cols
		mut c := 0
		for c + 4 <= cols {
			s0 += w[base + c] * x[c]
			s1 += w[base + c + 1] * x[c + 1]
			s2 += w[base + c + 2] * x[c + 2]
			s3 += w[base + c + 3] * x[c + 3]
			c += 4
		}
		for c < cols {
			s0 += w[base + c] * x[c]
			c++
		}
		out[r] = s0 + s1 + s2 + s3
	}
}

@[direct_array_access]
fn v_matvec_blocked(w []f32, x []f32, rows int, cols int, mut out []f32) {
	for r in 0 .. rows {
		out[r] = 0
	}
	tile := 32
	mut c0 := 0
	for c0 < cols {
		c1 := if c0 + tile < cols { c0 + tile } else { cols }
		for r in 0 .. rows {
			mut s := out[r]
			base := r * cols
			for c in c0 .. c1 {
				s += w[base + c] * x[c]
			}
			out[r] = s
		}
		c0 = c1
	}
}

struct BenchRng {
mut:
	state u64
}

fn (mut r BenchRng) next_f32() f32 {
	mut x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	w := x * 0x2545f4914f6cdd1d
	return f32(w >> 11) * (1.0 / 9007199254740992.0)
}

@[direct_array_access]
fn v_softmax_scaled(mut xs []f32, scale f32) {
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
fn v_sift_down(work []f32, mut heap []int, root int, n int) {
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

@[direct_array_access]
fn v_heap_topk(work []f32, k int, mut heap []int) {
	for i in 0 .. k {
		heap[i] = i
	}
	mut i := k / 2 - 1
	for i >= 0 {
		v_sift_down(work, mut heap, i, k)
		i--
	}
	for j in k .. work.len {
		if work[j] > work[heap[0]] {
			heap[0] = j
			v_sift_down(work, mut heap, 0, k)
		}
	}
}

// Exact sampler mirror (selection sort <=256, heap above — as shipped).
// fast=true swaps the softmax for the shipped C fast-softmax, mirroring
// sample_into_fast; the top-k/p/multinomial tail is identical.
@[direct_array_access]
fn v_sample_into(mut work []f32, mut idx []int, temp f32, top_k int, top_p f32, seed u64, fast bool) int {
	if temp <= 0.0 {
		mut best := 0
		for i, v in work {
			if v > work[best] {
				best = i
			}
		}
		return best
	}
	if fast {
		C.vllm_softmax_scaled_fast_f32(work.data, work.len, 1.0 / temp)
	} else {
		v_softmax_scaled(mut work, 1.0 / temp)
	}
	mut rng := BenchRng{state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed }}
	k := if top_k > 0 && top_k < work.len { top_k } else { work.len }
	if work.len > 256 {
		v_heap_topk(work, k, mut idx)
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
	if top_p < 1.0 {
		mut cum := f32(0)
		for i in 0 .. k {
			cum += work[idx[i]]
			if cum >= top_p {
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

// ---------- benches ----------
fn bench_matvec_small() {
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
	mut out := []f32{len: rows}
	for _ in 0 .. 3 {
		v_matvec(w, x, rows, cols, mut out)
	}
	sw := time.new_stopwatch()
	for _ in 0 .. 20 {
		v_matvec(w, x, rows, cols, mut out)
	}
	println('v matvec 256x512 x20 naive: ${sw.elapsed().microseconds()}us')
	sw2 := time.new_stopwatch()
	for _ in 0 .. 20 {
		v_matvec_blocked(w, x, rows, cols, mut out)
	}
	println('v matvec 256x512 x20 blocked: ${sw2.elapsed().microseconds()}us')
	for _ in 0 .. 3 {
		C.vllm_matvec_f32(w.data, x.data, out.data, rows, cols)
	}
	sw3 := time.new_stopwatch()
	for _ in 0 .. 20 {
		C.vllm_matvec_f32(w.data, x.data, out.data, rows, cols)
	}
	println('v matvec 256x512 x20 c-kernel: ${sw3.elapsed().microseconds()}us (avx2=${C.vllm_has_avx2()})')
}

fn bench_matvec_wide() {
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
	mut out := []f32{len: rows}
	v_matvec(w, x, rows, cols, mut out) // warmup
	sw := time.new_stopwatch()
	for _ in 0 .. 3 {
		v_matvec(w, x, rows, cols, mut out)
	}
	println('v matvec 2048x6144 x3 naive: ${sw.elapsed().milliseconds()}ms')
	sw2 := time.new_stopwatch()
	for _ in 0 .. 3 {
		C.vllm_matvec_f32(w.data, x.data, out.data, rows, cols)
	}
	println('v matvec 2048x6144 x3 c-kernel: ${sw2.elapsed().milliseconds()}ms (avx2=${C.vllm_has_avx2()})')
}

// Threaded mirror of the shipped matvec_parallel (SYNC: src/infer/parallel.v
// + simd_c.v matvec_c_range): row-block split, C AVX2 kernel per chunk.
// The tool passes nt explicitly for threshold proof; the tree resolves it
// from VLLM_THREADS with the min_rows_parallel floor.
fn tool_par_chunk(w []f32, x []f32, cols int, r0 int, n int, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	ws := w[r0 * cols..(r0 + n) * cols]
	mut outs := unsafe { out[r0..r0 + n] }
	C.vllm_matvec_f32(ws.data, x.data, outs.data, n, cols)
}

fn tool_matvec_parallel(w []f32, x []f32, rows int, cols int, mut out []f32, nt int) {
	if nt <= 1 {
		C.vllm_matvec_f32(w.data, x.data, out.data, rows, cols)
		return
	}
	mut wg := sync.new_waitgroup()
	chunk := rows / nt
	for t in 0 .. nt {
		r0 := t * chunk
		r1 := if t == nt - 1 { rows } else { r0 + chunk }
		wg.add(1)
		spawn tool_par_chunk(w, x, cols, r0, r1 - r0, mut out, mut wg)
	}
	wg.wait()
}

fn bench_matvec_parallel() {
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
	mut out := []f32{len: rows}
	tool_matvec_parallel(w, x, rows, cols, mut out, 4) // warmup
	for nt in [2, 4, 8] {
		sw := time.new_stopwatch()
		for _ in 0 .. 3 {
			tool_matvec_parallel(w, x, rows, cols, mut out, nt)
		}
		println('v matvec 2048x6144 x3 parallel-${nt}: ${sw.elapsed().milliseconds()}ms (avx2=${C.vllm_has_avx2()})')
	}
}

fn bench_sample64() {
	mut logits := []f32{len: 64}
	mut idx := []int{len: 64}
	for _ in 0 .. 100 {
		v_sample_into(mut logits, mut idx, 0.8, 40, 0.9, 1, false)
	}
	sw := time.new_stopwatch()
	for i in 0 .. 20000 {
		logits[i % 64] = -f32((i + 7) % 64)
		v_sample_into(mut logits, mut idx, 0.8, 40, 0.9, 1234 + u64(i), false)
	}
	println('v sample 64-logit x20k exact: ${sw.elapsed().milliseconds()}ms')
}

fn bench_sample_vocab() {
	n := 151936
	mut logits := []f32{len: n}
	for i in 0 .. n {
		logits[i] = -f32(i % 1000) * 0.01
	}
	mut idx := []int{len: n}
	v_sample_into(mut logits, mut idx, 0.8, 40, 0.9, 7, false) // warmup
	sw := time.new_stopwatch()
	for i in 0 .. 20 {
		logits[(i * 7919) % n] = -f32(i % 50) * 0.01
		v_sample_into(mut logits, mut idx, 0.8, 40, 0.9, 99 + u64(i), false)
	}
	println('v sample 151936-logit x20 exact: ${sw.elapsed().milliseconds()}ms')
	sw2 := time.new_stopwatch()
	for i in 0 .. 20 {
		logits[(i * 7919) % n] = -f32(i % 50) * 0.01
		v_sample_into(mut logits, mut idx, 0.8, 40, 0.9, 99 + u64(i), true)
	}
	println('v sample 151936-logit x20 fast: ${sw2.elapsed().milliseconds()}ms')
}

fn main() {
	bench_matvec_small()
	bench_matvec_wide()
	bench_matvec_parallel()
	bench_sample64()
	bench_sample_vocab()
}
