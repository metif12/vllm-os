module infer

import math

// Scaled dot-product attention over a KV cache — the phase-1b compute
// building block. Pure V, zero allocation on the hot path (caller-owned
// buffers, shapes validated at cache build, never per token).
//
// Composition (done by the phase-1b forward, not here): RoPE the q/k head
// slices with rope_in_place, append to the caches, call attention_step
// per head (GQA = several q heads share one kv head), project, repeat.
// RMSNorm + SwiGLU + matvec already exist in nn.v; this file closes the
// remaining gap to a full decoder layer.

// KVCache is a flat row-major [cap x dim] arena with a length cursor.
// One instance holds keys, another values (same type, same dim). append
// copies a row in; row borrows without allocation. Not thread-safe by
// design: one cache per decode worker, like the request scratch buffers.
pub struct KVCache {
pub:
	dim int
	cap int
mut:
	len  int
	data []f32
}

// new_kv_cache allocates a [cap x dim] f32 arena, empty.
pub fn new_kv_cache(dim int, cap_ int) !KVCache {
	if dim <= 0 || cap_ <= 0 {
		return error('kvcache: bad shape dim=${dim} cap=${cap_}')
	}
	return KVCache{
		dim:  dim
		cap:  cap_
		data: []f32{len: dim * cap_}
	}
}

// len returns the number of stored rows.
pub fn (c KVCache) len() int {
	return c.len
}

// append copies one dim-length row. Errors when full — the caller sizes
// cap from ctx_cap once, so overflow is a config bug, not a runtime case.
pub fn (mut c KVCache) append(row []f32) ! {
	if row.len != c.dim {
		return error('kvcache: row len ${row.len} vs dim ${c.dim}')
	}
	if c.len >= c.cap {
		return error('kvcache: full at cap ${c.cap}')
	}
	off := c.len * c.dim
	for i in 0 .. c.dim {
		c.data[off + i] = row[i]
	}
	c.len++
}

// row borrows row i without allocation (bounds-checked).
pub fn (c KVCache) row(i int) ![]f32 {
	if i < 0 || i >= c.len {
		return error('kvcache: row ${i} out of ${c.len}')
	}
	off := i * c.dim
	return c.data[off..off + c.dim]
}

// reset empties the cache without freeing the arena (prefill reuse).
pub fn (mut c KVCache) reset() {
	c.len = 0
}

// attention_step computes one query head output against full caches:
// scores = softmax(q.K^T * scale) into probs, out = Σ probs * V.
// probs must hold ≥ k.len() elems; out must hold dim elems. Single head —
// the forward loops heads (GQA shares one kv head across q heads).
@[direct_array_access]
pub fn attention_step(q []f32, k &KVCache, v &KVCache, scale f32, mut probs []f32, mut out []f32) ! {
	n := k.len()
	if q.len != k.dim || v.dim != k.dim || out.len != k.dim {
		return error('attention: dim mismatch q=${q.len} kdim=${k.dim} vdim=${v.dim} out=${out.len}')
	}
	if v.len() != n {
		return error('attention: kv length mismatch k=${n} v=${v.len()}')
	}
	if n == 0 {
		return error('attention: empty cache')
	}
	if probs.len < n {
		return error('attention: probs holds ${probs.len}, need ${n}')
	}
	// scores = q.K^T * scale, fused max for the softmax that follows.
	mut m := f32(-1e30)
	for i in 0 .. n {
		kr := k.row(i)!
		mut s := f32(0)
		for d in 0 .. q.len {
			s += q[d] * kr[d]
		}
		s *= scale
		probs[i] = s
		if s > m {
			m = s
		}
	}
	// softmax in place over probs[0..n] (exact exp: scores gate quality).
	mut s := f32(0)
	for i in 0 .. n {
		e := f32(math.exp(f64(probs[i] - m)))
		probs[i] = e
		s += e
	}
	inv := f32(1.0 / f64(s))
	for i in 0 .. n {
		probs[i] *= inv
	}
	// out = P * V
	for d in 0 .. out.len {
		out[d] = 0
	}
	for i in 0 .. n {
		vr := v.row(i)!
		pi := probs[i]
		for d in 0 .. out.len {
			out[d] += pi * vr[d]
		}
	}
}
