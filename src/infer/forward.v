module infer

import math

// Phase-1b transformer forward (Qwen3-style decoder, GQA + QK-norm + RoPE).
// Pure V, caller-owned buffers, no allocation inside the token loop:
// ForwardCtx owns arenas (KV caches) and scratch; decode_step reuses them.
// Weights stream from the GGUF blob via fused Q4/Q8 dots (never
// materialized). The v1 1.7B target is Q4_K_M (super-blocks, follow-up);
// this forward runs the Q4_0/Q4_1/F32/i8 types real small models ship.

// q4_0_matvec computes out = W*x for a Q4_0 row-major [rows x cols]
// matrix stored as packed 18-byte blocks (cols must divide 32).
@[direct_array_access]
pub fn q4_0_matvec(data []u8, rows int, cols int, x []f32, mut out []f32) {
	blocks_per_row := cols / 32
	for r in 0 .. rows {
		mut s := f32(0)
		base := r * blocks_per_row
		for b in 0 .. blocks_per_row {
			s += q4_0_dot_row(block_q40_at(data, base + b), x, b * 32)
		}
		out[r] = s
	}
}

// q4_1_matvec computes out = W*x for a Q4_1 row-major matrix (20B blocks).
@[direct_array_access]
pub fn q4_1_matvec(data []u8, rows int, cols int, x []f32, mut out []f32) {
	blocks_per_row := cols / 32
	for r in 0 .. rows {
		mut s := f32(0)
		base := r * blocks_per_row
		for b in 0 .. blocks_per_row {
			s += q4_1_dot_row(block_q41_at(data, base + b), x, b * 32)
		}
		out[r] = s
	}
}

// i8_matvec computes out = W*x for a plain-int8 row-major matrix
// (quantized token embeddings / tied output head).
@[direct_array_access]
pub fn i8_matvec(data []u8, rows int, cols int, x []f32, mut out []f32) {
	for r in 0 .. rows {
		base := r * cols
		mut s := f32(0)
		for c in 0 .. cols {
			s += f32(i8(data[base + c])) * x[c]
		}
		out[r] = s
	}
}

// embed_gather_i8 copies row id of an i8 [vocab x dim] matrix as f32.
@[direct_array_access]
pub fn embed_gather_i8(data []u8, dim int, id int, mut out []f32) {
	base := id * dim
	for d in 0 .. out.len {
		out[d] = f32(i8(data[base + d]))
	}
}

// embed_gather decodes row id of the token embedding matrix as f32,
// dispatching on the stored quant (Unsloth ships Q6_K super-blocks;
// older files may carry plain i8).
@[direct_array_access]
pub fn embed_gather(blob []u8, gguf GgufModel, dim int, id int, mut out []f32) ! {
	info := gguf.find('token_embd.weight') or { return error('gguf: missing token_embd.weight') }
	data := tensor_bytes(blob, gguf, 'token_embd.weight')!
	if info.typ == .q6_k {
		embed_gather_q6_k(data, dim, id, mut out)
	} else {
		embed_gather_i8(data, dim, id, mut out)
	}
}

// rope_q_in_place rotates one q-head; the kv-head k slice is rotated once
// per group via rope_in_place with the group's first q head. Split-half
// pairing matches rope_in_place (HF/Qwen3 convention).
@[direct_array_access]
fn rope_q_in_place(mut q []f32, cos_sin []f32, head_dim int) {
	half := head_dim / 2
	for h in 0 .. half {
		c := cos_sin[h]
		s := cos_sin[half + h]
		if h < q.len && half + h < q.len {
			a := q[h]
			b := q[half + h]
			q[h] = a * c - b * s
			q[half + h] = a * s + b * c
		}
	}
}

// ForwardCtx owns everything the token loop touches: config, weight blob
// + table, KV arenas, and reused scratch (no per-token allocation).
pub struct ForwardCtx {
pub mut:
	cfg     Qwen3Config
	gguf    GgufModel
	blob    []u8
	k_cache []KVCache // layers * heads_kv, dim = head_dim
	v_cache []KVCache
	pos     int
	// scratch buffers, sized once at build
	hidden   []f32
	normed   []f32
	q        []f32
	k        []f32
	v        []f32
	attn_out []f32
	proj     []f32
	gate     []f32
	up       []f32
	down     []f32
	probs    []f32
	cos_sin  []f32
	logits   []f32
}

// new_forward_ctx allocates arenas for ctx_cap tokens of KV per head.
pub fn new_forward_ctx(cfg Qwen3Config, gguf GgufModel, blob []u8) !ForwardCtx {
	h := cfg.hidden
	nh := cfg.heads_q
	nkv := cfg.heads_kv
	hd := cfg.head_dim
	vd := cfg.v_dim
	mut k_cache := []KVCache{cap: cfg.layers * nkv}
	mut v_cache := []KVCache{cap: cfg.layers * nkv}
	for _ in 0 .. cfg.layers * nkv {
		k_cache << new_kv_cache(hd, cfg.ctx_cap)!
		v_cache << new_kv_cache(vd, cfg.ctx_cap)!
	}
	inter := cfg.intermediate
	vocab := vocab_size(gguf)
	return ForwardCtx{
		cfg:      cfg
		gguf:     gguf
		blob:     blob
		k_cache:  k_cache
		v_cache:  v_cache
		hidden:   []f32{len: h}
		normed:   []f32{len: h}
		q:        []f32{len: nh * hd}
		k:        []f32{len: nkv * hd}
		v:        []f32{len: nkv * vd}
		attn_out: []f32{len: nh * vd}
		proj:     []f32{len: h}
		gate:     []f32{len: inter}
		up:       []f32{len: inter}
		down:     []f32{len: h}
		probs:    []f32{len: cfg.ctx_cap}
		cos_sin:  []f32{len: hd}
		logits:   []f32{len: vocab}
	}
}

fn vocab_size(g GgufModel) int {
	for t in g.tensors {
		if t.name == 'token_embd.weight' && t.shape.len == 2 {
			return int(t.shape[1])
		}
	}
	return 151936
}

// layer_tensors borrows one block's weight slices from the blob.
struct LayerTensors {
	q    []u8
	k    []u8
	v    []u8
	o    []u8
	qn   []f32
	kn   []f32
	an   []f32
	gate []u8
	up   []u8
	down []u8
	// down_typ selects the ffn_down kernel: Unsloth mixes Q4_1 (early
	// layers) with Q4_0 (later layers) in one file, so the forward must
	// dispatch per tensor instead of assuming one quant.
	down_typ GgmlType
	fn_      []f32
}

fn (f ForwardCtx) layer_tensors(l int) !LayerTensors {
	p := 'blk.${l}.'
	down_info := f.gguf.find(p + 'ffn_down.weight') or {
		return error('gguf: missing tensor ${p}ffn_down.weight')
	}
	return LayerTensors{
		q:        tensor_bytes(f.blob, f.gguf, p + 'attn_q.weight')!
		k:        tensor_bytes(f.blob, f.gguf, p + 'attn_k.weight')!
		v:        tensor_bytes(f.blob, f.gguf, p + 'attn_v.weight')!
		o:        tensor_bytes(f.blob, f.gguf, p + 'attn_output.weight')!
		qn:       f32_row(tensor_bytes(f.blob, f.gguf, p + 'attn_q_norm.weight')!)
		kn:       f32_row(tensor_bytes(f.blob, f.gguf, p + 'attn_k_norm.weight')!)
		an:       f32_row(tensor_bytes(f.blob, f.gguf, p + 'attn_norm.weight')!)
		gate:     tensor_bytes(f.blob, f.gguf, p + 'ffn_gate.weight')!
		up:       tensor_bytes(f.blob, f.gguf, p + 'ffn_up.weight')!
		down:     tensor_bytes(f.blob, f.gguf, p + 'ffn_down.weight')!
		down_typ: down_info.typ
		fn_:      f32_row(tensor_bytes(f.blob, f.gguf, p + 'ffn_norm.weight')!)
	}
}

// f32_row reinterprets a little-endian f32 blob slice (caller-sized).
pub fn f32_row(data []u8) []f32 {
	n := data.len / 4
	mut out := []f32{len: n}
	for i in 0 .. n {
		bits := u32(data[i * 4]) | (u32(data[i * 4 + 1]) << 8) | (u32(data[i * 4 + 2]) << 16) | (u32(data[
			i * 4 + 3]) << 24)
		out[i] = *unsafe { &f32(&bits) }
	}
	return out
}

// decode_step runs one full forward for the token already in hidden,
// appends kv at pos, and leaves logits over the vocab. No allocation:
// all temporaries are ForwardCtx scratch.
pub fn (mut f ForwardCtx) decode_step() ! {
	h := f.cfg.hidden
	nh := f.cfg.heads_q
	nkv := f.cfg.heads_kv
	hd := f.cfg.head_dim
	vd := f.cfg.v_dim
	inter := f.cfg.intermediate
	scale := f32(1.0 / math.sqrt(f64(hd)))
	rope_table(mut f.cos_sin, f.pos, hd, f.cfg.rope_theta)
	for l in 0 .. f.cfg.layers {
		lt := f.layer_tensors(l)!
		// --- attention block (Pre-Norm: norm into scratch, residual adds
		// back the untouched hidden; never norm hidden in place) ---
		for i in 0 .. h {
			f.normed[i] = f.hidden[i]
		}
		rmsnorm_in_place(mut f.normed, lt.an, f.cfg.rms_norm_eps)
		q4_0_matvec(lt.q, nh * hd, h, f.normed, mut f.q)
		q4_0_matvec(lt.k, nkv * hd, h, f.normed, mut f.k)
		q4_0_matvec(lt.v, nkv * vd, h, f.normed, mut f.v)
		// QK-norm per head (slices passed directly as mut args borrow;
		// never `mut qs := ...` — that implicitly clones and drops the write).
		for qh in 0 .. nh {
			rmsnorm_in_place(mut f.q[qh * hd..(qh + 1) * hd], lt.qn, f.cfg.rms_norm_eps)
		}
		for kh in 0 .. nkv {
			rmsnorm_in_place(mut f.k[kh * hd..(kh + 1) * hd], lt.kn, f.cfg.rms_norm_eps)
		}
		// RoPE: k rotated once per kv head (with first q head), rest q-only.
		gqa := nh / nkv
		for kh in 0 .. nkv {
			rope_in_place(mut f.q[kh * gqa * hd..(kh * gqa + 1) * hd], mut
				f.k[kh * hd..(kh + 1) * hd], f.cos_sin, hd)
			for g in 1 .. gqa {
				rope_q_in_place(mut f.q[(kh * gqa + g) * hd..(kh * gqa + g + 1) * hd], f.cos_sin,
					hd)
			}
		}
		// append kv rows, attend per q head
		for kh in 0 .. nkv {
			f.k_cache[l * nkv + kh].append(f.k[kh * hd..(kh + 1) * hd])!
			f.v_cache[l * nkv + kh].append(f.v[kh * vd..(kh + 1) * vd])!
		}
		for qh in 0 .. nh {
			kh := qh / gqa
			attention_step(f.q[qh * hd..(qh + 1) * hd], &f.k_cache[l * nkv + kh], &f.v_cache[
				l * nkv + kh], scale, mut f.probs, mut f.attn_out[qh * vd..(qh + 1) * vd])!
		}
		q4_0_matvec(lt.o, h, nh * vd, f.attn_out, mut f.proj)
		for i in 0 .. h {
			f.hidden[i] += f.proj[i]
		}
		// --- FFN block (SwiGLU, same Pre-Norm scratch pattern) ---
		for i in 0 .. h {
			f.normed[i] = f.hidden[i]
		}
		rmsnorm_in_place(mut f.normed, lt.fn_, f.cfg.rms_norm_eps)
		q4_0_matvec(lt.gate, inter, h, f.normed, mut f.gate)
		q4_0_matvec(lt.up, inter, h, f.normed, mut f.up)
		silu_in_place(mut f.gate)
		for i in 0 .. inter {
			f.up[i] *= f.gate[i]
		}
		if lt.down_typ == .q4_1 {
			q4_1_matvec(lt.down, h, inter, f.up, mut f.down)
		} else {
			q4_0_matvec(lt.down, h, inter, f.up, mut f.down)
		}
		for i in 0 .. h {
			f.hidden[i] += f.down[i]
		}
	}
	// vocab head (tied embeddings): final norm into scratch, then project.
	// The head quant matches token_embd (Q6_K on Unsloth files).
	onorm := f32_row(tensor_bytes(f.blob, f.gguf, 'output_norm.weight')!)
	for i in 0 .. h {
		f.normed[i] = f.hidden[i]
	}
	rmsnorm_in_place(mut f.normed, onorm, f.cfg.rms_norm_eps)
	emb_info := f.gguf.find('token_embd.weight') or {
		return error('gguf: missing token_embd.weight')
	}
	emb := tensor_bytes(f.blob, f.gguf, 'token_embd.weight')!
	if emb_info.typ == .q6_k {
		q6_k_matvec(emb, f.logits.len, h, f.normed, mut f.logits)
	} else {
		i8_matvec(emb, f.logits.len, h, f.normed, mut f.logits)
	}
	f.pos++
}
