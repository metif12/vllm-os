module infer

// Dequant kernels: Q4_0, Q8_0, F16, F32 -> F32 rows.
// Pure V, no GC pressure (caller owns `out`). Matches GGML block layouts.
// f16 values are carried as u16 bits (V has no f16 scalar); conversion is
// software-only, no F16C dependency (bare-metal safe).

pub const qk4_0 = 32
pub const qk8_0 = 32
pub const qk4_1 = 32
pub const qk_k = 256
pub const q6_k_block_bytes = 210

pub struct BlockQ40 {
pub:
	d  u16    // f16 bits
	qs [16]u8 // 32 nibbles packed
}

pub struct BlockQ41 {
pub:
	d  u16    // f16 scale bits
	m  u16    // f16 offset bits
	qs [16]u8 // 32 nibbles packed (0..15 unsigned)
}

pub struct BlockQ80 {
pub:
	d  u16 // f16 bits
	qs [32]i8
}

// q80_scale decodes a Q8_0 block's f16 scale to f32 (shared by the V and
// C dot paths so both stay bit-identical on the scale factor).
pub fn q80_scale(block BlockQ80) f32 {
	return f16bits_to_f32(block.d)
}

// q40_scale decodes a Q4_0 block's f16 scale to f32 (shared by the V and
// C fused-dot paths, mirroring q80_scale).
pub fn q40_scale(block BlockQ40) f32 {
	return f16bits_to_f32(block.d)
}

// q41_scale decodes a Q4_1 block's f16 scale to f32.
pub fn q41_scale(block BlockQ41) f32 {
	return f16bits_to_f32(block.d)
}

// q41_min decodes a Q4_1 block's f16 offset to f32.
pub fn q41_min(block BlockQ41) f32 {
	return f16bits_to_f32(block.m)
}

// block_q40_at parses one 18-byte Q4_0 block (caller bounds-checks).
@[direct_array_access]
pub fn block_q40_at(data []u8, blk int) BlockQ40 {
	off := blk * 18
	mut qs := [16]u8{}
	for i in 0 .. 16 {
		qs[i] = data[off + 2 + i]
	}
	return BlockQ40{
		d:  u16(data[off]) | (u16(data[off + 1]) << 8)
		qs: qs
	}
}

// block_q41_at parses one 20-byte Q4_1 block (caller bounds-checks).
@[direct_array_access]
pub fn block_q41_at(data []u8, blk int) BlockQ41 {
	off := blk * 20
	mut qs := [16]u8{}
	for i in 0 .. 16 {
		qs[i] = data[off + 4 + i]
	}
	return BlockQ41{
		d:  u16(data[off]) | (u16(data[off + 1]) << 8)
		m:  u16(data[off + 2]) | (u16(data[off + 3]) << 8)
		qs: qs
	}
}

// dequant_q4_1_row decodes one 32-elem Q4_1 block: out = q * scale + min.
// GGML split-half layout: byte i holds q[i] (low nibble) and q[i+16]
// (high nibble) — never interleaved pairs.
@[direct_array_access]
pub fn dequant_q4_1_row(block BlockQ41, mut out []f32) {
	scale := f16bits_to_f32(block.d)
	min := f16bits_to_f32(block.m)
	for i in 0 .. 16 {
		b := block.qs[i]
		out[i] = f32(int(b & 0x0f)) * scale + min
		out[i + 16] = f32(int((b >> 4) & 0x0f)) * scale + min
	}
}

// q4_1_dot_row fuses dot(dequant(q_row), x) without materializing the row.
// Same split-half layout as dequant_q4_1_row (GGML vec_dot_q4_1).
@[direct_array_access]
pub fn q4_1_dot_row(block BlockQ41, x []f32, x_off int) f32 {
	scale := f16bits_to_f32(block.d)
	min := f16bits_to_f32(block.m)
	mut s := f32(0)
	mut xsum := f32(0)
	for i in 0 .. 16 {
		b := block.qs[i]
		lo := f32(int(b & 0x0f))
		hi := f32(int((b >> 4) & 0x0f))
		s += lo * x[x_off + i] + hi * x[x_off + i + 16]
		xsum += x[x_off + i] + x[x_off + i + 16]
	}
	return s * scale + xsum * min
}

// i8_dot_row is the plain int8 dot for embedding/head rows (no scale).
@[direct_array_access]
pub fn i8_dot_row(qs []i8, x []f32, x_off int, n int) f32 {
	mut s := f32(0)
	for i in 0 .. n {
		s += f32(qs[i]) * x[x_off + i]
	}
	return s
}

// Q6_K super-block (256 elems, 210 bytes): ql[128] low nibbles,
// qh[64] high 2-bits (4 per byte), scales[16] int8 per-16 group,
// f16 super-scale d. Layout mirrors ggml vec_dot_q6_K: the 256 values
// split into two 128-halves; byte l of each half's ql/qh covers
// positions l, l+32, l+64, l+96 of that half.
pub struct BlockQ6K {
pub:
	ql     [128]u8
	qh     [64]u8
	scales [16]i8
	d      u16 // f16 super-scale bits
}

// block_q6_k_at parses one 210-byte Q6_K super-block (caller bounds-checks).
@[direct_array_access]
pub fn block_q6_k_at(data []u8, blk int) BlockQ6K {
	off := blk * q6_k_block_bytes
	mut ql := [128]u8{}
	mut qh := [64]u8{}
	mut scales := [16]i8{}
	for i in 0 .. 128 {
		ql[i] = data[off + i]
	}
	for i in 0 .. 64 {
		qh[i] = data[off + 128 + i]
	}
	for i in 0 .. 16 {
		scales[i] = i8(data[off + 192 + i])
	}
	return BlockQ6K{
		ql:     ql
		qh:     qh
		scales: scales
		d:      u16(data[off + 208]) | (u16(data[off + 209]) << 8)
	}
}

// dequant_q6_k_block decodes one 256-elem super-block into out[0..256].
// value = d * scale[group] * (q6 - 32), q6 = (qh_2bits << 4) | ql_nibble.
@[direct_array_access]
pub fn dequant_q6_k_block(block BlockQ6K, mut out []f32) {
	d := f16bits_to_f32(block.d)
	for j in 0 .. 2 {
		for l in 0 .. 32 {
			qh := block.qh[j * 32 + l]
			q0 := block.ql[j * 64 + l]
			q1 := block.ql[j * 64 + l + 32]
			// 2-bit fields extracted in the unsigned domain, then combined.
			f0 := (qh & 3) << 4
			f1 := ((qh >> 2) & 3) << 4
			f2 := ((qh >> 4) & 3) << 4
			f3 := ((qh >> 6) & 3) << 4
			v0 := int((q0 & 0x0f) | f0) - 32
			v1 := int((q1 & 0x0f) | f1) - 32
			v2 := int((q0 >> 4) | f2) - 32
			v3 := int((q1 >> 4) | f3) - 32
			p0 := j * 128 + l
			out[p0] = f32(v0) * f32(block.scales[p0 / 16]) * d
			out[p0 + 32] = f32(v1) * f32(block.scales[(p0 + 32) / 16]) * d
			out[p0 + 64] = f32(v2) * f32(block.scales[(p0 + 64) / 16]) * d
			out[p0 + 96] = f32(v3) * f32(block.scales[(p0 + 96) / 16]) * d
		}
	}
}

// q6_k_dot_block fuses dot(dequant(super_block), x) without materializing.
@[direct_array_access]
pub fn q6_k_dot_block(block BlockQ6K, x []f32, x_off int) f32 {
	d := f16bits_to_f32(block.d)
	mut s := f32(0)
	for j in 0 .. 2 {
		for l in 0 .. 32 {
			qh := block.qh[j * 32 + l]
			q0 := block.ql[j * 64 + l]
			q1 := block.ql[j * 64 + l + 32]
			f0 := (qh & 3) << 4
			f1 := ((qh >> 2) & 3) << 4
			f2 := ((qh >> 4) & 3) << 4
			f3 := ((qh >> 6) & 3) << 4
			v0 := int((q0 & 0x0f) | f0) - 32
			v1 := int((q1 & 0x0f) | f1) - 32
			v2 := int((q0 >> 4) | f2) - 32
			v3 := int((q1 >> 4) | f3) - 32
			p0 := j * 128 + l
			s += f32(v0) * f32(block.scales[p0 / 16]) * x[x_off + p0] +
				f32(v1) * f32(block.scales[(p0 + 32) / 16]) * x[x_off + p0 + 32] +
				f32(v2) * f32(block.scales[(p0 + 64) / 16]) * x[x_off + p0 + 64] +
				f32(v3) * f32(block.scales[(p0 + 96) / 16]) * x[x_off + p0 + 96]
		}
	}
	return s * d
}

// q6_k_matvec computes out = W*x for a Q6_K row-major [rows x cols]
// matrix (cols must divide 256).
@[direct_array_access]
pub fn q6_k_matvec(data []u8, rows int, cols int, x []f32, mut out []f32) {
	blocks_per_row := cols / 256
	for r in 0 .. rows {
		mut s := f32(0)
		base := r * blocks_per_row
		for b in 0 .. blocks_per_row {
			s += q6_k_dot_block(block_q6_k_at(data, base + b), x, b * 256)
		}
		out[r] = s
	}
}

// embed_gather_q6_k decodes row id of a Q6_K [vocab x dim] matrix as f32
// (dim must divide 256; Qwen3-0.6B dim 1024 = 4 super-blocks).
@[direct_array_access]
pub fn embed_gather_q6_k(data []u8, dim int, id int, mut out []f32) {
	blocks_per_row := dim / 256
	base := id * blocks_per_row
	for b in 0 .. blocks_per_row {
		mut seg := unsafe { out[b * 256..(b + 1) * 256] }
		dequant_q6_k_block(block_q6_k_at(data, base + b), mut seg)
	}
}

// f16_to_f32 exposes the software f16 decode for probes/tools (same path
// as the quant kernels; bare-metal safe, no F16C).
pub fn f16_to_f32(bits u16) f32 {
	return f16bits_to_f32(bits)
}

fn f16bits_to_f32(bits u16) f32 {
	sign := u32(bits >> 15) << 31
	exp := int((bits >> 10) & 0x1f)
	mant := u32(bits & 0x3ff)
	mut fbits := u32(0)
	if exp == 0 {
		if mant == 0 {
			fbits = sign
		} else {
			// Subnormal: normalize.
			mut e := -14
			mut m := mant
			for m & 0x400 == 0 {
				m <<= 1
				e--
			}
			m &= 0x3ff
			fbits = sign | (u32(e + 127) << 23) | (m << 13)
		}
	} else if exp == 31 {
		fbits = sign | 0x7f800000 | (mant << 13)
	} else {
		fbits = sign | (u32(exp - 15 + 127) << 23) | (mant << 13)
	}
	return *unsafe { &f32(&fbits) }
}

// dequant_q4_0_row decodes one 32-elem block into out[0..32].
// GGML split-half layout: byte i holds q[i] (low nibble) and q[i+16]
// (high nibble) — matches vec_dot_q4_0, never interleaved pairs.
@[direct_array_access]
pub fn dequant_q4_0_row(block BlockQ40, mut out []f32) {
	scale := f16bits_to_f32(block.d)
	for i in 0 .. 16 {
		b := block.qs[i]
		out[i] = f32(int(b & 0x0f) - 8) * scale
		out[i + 16] = f32(int((b >> 4) & 0x0f) - 8) * scale
	}
}

// dequant_q8_0_row decodes one 32-elem block into out[0..32].
@[direct_array_access]
pub fn dequant_q8_0_row(block BlockQ80, mut out []f32) {
	scale := f16bits_to_f32(block.d)
	for i in 0 .. 32 {
		out[i] = f32(block.qs[i]) * scale
	}
}

// q8_0_dot_row computes dot(q_row_int8, x_f32) * scale without
// materializing the dequantized row: the forward pass never needs the
// intermediate f32 vector, saving a full hidden-size store per row.
@[direct_array_access]
pub fn q8_0_dot_row(block BlockQ80, x []f32, x_off int) f32 {
	scale := f16bits_to_f32(block.d)
	mut s := f32(0)
	for i in 0 .. 32 {
		s += f32(block.qs[i]) * x[x_off + i]
	}
	return s * scale
}

// q4_0_dot_row computes dot(dequant(q_row_4bit), x_f32) * scale without
// materializing the dequantized row: same fused idea as q8_0_dot_row,
// but for the Q4_0 weights the v1 model ships (1.1 GB traffic per token
// at Q4_K_M scale; fused saves a full hidden-size f32 store per row).
// Split-half nibble layout matches dequant_q4_0_row and GGML vec_dot.
@[direct_array_access]
pub fn q4_0_dot_row(block BlockQ40, x []f32, x_off int) f32 {
	scale := f16bits_to_f32(block.d)
	mut s := f32(0)
	for i in 0 .. 16 {
		b := block.qs[i]
		s += f32(int(b & 0x0f) - 8) * x[x_off + i]
		s += f32(int((b >> 4) & 0x0f) - 8) * x[x_off + i + 16]
	}
	return s * scale
}

// dequant_f16_row converts n f16 lanes (as u16 bits) to f32.
@[direct_array_access]
pub fn dequant_f16_row(src []u16, mut out []f32) {
	for i in 0 .. src.len {
		out[i] = f16bits_to_f32(src[i])
	}
}
