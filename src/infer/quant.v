module infer

// Dequant kernels: Q4_0, Q8_0, F16, F32 -> F32 rows.
// Pure V, no GC pressure (caller owns `out`). Matches GGML block layouts.
// f16 values are carried as u16 bits (V has no f16 scalar); conversion is
// software-only, no F16C dependency (bare-metal safe).

pub const qk4_0 = 32
pub const qk8_0 = 32

pub struct BlockQ40 {
pub:
	d  u16 // f16 bits
	qs [16]u8 // 32 nibbles packed
}

pub struct BlockQ80 {
pub:
	d  u16 // f16 bits
	qs [32]i8
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
pub fn dequant_q4_0_row(block BlockQ40, mut out []f32) {
	scale := f16bits_to_f32(block.d)
	for i in 0 .. 16 {
		b := block.qs[i]
		lo := f32(int(b & 0x0f) - 8) * scale
		hi := f32(int((b >> 4) & 0x0f) - 8) * scale
		out[i * 2] = lo
		out[i * 2 + 1] = hi
	}
}

// dequant_q8_0_row decodes one 32-elem block into out[0..32].
pub fn dequant_q8_0_row(block BlockQ80, mut out []f32) {
	scale := f16bits_to_f32(block.d)
	for i in 0 .. 32 {
		out[i] = f32(block.qs[i]) * scale
	}
}

// dequant_f16_row converts n f16 lanes (as u16 bits) to f32.
pub fn dequant_f16_row(src []u16, mut out []f32) {
	for i in 0 .. src.len {
		out[i] = f16bits_to_f32(src[i])
	}
}
