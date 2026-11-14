module infer

// Phase-1b kernel tests: block parsing, fused Q4/Q8/i8 matvecs, embed
// gather, rope-q agreement. All synthetic and fast; real-weight parity
// runs through tools/parity_llama (greedy tokens vs llama.cpp).

fn q40_bytes(scale_bits u16, nibbles []u8) []u8 {
	mut out := []u8{len: 18}
	out[0] = u8(scale_bits & 0xff)
	out[1] = u8(scale_bits >> 8)
	for i in 0 .. 16 {
		out[2 + i] = nibbles[i]
	}
	return out
}

fn test_block_q40_round_trip() {
	nibbles := [u8(0x12), 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a,
		0xbc, 0xde, 0xf0]
	data := q40_bytes(u16(0x3c00), nibbles)
	b := block_q40_at(data, 0)
	assert b.d == u16(0x3c00)
	for i in 0 .. 16 {
		assert b.qs[i] == nibbles[i]
	}
}

// GGML split-half layout regression: byte 0 = 0xAB decodes to q[0]=0xB-8
// and q[16]=0xA-8 (low nibble first half, high nibble second half).
// The old interleaved code put them at q[0],q[1] and passed every
// self-consistency test while disagreeing with llama.cpp.
fn test_q4_0_split_half_layout() {
	mut data := []u8{len: 18}
	data[0] = u8(0x00)
	data[1] = u8(0x3c) // scale 1.0
	data[2] = u8(0xab) // byte 0: lo=0xB, hi=0xA
	for i in 3 .. 18 {
		data[i] = u8(0x88) // rest decodes to 0
	}
	mut out := []f32{len: 32}
	dequant_q4_0_row(block_q40_at(data, 0), mut out)
	assert out[0] == f32(0xb - 8)
	assert out[16] == f32(0xa - 8)
	assert out[1] == f32(0)
	assert out[17] == f32(0)
	// fused dot agrees on split-half positions
	mut x := []f32{len: 32}
	x[0] = 1.0
	x[16] = 2.0
	got := q4_0_dot_row(block_q40_at(data, 0), x, 0)
	assert got == f32(0xb - 8) * 1.0 + f32(0xa - 8) * 2.0
}

fn test_q4_1_split_half_layout() {
	mut data := []u8{len: 20}
	data[0] = u8(0x00)
	data[1] = u8(0x3c) // scale 1.0
	data[2] = u8(0x00)
	data[3] = u8(0x00) // min 0.0
	data[4] = u8(0xab)
	for i in 5 .. 20 {
		data[i] = u8(0x00)
	}
	mut out := []f32{len: 32}
	dequant_q4_1_row(block_q41_at(data, 0), mut out)
	assert out[0] == f32(0xb)
	assert out[16] == f32(0xa)
	assert out[1] == f32(0)
	mut x := []f32{len: 32}
	x[0] = 1.0
	x[16] = 2.0
	got := q4_1_dot_row(block_q41_at(data, 0), x, 0)
	assert got == f32(0xb) * 1.0 + f32(0xa) * 2.0
}

// Q6_K super-block layout regression (GGML block_q6_K: ql[128], qh[64],
// scales[16], f16 d). Byte 0 of ql/qh addresses positions 0/32/64/96.
fn test_q6_k_layout() {
	mut data := []u8{len: 210}
	data[0] = u8(0xab) // ql[0]: low=0xB (pos 0), high=0xA (pos 64)
	data[128] = u8(0x1b) // qh[0]: pos0->3, pos32->2, pos64->1, pos96->0
	for i in 192 .. 208 {
		data[i] = u8(0x01) // all scales = 1
	}
	data[208] = u8(0x00)
	data[209] = u8(0x3c) // d = 1.0
	mut out := []f32{len: 256}
	dequant_q6_k_block(block_q6_k_at(data, 0), mut out)
	assert out[0] == f32((0x3b - 32)) // (3<<4)|0xB = 59 - 32 = 27
	assert out[32] == f32((0x20 - 32)) // (2<<4)|0 = 0
	assert out[64] == f32((0x1a - 32)) // (1<<4)|0xA = 26 - 32 = -6
	assert out[96] == f32(-32)
	assert out[1] == f32(-32)
	mut x := []f32{len: 256}
	x[0] = 1.0
	x[32] = 2.0
	x[64] = 3.0
	x[96] = 4.0
	got := q6_k_dot_block(block_q6_k_at(data, 0), x, 0)
	assert got == f32(27 * 1 + 0 * 2 - 6 * 3 - 32 * 4)
}

fn test_q4_0_matvec_matches_dequant() {
	rows := 4
	cols := 64
	mut data := []u8{len: rows * (cols / 32) * 18}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	// fix up f16 scales to 1.0 so magnitudes stay sane
	for r in 0 .. rows {
		for b in 0 .. cols / 32 {
			off := (r * (cols / 32) + b) * 18
			data[off] = u8(0x00)
			data[off + 1] = u8(0x3c)
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. cols {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut got := []f32{len: rows}
	q4_0_matvec(data, rows, cols, x, mut got)
	// reference: materialize then naive matvec
	mut w := []f32{len: rows * cols}
	mut row := []f32{len: 32}
	for r in 0 .. rows {
		for b in 0 .. cols / 32 {
			dequant_q4_0_row(block_q40_at(data, r * (cols / 32) + b), mut row)
			for i in 0 .. 32 {
				w[r * cols + b * 32 + i] = row[i]
			}
		}
	}
	mut want := []f32{len: rows}
	matvec(w, x, rows, cols, mut want)
	for i in 0 .. rows {
		d := got[i] - want[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_q4_1_matvec_matches_dequant() {
	rows := 3
	cols := 64
	mut data := []u8{len: rows * (cols / 32) * 20}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / 32 {
			off := (r * (cols / 32) + b) * 20
			data[off] = u8(0x00)
			data[off + 1] = u8(0x3c) // scale 1.0
			data[off + 2] = u8(0x00)
			data[off + 3] = u8(0x00) // min 0.0
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. cols {
		x[i] = f32(i % 13) * 0.07 - 0.3
	}
	mut got := []f32{len: rows}
	q4_1_matvec(data, rows, cols, x, mut got)
	mut w := []f32{len: rows * cols}
	mut row := []f32{len: 32}
	for r in 0 .. rows {
		for b in 0 .. cols / 32 {
			dequant_q4_1_row(block_q41_at(data, r * (cols / 32) + b), mut row)
			for i in 0 .. 32 {
				w[r * cols + b * 32 + i] = row[i]
			}
		}
	}
	mut want := []f32{len: rows}
	matvec(w, x, rows, cols, mut want)
	for i in 0 .. rows {
		d := got[i] - want[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_i8_matvec_and_gather() {
	rows := 3
	cols := 16
	mut data := []u8{len: rows * cols}
	for i in 0 .. data.len {
		data[i] = u8((i * 29 + 5) % 200)
	}
	mut x := []f32{len: cols}
	for i in 0 .. cols {
		x[i] = f32(i) * 0.1 - 0.5
	}
	mut got := []f32{len: rows}
	i8_matvec(data, rows, cols, x, mut got)
	for r in 0 .. rows {
		mut want := f32(0)
		for c in 0 .. cols {
			want += f32(i8(data[r * cols + c])) * x[c]
		}
		d := got[r] - want
		assert d < 1e-4 && d > -1e-4
	}
	mut row := []f32{len: cols}
	embed_gather_i8(data, cols, 1, mut row)
	for c in 0 .. cols {
		assert row[c] == f32(i8(data[cols + c]))
	}
}

fn test_rope_q_matches_pair_q_half() {
	hd := 128
	mut cos_sin := []f32{len: hd}
	rope_table(mut cos_sin, 7, hd, 1000000.0)
	mut q1 := []f32{len: hd}
	mut q2 := []f32{len: hd}
	mut k := []f32{len: hd}
	for i in 0 .. hd {
		q1[i] = f32(i % 19) * 0.03 - 0.2
		q2[i] = q1[i]
		k[i] = f32(i % 11) * 0.04 - 0.1
	}
	mut k_copy := k.clone()
	rope_in_place(mut q1, mut k, cos_sin, hd)
	rope_q_in_place(mut q2, cos_sin, hd)
	assert k != k_copy // pair path really rotated k (sanity)
	for i in 0 .. hd {
		d := q1[i] - q2[i]
		assert d < 1e-6 && d > -1e-6
	}
}

// Split-half lock-in: pair (h, h+half) shares freq h (HF/Qwen3). With
// cos=[1,0..], sin=[1,0..] (pos chosen so), q=[1,0,0,0] must rotate the
// 0/2 pair, leaving 1 and 3 untouched. Adjacent-pair (NeoX) code puts
// the sine term at index 1 instead — this test pins the convention.
fn test_rope_split_half_pairing() {
	hd := 4
	// freq[0]=1, freq[1] arbitrary; pos=pi/2 → cos0=0,sin0=1; pick table directly
	mut cos_sin := [f32(0.0), 1.0, 1.0, 0.0] // cos=[0,1], sin=[1,0]
	mut q := [f32(1.0), 0.0, 0.0, 0.0]
	mut k := [f32(0.0), 0.0, 0.0, 0.0]
	rope_in_place(mut q, mut k, cos_sin, hd)
	assert q[0] == f32(0.0) // 1*0 - 0*1
	assert q[2] == f32(1.0) // 1*1 + 0*0
	assert q[1] == f32(0.0) // untouched by pair (1,3)
	assert q[3] == f32(0.0) // untouched by pair (1,3)
}
