module infer

import os

// Q2_0 correctness gates. GGML type 42, 18-byte block (f16 d + 16 bytes of
// 64 two-bit codes), value = (code-1)*d. Provenance in quant.v.

fn q20_abs(x f32) f32 {
	return if x < 0 { -x } else { x }
}

fn test_q2_0_code_and_dequant() {
	assert q2_0_block_bytes == 18
	assert qk2_0 == 64
	mut data := []u8{len: q2_0_block_bytes}
	data[0] = 0x00
	data[1] = 0x3c // d = 1.0
	// Byte 0 codes: j=0 -> 0 (-1), j=1 -> 1 (0), j=2 -> 2 (+1), j=3 -> 3 (+2).
	data[2] = 0xe4
	blk := block_q2_0_at(data, 0)
	assert q2_0_code(blk, 0) == 0
	assert q2_0_code(blk, 1) == 1
	assert q2_0_code(blk, 2) == 2
	assert q2_0_code(blk, 3) == 3
	mut out := []f32{len: qk2_0}
	dequant_q2_0_row(blk, mut out)
	assert out[0] == -1.0
	assert out[1] == 0.0
	assert out[2] == 1.0
	assert out[3] == 2.0
	// Remaining bytes are zero -> code 0 -> -1.
	for j in 4 .. qk2_0 {
		assert out[j] == -1.0
	}
}

fn test_q2_0_fused_dot_matches_dequant() {
	mut data := []u8{len: q2_0_block_bytes * 4}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	scales := [u16(0x3800), 0x3c00, 0x4000, 0x4400]
	for b in 0 .. 4 {
		off := b * q2_0_block_bytes
		data[off] = u8(scales[b] & 0xff)
		data[off + 1] = u8(scales[b] >> 8)
	}
	mut x := []f32{len: qk2_0}
	for i in 0 .. x.len {
		x[i] = f32(i % 11) * 0.07 - 0.35
	}
	mut row := []f32{len: qk2_0}
	for b in 0 .. 4 {
		blk := block_q2_0_at(data, b)
		dequant_q2_0_row(blk, mut row)
		mut ref_dot := f32(0)
		for i in 0 .. qk2_0 {
			ref_dot += row[i] * x[i]
		}
		got := q2_0_dot_row(blk, x, 0)
		mut diff := got - ref_dot
		if diff < 0 {
			diff = -diff
		}
		assert diff < 1e-3 * q20_abs(ref_dot) + 1e-2
	}
}

fn test_q2_0_matvec_matches_reference() {
	rows := 20
	cols := 256
	mut data := []u8{len: rows * (cols / qk2_0) * q2_0_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk2_0 {
			off := (r * (cols / qk2_0) + b) * q2_0_block_bytes
			bits := u16(13 + (r + b) % 3) << 10
			data[off] = u8(bits & 0xff)
			data[off + 1] = u8(bits >> 8)
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut got := []f32{len: rows}
	q2_0_matvec(data, rows, cols, x, mut got)
	mut row := []f32{len: qk2_0}
	for r in 0 .. rows {
		mut ref_dot := f32(0)
		for b in 0 .. cols / qk2_0 {
			dequant_q2_0_row(block_q2_0_at(data, r * (cols / qk2_0) + b), mut row)
			for i in 0 .. qk2_0 {
				ref_dot += row[i] * x[b * qk2_0 + i]
			}
		}
		mut diff := got[r] - ref_dot
		if diff < 0 {
			diff = -diff
		}
		assert diff < 1e-3 * q20_abs(ref_dot) + 1e-2
	}
}

fn test_q2_0_c_matches_v() {
	rows := 12
	cols := 192
	mut data := []u8{len: rows * (cols / qk2_0) * q2_0_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 41 + 13) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk2_0 {
			off := (r * (cols / qk2_0) + b) * q2_0_block_bytes
			bits := u16(13 + (r + b) % 3) << 10
			data[off] = u8(bits & 0xff)
			data[off + 1] = u8(bits >> 8)
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 19) * 0.04 - 0.38
	}
	mut a := []f32{len: rows}
	mut b := []f32{len: rows}
	q2_0_matvec(data, rows, cols, x, mut a)
	q2_0_matvec_c(data, rows, cols, x, mut b)
	for i in 0 .. rows {
		mut diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		assert diff < 1e-3 * q20_abs(a[i]) + 1e-2
	}
	mut c := []f32{len: rows}
	q2_0_matvec_c_range(data, rows, cols, x, 0, rows, mut c)
	for i in 0 .. rows {
		mut diff := c[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		assert diff < 1e-3 * q20_abs(b[i]) + 1e-2
	}
}

fn test_q2_0_threaded_matches_serial() {
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '4', true)
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	rows := 2101
	cols := 256
	mut data := []u8{len: rows * (cols / qk2_0) * q2_0_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk2_0 {
			off := (r * (cols / qk2_0) + b) * q2_0_block_bytes
			data[off] = 0x00
			data[off + 1] = 0x38 // d = 0.5
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 13) * 0.05 - 0.3
	}
	mut serial := []f32{len: rows}
	mut threaded := []f32{len: rows}
	mut pooled := []f32{len: rows}
	q2_0_matvec(data, rows, cols, x, mut serial)
	q2_0_matvec_parallel(data, rows, cols, x, mut threaded)
	pool := new_worker_pool(4)
	q2_0_matvec_parallel_pool(pool, data, rows, cols, x, mut pooled)
	for i in 0 .. rows {
		mut d1 := serial[i] - threaded[i]
		if d1 < 0 {
			d1 = -d1
		}
		mut d2 := serial[i] - pooled[i]
		if d2 < 0 {
			d2 = -d2
		}
		assert d1 < 1e-3 * q20_abs(serial[i]) + 1e-2
		assert d2 < 1e-3 * q20_abs(serial[i]) + 1e-2
	}
}

// R8 repack: codes become signed bytes and the dot reuses the Q8 int8 math.
// The repacked kernel must match the native Q2_0 kernel numerically.
fn test_q2_0_r8_repack_matches_native() {
	blocks := 4
	mut src := []u8{len: blocks * q2_0_block_bytes}
	for i in 0 .. src.len {
		src[i] = u8((i * 37 + 11) % 256)
	}
	for b in 0 .. blocks {
		bits := u16(13 + b % 3) << 10
		src[b * q2_0_block_bytes] = u8(bits & 0xff)
		src[b * q2_0_block_bytes + 1] = u8(bits >> 8)
	}
	r8 := repack_q2_0_r8(src, blocks)
	assert r8.len == blocks * q2_0_r8_block_bytes
	assert q2_0_r8_block_bytes == 66
	// Scale survived, codes are signed {-1,0,1,2}.
	for b in 0 .. blocks {
		assert r8[b * q2_0_r8_block_bytes] == src[b * q2_0_block_bytes]
		assert r8[b * q2_0_r8_block_bytes + 1] == src[b * q2_0_block_bytes + 1]
		for j in 0 .. qk2_0 {
			want := q2_0_code(block_q2_0_at(src, b), j) - 1
			assert i8(r8[b * q2_0_r8_block_bytes + 2 + j]) == i8(want)
		}
	}
	mut x := []f32{len: qk2_0 * blocks}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut native := []f32{len: 1}
	mut packed := []f32{len: 1}
	q2_0_matvec(src, 1, qk2_0 * blocks, x, mut native)
	q2_0_r8_matvec(r8, 1, qk2_0 * blocks, x, mut packed)
	mut diff := native[0] - packed[0]
	if diff < 0 {
		diff = -diff
	}
	assert diff < 1e-3 * q20_abs(native[0]) + 1e-2
}

fn test_q2_0_r8_c_matches_v() {
	blocks := 4
	mut src := []u8{len: blocks * q2_0_block_bytes}
	for i in 0 .. src.len {
		src[i] = u8((i * 53 + 7) % 256)
	}
	for b in 0 .. blocks {
		src[b * q2_0_block_bytes] = 0x00
		src[b * q2_0_block_bytes + 1] = 0x3c
	}
	r8 := repack_q2_0_r8(src, blocks)
	mut x := []f32{len: qk2_0 * blocks}
	for i in 0 .. x.len {
		x[i] = f32(i % 13) * 0.04 - 0.26
	}
	mut a := []f32{len: 1}
	mut b := []f32{len: 1}
	q2_0_r8_matvec(r8, 1, qk2_0 * blocks, x, mut a)
	q2_0_r8_matvec_c(r8, 1, qk2_0 * blocks, x, mut b)
	mut diff := a[0] - b[0]
	if diff < 0 {
		diff = -diff
	}
	assert diff < 1e-3 * q20_abs(a[0]) + 1e-2
}

fn test_q2_0_embed_gather() {
	dim := 128
	id := 3
	mut data := []u8{len: 10 * (dim / qk2_0) * q2_0_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 29 + 5) % 256)
	}
	for b in 0 .. 10 * (dim / qk2_0) {
		off := b * q2_0_block_bytes
		data[off] = 0x00
		data[off + 1] = 0x3c
	}
	mut out := []f32{len: dim}
	embed_gather_q2_0(data, dim, id, mut out)
	for b in 0 .. dim / qk2_0 {
		mut row := []f32{len: qk2_0}
		dequant_q2_0_row(block_q2_0_at(data, id * (dim / qk2_0) + b), mut row)
		for i in 0 .. qk2_0 {
			assert out[b * qk2_0 + i] == row[i]
		}
	}
}
