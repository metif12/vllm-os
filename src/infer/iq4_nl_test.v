module infer

import os

// IQ4_NL correctness gates. The 16-value codebook and 18-byte split-half
// block layout are pinned to ggml-common.h / ggml-quants.c (provenance in
// quant.v). Fused dot must agree with dequant-then-dot; the C/AVX2 wrapper
// agreement is gated in simd_test.v / this file under -prod.

fn iq4_abs(x f32) f32 {
	return if x < 0 { -x } else { x }
}

fn test_iq4_nl_grid_pinned_to_ggml() {
	expected := [f32(-127), -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
	for i in 0 .. 16 {
		assert iq4_nl_grid(i) == expected[i]
	}
	assert iq4_nl_block_bytes == 18
	assert qk4_nl == 32
}

fn test_iq4_nl_dequant_split_half() {
	// d = 1.0, byte 0 codes: low=0 (-127), high=1 (-104).
	mut data := []u8{len: iq4_nl_block_bytes}
	data[0] = 0x00
	data[1] = 0x3c
	data[2] = 0x10
	mut out := []f32{len: qk4_nl}
	dequant_iq4_nl_row(block_iq4_nl_at(data, 0), mut out)
	assert out[0] == -127.0
	assert out[16] == -104.0
	for i in 1 .. 16 {
		assert out[i] == iq4_nl_grid(0)
		assert out[i + 16] == iq4_nl_grid(0)
	}
}

fn test_iq4_nl_fused_dot_matches_dequant() {
	mut data := []u8{len: iq4_nl_block_bytes * 4}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	// Valid finite f16 scales in each block (0.5, 1.0, 2.0, 4.0).
	scales := [u16(0x3800), 0x3c00, 0x4000, 0x4400]
	for b in 0 .. 4 {
		off := b * iq4_nl_block_bytes
		data[off] = u8(scales[b] & 0xff)
		data[off + 1] = u8(scales[b] >> 8)
	}
	mut x := []f32{len: qk4_nl}
	for i in 0 .. x.len {
		x[i] = f32(i % 11) * 0.07 - 0.35
	}
	mut row := []f32{len: qk4_nl}
	for b in 0 .. 4 {
		blk := block_iq4_nl_at(data, b)
		dequant_iq4_nl_row(blk, mut row)
		mut ref_dot := f32(0)
		for i in 0 .. qk4_nl {
			ref_dot += row[i] * x[i]
		}
		got := iq4_nl_dot_row(blk, x, 0)
		mut diff := got - ref_dot
		if diff < 0 {
			diff = -diff
		}
		// f32 accumulation order differs; allow 1e-3 relative + floor.
		mut mag := ref_dot
		if mag < 0 {
			mag = -mag
		}
		assert diff < 1e-3 * mag + 1e-2
	}
}

fn test_iq4_nl_matvec_matches_reference() {
	rows := 40
	cols := 128 // 4 blocks per row
	mut data := []u8{len: rows * (cols / qk4_nl) * iq4_nl_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk4_nl {
			off := (r * (cols / qk4_nl) + b) * iq4_nl_block_bytes
			// d in [0.25, 2.0]: f16 exponent bits 13..15, zero mantissa.
			d_bits := u16(13 + (r + b) % 3) << 10
			data[off] = u8(d_bits & 0xff)
			data[off + 1] = u8(d_bits >> 8)
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut got := []f32{len: rows}
	iq4_nl_matvec(data, rows, cols, x, mut got)
	mut row := []f32{len: qk4_nl}
	for r in 0 .. rows {
		mut ref_dot := f32(0)
		for b in 0 .. cols / qk4_nl {
			dequant_iq4_nl_row(block_iq4_nl_at(data, r * (cols / qk4_nl) + b), mut row)
			for i in 0 .. qk4_nl {
				ref_dot += row[i] * x[b * qk4_nl + i]
			}
		}
		mut diff := got[r] - ref_dot
		if diff < 0 {
			diff = -diff
		}
		mut mag := ref_dot
		if mag < 0 {
			mag = -mag
		}
		assert diff < 1e-3 * mag + 1e-2
	}
}

// In -prod on an AVX2 host this compares the pshufb C kernel against the
// scalar V reference. Outside -prod the `_c` wrappers are the V kernels, so
// the test is a no-op agreement check (same as the other simd gates).
fn test_iq4_nl_c_matches_v() {
	rows := 12
	cols := 128
	mut data := []u8{len: rows * (cols / qk4_nl) * iq4_nl_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 41 + 13) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk4_nl {
			off := (r * (cols / qk4_nl) + b) * iq4_nl_block_bytes
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
	iq4_nl_matvec(data, rows, cols, x, mut a)
	iq4_nl_matvec_c(data, rows, cols, x, mut b)
	for i in 0 .. rows {
		mut diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		mut mag := a[i]
		if mag < 0 {
			mag = -mag
		}
		assert diff < 1e-3 * mag + 1e-2
	}
	// Range form must agree with the whole-matrix C call.
	mut c := []f32{len: rows}
	iq4_nl_matvec_c_range(data, rows, cols, x, 0, rows, mut c)
	for i in 0 .. rows {
		mut diff := c[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		assert diff < 1e-3 * iq4_abs(b[i]) + 1e-2
	}
	// Fused dot wrapper agrees with the scalar V dot on one block.
	blk := block_iq4_nl_at(data, 0)
	dv := iq4_nl_dot_row(blk, x, 0)
	dc := iq4_nl_dot_row_c(blk, x, 0)
	mut dd := dv - dc
	if dd < 0 {
		dd = -dd
	}
	assert dd < 1e-3 * iq4_abs(dv) + 1e-2
}

// Threaded row-block head (spawn + pool forms) must match the serial kernel,
// including the remainder-row partition at rows % workers != 0.
fn test_iq4_nl_parallel_matches_serial() {
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
	mut data := []u8{len: rows * (cols / qk4_nl) * iq4_nl_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk4_nl {
			off := (r * (cols / qk4_nl) + b) * iq4_nl_block_bytes
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
	iq4_nl_matvec(data, rows, cols, x, mut serial)
	iq4_nl_matvec_parallel(data, rows, cols, x, mut threaded)
	pool := new_worker_pool(4)
	iq4_nl_matvec_parallel_pool(pool, data, rows, cols, x, mut pooled)
	for i in 0 .. rows {
		mut d1 := serial[i] - threaded[i]
		if d1 < 0 {
			d1 = -d1
		}
		mut d2 := serial[i] - pooled[i]
		if d2 < 0 {
			d2 = -d2
		}
		assert d1 < 1e-3 * iq4_abs(serial[i]) + 1e-2
		assert d2 < 1e-3 * iq4_abs(serial[i]) + 1e-2
	}
}

fn test_iq4_nl_embed_gather() {
	dim := 64
	id := 3
	mut data := []u8{len: 10 * (dim / qk4_nl) * iq4_nl_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 29 + 5) % 256)
	}
	for b in 0 .. 10 * (dim / qk4_nl) {
		off := b * iq4_nl_block_bytes
		data[off] = 0x00
		data[off + 1] = 0x3c // d = 1.0
	}
	mut out := []f32{len: dim}
	embed_gather_iq4_nl(data, dim, id, mut out)
	for b in 0 .. dim / qk4_nl {
		mut row := []f32{len: qk4_nl}
		dequant_iq4_nl_row(block_iq4_nl_at(data, id * (dim / qk4_nl) + b), mut row)
		for i in 0 .. qk4_nl {
			assert out[b * qk4_nl + i] == row[i]
		}
	}
}
