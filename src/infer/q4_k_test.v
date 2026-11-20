module infer

import os

// Q4_K gates: block layout, dequant hand values, fused-vs-dequant, matvec
// reference, C-vs-V, threaded/pool. Primary-source layout in quant.v.

fn q4abs(x f32) f32 {
	return if x < 0 { -x } else { x }
}

fn test_q4_k_block_and_dequant() {
	assert q4_k_block_bytes == 142
	assert qk_k == 256
	mut data := []u8{len: q4_k_block_bytes}
	data[0] = 0x00
	data[1] = 0x3c // d = 1.0
	data[2] = 0x00
	data[3] = 0x00 // dmin = 0
	data[4] = 63 // scales[0]: sub-block 0 scale = 63
	data[5] = 63 // scales[1]: sub-block 1 scale = 63
	data[8] = 0 // scales[4]: sub-block 0 min = 0
	data[9] = 0 // scales[5]: sub-block 1 min = 0
	data[16] = 0x12 // qs[0]: q[0]=2, q[32]=1
	blk := block_q4_k_at(data, 0)
	assert blk.d == 0x3c00
	assert blk.dmin == 0
	assert blk.scales[0] == 63
	assert blk.qs[0] == 0x12
	mut out := []f32{len: qk_k}
	dequant_q4_k_block(blk, mut out)
	assert q4abs(out[0] - 126.0) < 1e-5
	assert q4abs(out[32] - 63.0) < 1e-5
	for i in 1 .. 32 {
		assert q4abs(out[i]) < 1e-6
	}
}

fn test_q4_k_fused_dot_matches_dequant() {
	mut data := []u8{len: q4_k_block_bytes * 3}
	for i in 0 .. data.len {
		data[i] = u8((i * 37 + 11) % 256)
	}
	for b in 0 .. 3 {
		off := b * q4_k_block_bytes
		data[off] = 0x00
		data[off + 1] = 0x38 // d = 0.5
		data[off + 2] = 0x00
		data[off + 3] = 0x38 // dmin = 0.5
	}
	mut x := []f32{len: qk_k}
	for i in 0 .. x.len {
		x[i] = f32(i % 11) * 0.07 - 0.35
	}
	mut row := []f32{len: qk_k}
	for b in 0 .. 3 {
		blk := block_q4_k_at(data, b)
		dequant_q4_k_block(blk, mut row)
		mut ref_dot := f32(0)
		for i in 0 .. qk_k {
			ref_dot += row[i] * x[i]
		}
		got := q4_k_dot_block(blk, x, 0)
		assert q4abs(got - ref_dot) < 1e-3 * q4abs(ref_dot) + 1e-2
	}
}

fn q4_k_make_data(rows int, cols int) []u8 {
	mut data := []u8{len: rows * (cols / qk_k) * q4_k_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk_k {
			off := (r * (cols / qk_k) + b) * q4_k_block_bytes
			data[off] = 0x00
			data[off + 1] = 0x38 // d = 0.5
			data[off + 2] = 0x00
			data[off + 3] = 0x38 // dmin = 0.5
		}
	}
	return data
}

fn test_q4_k_c_matches_v() {
	rows := 5
	cols := 512
	data := q4_k_make_data(rows, cols)
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut a := []f32{len: rows}
	mut b := []f32{len: rows}
	q4_k_matvec(data, rows, cols, x, mut a)
	q4_k_matvec_c(data, rows, cols, x, mut b)
	for i in 0 .. rows {
		assert q4abs(a[i] - b[i]) < 1e-3 * q4abs(a[i]) + 1e-2
	}
	mut c := []f32{len: rows}
	q4_k_matvec_c_range(data, rows, cols, x, 0, rows, mut c)
	for i in 0 .. rows {
		assert q4abs(c[i] - b[i]) < 1e-3 * q4abs(b[i]) + 1e-2
	}
}

fn test_q4_k_threaded_matches_serial() {
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '1', true)
	rows := 2101
	cols := 256
	data := q4_k_make_data(rows, cols)
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 13) * 0.05 - 0.3
	}
	mut serial := []f32{len: rows}
	mut threaded := []f32{len: rows}
	mut pooled := []f32{len: rows}
	q4_k_matvec(data, rows, cols, x, mut serial)
	q4_k_matvec_parallel(data, rows, cols, x, mut threaded)
	pool := new_worker_pool(4)
	q4_k_matvec_parallel_pool(pool, data, rows, cols, x, mut pooled)
	for i in 0 .. rows {
		assert q4abs(serial[i] - threaded[i]) < 1e-3 * q4abs(serial[i]) + 1e-2
		assert q4abs(serial[i] - pooled[i]) < 1e-3 * q4abs(serial[i]) + 1e-2
	}
	if v := orig {
		os.setenv('VLLM_THREADS', v, true)
	} else {
		os.unsetenv('VLLM_THREADS')
	}
}

fn test_q4_k_matvec_matches_reference() {
	rows := 5
	cols := 512
	mut data := []u8{len: rows * (cols / qk_k) * q4_k_block_bytes}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		for b in 0 .. cols / qk_k {
			off := (r * (cols / qk_k) + b) * q4_k_block_bytes
			data[off] = 0x00
			data[off + 1] = 0x38
			data[off + 2] = 0x00
			data[off + 3] = 0x38
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut got := []f32{len: rows}
	q4_k_matvec(data, rows, cols, x, mut got)
	mut row := []f32{len: qk_k}
	for r in 0 .. rows {
		mut ref_dot := f32(0)
		for b in 0 .. cols / qk_k {
			dequant_q4_k_block(block_q4_k_at(data, r * (cols / qk_k) + b), mut row)
			for i in 0 .. qk_k {
				ref_dot += row[i] * x[b * qk_k + i]
			}
		}
		assert q4abs(got[r] - ref_dot) < 1e-3 * q4abs(ref_dot) + 1e-2
	}
}
