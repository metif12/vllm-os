module infer

import os
import sync

// Correctness gates for the persistent worker pool (phase 7). Timing is NOT
// gated here (CI boxes vary) — tools/forward_bench.v measures the win.
// Every pool path must agree with the serial kernel it replaces; the
// fork-join members must be BIT-EXACT (disjoint outs, deterministic kernel).

fn test_pool_matvec_matches_serial() {
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '4', true)
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	pool := new_worker_pool(4)
	assert pool != unsafe { nil }
	rows := 2100 // remainder rows spread across 4 workers
	cols := 64
	mut w := []f32{len: rows * cols}
	mut x := []f32{len: cols}
	for i in 0 .. w.len {
		w[i] = f32(i % 13) * 0.125 - 0.5
	}
	for i in 0 .. x.len {
		x[i] = f32(i % 7) * 0.25 - 0.125
	}
	mut serial := []f32{len: rows}
	mut pooled := []f32{len: rows}
	matvec_c(w, x, rows, cols, mut serial)
	matvec_parallel_pool(pool, w, x, rows, cols, mut pooled)
	for i in 0 .. rows {
		d := pooled[i] - serial[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_pool_q6k_matches_serial() {
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '4', true)
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	pool := new_worker_pool(4)
	assert pool != unsafe { nil }
	rows := 2101
	cols := 256
	mut data := []u8{len: rows * 210}
	for i in 0 .. data.len {
		data[i] = u8((i * 53 + 7) % 256)
	}
	for r in 0 .. rows {
		off := r * 210
		for i in 192 .. 208 {
			data[off + i] = u8(((i * 7 + r) % 5 + 254) % 256)
		}
		data[off + 208] = 0
		data[off + 209] = 0x38 // d = 0.5
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 41) * 0.05 - 1.0
	}
	mut serial := []f32{len: rows}
	mut pooled := []f32{len: rows}
	q6_k_matvec(data, rows, cols, x, mut serial)
	q6_k_matvec_parallel_pool(pool, data, rows, cols, x, mut pooled)
	for i in 0 .. rows {
		d := pooled[i] - serial[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_pool_q40_forkjoin_bit_exact() {
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '4', true)
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	pool := new_worker_pool(4)
	assert pool != unsafe { nil }
	ra := 64
	rb := 48
	rc := 32
	cols := 128
	mut da := []u8{len: ra * (cols / 32) * 18}
	mut db := []u8{len: rb * (cols / 32) * 18}
	mut dc := []u8{len: rc * (cols / 32) * 18}
	for i in 0 .. da.len {
		da[i] = u8((i * 37 + 11) % 256)
	}
	for i in 0 .. db.len {
		db[i] = u8((i * 53 + 7) % 256)
	}
	for i in 0 .. dc.len {
		dc[i] = u8((i * 29 + 5) % 200)
	}
	for r in 0 .. ra {
		for b in 0 .. cols / 32 {
			off := (r * (cols / 32) + b) * 18
			da[off] = 0
			da[off + 1] = 0x3c
		}
	}
	for r in 0 .. rb {
		for b in 0 .. cols / 32 {
			off := (r * (cols / 32) + b) * 18
			db[off] = 0
			db[off + 1] = 0x3c
		}
	}
	for r in 0 .. rc {
		for b in 0 .. cols / 32 {
			off := (r * (cols / 32) + b) * 18
			dc[off] = 0
			dc[off + 1] = 0x3c
		}
	}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 17) * 0.05 - 0.4
	}
	mut sa := []f32{len: ra}
	mut sb := []f32{len: rb}
	mut sc := []f32{len: rc}
	mut pa := []f32{len: ra}
	mut pb := []f32{len: rb}
	mut pc := []f32{len: rc}
	q4_0_matvec_c(da, ra, cols, x, mut sa)
	q4_0_matvec_c(db, rb, cols, x, mut sb)
	q4_0_matvec_c(dc, rc, cols, x, mut sc)
	q4_0_matvec_3way_pool(pool, da, db, dc, ra, rb, rc, cols, x, mut pa, mut pb, mut pc)
	for i in 0 .. ra {
		assert pa[i] == sa[i]
	}
	for i in 0 .. rb {
		assert pb[i] == sb[i]
	}
	for i in 0 .. rc {
		assert pc[i] == sc[i]
	}
	mut s2a := []f32{len: ra}
	mut s2b := []f32{len: rb}
	mut p2a := []f32{len: ra}
	mut p2b := []f32{len: rb}
	q4_0_matvec_c(da, ra, cols, x, mut s2a)
	q4_0_matvec_c(db, rb, cols, x, mut s2b)
	q4_0_matvec_2way_pool(pool, da, db, ra, rb, cols, x, mut p2a, mut p2b)
	for i in 0 .. ra {
		assert p2a[i] == s2a[i]
	}
	for i in 0 .. rb {
		assert p2b[i] == s2b[i]
	}
}

fn pool_concurrent_worker(pool &WorkerPool, w []f32, x []f32, mut out []f32, wg &sync.WaitGroup) {
	defer {
		wg.done()
	}
	matvec_parallel_pool(pool, w, x, 2048, 64, mut out)
}

// Several submitters share one pool: each batch has its own WaitGroup, so
// interleaved jobs must still land in the right caller's output buffer.
fn test_pool_concurrent_batches() {
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '4', true)
	defer {
		if v := orig {
			os.setenv('VLLM_THREADS', v, true)
		} else {
			os.unsetenv('VLLM_THREADS')
		}
	}
	pool := new_worker_pool(4)
	assert pool != unsafe { nil }
	n := 4
	rows := 2048
	cols := 64
	mut weights := [][]f32{cap: n}
	mut results := [][]f32{cap: n}
	mut serials := [][]f32{cap: n}
	mut x := []f32{len: cols}
	for i in 0 .. x.len {
		x[i] = f32(i % 7) * 0.25
	}
	for t in 0 .. n {
		mut w := []f32{len: rows * cols}
		for i in 0 .. w.len {
			w[i] = f32((i + t) % 13) * 0.125
		}
		weights << w
		results << []f32{len: rows}
		serials << []f32{len: rows}
		matvec_c(w, x, rows, cols, mut serials[t])
	}
	wg := sync.new_waitgroup()
	wg.add(n)
	for t in 0 .. n {
		spawn pool_concurrent_worker(pool, weights[t], x, mut results[t], wg)
	}
	wg.wait()
	for t in 0 .. n {
		for i in 0 .. rows {
			d := results[t][i] - serials[t][i]
			assert d < 1e-3 && d > -1e-3
		}
	}
}
