module infer

import os
import sync

// Parallel matvec: row-block partitioned across worker threads, each
// running the dispatched kernel (SIMD when available) on its chunk.
// Threads + SIMD compose: threading multiplies available DRAM bandwidth
// up to the socket wall, SIMD minimizes per-byte cost.
//
// Cost model: one spawn-storm per call (no persistent pool yet — see
// below), so the auto backend only selects this for big shapes where
// spawn (~10-50us) vanishes next to milliseconds of streaming. Small
// projections stay serial (they already beat BLAS single-threaded).
//
// Gating (three independent levers, same contract as simd_c.v):
//  1. Compile-time: -freestanding builds exclude spawn entirely and run
//     the serial kernel (bare-metal schedulers are not pthreads).
//  2. Runtime env: VLLM_THREADS=1 forces serial even on host builds.
//  3. Config: `backend: cpu | cpu-blocked | cpu-simd` never spawns;
//     `auto` picks parallel for big shapes, `cpu-parallel` forces it.
//
// Follow-up (documented, not done): a persistent worker pool pinned per
// decode worker would remove the per-call spawn cost for the phase-1b
// forward (28 layers x projections per token). Revisit when the forward
// lands and per-token spawn shows up in profiles.

// min_rows_parallel is the auto-select floor: below this, spawn costs
// more than threading saves (measured: 256x512 regresses with threads).
pub const min_rows_parallel = 2048
pub const min_cols_parallel = 1024

// parallel_threads resolves the worker count: VLLM_THREADS when set to
// 1..64, else a fixed default of 4 (bandwidth-bound streaming saturates
// early; more threads add contention, not bytes — see docs/perf.md).
pub fn parallel_threads() int {
	if v := os.getenv_opt('VLLM_THREADS') {
		n := v.int()
		if n >= 1 && n <= 64 {
			return n
		}
	}
	return 4
}

// matvec_parallel computes out = W*x, W row-major [rows x cols], split
// across parallel_threads() workers. Chunk slices share the caller's
// buffers (no copy); out ranges are disjoint per worker. Falls back to
// the serial kernel when threads == 1 or under -freestanding.
pub fn matvec_parallel(w []f32, x []f32, rows int, cols int, mut out []f32) {
	$if freestanding {
		matvec_c(w, x, rows, cols, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 {
			matvec_c(w, x, rows, cols, mut out)
			return
		}
		// Cap workers so every chunk amortizes spawn (>= 512 rows).
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			matvec_c(w, x, rows, cols, mut out)
			return
		}
		mut wg := sync.new_waitgroup()
		// Balanced partition: distribute rows % n remainder rows so block
		// sizes differ by at most one (same fix as VSL gemv_n: a naive
		// chunk + last-takes-rest leaves one worker with the whole tail).
		base := rows / n
		rem := rows % n
		for t in 0 .. n {
			r0 := t * base + (if t < rem { t } else { rem })
			r1 := r0 + base + (if t < rem { 1 } else { 0 })
			wg.add(1)
			spawn matvec_chunk(w, x, rows, cols, r0, r1 - r0, mut out, mut wg)
		}
		wg.wait()
	}
}

// q6_k_matvec_parallel computes out = W*x for a Q6_K row-major matrix,
// split across parallel_threads() workers (row-block partitioned, each
// running the dispatched C/V kernel on its chunk via q6_k_matvec_c_range).
// The phase-1b vocab head (151936 rows, single call site) is the target:
// small projections stay serial (spawn vanishes only next to streaming).
// Gating mirrors matvec_parallel: freestanding and VLLM_THREADS=1 run
// the serial kernel (the bare-metal guest pins threads to 1).
pub fn q6_k_matvec_parallel(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if freestanding {
		q6_k_matvec_c(data, rows, cols, x, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 {
			q6_k_matvec_c(data, rows, cols, x, mut out)
			return
		}
		// Cap workers so every chunk amortizes spawn (>= 512 rows).
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			q6_k_matvec_c(data, rows, cols, x, mut out)
			return
		}
		mut wg := sync.new_waitgroup()
		// Balanced partition: rows % n remainder rows spread so block
		// sizes differ by at most one (same as matvec_parallel).
		base := rows / n
		rem := rows % n
		for t in 0 .. n {
			r0 := t * base + (if t < rem { t } else { rem })
			r1 := r0 + base + (if t < rem { 1 } else { 0 })
			wg.add(1)
			spawn q6_k_chunk(data, rows, cols, x, r0, r1 - r0, mut out, mut wg)
		}
		wg.wait()
	}
}

// q6_k_chunk runs the dispatched range kernel on one row block [r0, r0+n).
// Whole slices cross the spawn boundary; offsets resolve inside via
// q6_k_matvec_c_range (same discipline as matvec_chunk).
fn q6_k_chunk(data []u8, rows int, cols int, x []f32, r0 int, n int, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	q6_k_matvec_c_range(data, rows, cols, x, r0, n, mut out)
}

// q4_0_chunk_full runs one whole-matrix Q4_0 matvec on a worker thread.
// Each fork-join member owns a disjoint out buffer (no row-splitting:
// these shapes are too small for spawn-per-row-block to win — measured.
// The win is concurrent independent streams, not partitioned rows).
fn q4_0_chunk_full(data []u8, rows int, cols int, x []f32, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	q4_0_matvec_c(data, rows, cols, x, mut out)
}

// q4_0_matvec_2way runs two independent Q4_0 matvecs (same x, disjoint
// outs) concurrently: the FFN gate/up pair. Serial under freestanding /
// VLLM_THREADS=1 (the bare-metal guest stays single-threaded).
pub fn q4_0_matvec_2way(a []u8, b []u8, rows_a int, rows_b int, cols int, x []f32, mut out_a []f32, mut out_b []f32) {
	$if freestanding {
		q4_0_matvec_c(a, rows_a, cols, x, mut out_a)
		q4_0_matvec_c(b, rows_b, cols, x, mut out_b)
		return
	} $else {
		if parallel_threads() <= 1 {
			q4_0_matvec_c(a, rows_a, cols, x, mut out_a)
			q4_0_matvec_c(b, rows_b, cols, x, mut out_b)
			return
		}
		mut wg := sync.new_waitgroup()
		wg.add(2)
		spawn q4_0_chunk_full(a, rows_a, cols, x, mut out_a, mut wg)
		spawn q4_0_chunk_full(b, rows_b, cols, x, mut out_b, mut wg)
		wg.wait()
	}
}

// q4_0_matvec_3way runs three independent Q4_0 matvecs (same x, disjoint
// outs) concurrently: the attention q/k/v projections. Same gating as
// the 2-way entry (freestanding / VLLM_THREADS=1 pin the serial path).
pub fn q4_0_matvec_3way(a []u8, b []u8, c []u8, rows_a int, rows_b int, rows_c int, cols int, x []f32, mut out_a []f32, mut out_b []f32, mut out_c []f32) {
	$if freestanding {
		q4_0_matvec_c(a, rows_a, cols, x, mut out_a)
		q4_0_matvec_c(b, rows_b, cols, x, mut out_b)
		q4_0_matvec_c(c, rows_c, cols, x, mut out_c)
		return
	} $else {
		if parallel_threads() <= 1 {
			q4_0_matvec_c(a, rows_a, cols, x, mut out_a)
			q4_0_matvec_c(b, rows_b, cols, x, mut out_b)
			q4_0_matvec_c(c, rows_c, cols, x, mut out_c)
			return
		}
		mut wg := sync.new_waitgroup()
		wg.add(3)
		spawn q4_0_chunk_full(a, rows_a, cols, x, mut out_a, mut wg)
		spawn q4_0_chunk_full(b, rows_b, cols, x, mut out_b, mut wg)
		spawn q4_0_chunk_full(c, rows_c, cols, x, mut out_c, mut wg)
		wg.wait()
	}
}

// matvec_chunk runs the dispatched kernel on one row block [r0, r0+n).
// Whole slices cross the spawn boundary (as in the bench harness);
// offsets resolve inside via matvec_c_range.
fn matvec_chunk(w []f32, x []f32, rows int, cols int, r0 int, n int, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	matvec_c_range(w, x, rows, cols, r0, n, mut out)
}

// --- Persistent-pool variants (phase 7) ---------------------------------
//
// The originals above stay spawn-based so tools/tests keep their existing
// fallback. These variants run the same partition on a pre-spawned pool,
// removing ~140 OS-thread spawns per decode token on the Q4 path. A nil
// pool (freestanding / VLLM_THREADS=1 / no pool wired) delegates straight
// to the original spawn-or-serial entry, so numerics are bit-identical.

// matvec_parallel_pool computes out = W*x using the persistent pool.
pub fn matvec_parallel_pool(pool &WorkerPool, w []f32, x []f32, rows int, cols int, mut out []f32) {
	$if freestanding {
		matvec_c(w, x, rows, cols, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 || pool == unsafe { nil } {
			matvec_parallel(w, x, rows, cols, mut out)
			return
		}
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			matvec_c(w, x, rows, cols, mut out)
			return
		}
		wg := pool_batch(n)
		for t in 0 .. n {
			r0, r1 := pool_bounds(rows, n, t)
			pool.jobs <- PoolJob{
				kind:  PoolJobKind.matvec_range
				words: w
				x:     x
				out:   out
				rows:  rows
				cols:  cols
				r0:    r0
				n:     r1 - r0
				wg:    wg
			}
		}
		wg.wait()
	}
}

// q6_k_matvec_parallel_pool computes the Q6_K vocab head on the pool.
pub fn q6_k_matvec_parallel_pool(pool &WorkerPool, data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if freestanding {
		q6_k_matvec_c(data, rows, cols, x, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 || pool == unsafe { nil } {
			q6_k_matvec_parallel(data, rows, cols, x, mut out)
			return
		}
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			q6_k_matvec_c(data, rows, cols, x, mut out)
			return
		}
		wg := pool_batch(n)
		for t in 0 .. n {
			r0, r1 := pool_bounds(rows, n, t)
			pool.jobs <- PoolJob{
				kind: PoolJobKind.q6k_range
				data: data
				x:    x
				out:  out
				rows: rows
				cols: cols
				r0:   r0
				n:    r1 - r0
				wg:   wg
			}
		}
		wg.wait()
	}
}

// q4_0_matvec_2way_pool runs the gate/up pair on the pool (one job each).
pub fn q4_0_matvec_2way_pool(pool &WorkerPool, a []u8, b []u8, rows_a int, rows_b int, cols int, x []f32, mut out_a []f32, mut out_b []f32) {
	$if freestanding {
		q4_0_matvec_2way(a, b, rows_a, rows_b, cols, x, mut out_a, mut out_b)
		return
	} $else {
		if parallel_threads() <= 1 || pool == unsafe { nil } {
			q4_0_matvec_2way(a, b, rows_a, rows_b, cols, x, mut out_a, mut out_b)
			return
		}
		wg := pool_batch(2)
		pool.jobs <- PoolJob{
			kind: PoolJobKind.q4_0_full
			data: a
			x:    x
			out:  out_a
			rows: rows_a
			cols: cols
			wg:   wg
		}
		pool.jobs <- PoolJob{
			kind: PoolJobKind.q4_0_full
			data: b
			x:    x
			out:  out_b
			rows: rows_b
			cols: cols
			wg:   wg
		}
		wg.wait()
	}
}

// q4_0_matvec_3way_pool runs the q/k/v triple on the pool (one job each).
pub fn q4_0_matvec_3way_pool(pool &WorkerPool, a []u8, b []u8, c []u8, rows_a int, rows_b int, rows_c int, cols int, x []f32, mut out_a []f32, mut out_b []f32, mut out_c []f32) {
	$if freestanding {
		q4_0_matvec_3way(a, b, c, rows_a, rows_b, rows_c, cols, x, mut out_a, mut out_b, mut out_c)
		return
	} $else {
		if parallel_threads() <= 1 || pool == unsafe { nil } {
			q4_0_matvec_3way(a, b, c, rows_a, rows_b, rows_c, cols, x, mut out_a, mut out_b, mut
				out_c)
			return
		}
		wg := pool_batch(3)
		pool.jobs <- PoolJob{
			kind: PoolJobKind.q4_0_full
			data: a
			x:    x
			out:  out_a
			rows: rows_a
			cols: cols
			wg:   wg
		}
		pool.jobs <- PoolJob{
			kind: PoolJobKind.q4_0_full
			data: b
			x:    x
			out:  out_b
			rows: rows_b
			cols: cols
			wg:   wg
		}
		pool.jobs <- PoolJob{
			kind: PoolJobKind.q4_0_full
			data: c
			x:    x
			out:  out_c
			rows: rows_c
			cols: cols
			wg:   wg
		}
		wg.wait()
	}
}

// iq4_nl_matvec_parallel computes the IQ4_NL vocab head row-block split
// across workers (spawn-per-call form; the _pool variant below is what the
// server uses). Gating mirrors q6_k_matvec_parallel.
pub fn iq4_nl_matvec_parallel(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if freestanding {
		iq4_nl_matvec_c(data, rows, cols, x, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 {
			iq4_nl_matvec_c(data, rows, cols, x, mut out)
			return
		}
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			iq4_nl_matvec_c(data, rows, cols, x, mut out)
			return
		}
		mut wg := sync.new_waitgroup()
		base := rows / n
		rem := rows % n
		for t in 0 .. n {
			r0 := t * base + (if t < rem { t } else { rem })
			r1 := r0 + base + (if t < rem { 1 } else { 0 })
			wg.add(1)
			spawn iq4_nl_chunk(data, rows, cols, x, r0, r1 - r0, mut out, mut wg)
		}
		wg.wait()
	}
}

fn iq4_nl_chunk(data []u8, rows int, cols int, x []f32, r0 int, n int, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	iq4_nl_matvec_c_range(data, rows, cols, x, r0, n, mut out)
}

// iq4_nl_matvec_parallel_pool is the persistent-pool form of the IQ4_NL
// vocab head. A nil pool (guest / no pool wired) delegates to the spawn or
// serial entry, so numerics are identical.
pub fn iq4_nl_matvec_parallel_pool(pool &WorkerPool, data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if freestanding {
		iq4_nl_matvec_c(data, rows, cols, x, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 || pool == unsafe { nil } {
			iq4_nl_matvec_parallel(data, rows, cols, x, mut out)
			return
		}
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			iq4_nl_matvec_c(data, rows, cols, x, mut out)
			return
		}
		wg := pool_batch(n)
		for t in 0 .. n {
			r0, r1 := pool_bounds(rows, n, t)
			pool.jobs <- PoolJob{
				kind: PoolJobKind.iq4_nl_range
				data: data
				x:    x
				out:  out
				rows: rows
				cols: cols
				r0:   r0
				n:    r1 - r0
				wg:   wg
			}
		}
		wg.wait()
	}
}

// q2_0_matvec_parallel computes the Q2_0 vocab head row-block split across
// workers (spawn-per-call form). Gating mirrors the other quantized heads.
pub fn q2_0_matvec_parallel(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if freestanding {
		q2_0_matvec_c(data, rows, cols, x, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 {
			q2_0_matvec_c(data, rows, cols, x, mut out)
			return
		}
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			q2_0_matvec_c(data, rows, cols, x, mut out)
			return
		}
		mut wg := sync.new_waitgroup()
		base := rows / n
		rem := rows % n
		for t in 0 .. n {
			r0 := t * base + (if t < rem { t } else { rem })
			r1 := r0 + base + (if t < rem { 1 } else { 0 })
			wg.add(1)
			spawn q2_0_chunk(data, rows, cols, x, r0, r1 - r0, mut out, mut wg)
		}
		wg.wait()
	}
}

fn q2_0_chunk(data []u8, rows int, cols int, x []f32, r0 int, n int, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	q2_0_matvec_c_range(data, rows, cols, x, r0, n, mut out)
}

// q2_0_matvec_parallel_pool is the persistent-pool form of the Q2_0 head.
// A nil pool (guest / no pool wired) delegates to the spawn or serial entry.
pub fn q2_0_matvec_parallel_pool(pool &WorkerPool, data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if freestanding {
		q2_0_matvec_c(data, rows, cols, x, mut out)
		return
	} $else {
		nt := parallel_threads()
		if nt <= 1 || rows < 2 || pool == unsafe { nil } {
			q2_0_matvec_parallel(data, rows, cols, x, mut out)
			return
		}
		mut n := nt
		for n > 1 && rows / n < 512 {
			n--
		}
		if n <= 1 {
			q2_0_matvec_c(data, rows, cols, x, mut out)
			return
		}
		wg := pool_batch(n)
		for t in 0 .. n {
			r0, r1 := pool_bounds(rows, n, t)
			pool.jobs <- PoolJob{
				kind: PoolJobKind.q2_0_range
				data: data
				x:    x
				out:  out
				rows: rows
				cols: cols
				r0:   r0
				n:    r1 - r0
				wg:   wg
			}
		}
		wg.wait()
	}
}
