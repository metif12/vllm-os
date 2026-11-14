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

// matvec_chunk runs the dispatched kernel on one row block [r0, r0+n).
// Whole slices cross the spawn boundary (as in the bench harness);
// offsets resolve inside via matvec_c_range.
fn matvec_chunk(w []f32, x []f32, rows int, cols int, r0 int, n int, mut out []f32, mut wg sync.WaitGroup) {
	defer {
		wg.done()
	}
	matvec_c_range(w, x, rows, cols, r0, n, mut out)
}
