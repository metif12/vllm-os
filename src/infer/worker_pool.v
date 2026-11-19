module infer

import sync

// Persistent worker pool for the decode hot path. The phase-1b forward
// previously spawned 2-4 OS threads per fork-join (~57 fork-joins per token
// on the 28-layer shape) — a spawn storm that shows up next to sub-ms
// quantized kernels. Here a fixed set of long-lived threads drains a shared
// job channel; each batch carries its own WaitGroup so concurrent requests
// can interleave jobs without cross-batch barriers.
//
// Contract (kept deliberately small for the pinned V 0.5.2):
//   - workers are spawned once at boot and never joined (process lifetime);
//   - a job is a value struct: slices share the caller's backing arrays,
//     out ranges are disjoint per worker, so no data locks are needed;
//   - the submitter wg.add(N), workers wg.done(), submitter wg.wait();
//   - workers never submit nested jobs, so a full queue cannot deadlock;
//   - -freestanding and VLLM_THREADS=1 produce no pool (serial fallback),
//     keeping the bare-metal guest single-threaded.
pub struct WorkerPool {
mut:
	jobs chan PoolJob
	n    int
}

enum PoolJobKind {
	matvec_range // f32 row-block: matvec_c_range
	q6k_range    // Q6_K row-block: q6_k_matvec_c_range
	iq4_nl_range // IQ4_NL row-block: iq4_nl_matvec_c_range
	q4_0_full    // whole Q4_0 matrix: q4_0_matvec_c
}

struct PoolJob {
	kind PoolJobKind
	// f32 path
	words []f32
	// quantized paths
	data []u8
	// shared
	x    []f32
	rows int
	cols int
	r0   int
	n    int
	wg   &sync.WaitGroup = unsafe { nil }
mut:
	// out is mutable so workers can pass it to kernels; the backing array is
	// shared with the caller and each job owns a disjoint output range.
	out []f32
}

const pool_queue_cap = 4096

// new_worker_pool spawns n long-lived workers and returns the pool. It
// returns nil under -freestanding (no pthread-style scheduler in the guest)
// and when n <= 1 (VLLM_THREADS=1 pins the serial kernels). Callers treat a
// nil pool as "use the spawn/serial fallback" so behaviour is unchanged.
pub fn new_worker_pool(n int) &WorkerPool {
	$if freestanding {
		return unsafe { nil }
	} $else {
		if n <= 1 {
			return unsafe { nil }
		}
		mut p := &WorkerPool{
			jobs: chan PoolJob{cap: pool_queue_cap}
			n:    n
		}
		for i in 0 .. n {
			spawn pool_worker(p, i)
		}
		return p
	}
}

// pool_worker is the worker loop: take a job, run the dispatched kernel on
// its (possibly row-ranged) chunk, signal completion. It never exits.
fn pool_worker(p &WorkerPool, id int) {
	_ = id
	for {
		mut job := <-p.jobs
		match job.kind {
			.matvec_range {
				matvec_c_range(job.words, job.x, job.rows, job.cols, job.r0, job.n, mut job.out)
			}
			.q6k_range {
				q6_k_matvec_c_range(job.data, job.rows, job.cols, job.x, job.r0, job.n, mut job.out)
			}
			.iq4_nl_range {
				iq4_nl_matvec_c_range(job.data, job.rows, job.cols, job.x, job.r0, job.n, mut
					job.out)
			}
			.q4_0_full {
				q4_0_matvec_c(job.data, job.rows, job.cols, job.x, mut job.out)
			}
		}

		job.wg.done()
	}
}

// pool_batch returns a WaitGroup pre-armed for n jobs in a batch.
fn pool_batch(n int) &sync.WaitGroup {
	wg := sync.new_waitgroup()
	wg.add(n)
	return wg
}

// pool_bounds computes the balanced row range for worker t of n, spreading
// the rows % n remainder so block sizes differ by at most one (same
// partition as the spawn-based parallel.v paths).
fn pool_bounds(rows int, n int, t int) (int, int) {
	base := rows / n
	rem := rows % n
	r0 := t * base + (if t < rem { t } else { rem })
	r1 := r0 + base + (if t < rem { 1 } else { 0 })
	return r0, r1
}
