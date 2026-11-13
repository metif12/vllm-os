module infer

// Backend abstraction: the forward pass (phase 1b) calls these kernels
// instead of free functions, so SIMD/GPU ports slot in without touching
// model or server code. Selection lives in select_backend; the registry
// is the single place new backends are added.

pub interface Backend {
	name() string
	// matvec computes out = W*x, W row-major [rows x cols].
	matvec(w []f32, x []f32, rows int, cols int, mut out []f32)
}

pub struct CpuNaiveBackend {}

// name reports the registry key for this backend.
pub fn (b CpuNaiveBackend) name() string {
	return 'cpu'
}

// matvec computes out = W*x via the unrolled pure-V kernel.
pub fn (b CpuNaiveBackend) matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	matvec(w, x, rows, cols, mut out)
}

pub struct CpuBlockedBackend {}

// name reports the registry key for this backend.
pub fn (b CpuBlockedBackend) name() string {
	return 'cpu-blocked'
}

// matvec computes out = W*x via the cache-tiled pure-V kernel.
pub fn (b CpuBlockedBackend) matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	matvec_blocked(w, x, rows, cols, mut out)
}

// CpuSimdBackend routes matvec through the C AVX2+FMA kernel, falling
// back to pure V per call when the binary lacks the C layer (non-prod),
// the CPU lacks AVX2, or VLLM_NO_SIMD=1. Host-only speed path: the
// bare-metal port selects cpu/cpu-blocked explicitly.
pub struct CpuSimdBackend {}

// CpuParallelBackend splits rows across worker threads (each running
// the dispatched SIMD-or-V kernel on its chunk). Host-only throughput
// path: under -freestanding it runs the serial kernel. Spawns per call
// (no pool yet); the auto backend only selects it where spawn vanishes
// next to streaming time. Validate with VLLM_THREADS=1 to force serial.
pub struct CpuParallelBackend {}

// name reports the registry key for this backend.
pub fn (b CpuParallelBackend) name() string {
	return 'cpu-parallel'
}

// matvec computes out = W*x via row-block partitioned workers.
pub fn (b CpuParallelBackend) matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	matvec_parallel(w, x, rows, cols, mut out)
}

// name reports the registry key for this backend.
pub fn (b CpuSimdBackend) name() string {
	return 'cpu-simd'
}

// matvec computes out = W*x via the runtime-dispatched C kernel.
pub fn (b CpuSimdBackend) matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	matvec_c(w, x, rows, cols, mut out)
}

// list_backends is the registry: add new backends here and in select_backend.
pub fn list_backends() []string {
	return ['cpu', 'cpu-blocked', 'cpu-simd', 'cpu-parallel']
}

// select_backend picks by name, or 'auto' chooses the fastest available
// kernel for the shape: threads + SIMD for big layers on host builds,
// SIMD (when has_avx2()) for wide single layers, blocked for mid-size,
// naive for the rest.
pub fn select_backend(name string, rows int, cols int) Backend {
	if name == 'cpu' {
		return CpuNaiveBackend{}
	}
	if name == 'cpu-blocked' {
		return CpuBlockedBackend{}
	}
	if name == 'cpu-simd' {
		return CpuSimdBackend{}
	}
	if name == 'cpu-parallel' {
		return CpuParallelBackend{}
	}
	// auto (and unknown names, documented fallback)
	$if freestanding {
		if u64(rows) * u64(cols) >= 1 << 20 {
			return CpuBlockedBackend{}
		}
		return CpuNaiveBackend{}
	} $else {
		if rows >= min_rows_parallel && cols >= min_cols_parallel && parallel_threads() > 1 {
			return CpuParallelBackend{}
		}
		if has_avx2() && u64(rows) * u64(cols) >= 1 << 20 {
			return CpuSimdBackend{}
		}
		if u64(rows) * u64(cols) >= 1 << 20 {
			return CpuBlockedBackend{}
		}
		return CpuNaiveBackend{}
	}
}
