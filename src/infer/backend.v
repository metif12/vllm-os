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

pub fn (b CpuNaiveBackend) name() string {
	return 'cpu'
}

pub fn (b CpuNaiveBackend) matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	matvec(w, x, rows, cols, mut out)
}

pub struct CpuBlockedBackend {}

pub fn (b CpuBlockedBackend) name() string {
	return 'cpu-blocked'
}

pub fn (b CpuBlockedBackend) matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	matvec_blocked(w, x, rows, cols, mut out)
}

// list_backends is the registry: add new backends here and in select_backend.
pub fn list_backends() []string {
	return ['cpu', 'cpu-blocked']
}

// select_backend picks by name, or 'auto' chooses the tiled kernel once the
// matrix is wide enough for tiling to pay off (vocab/FFN projections).
pub fn select_backend(name string, rows int, cols int) Backend {
	if name == 'cpu' {
		return CpuNaiveBackend{}
	}
	if name == 'cpu-blocked' {
		return CpuBlockedBackend{}
	}
	// auto
	if u64(rows) * u64(cols) >= 1 << 20 {
		return CpuBlockedBackend{}
	}
	return CpuNaiveBackend{}
}
