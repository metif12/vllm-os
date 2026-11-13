module infer

fn test_backend_registry() {
	names := list_backends()
	assert 'cpu' in names
	assert 'cpu-blocked' in names
	assert 'cpu-simd' in names
	assert 'cpu-parallel' in names
}

fn test_select_backend_explicit() {
	b := select_backend('cpu-blocked', 4, 4)
	assert b.name() == 'cpu-blocked'
	s := select_backend('cpu-simd', 4, 4)
	assert s.name() == 'cpu-simd'
	p := select_backend('cpu-parallel', 4, 4)
	assert p.name() == 'cpu-parallel'
}

fn test_select_backend_auto_threshold() {
	small := select_backend('auto', 64, 64)
	assert small.name() == 'cpu'
	// Wide layer: parallel on host builds with threads enabled, else simd
	// when the binary has the C layer and the CPU has AVX2, else blocked.
	// All three are correct answers — the test pins the rule, not the
	// machine (dev builds without -prod always say cpu-blocked here).
	wide := select_backend('auto', 2048, 6144)
	assert wide.name() in ['cpu-parallel', 'cpu-simd', 'cpu-blocked']
}

fn test_backends_agree_on_matvec() {
	w := [f32(1), 2, 3, 4, 5, 6]
	x := [f32(1), 0.5, -1.0]
	mut a := []f32{len: 2}
	mut b := []f32{len: 2}
	select_backend('cpu', 2, 3).matvec(w, x, 2, 3, mut a)
	select_backend('cpu-blocked', 2, 3).matvec(w, x, 2, 3, mut b)
	assert a == b
	mut c := []f32{len: 2}
	select_backend('cpu-simd', 2, 3).matvec(w, x, 2, 3, mut c)
	for i in 0 .. 2 {
		d := c[i] - a[i]
		assert d < 1e-5 && d > -1e-5
	}
	mut d_ := []f32{len: 2}
	select_backend('cpu-parallel', 2, 3).matvec(w, x, 2, 3, mut d_)
	for i in 0 .. 2 {
		dd := d_[i] - a[i]
		assert dd < 1e-5 && dd > -1e-5
	}
}
