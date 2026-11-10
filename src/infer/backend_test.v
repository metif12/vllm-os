module infer

fn test_backend_registry_lists_both() {
	names := list_backends()
	assert 'cpu' in names
	assert 'cpu-blocked' in names
}

fn test_select_backend_explicit() {
	b := select_backend('cpu-blocked', 4, 4)
	assert b.name() == 'cpu-blocked'
}

fn test_select_backend_auto_threshold() {
	small := select_backend('auto', 64, 64)
	assert small.name() == 'cpu'
	wide := select_backend('auto', 2048, 6144)
	assert wide.name() == 'cpu-blocked'
}

fn test_backends_agree_on_matvec() {
	w := [f32(1), 2, 3, 4, 5, 6]
	x := [f32(1), 0.5, -1.0]
	mut a := []f32{len: 2}
	mut b := []f32{len: 2}
	select_backend('cpu', 2, 3).matvec(w, x, 2, 3, mut a)
	select_backend('cpu-blocked', 2, 3).matvec(w, x, 2, 3, mut b)
	assert a == b
}
