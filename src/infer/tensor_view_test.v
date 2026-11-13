module infer

fn test_view_from_slice_ok() {
	data := [f32(1), 2, 3, 4, 5, 6]
	v := view_from_slice(data, [2, 3])!
	assert v.rank() == 2
	assert v.numel() == 6
	assert v.is_row_major_contiguous()
	assert v.get([1, 2])! == 6.0
}

fn test_view_rejects_bad_shapes() {
	data := [f32(1), 2, 3]
	if _ := view_from_slice(data, [2, 2]) {
		assert false, 'expected numel mismatch'
	} else {
		assert true
	}
	if _ := view_from_slice(data, [0, 3]) {
		assert false, 'expected bad dim'
	} else {
		assert true
	}
	if _ := view_from_slice(data, []int{}) {
		assert false, 'expected empty shape'
	} else {
		assert true
	}
}

fn test_view_reshape_with_infer() {
	data := [f32(1), 2, 3, 4, 5, 6]
	v := view_from_slice(data, [6])!
	r := v.reshape([2, -1])!
	assert r.shape == [2, 3]
	assert r.get([1, 0])! == 4.0
	// Round trip back to flat.
	f := r.reshape([6])!
	assert f.get([5])! == 6.0
}

fn test_matvec_view_matches_matvec() {
	mut w := []f32{len: 4 * 6}
	mut x := []f32{len: 6}
	for i in 0 .. w.len {
		w[i] = f32(i % 9) * 0.25 - 1.0
	}
	for i in 0 .. x.len {
		x[i] = f32(i % 5) * 0.5
	}
	wv := view_from_slice(w, [4, 6])!
	mut a := []f32{len: 4}
	mut b := []f32{len: 4}
	matvec(w, x, 4, 6, mut a)
	matvec_view(wv, x, mut b)!
	for i in 0 .. 4 {
		d := b[i] - a[i]
		assert d < 1e-4 && d > -1e-4
	}
}

fn test_matvec_view_rejects_strided() {
	data := [f32(1), 2, 3, 4]
	// Hand-built strided view (column-major): kernels need row-major.
	wv := TensorView{
		data:    data
		shape:   [2, 2]
		strides: [1, 2]
	}
	assert !wv.is_row_major_contiguous()
	mut out := []f32{len: 2}
	if _ := matvec_view(wv, [f32(1), 1], mut out) {
		assert false, 'expected layout rejection'
	} else {
		assert true
	}
}
