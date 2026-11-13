module infer

// VTL/VSL parity: the hand-rolled kernels must agree with the reference
// libraries element-for-element (within f32 tolerance).
//
// Needs: `v install vtl vsl` (host-only dev dependency; the server binary
// and the bare-metal port never link VTL/VSL — see tensor_view.v for why
// the token loop stays pure V + C).
// Run: `v -cc gcc test src/infer` (any profile; C layer optional).

import vtl
import vtl.la
import vsl.la as vsl_la

fn parity_data() ([]f32, []f32, int, int) {
	rows := 8
	cols := 16
	mut w := []f32{len: rows * cols}
	mut x := []f32{len: cols}
	for i in 0 .. w.len {
		w[i] = f32(i % 13) * 0.125 - 0.75
	}
	for i in 0 .. x.len {
		x[i] = f32(i % 7) * 0.25 - 0.5
	}
	return w, x, rows, cols
}

fn test_vtl_tensor_matmul_matches_matvec() {
	w, x, rows, cols := parity_data()
	mut expect := []f32{len: rows}
	matvec(w, x, rows, cols, mut expect)
	wt := vtl.from_array(w, [rows, cols])!
	xt := vtl.from_array(x, [cols, 1])!
	ct := la.matmul(wt, xt)!
	got := ct.to_array()
	assert got.len == rows
	for i in 0 .. rows {
		d := got[i] - expect[i]
		assert d < 1e-4 && d > -1e-4
	}
}

fn test_vsl_blas_gemm_matches_matvec() {
	w, x, rows, cols := parity_data()
	mut expect := []f32{len: rows}
	matvec(w, x, rows, cols, mut expect)
	mut wf := []f64{len: w.len}
	mut xf := []f64{len: x.len}
	for i in 0 .. w.len {
		wf[i] = w[i]
	}
	for i in 0 .. x.len {
		xf[i] = x[i]
	}
	mam := vsl_la.Matrix.raw(rows, cols, wf)
	mbm := vsl_la.Matrix.raw(cols, 1, xf)
	mut dm := vsl_la.Matrix.new[f64](rows, 1)
	vsl_la.matrix_matrix_mul(mut dm, 1.0, mam, mbm)
	for i in 0 .. rows {
		d := f32(dm.get(i, 0)) - expect[i]
		assert d < 1e-3 && d > -1e-3
	}
}

fn test_tensor_view_matches_vtl_tensor() {
	w, _, rows, cols := parity_data()
	v := view_from_slice(w, [rows, cols])!
	wt := vtl.from_array(w, [rows, cols])!
	// Same buffer, same layout: flat contents must be identical.
	flat := wt.to_array()
	assert flat.len == v.numel()
	for i, val in flat {
		assert val == v.data[i]
	}
	// Reshape parity: VTL reshape vs view reshape on the same buffer.
	vr := v.reshape([4, 32])!
	tr := wt.reshape([4, 32])!
	tflat := tr.to_array()
	assert tflat.len == vr.numel()
	for i, val in tflat {
		assert val == vr.data[i]
	}
}
