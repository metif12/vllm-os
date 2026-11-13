module infer

// TensorView is the VTL-aligned, zero-copy tensor view over caller-owned
// f32 buffers. Pure V, no allocation, no external dependency: the
// bare-metal-safe subset of the VTL Tensor API (shape / strides /
// row-major reshape / matvec) that the forward pass needs.
//
// Why not import VTL here: VTL Tensors are GC-managed and VSL-backed,
// which violates the token loop's `-gc none`, zero-alloc contract.
// Instead this view mirrors VTL naming and semantics, and the host-only
// `tools/vtl_parity.v` checks every op below against real VTL Tensors
// element-for-element. Swapping a host reference path to VTL later
// needs no signature changes on this side.
//
// Contract: views never own data. The caller keeps the backing []f32
// alive; all bounds are validated once at view creation, never per
// element (same rule as the hot kernels).

pub struct TensorView {
pub:
	data    []f32
	shape   []int
	strides []int
}

// view_from_slice builds a row-major view over data with the given shape.
// Returns an error when shape is empty, non-positive, or mismatches numel.
pub fn view_from_slice(data []f32, shape []int) !TensorView {
	if shape.len == 0 {
		return error('view: empty shape')
	}
	mut n := 1
	for d in shape {
		if d <= 0 {
			return error('view: bad dim ${d}')
		}
		n *= d
	}
	if n != data.len {
		return error('view: shape ${shape} needs ${n} elems, got ${data.len}')
	}
	mut strides := []int{len: shape.len}
	mut s := 1
	for i := shape.len - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return TensorView{
		data:    data
		shape:   shape.clone()
		strides: strides
	}
}

// rank returns the number of dimensions.
pub fn (v TensorView) rank() int {
	return v.shape.len
}

// numel returns the total element count.
pub fn (v TensorView) numel() int {
	mut n := 1
	for d in v.shape {
		n *= d
	}
	return n
}

// is_row_major_contiguous reports C-order dense layout (the only layout
// the kernels accept; GGUF weight rows are always dense).
pub fn (v TensorView) is_row_major_contiguous() bool {
	mut s := 1
	for i := v.shape.len - 1; i >= 0; i-- {
		if v.strides[i] != s {
			return false
		}
		s *= v.shape[i]
	}
	return true
}

// offset flattens a multi-index to a data offset (bounds-checked).
pub fn (v TensorView) offset(idx []int) !int {
	if idx.len != v.shape.len {
		return error('view: rank mismatch ${idx.len} vs ${v.shape.len}')
	}
	mut off := 0
	for i, ix in idx {
		if ix < 0 || ix >= v.shape[i] {
			return error('view: index ${ix} out of bounds for dim ${v.shape[i]}')
		}
		off += ix * v.strides[i]
	}
	return off
}

// get reads one element by multi-index.
pub fn (v TensorView) get(idx []int) !f32 {
	return v.data[v.offset(idx)!]
}

// reshape returns a new view over the same data with a new shape.
// -1 infers one dimension (VTL/NumPy convention); total must match.
pub fn (v TensorView) reshape(new_shape []int) !TensorView {
	mut shape := new_shape.clone()
	mut known := 1
	mut infer := -1
	for i, d in shape {
		if d == -1 {
			if infer >= 0 {
				return error('view: only one -1 allowed')
			}
			infer = i
		} else if d <= 0 {
			return error('view: bad dim ${d}')
		} else {
			known *= d
		}
	}
	if infer >= 0 {
		if v.numel() % known != 0 {
			return error('view: cannot infer -1 for ${shape}')
		}
		shape[infer] = v.numel() / known
	}
	return view_from_slice(v.data, shape)
}

// matvec_view computes out = W*x where W is a rank-2 row-major view.
// Dispatches to the same kernel the backends use (SIMD when available).
pub fn matvec_view(w TensorView, x []f32, mut out []f32) ! {
	if w.rank() != 2 {
		return error('view matvec: want rank-2 weight, got ${w.rank()}')
	}
	if !w.is_row_major_contiguous() {
		return error('view matvec: weight must be row-major contiguous')
	}
	rows := w.shape[0]
	cols := w.shape[1]
	if x.len != cols {
		return error('view matvec: x len ${x.len} vs cols ${cols}')
	}
	if out.len != rows {
		return error('view matvec: out len ${out.len} vs rows ${rows}')
	}
	matvec_c(w.data, x, rows, cols, mut out)
}
