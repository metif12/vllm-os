module infer

// GGUF container parser: header + KV store + tensor table.
// Host + bare-metal compatible. Little-endian only.
// Spec: magic 'GGUF'(u32) version(u32) n_tensors(u64) n_kv(u64),
// then KV pairs, then tensor infos, then aligned tensor data.

pub const gguf_magic = u32(0x46554747) // 'GGUF' little-endian
pub const gguf_version_supported = u32(3)
pub const gguf_default_align = u64(32)

pub enum GgufType as u32 {
	u8
	i8
	u16
	i16
	u32
	i32
	f32
	boolean
	string
	array
	u64
	i64
	f64
}

// GGML quant block types relevant for Qwen3 quants.
pub enum GgmlType as u32 {
	f32
	f16
	q4_0
	q4_1
	q5_0
	q5_1
	q8_0
	q8_1
	q2_k
	q3_k
	q4_k
	q5_k
	q6_k
	q8_k
	i8
	i16
	i32
	i64
	f64
	iq2_xxs
	iq2_xs
	iq3_xxs
	iq3_s
	iq2_s
	iq4_nl
	iq4_xs
	iq8_bb
	sparse
}

pub struct GgufMeta {
pub:
	version   u32
	n_tensors u64
	n_kv      u64
	alignment u64 = gguf_default_align
}

pub struct GgufTensorInfo {
pub:
	name   string
	typ    GgmlType
	shape  []u64
	offset u64 // offset from start of tensor data section
	size   u64 // byte size of this tensor's data
}

pub struct GgufModel {
pub:
	path    string
	meta    GgufMeta
	tensors []GgufTensorInfo
	kv      map[string]string // stringified KV (numbers rendered as text)
}

struct Reader {
mut:
	buf []u8
	off int
}

fn (mut r Reader) need(n int) ! {
	if r.off + n > r.buf.len {
		return error('gguf: truncated file at offset ${r.off}+${n}')
	}
}

fn (mut r Reader) u8() !u8 {
	r.need(1)!
	v := r.buf[r.off]
	r.off++
	return v
}

fn (mut r Reader) u16_le() !u16 {
	r.need(2)!
	v := u16(r.buf[r.off]) | (u16(r.buf[r.off + 1]) << 8)
	r.off += 2
	return v
}

fn (mut r Reader) u32_le() !u32 {
	r.need(4)!
	v := u32(r.buf[r.off]) | (u32(r.buf[r.off + 1]) << 8) | (u32(r.buf[r.off + 2]) << 16) | (u32(r.buf[r.off + 3]) << 24)
	r.off += 4
	return v
}

fn (mut r Reader) u64_le() !u64 {
	r.need(8)!
	mut v := u64(0)
	for i in 0 .. 8 {
		v |= u64(r.buf[r.off + i]) << u64(i * 8)
	}
	r.off += 8
	return v
}

fn (mut r Reader) i8_() !i8 {
	return i8(r.u8()!)
}

fn (mut r Reader) i16_le() !i16 {
	return i16(r.u16_le()!)
}

fn (mut r Reader) i32_le() !i32 {
	return i32(r.u32_le()!)
}

fn (mut r Reader) i64_le() !i64 {
	return i64(r.u64_le()!)
}

fn (mut r Reader) f32_le() !f32 {
	bits := r.u32_le()!
	return *unsafe { &f32(&bits) }
}

fn (mut r Reader) f64_le() !f64 {
	bits := r.u64_le()!
	return *unsafe { &f64(&bits) }
}

fn (mut r Reader) gguf_str() !string {
	n := r.u64_le()!
	if n > u64(1 << 26) {
		return error('gguf: absurd string len ${n}')
	}
	ni := int(n)
	r.need(ni)!
	s := r.buf[r.off..r.off + ni].bytestr()
	r.off += ni
	return s
}

// Header-only probe (cheap, used by boot path before full parse).
// load_gguf_header reads the 24-byte prefix: magic, version, counts.
pub fn load_gguf_header(path string) !GgufModel {
	data := read_file_bytes(path, 4096)!
	if data.len < 24 {
		return error('gguf: file too small: ${path}')
	}
	magic := u32(data[0]) | (u32(data[1]) << 8) | (u32(data[2]) << 16) | (u32(data[3]) << 24)
	if magic != gguf_magic {
		return error('gguf: bad magic, not a GGUF file: ${path}')
	}
	version := u32(data[4]) | (u32(data[5]) << 8) | (u32(data[6]) << 16) | (u32(data[7]) << 24)
	mut n_tensors := u64(0)
	mut n_kv := u64(0)
	for i in 0 .. 8 {
		n_tensors |= u64(data[8 + i]) << u64(i * 8)
		n_kv |= u64(data[16 + i]) << u64(i * 8)
	}
	return GgufModel{
		path: path
		meta: GgufMeta{
			version:   version
			n_tensors: n_tensors
			n_kv:      n_kv
		}
	}
}

// Full parse of KV + tensor table. Reads the whole prefix up to the end of
// the tensor table (NOT the weight blobs). For a 1.7B model this prefix is
// small (tens of KB); blobs stay on disk / get mmap'd by the kernel port.
// load_gguf_table is the boot-time index loader.
pub fn load_gguf_table(path string) !GgufModel {
	// Read enough to cover header + KV + tensor infos. Real files carry
	// large KV arrays (Qwen3 tokenizers: 152k strings, ~2-8 MB), so the
	// window is 32 MB; absurd length fields are still rejected below, so
	// a corrupt file cannot OOM the host.
	data := read_file_bytes(path, 32 * 1024 * 1024)!
	mut r := Reader{
		buf: data
	}
	magic := r.u32_le()!
	if magic != gguf_magic {
		return error('gguf: bad magic, not a GGUF file: ${path}')
	}
	version := r.u32_le()!
	if version != gguf_version_supported {
		return error('gguf: unsupported version ${version}, want ${gguf_version_supported}')
	}
	n_tensors := r.u64_le()!
	n_kv := r.u64_le()!
	if n_tensors > 2048 || n_kv > 4096 {
		return error('gguf: absurd counts tensors=${n_tensors} kv=${n_kv}')
	}
	mut kv := map[string]string{}
	mut alignment := gguf_default_align
	for _ in 0 .. int(n_kv) {
		key := r.gguf_str()!
		typ := r.u32_le()!
		val := read_kv_value(mut r, typ)!
		if key == 'general.alignment' {
			alignment = val.u64()
		}
		kv[key] = val
	}
	mut tensors := []GgufTensorInfo{cap: int(n_tensors)}
	for _ in 0 .. int(n_tensors) {
		name := r.gguf_str()!
		ndims := r.u32_le()!
		if ndims > 4 {
			return error('gguf: tensor ${name} has absurd ndims=${ndims}')
		}
		mut shape := []u64{cap: int(ndims)}
		for _ in 0 .. int(ndims) {
			shape << r.u64_le()!
		}
		typ := unsafe { GgmlType(r.u32_le()!) }
		offset := r.u64_le()!
		size := ggml_tensor_bytes(typ, numel(shape))
		tensors << GgufTensorInfo{
			name:   name
			typ:    typ
			shape:  shape
			offset: offset
			size:   size
		}
	}
	return GgufModel{
		path:    path
		meta:    GgufMeta{
			version:   version
			n_tensors: n_tensors
			n_kv:      n_kv
			alignment: alignment
		}
		tensors: tensors
		kv:      kv
	}
}

fn read_kv_value(mut r Reader, typ u32) !string {
	et_outer := unsafe { GgufType(typ) }
	match et_outer {
		.u8 { return r.u8()!.str() }
		.i8 { return r.i8_()!.str() }
		.u16 { return r.u16_le()!.str() }
		.i16 { return r.i16_le()!.str() }
		.u32 { return r.u32_le()!.str() }
		.i32 { return r.i32_le()!.str() }
		.f32 { return r.f32_le()!.str() }
		.boolean { return (r.u8()! != 0).str() }
		.string { return r.gguf_str()! }
		.array {
			et := r.u32_le()!
			n := r.u64_le()!
			if n > 4 << 20 {
				return error('gguf: absurd array len ${n}')
			}
			if n > 4096 {
				// Large arrays (tokenizer tokens/merges: 50-150k strings)
				// are walked but not retained: the boot path only needs
				// arch KV (tokenizer comes from the sidecar tokenizer.json).
				// Offsets must still advance, so each element is parsed
				// and discarded. Loading tokens from GGUF is follow-up.
				for _ in 0 .. int(n) {
					read_kv_value(mut r, et)!
				}
				return 'array[${n}]'
			}
			mut parts := []string{cap: int(n)}
			for _ in 0 .. int(n) {
				parts << read_kv_value(mut r, et)!
			}
			return '[' + parts.join(',') + ']'
		}
		.u64 { return r.u64_le()!.str() }
		.i64 { return r.i64_le()!.str() }
		.f64 { return r.f64_le()!.str() }
	}
}

fn numel(shape []u64) u64 {
	mut n := u64(1)
	for d in shape {
		n *= d
	}
	return n
}

// Byte size of one element for unquantized types; for quantized types this
// returns the per-element average (block_size / block_bytes inverted) as a
// fraction-safe integer division used only for size estimates. Exact block
// math lives in quant.v.
fn ggml_type_size(t GgmlType) u64 {
	match t {
		.f32 { return 4 }
		.f16 { return 2 }
		.q4_0, .q4_1 { return 1 } // ~0.5B/elem effective; table size fixed up below
		.q5_0, .q5_1 { return 1 }
		.q8_0, .q8_1 { return 1 }
		.q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k { return 1 }
		.i8 { return 1 }
		.i16 { return 2 }
		.i32 { return 4 }
		.i64 { return 8 }
		.f64 { return 8 }
		else { return 1 }
	}
}

// ggml_tensor_bytes is the exact byte size for fixed-block quant types
// (Q4_0: 2 scale + 16 nibble bytes per 32 elems, etc.); unquantized types
// fall back to ggml_type_size * n. K-quants use super-blocks and stay on
// the estimate path (follow-up when a K-quant weight ships in v1).
fn ggml_tensor_bytes(t GgmlType, n u64) u64 {
	if t == .q4_0 || t == .q4_1 || t == .q5_0 || t == .q5_1 || t == .q8_0 || t == .q8_1 {
		blocks := (n + 31) / 32
		per := match t {
			.q4_0 { 18 }
			.q4_1 { 20 }
			.q5_0 { 22 }
			.q5_1 { 24 }
			else { 34 }
		}
		return blocks * per
	}
	return ggml_type_size(t) * n
}

// find returns the tensor info by name or none.
pub fn (m GgufModel) find(name string) ?GgufTensorInfo {
	for t in m.tensors {
		if t.name == name {
			return t
		}
	}
	return none
}
