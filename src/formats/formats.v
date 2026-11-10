module formats

// Unified weight-format layer. New formats land here behind WeightFormat +
// TensorMeta; inference code only sees ModelFile and never touches bytes.
// GGUF parsing lives in `infer` (already tested); this module owns
// detection + SafeTensors and the shared metadata shape.

// WeightFormat enumerates every container the loader accepts.
pub enum WeightFormat as u8 {
	unknown
	gguf
	safetensors
	pytorch_bin
}

pub fn (f WeightFormat) str() string {
	return match f {
		.unknown { 'unknown' }
		.gguf { 'gguf' }
		.safetensors { 'safetensors' }
		.pytorch_bin { 'pytorch_bin' }
	}
}

// DType is the logical element type shared by all containers.
pub enum DType as u8 {
	f32
	f16
	bf16
	u8
	i8
	i16
	i32
	q4_0
	q8_0
	other
}

pub fn dtype_from_str(s string) DType {
	return match s.to_upper() {
		'F32', 'FLOAT32' { .f32 }
		'F16', 'FLOAT16', 'FP16' { .f16 }
		'BF16', 'BFLOAT16' { .bf16 }
		'U8' { .u8 }
		'I8' { .i8 }
		'I16' { .i16 }
		'I32' { .i32 }
		'Q4_0' { .q4_0 }
		'Q8_0' { .q8_0 }
		else { .other }
	}
}

pub fn dtype_nbytes(dt DType) u64 {
	return match dt {
		.f32 { 4 }
		.f16 { 2 }
		.bf16 { 2 }
		.u8 { 1 }
		.i8 { 1 }
		.i16 { 2 }
		.i32 { 4 }
		// Quantized averages handled block-wise by infer/quant; 1 keeps
		// size estimates conservative without leaking GGML layouts here.
		.q4_0 { 1 }
		.q8_0 { 1 }
		.other { 1 }
	}
}

// TensorMeta describes one weight tensor without touching blob bytes.
pub struct TensorMeta {
pub:
	name     string
	dtype    DType
	shape    []u64
	data_off u64 // offset from start of the data section
	nbytes   u64
}

pub fn numel(shape []u64) u64 {
	mut n := u64(1)
	for d in shape {
		n *= d
	}
	return n
}

// ModelFile is the unified view every loader returns.
pub struct ModelFile {
pub:
	path    string
	format  WeightFormat
	tensors []TensorMeta
	meta    map[string]string
}

pub fn (f ModelFile) find(name string) ?TensorMeta {
	for t in f.tensors {
		if t.name == name {
			return t
		}
	}
	return none
}

pub fn (f ModelFile) count() int {
	return f.tensors.len
}

// detect sniffs the container: GGUF magic, SafeTensors length-prefixed
// JSON header, or PyTorch extension fallback. Never reads past 16 bytes.
pub fn detect(path string) !WeightFormat {
	data := read_prefix(path, 16)!
	if data.len >= 4 && data[0] == `G` && data[1] == `G` && data[2] == `U` && data[3] == `F` {
		return .gguf
	}
	if data.len >= 10 {
		mut n := u64(0)
		for i in 0 .. 8 {
			n |= u64(data[i]) << u64(i * 8)
		}
		// SafeTensors: u64 header len, then JSON starting with '{"'.
		if n > 2 && n < 256 * 1024 * 1024 && data[8] == `{` && data[9] == `"` {
			return .safetensors
		}
	}
	lower := path.to_lower()
	if lower.ends_with('.bin') || lower.ends_with('.pt') || lower.ends_with('.pth') {
		return .pytorch_bin
	}
	return .unknown
}

// load_safetensors parses the header only (JSON index, no blob reads).
// Files: [u64 header_len][header JSON][tensor data...].
pub fn load_safetensors(path string) !ModelFile {
	prefix := read_prefix(path, 8)!
	if prefix.len < 8 {
		return error('safetensors: file too small: ${path}')
	}
	mut n := u64(0)
	for i in 0 .. 8 {
		n |= u64(prefix[i]) << u64(i * 8)
	}
	if n < 2 || n > 256 * 1024 * 1024 {
		return error('safetensors: absurd header len ${n}')
	}
	hdr := read_at(path, 8, int(n))!
	tensors, meta := parse_st_header(hdr.bytestr())!
	return ModelFile{
		path:    path
		format:  .safetensors
		tensors: tensors
		meta:    meta
	}
}

// parse_st_header scans the JSON index with a purpose-built scanner:
// entries look like "name":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}.
// A hand scanner keeps this dependency-free for the bare-metal port.
fn parse_st_header(hdr string) !([]TensorMeta, map[string]string) {
	mut tensors := []TensorMeta{}
	mut meta := map[string]string{}
	mut i := 0
	for i < hdr.len {
		// Find next "key" : value where value starts with '{' (tensor) or
		// the __metadata__ object.
		if hdr[i] != `"` {
			i++
			continue
		}
		key, next := scan_quoted(hdr, i)!
		i = next
		i = skip_ws_colon(hdr, i)!
		if i >= hdr.len {
			break
		}
		if hdr[i] != `{` {
			// Scalar metadata value: record raw and skip it.
			val, after := scan_value(hdr, i)!
			meta[key] = val
			i = after
			continue
		}
		// Object: tensor entry or __metadata__.
		obj, after := scan_object(hdr, i)!
		i = after
		if key == '__metadata__' {
			for k, v in split_flat_object(obj) {
				meta[k] = v
			}
			continue
		}
		dtype_s := obj_field(obj, 'dtype') or { continue }
		shape_s := obj_field(obj, 'shape') or { continue }
		offs_s := obj_field(obj, 'data_offsets') or { continue }
		dtype := dtype_from_str(unquote(dtype_s))
		shape := parse_u64_list(shape_s)!
		offs := parse_u64_list(offs_s)!
		if offs.len != 2 || offs[1] < offs[0] {
			return error('safetensors: bad data_offsets for ${key}')
		}
		tensors << TensorMeta{
			name:     key
			dtype:    dtype
			shape:    shape
			data_off: offs[0]
			nbytes:   offs[1] - offs[0]
		}
	}
	return tensors, meta
}

fn scan_quoted(s string, start int) !(string, int) {
	// start at opening quote; returns (unescaped content, index past closing quote).
	mut out := []u8{cap: 32}
	mut i := start + 1
	for i < s.len {
		c := s[i]
		if c == `\\` && i + 1 < s.len {
			out << s[i + 1]
			i += 2
			continue
		}
		if c == `"` {
			return out.bytestr(), i + 1
		}
		out << c
		i++
	}
	return error('safetensors: unterminated string')
}

fn skip_ws_colon(s string, i int) !int {
	mut j := i
	for j < s.len && (s[j] == ` ` || s[j] == `\n` || s[j] == `\r` || s[j] == `\t`) {
		j++
	}
	if j >= s.len || s[j] != `:` {
		return error('safetensors: expected colon')
	}
	j++
	for j < s.len && (s[j] == ` ` || s[j] == `\n` || s[j] == `\r` || s[j] == `\t`) {
		j++
	}
	return j
}

// scan_value captures a scalar (string/number/true/false/null) raw.
fn scan_value(s string, start int) !(string, int) {
	if s[start] == `"` {
		v, next := scan_quoted(s, start)!
		return '"' + v + '"', next
	}
	mut j := start
	for j < s.len && s[j] != `,` && s[j] != `}` && s[j] != `]` {
		j++
	}
	return s[start..j].trim_space(), j
}

// scan_object captures a balanced {...} substring (no nesting expected
// inside tensor entries, but handles one level for __metadata__).
fn scan_object(s string, start int) !(string, int) {
	mut depth := 0
	mut in_str := false
	mut j := start
	for j < s.len {
		c := s[j]
		if in_str {
			if c == `\\` {
				j += 2
				continue
			}
			if c == `"` {
				in_str = false
			}
			j++
			continue
		}
		if c == `"` {
			in_str = true
		} else if c == `{` {
			depth++
		} else if c == `}` {
			depth--
			if depth == 0 {
				return s[start..j + 1], j + 1
			}
		}
		j++
	}
	return error('safetensors: unterminated object')
}

// obj_field extracts the raw value of "field" from a flat {...} object.
fn obj_field(obj string, field string) ?string {
	needle := '"' + field + '"'
	fi := obj.index(needle) or { return none }
	mut j := fi + needle.len
	for j < obj.len && obj[j] != `:` {
		j++
	}
	if j >= obj.len {
		return none
	}
	j++
	for j < obj.len && (obj[j] == ` ` || obj[j] == `\t`) {
		j++
	}
	if obj[j] == `"` {
		v, _ := scan_quoted(obj, j) or { return none }
		return '"' + v + '"'
	}
	mut k := j
	mut depth := 0
	mut in_s := false
	for k < obj.len {
		c := obj[k]
		if in_s {
			if c == `\\` {
				k += 2
				continue
			}
			if c == `"` {
				in_s = false
			}
			k++
			continue
		}
		if c == `"` {
			in_s = true
		} else if c == `[` {
			depth++
		} else if c == `]` {
			depth--
		} else if (c == `,` || c == `}`) && depth == 0 {
			break
		}
		k++
	}
	return obj[j..k].trim_space()
}

fn unquote(s string) string {
	t := s.trim_space()
	if t.len >= 2 && t[0] == `"` && t[t.len - 1] == `"` {
		return t[1..t.len - 1]
	}
	return t
}

fn parse_u64_list(s string) ![]u64 {
	t := s.trim_space()
	if !t.starts_with('[') || !t.ends_with(']') {
		return error('safetensors: expected list, got ${t}')
	}
	inner := t[1..t.len - 1].trim_space()
	if inner == '' {
		return []u64{}
	}
	parts := inner.split(',')
	mut out := []u64{cap: parts.len}
	for p in parts {
		out << p.trim_space().u64()
	}
	return out
}

// split_flat_object splits {"a":"b","c":1} into string pairs (raw values).
fn split_flat_object(obj string) map[string]string {
	mut m := map[string]string{}
	inner := obj.trim_space()
	if inner.len < 2 {
		return m
	}
	body := inner[1..inner.len - 1]
	mut i := 0
	for i < body.len {
		for i < body.len && body[i] != `"` {
			i++
		}
		if i >= body.len {
			break
		}
		k, next := scan_quoted(body, i) or { break }
		i = next
		for i < body.len && body[i] != `:` {
			i++
		}
		if i >= body.len {
			break
		}
		i++
		v, after := scan_value(body, i) or { break }
		m[k] = unquote(v)
		i = after
	}
	return m
}
