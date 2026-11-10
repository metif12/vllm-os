module infer

// GGUF container parser (host + bare-metal compatible, no GC tricks here).
// Spec: magic 'GGUF', u32 version, u64 n_tensors, u64 n_kv.

pub const gguf_magic = u32(0x46554747) // 'GGUF' little-endian

pub struct GgufMeta {
pub:
	version   u32
	n_tensors u64
	n_kv      u64
}

pub struct GgufModel {
pub:
	path string
	meta GgufMeta
	size int
}

// Parse just the 24-byte header. Full tensor table = phase 2.
pub fn load_gguf_header(path string) !GgufModel {
	data := read_file_bytes(path, 32)!
	if data.len < 24 {
		return error('gguf: file too small: ${path}')
	}
	magic := u32(data[0]) | (u32(data[1]) << 8) | (u32(data[2]) << 16) | (u32(data[3]) << 24)
	if magic != gguf_magic {
		return error('gguf: bad magic, not a GGUF file: ${path}')
	}
	version := u32(data[4]) | (u32(data[5]) << 8) | (u32(data[6]) << 16) | (u32(data[7]) << 24)
	n_tensors := read_u64_le(data, 8)
	n_kv := read_u64_le(data, 16)
	return GgufModel{
		path: path
		size: data.len
		meta: GgufMeta{
			version:   version
			n_tensors: n_tensors
			n_kv:      n_kv
		}
	}
}

fn read_u64_le(b []u8, off int) u64 {
	mut v := u64(0)
	for i in 0 .. 8 {
		v |= u64(b[off + i]) << u64(i * 8)
	}
	return v
}

// read_file_bytes reads at most `cap` bytes (header probe, not full mmap).
fn read_file_bytes(path string, cap int) ![]u8 {
	mut f := open_file_rb(path)!
	defer {
		f.close()
	}
	return f.read_bytes(cap)
}
