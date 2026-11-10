module formats

import os

// read_prefix reads at most n bytes (sniffing, never whole blobs).
pub fn read_prefix(path string, n int) ![]u8 {
	mut f := os.open(path)!
	defer {
		f.close()
	}
	mut buf := []u8{len: n}
	mut got := 0
	for got < n {
		mut chunk := []u8{len: n - got}
		r := f.read(mut chunk) or { break }
		if r <= 0 {
			break
		}
		for i in 0 .. r {
			buf[got + i] = chunk[i]
		}
		got += r
	}
	unsafe {
		buf.len = got
	}
	return buf
}

// read_at reads n bytes at absolute offset off (header-only access).
pub fn read_at(path string, off u64, n int) ![]u8 {
	if n <= 0 || n > 256 * 1024 * 1024 {
		return error('refusing read of ${n} bytes')
	}
	mut f := os.open(path)!
	defer {
		f.close()
	}
	f.seek(int(off), .start)!
	mut buf := []u8{len: n}
	mut got := 0
	for got < n {
		mut chunk := []u8{len: n - got}
		r := f.read(mut chunk) or { break }
		if r <= 0 {
			break
		}
		for i in 0 .. r {
			buf[got + i] = chunk[i]
		}
		got += r
	}
	if got != n {
		return error('short read: want ${n}, got ${got}')
	}
	return buf
}
