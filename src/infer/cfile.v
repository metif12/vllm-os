module infer

import os

// Minimal C file reader: header probe without pulling a 1GB GGUF into RAM.
// Bare-metal friendly (no `os` dependency in this unit).
//
// NOTE: the C symbols used here (FILE/fopen/fread/fclose) are declared in
// vlib's builtin/cfns, so this file declares none of its own. A local
// re-declaration once shadowed vlib's `fread` on some toolchains and broke
// vlib/os itself (`infer.uint` mismatch) — do not re-add them.

#include <stdio.h>

struct RawFile {
	f &C.FILE
}

fn open_file_rb(path string) !RawFile {
	f := C.fopen(path.str, c'rb')
	if f == unsafe { nil } {
		return error('cannot open file: ${path}')
	}
	return RawFile{
		f: f
	}
}

fn (mut r RawFile) read_bytes(n int) []u8 {
	mut buf := []u8{len: n}
	// Same call shape as vlib/os: int literal + int length, converted by
	// each toolchain against its own builtin/cfns declaration.
	got := C.fread(buf.data, 1, n, r.f)
	unsafe {
		buf.len = int(got)
	}
	return buf
}

fn (mut r RawFile) close() {
	C.fclose(r.f)
}

// read_file_bytes reads at most `cap_` bytes (header/table probe, not full mmap).
fn read_file_bytes(path string, cap_ int) ![]u8 {
	mut f := open_file_rb(path)!
	defer {
		f.close()
	}
	return f.read_bytes(cap_)
}

// read_whole_file is for SMALL sidecar files only (tokenizer.json, config).
// Never use it for *.gguf blobs.
fn read_whole_file(path string) ![]u8 {
	return os.read_bytes(path)!
}
