module infer

import os

// Minimal C file reader: header probe without pulling a 1GB GGUF into RAM.
// Bare-metal friendly (no `os` dependency in this unit).

#include <stdio.h>

struct C.FILE {}

fn C.fopen(path &char, mode &char) &C.FILE
fn C.fread(ptr voidptr, size uint, nmemb uint, stream &C.FILE) uint
fn C.fclose(stream &C.FILE) int

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
	got := C.fread(buf.data, uint(1), uint(n), r.f)
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
