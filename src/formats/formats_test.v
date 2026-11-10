module formats

import os

fn write_tmp(name string, data []u8) ! {
	os.write_bytes(name, data)!
}

fn test_detect_gguf_magic() ! {
	// Minimal GGUF header: magic + version(3) + 0 tensors + 0 kv.
	mut hdr := []u8{len: 24}
	hdr[0] = `G`
	hdr[1] = `G`
	hdr[2] = `U`
	hdr[3] = `F`
	hdr[4] = 3
	write_tmp('st_test_gguf.bin', hdr)!
	defer {
		os.rm('st_test_gguf.bin') or {}
	}
	assert detect('st_test_gguf.bin')! == .gguf
}

fn test_detect_safetensors_header() ! {
	body := '{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
	mut f := []u8{}
	n := u64(body.len)
	for i in 0 .. 8 {
		f << u8((n >> u64(i * 8)) & 0xff)
	}
	f << body.bytes()
	f << []u8{len: 16}
	write_tmp('st_test_st.bin', f)!
	defer {
		os.rm('st_test_st.bin') or {}
	}
	assert detect('st_test_st.bin')! == .safetensors
	mf := load_safetensors('st_test_st.bin')!
	assert mf.count() == 1
	t := mf.find('w') or { panic('missing tensor w') }
	assert t.dtype == .f32
	assert t.shape == [u64(2), u64(2)]
	assert t.nbytes == 16
	assert mf.meta['format'] == 'pt'
}

fn test_detect_unknown_and_pytorch_hint() ! {
	write_tmp('st_test_txt.dat', 'hello world, no magic here.........'.bytes())!
	defer {
		os.rm('st_test_txt.dat') or {}
	}
	assert detect('st_test_txt.dat')! == .unknown
	// Extension fallback for legacy pytorch files (loader errors clearly).
	write_tmp('st_test_model.bin', 'torch garbage........................'.bytes())!
	defer {
		os.rm('st_test_model.bin') or {}
	}
	assert detect('st_test_model.bin')! == .pytorch_bin
}

fn test_dtype_from_str() {
	assert dtype_from_str('F32') == .f32
	assert dtype_from_str('bf16') == .bf16
	assert dtype_from_str('Q4_0') == .q4_0
	assert dtype_from_str('nope') == .other
	assert dtype_nbytes(.f32) == 4
	assert dtype_nbytes(.f16) == 2
}
