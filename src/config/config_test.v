module config

import os

fn test_defaults_validate() ! {
	c := defaults()
	c.validate()!
	assert c.server.port == 8000
	assert c.model.ctx_cap == 4096
	assert c.model.backend == 'cpu'
	assert c.model.format == 'auto'
}

fn test_partial_file_falls_back_to_defaults() ! {
	os.write_file('cfg_test.json', '{"server":{"port":9000}}')!
	defer {
		os.rm('cfg_test.json') or {}
	}
	c := load('cfg_test.json')!
	assert c.server.port == 9000
	assert c.server.host == '0.0.0.0'
	assert c.model.weights == 'models/qwen3-1.7b-q4_k_m.gguf'
	assert c.model.ctx_cap == 4096
}

fn test_bom_prefix_is_stripped() ! {
	// Windows editors (Notepad) save UTF-8 with BOM; the loader must cope.
	bom := [u8(0xEF), 0xBB, 0xBF]
	os.write_bytes('cfg_test_bom.json', [...bom, ...'{"server":{"port":9001}}'.bytes()])!
	defer {
		os.rm('cfg_test_bom.json') or {}
	}
	c := load('cfg_test_bom.json')!
	assert c.server.port == 9001
}

fn test_rejects_bad_values() {
	mut c := defaults()
	c.server.port = 99999
	if _ := c.validate() {
		assert false, 'expected port rejection'
	} else {
		assert true
	}
	mut c2 := defaults()
	c2.model.backend = 'cuda'
	if _ := c2.validate() {
		assert false, 'expected backend rejection'
	} else {
		assert true
	}
}
