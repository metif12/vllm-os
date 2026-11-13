module cfg

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

fn test_env_overrides_file_and_defaults() {
	os.setenv('VLLM_PORT', '9001', true)
	os.setenv('VLLM_BACKEND', 'cpu-simd', true)
	os.setenv('VLLM_PRESET', 'creative', true)
	defer {
		os.unsetenv('VLLM_PORT')
		os.unsetenv('VLLM_BACKEND')
		os.unsetenv('VLLM_PRESET')
	}
	mut c := defaults()
	c.apply_env()!
	assert c.server.port == 9001
	assert c.model.backend == 'cpu-simd'
	assert c.sampler.preset == 'creative'
	// Untouched keys keep file/default values.
	assert c.model.weights == 'models/qwen3-1.7b-q4_k_m.gguf'
	assert c.server.host == '0.0.0.0'
}

fn test_env_bad_value_rejects() {
	os.setenv('VLLM_PORT', '99999', true)
	defer {
		os.unsetenv('VLLM_PORT')
	}
	mut c := defaults()
	if _ := c.apply_env() {
		assert false, 'expected port rejection'
	} else {
		assert true
	}
}

fn test_sampler_preset_resolve() {
	mut c := defaults()
	c.sampler.preset = 'creative'
	p := c.sampler_params()
	assert p.temp == f32(1.1)
	assert p.top_k == 100
	assert p.top_p == f32(0.95)
}

fn test_sampler_explicit_overrides_preset() {
	mut c := defaults()
	c.sampler.preset = 'creative'
	c.sampler.temp = 0.2
	p := c.sampler_params()
	assert p.temp == f32(0.2)
	assert p.top_k == 100
}

fn test_unknown_preset_rejects() {
	mut c := defaults()
	c.sampler.preset = 'wild'
	if _ := c.validate() {
		assert false, 'expected preset rejection'
	} else {
		assert true
	}
}

fn test_sampler_bounds_reject() {
	mut c := defaults()
	c.sampler.temp = 5.0
	if _ := c.validate() {
		assert false, 'expected temp rejection'
	} else {
		assert true
	}
	mut c2 := defaults()
	c2.sampler.min_p = 1.0
	if _ := c2.validate() {
		assert false, 'expected min_p rejection'
	} else {
		assert true
	}
	mut c3 := defaults()
	c3.sampler.top_k = -1
	if _ := c3.validate() {
		assert false, 'expected top_k rejection'
	} else {
		assert true
	}
}

fn test_server_limits() {
	mut c := defaults()
	c.server.max_conns = 5000
	if _ := c.validate() {
		assert false, 'expected max_conns rejection'
	} else {
		assert true
	}
	mut c2 := defaults()
	c2.server.max_body = 100
	if _ := c2.validate() {
		assert false, 'expected max_body rejection'
	} else {
		assert true
	}
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

fn test_accepts_simd_and_auto_backends() {
	mut c := defaults()
	c.model.backend = 'cpu-simd'
	c.validate()!
	c.model.backend = 'cpu-parallel'
	c.validate()!
	c.model.backend = 'auto'
	c.validate()!
	c.model.backend = 'cpu-blocked'
	c.validate()!
}
