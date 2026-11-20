module cfg

import os

// Phase 10 attention-window knob gates: defaults, env overrides, validation.

fn test_attn_window_defaults() {
	c := defaults()
	assert c.model.attn_window == 0
	assert c.model.attn_sinks == 4
	c.validate() or { panic(err) }
}

fn test_attn_window_env() {
	orig_w := os.getenv_opt('VLLM_ATTN_WINDOW')
	orig_s := os.getenv_opt('VLLM_ATTN_SINKS')
	os.setenv('VLLM_ATTN_WINDOW', '512', true)
	os.setenv('VLLM_ATTN_SINKS', '8', true)
	mut c := defaults()
	c.apply_env() or { panic(err) }
	assert c.model.attn_window == 512
	assert c.model.attn_sinks == 8
	if v := orig_w {
		os.setenv('VLLM_ATTN_WINDOW', v, true)
	} else {
		os.unsetenv('VLLM_ATTN_WINDOW')
	}
	if v := orig_s {
		os.setenv('VLLM_ATTN_SINKS', v, true)
	} else {
		os.unsetenv('VLLM_ATTN_SINKS')
	}
}

fn test_attn_window_validation() {
	mut c := defaults()
	c.model.attn_window = 16
	mut rejected := false
	c.validate() or { rejected = true }
	assert rejected
	c.model.attn_window = 0
	c.model.attn_sinks = 100
	rejected = false
	c.validate() or { rejected = true }
	assert rejected
}
