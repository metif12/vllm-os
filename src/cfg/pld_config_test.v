module cfg

import os

// Phase 8 PLD knob gates: defaults, env overrides, normalization and
// validation. Kept in a separate file so the existing cfg test list stays
// untouched; `v test src` picks it up automatically.

fn test_pld_defaults_and_resolution() {
	c := defaults()
	assert !c.sampler.pld
	assert c.sampler.pld_ngram == 3
	assert c.sampler.pld_continuation == 10
	p := c.pld_config()
	assert !p.enabled
	assert p.max_ngram == 3
	assert p.continuation == 10
}

fn test_pld_env_overrides() {
	orig_on := os.getenv_opt('VLLM_PLD')
	orig_n := os.getenv_opt('VLLM_PLD_NGRAM')
	orig_c := os.getenv_opt('VLLM_PLD_CONTINUATION')
	os.setenv('VLLM_PLD', '1', true)
	os.setenv('VLLM_PLD_NGRAM', '5', true)
	os.setenv('VLLM_PLD_CONTINUATION', '7', true)
	mut c := defaults()
	c.apply_env() or { panic(err) }
	p := c.pld_config()
	assert p.enabled
	assert p.max_ngram == 5
	assert p.continuation == 7
	// Restore explicitly (defer with optional capture breaks release-V cgen).
	if v := orig_on {
		os.setenv('VLLM_PLD', v, true)
	} else {
		os.unsetenv('VLLM_PLD')
	}
	if v := orig_n {
		os.setenv('VLLM_PLD_NGRAM', v, true)
	} else {
		os.unsetenv('VLLM_PLD_NGRAM')
	}
	if v := orig_c {
		os.setenv('VLLM_PLD_CONTINUATION', v, true)
	} else {
		os.unsetenv('VLLM_PLD_CONTINUATION')
	}
}

fn test_pld_normalize_and_validate() {
	mut c := defaults()
	c.sampler.pld_ngram = 0
	c.sampler.pld_continuation = 0
	c.normalize()
	assert c.sampler.pld_ngram == 3
	assert c.sampler.pld_continuation == 10
	c.sampler.pld_ngram = 100
	mut rejected := false
	c.validate() or { rejected = true }
	assert rejected
	c.sampler.pld_ngram = 3
	c.sampler.pld_continuation = 100
	rejected = false
	c.validate() or { rejected = true }
	assert rejected
}
