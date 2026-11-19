module infer

import os

// PLD gates: draft selection plus greedy bit-exactness on/off (real weights
// when present). The wall-time gate is NOT tested here — serial verification
// is intentionally speed-neutral; see ROADMAP phase 8.

fn test_pld_draft_longest_match() {
	history := [10, 20, 30, 40, 50, 10, 20, 30]
	mut out := []int{len: 4}
	n := pld_draft(history, 3, 3, mut out)
	assert n == 3
	assert out[0] == 40
	assert out[1] == 50
	assert out[2] == 10
}

fn test_pld_draft_no_match() {
	history := [1, 2, 3, 4, 5, 6]
	mut out := []int{len: 4}
	n := pld_draft(history, 3, 4, mut out)
	assert n == 0
}

fn test_pld_draft_short_history() {
	history := [7]
	mut out := []int{len: 2}
	assert pld_draft(history, 3, 2, mut out) == 0
	assert pld_draft([], 3, 2, mut out) == 0
}

fn test_pld_draft_continuation_cap() {
	history := [1, 2, 3, 4, 5, 6, 1, 2, 3]
	mut out := []int{len: 2}
	n := pld_draft(history, 3, 2, mut out)
	assert n == 2
	assert out[0] == 4
	assert out[1] == 5
}

fn test_pld_zero_stats() {
	s := PldStats{}
	assert s.accept_rate() == 0
	mut s2 := PldStats{}
	s2.drafts = 4
	s2.accepted = 3
	assert s2.accept_rate() > 0.74 && s2.accept_rate() < 0.76
}

// Bit-exactness on real weights: greedy output must not change when PLD is
// enabled, regardless of how many drafts are proposed or accepted.
fn test_pld_greedy_bit_exact_real_weights() {
	orig_threads := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '1', true)
	mut path := ''
	for c in ['models/Qwen3-0.6B-Q4_0.gguf', '../models/Qwen3-0.6B-Q4_0.gguf',
		'../../models/Qwen3-0.6B-Q4_0.gguf'] {
		if os.exists(c) {
			path = c
			break
		}
	}
	if path == '' {
		return
	}
	mut m := load_model(path, Qwen3Config{}) or { return }
	if m.blob.len == 0 {
		return
	}
	sp := SampleParams{
		temp: 0.0
	}
	prompt := 'The city council met on Tuesday to discuss the annual budget.'
	m.pld = PldCfg{
		enabled: false
	}
	plain := m.generate(prompt, 8, sp)
	m.pld = PldCfg{
		enabled:      true
		max_ngram:    3
		continuation: 10
	}
	spec := m.generate(prompt, 8, sp)
	if v := orig_threads {
		os.setenv('VLLM_THREADS', v, true)
	} else {
		os.unsetenv('VLLM_THREADS')
	}
	assert plain == spec
}
