module infer

import os
import time

// Prefix-cache gates: match detection and bit-identical generation with the
// cache on/off. Real weights are required for the latter (skip if absent).

fn test_prefix_match_len() {
	mut c := new_prefix_cache()
	ids := [1, 2, 3, 4, 5]
	// No cache yet.
	assert c.match_len(ids) == 0
	// Fill the entry directly for the match logic (no KV needed here).
	c.tokens = ids.clone()
	c.pos = ids.len
	assert c.match_len(ids) == 0 // must leave at least one token to prefill
	assert c.match_len([1, 2, 3, 4, 5, 6]) == 5
	assert c.match_len([1, 2, 9, 4, 5, 6]) == 0
	assert c.match_len([1, 2]) == 0
}

fn test_prefix_cache_generation_identical() {
	mut path := ''
	for p in ['models/Qwen3-0.6B-Q4_0.gguf', '../models/Qwen3-0.6B-Q4_0.gguf',
		'../../models/Qwen3-0.6B-Q4_0.gguf'] {
		if os.exists(p) {
			path = p
			break
		}
	}
	if path == '' {
		return
	}
	orig := os.getenv_opt('VLLM_THREADS')
	os.setenv('VLLM_THREADS', '1', true)
	mut m := load_model(path, Qwen3Config{}) or { return }
	if m.blob.len == 0 {
		return
	}
	sp := SampleParams{
		temp: 0.0
	}
	turn1 := 'hello world hello world hello world hello world hello world hello world hello world hello world'
	turn2 := turn1 + ' goodbye'
	m.prefix = unsafe { nil }
	mut sw := time.new_stopwatch()
	plain := m.generate(turn2, 8, sp)
	plain_ms := sw.elapsed().milliseconds()
	m.prefix = new_prefix_cache()
	_ = m.generate(turn1, 2, sp) // prime the cache with turn 1
	sw = time.new_stopwatch()
	cached := m.generate(turn2, 8, sp)
	cached_ms := sw.elapsed().milliseconds()
	eprintln('[prefix_bench] plain_ms=${plain_ms} cached_ms=${cached_ms}')
	hits, misses := m.prefix.stats()
	if v := orig {
		os.setenv('VLLM_THREADS', v, true)
	} else {
		os.unsetenv('VLLM_THREADS')
	}
	assert misses >= 1
	assert hits >= 1
	assert plain == cached
}
