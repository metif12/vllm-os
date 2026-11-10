module infer

import os

fn test_sample_greedy_picks_argmax() {
	assert sample_greedy([f32(0.1), 2.5, 1.0]) == 1
}

fn test_softmax_sums_to_one() {
	mut xs := [f32(1.0), 2.0, 3.0]
	softmax_in_place(mut xs)
	mut s := f32(0)
	for v in xs {
		s += v
	}
	assert s > 0.999 && s < 1.001
	assert xs[2] > xs[1] && xs[1] > xs[0]
}

fn test_sample_deterministic_with_seed() {
	logits := [f32(0.5), 1.5, 1.0, 0.2]
	a := sample(logits, SampleParams{
		temp:  0.8
		top_k: 4
		top_p: 0.9
		seed:  42
	})
	b := sample(logits, SampleParams{
		temp:  0.8
		top_k: 4
		top_p: 0.9
		seed:  42
	})
	assert a == b
}

fn test_sample_temp_zero_is_greedy() {
	logits := [f32(0.5), 1.5, 1.0]
	assert sample(logits, SampleParams{
		temp: 0
		seed: 1
	}) == 1
}

fn test_rmsnorm_unit_scale() {
	mut x := [f32(1.0), 1.0, 1.0, 1.0]
	w := [f32(1.0), 1.0, 1.0, 1.0]
	rmsnorm_in_place(mut x, w, 0.0)
	for v in x {
		assert v > 0.999 && v < 1.001
	}
}

fn test_dequant_q8_0_scale() {
	b := BlockQ80{
		d:  u16(0x3c00) // 1.0 in f16
		qs: [32]i8{init: 2}
	}
	mut out := []f32{len: 32}
	dequant_q8_0_row(b, mut out)
	for v in out {
		assert v > 1.99 && v < 2.01
	}
}

fn test_chat_template_shape() {
	s := apply_chat_template([ChatMsg{
		role:    'user'
		content: 'hi'
	}])
	assert s.contains('<|im_start|>user\nhi<|im_end|>')
	assert s.ends_with('<|im_start|>assistant\n')
}

fn test_load_gguf_header_rejects_garbage() {
	os.write_file('vllm_test_bad.gguf', 'not a gguf file at all..............')!
	defer {
		os.rm('vllm_test_bad.gguf') or {}
	}
	if _ := load_gguf_header('vllm_test_bad.gguf') {
		assert false, 'expected bad-magic error'
	} else {
		assert err.msg().contains('bad magic')
	}
}

fn test_config_from_kv_overrides() {
	base := Qwen3Config{}
	kv := {
		'qwen3.block_count':             '12'
		'qwen3.embedding_length':        '1024'
		'qwen3.attention.head_count':    '8'
		'qwen3.attention.head_count_kv': '4'
	}
	c := config_from_kv(base, kv)
	assert c.layers == 12
	assert c.hidden == 1024
	assert c.heads_q == 8
	assert c.heads_kv == 4
	// untouched fallback stays
	assert c.vocab == 151936
}
