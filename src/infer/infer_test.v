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

fn test_preset_params_values() {
	det := preset_params('deterministic')!
	assert det.temp == 0.0
	precise := preset_params('precise')!
	assert precise.temp == f32(0.3)
	assert precise.top_k == 20
	bal := preset_params('balanced')!
	assert bal == SampleParams{}
	cre := preset_params('creative')!
	assert cre.temp == f32(1.1)
	assert cre.top_k == 100
	assert cre.top_p == f32(0.95)
	assert sampler_presets().len == 4
	if _ := preset_params('wild') {
		assert false, 'expected unknown preset error'
	} else {
		assert err.msg().contains('wild')
	}
}

fn test_deterministic_preset_is_greedy() {
	logits := [f32(0.1), 2.5, 1.0]
	p := preset_params('deterministic')!
	mut work := logits.clone()
	mut idx := []int{len: 3}
	assert sample_into(mut work, mut idx, p) == 1
}

fn test_min_p_filters_tail() {
	logits := [f32(4.0), 1.0, 0.5, 0.0, -1.0]
	mut work := logits.clone()
	mut idx := []int{len: 5}
	// top-1 prob ~0.92; min_p 0.5 keeps only the winner deterministically.
	r := sample_into(mut work, mut idx, SampleParams{
		temp:  1.0
		top_k: 5
		top_p: 1.0
		min_p: 0.5
		seed:  1
	})
	assert r == 0
}

fn test_min_p_zero_keeps_all() {
	logits := [f32(4.0), 1.0, 0.5, 0.0, -1.0]
	mut w1 := logits.clone()
	mut w2 := logits.clone()
	mut i1 := []int{len: 5}
	mut i2 := []int{len: 5}
	a := sample_into(mut w1, mut i1, SampleParams{
		temp:  1.0
		top_k: 5
		top_p: 1.0
		seed:  3
	})
	b := sample_into(mut w2, mut i2, SampleParams{
		temp:  1.0
		top_k: 5
		top_p: 1.0
		min_p: 1e-9
		seed:  3
	})
	assert a == b
}

fn test_tokenizer_interface_dispatch() {
	b := Bpe{}
	t := Tokenizer(b)
	assert t.name() == 'bpe-stub'
	assert t.encode('hi there') == encode_stub('hi there')
	s := Tokenizer(StubTokenizer{})
	assert s.name() == 'stub'
	assert s.encode('a b') == encode_stub('a b')
	m := load_model_stub(Qwen3Config{})
	assert m.tok.name() == 'stub'
	assert m.tok.encode('hello').len > 0
}

fn test_weights_dir_handles_both_separators() {
	assert weights_dir('models/qwen3.gguf') == 'models'
	assert weights_dir('C:\\models\\qwen3.gguf') == 'C:\\models'
	assert weights_dir('qwen3.gguf') == '.'
}

// BPE golden vectors: exact llama.cpp parity on general text (spaces,
// punctuation, digits, newlines, contractions, unicode). The sidecar
// tokenizer.json is gitignored like the weights; skip when absent.
fn bpe_test_tok() ?Bpe {
	for c in ['models/tokenizer.json', '../models/tokenizer.json', '../../models/tokenizer.json'] {
		if os.exists(c) {
			return load_bpe(c)!
		}
	}
	if v := os.getenv_opt('VLLM_MODELS') {
		p := v + '/tokenizer.json'
		if os.exists(p) {
			return load_bpe(p)!
		}
	}
	return none
}

fn test_bpe_encode_matches_llama() {
	b := bpe_test_tok() or { return }
	cases := {
		'Hi':                   [13048]
		'Hello world':          [9707, 1879]
		'Hello, world!':        [9707, 11, 1879, 0]
		'The quick brown fox.': [785, 3974, 13876, 38835, 13]
		'123 456':              [16, 17, 18, 220, 19, 20, 21]
		"don't stop":           [15007, 944, 2936]
		'a  b':                 [64, 220, 293]
		'  leading':            [220, 6388]
		'trailing  ':           [376, 14277, 256]
		'Qwen3-0.6B':           [48, 16948, 18, 12, 15, 13, 21, 33]
		'café':                 [924, 58858]
		'naïve':                [3376, 37572, 586]
		'Hello 世界':           [9707, 220, 99489]
		'price: $19.99':        [6555, 25, 400, 16, 24, 13, 24, 24]
		'email me@example.com': [2332, 752, 35487, 905]
		'(a+b)*c':              [2877, 35093, 4806, 66]
		'x=1;y=2':              [87, 28, 16, 70863, 28, 17]
		'100%':                 [16, 15, 15, 4]
		'well-known':           [9157, 21309]
		"It's fine":            [2132, 594, 6915]
		'   three':             [256, 2326]
		'a   b':                [64, 256, 293]
		'12345':                [16, 17, 18, 19, 20]
		'3.14159':              [18, 13, 16, 19, 16, 20, 24]
	}
	for text, want in cases {
		got := b.encode(text)
		assert got == want, 'bpe mismatch for ${text}: got ${got}, want ${want}'
	}
}

fn test_bpe_decode_roundtrip() {
	b := bpe_test_tok() or { return }
	cases := ['Hi', 'Hello world', 'Hello, world!', 'price: $19.99', 'café', 'line1\nline2',
		'x=1;y=2', "don't stop", '100%']
	for text in cases {
		assert b.decode(b.encode(text)) == text, 'roundtrip failed for ${text}'
	}
}
