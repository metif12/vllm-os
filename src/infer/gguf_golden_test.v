module infer

import os

// Golden GGUF table tests: the two small reference models must parse to
// exact tensor counts, byte sizes, types, and key shapes. Catches parser
// regressions (array truncation, block-size math, shape decode) that only
// real files trigger — the stub fixtures in infer_test.v cannot.
//
// The weights are NOT committed (models/ is gitignored, ~1.1 GB total).
// When absent (CI, fresh clones) the tests skip with a pointer to
// scripts/fetch_small_models.sh instead of failing.

// golden_path resolves a models/ file from several plausible working
// directories (v test runs each file from its own dir) or VLLM_MODELS.
fn golden_path(name string) ?string {
	if v := os.getenv_opt('VLLM_MODELS') {
		p := v + '/' + name
		if os.exists(p) {
			return p
		}
	}
	for c in ['models/' + name, '../models/' + name, '../../models/' + name] {
		if os.exists(c) {
			return c
		}
	}
	return none
}

fn count_types(tensors []GgufTensorInfo) map[GgmlType]int {
	mut h := map[GgmlType]int{}
	for t in tensors {
		h[t.typ] = h[t.typ] + 1
	}
	return h
}

fn total_bytes(tensors []GgufTensorInfo) u64 {
	mut n := u64(0)
	for t in tensors {
		n += t.size
	}
	return n
}

fn test_gguf_golden_qwen_06b() {
	path := golden_path('Qwen3-0.6B-Q4_0.gguf') or {
		eprintln('[skip] Qwen3-0.6B-Q4_0.gguf absent; run scripts/fetch_small_models.sh')
		return
	}
	m := load_gguf_table(path)!
	assert m.meta.n_tensors == 310
	assert m.meta.n_kv == 32
	assert total_bytes(m.tensors) == 404160512
	h := count_types(m.tensors)
	assert h[.q4_0] == 193
	assert h[.f32] == 113
	assert h[.q4_1] == 3
	assert h[.i8] == 1
	cfg := config_from_kv(Qwen3Config{}, m.kv)
	assert cfg.hidden == 1024
	assert cfg.layers == 28
	assert cfg.heads_q == 16
	assert cfg.heads_kv == 8
	assert cfg.intermediate == 3072
	q := m.find('blk.0.attn_q.weight') or { panic('missing blk.0.attn_q.weight') }
	assert q.typ == .q4_0
	assert q.shape == [u64(1024), u64(2048)]
}

fn test_gguf_golden_llama_1b() {
	path := golden_path('Llama-3.2-1B-Q4_0.gguf') or {
		eprintln('[skip] Llama-3.2-1B-Q4_0.gguf absent; run scripts/fetch_small_models.sh')
		return
	}
	m := load_gguf_table(path)!
	assert m.meta.n_tensors == 147
	assert m.meta.n_kv == 35
	assert total_bytes(m.tensors) == 812392576
	h := count_types(m.tensors)
	assert h[.q4_0] == 110
	assert h[.f32] == 34
	assert h[.q4_1] == 2
	assert h[.i8] == 1
	// NOTE: config_from_kv only understands qwen3.* keys, so Llama arch
	// fields print Qwen defaults. Assert tensor-level facts instead;
	// generalizing the keys rides with tokenizer-from-GGUF.
	q := m.find('blk.0.attn_q.weight') or { panic('missing blk.0.attn_q.weight') }
	assert q.typ == .q4_0
	assert q.shape == [u64(2048), u64(2048)]
}
