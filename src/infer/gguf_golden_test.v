module infer

import os

// Golden GGUF table tests: the two small reference models must parse to
// exact tensor counts, byte sizes, types, and key shapes. Catches parser
// regressions (array truncation, block-size math, shape decode) that only
// real files trigger — the stub fixtures in infer_test.v cannot.
//
// The weights are NOT committed (models/ is gitignored, ~1.4 GB total).
// When absent (CI, fresh clones) the tests skip with a pointer to
// scripts/fetch_small_models.sh instead of failing.
// Qwen3-1.7B is Apache-2.0 via unsloth (no license gate, unlike Llama);
// the 0.6B tokenizer.json covers it too (same 151936 Qwen3 vocab).

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
	assert total_bytes(m.tensors) == 376204288
	h := count_types(m.tensors)
	assert h[.q4_0] == 193
	assert h[.f32] == 113
	assert h[.q4_1] == 3
	assert h[.q6_k] == 1
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

fn test_gguf_golden_qwen_17b() {
	path := golden_path('Qwen3-1.7B-Q4_0.gguf') or {
		eprintln('[skip] Qwen3-1.7B-Q4_0.gguf absent; run scripts/fetch_small_models.sh')
		return
	}
	m := load_gguf_table(path)!
	assert m.meta.n_tensors == 310
	assert m.meta.n_kv == 32
	assert total_bytes(m.tensors) == 1050830848
	h := count_types(m.tensors)
	assert h[.q4_0] == 193
	assert h[.f32] == 113
	assert h[.q4_1] == 3
	assert h[.q6_k] == 1
	cfg := config_from_kv(Qwen3Config{}, m.kv)
	assert cfg.hidden == 2048
	assert cfg.layers == 28
	assert cfg.heads_q == 16
	assert cfg.heads_kv == 8
	assert cfg.intermediate == 6144
	q := m.find('blk.0.attn_q.weight') or { panic('missing blk.0.attn_q.weight') }
	assert q.typ == .q4_0
	assert q.shape == [u64(2048), u64(2048)]
}

// Phase 11 target model: Q4_K_M. Skips when the file is absent (it is not
// fetched by fetch_small_models.sh, which currently pulls Q4_0). When
// present, this pins the 142-byte / 256-elem block sizing and the arch KV.
fn test_gguf_golden_qwen_17b_q4_k_m() {
	path := golden_path('Qwen3-1.7B-Q4_K_M.gguf') or {
		eprintln('[skip] Qwen3-1.7B-Q4_K_M.gguf absent; phase 11 real-weight gate pending')
		return
	}
	m := load_gguf_table(path)!
	assert m.meta.n_tensors == 310
	h := count_types(m.tensors)
	assert h[.q4_k] > 0
	q := m.find('blk.0.attn_q.weight') or { panic('missing blk.0.attn_q.weight') }
	assert q.typ == .q4_k
	assert q.shape == [u64(2048), u64(2048)]
	assert q.size == u64((2048 * 2048 / 256) * 142)
	cfg := config_from_kv(Qwen3Config{}, m.kv)
	assert cfg.hidden == 2048
	assert cfg.layers == 28
	assert cfg.intermediate == 6144
	emb := m.find('token_embd.weight') or { panic('missing token_embd.weight') }
	assert emb.typ == .q6_k
}

// load_model caches the weights blob at boot so the serve path never
// re-reads the file per request (that was a seconds-scale disk tax).
// Skips like the golden tests when models/ is absent.
fn test_load_model_caches_blob() {
	path := golden_path('Qwen3-0.6B-Q4_0.gguf') or {
		eprintln('[skip] Qwen3-0.6B-Q4_0.gguf absent; run scripts/fetch_small_models.sh')
		return
	}
	m := load_model(path, Qwen3Config{})!
	assert m.blob.len == 382156480
	// The cached blob serves real bytes: first tensor slices from it.
	t := tensor_bytes(m.blob, m.gguf, 'blk.0.attn_q.weight')!
	assert t.len > 0
}
