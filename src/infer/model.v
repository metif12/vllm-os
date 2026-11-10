module infer

import formats
import os

// Qwen3-1.7B arch defaults. Values are overridden from the GGUF KV store
// at boot; these are sane fallbacks for sizing the arenas.
pub struct Qwen3Config {
pub mut:
	model_id     string = 'Qwen/Qwen3-1.7B'
	hidden       int    = 2048
	layers       int    = 28
	heads_q      int    = 16
	heads_kv     int    = 8
	intermediate int    = 6144
	vocab        int    = 151936
	ctx_native   int    = 32768
	ctx_cap      int    = 4096
	rope_theta   f64    = 1000000.0
	rms_norm_eps f64    = 1e-6
	quant        string = 'Q4_K_M'
}

pub struct Model {
pub:
	cfg    Qwen3Config
	format formats.WeightFormat
	gguf   GgufModel         // valid when format == .gguf
	st     formats.ModelFile // valid when format == .safetensors
	tok    Bpe
}

// n_tensors is the unified tensor count across containers.
pub fn (m Model) n_tensors() int {
	return match m.format {
		.gguf { int(m.gguf.meta.n_tensors) }
		.safetensors { m.st.count() }
		else { 0 }
	}
}

// config_from_kv overrides arch fields from GGUF metadata when present.
pub fn config_from_kv(base Qwen3Config, kv map[string]string) Qwen3Config {
	mut c := base
	if v := kv['qwen3.embedding_length'] {
		c.hidden = v.int()
	}
	if v := kv['qwen3.block_count'] {
		c.layers = v.int()
	}
	if v := kv['qwen3.attention.head_count'] {
		c.heads_q = v.int()
	}
	if v := kv['qwen3.attention.head_count_kv'] {
		c.heads_kv = v.int()
	}
	if v := kv['qwen3.feed_forward_length'] {
		c.intermediate = v.int()
	}
	if v := kv['qwen3.context_length'] {
		c.ctx_native = v.int()
	}
	if v := kv['qwen3.rope.freq_base'] {
		c.rope_theta = v.f64()
	}
	if v := kv['tokenizer.ggml.tokens'] {
		_ = v
	}
	return c
}

// Boot path: detect container, parse index, load tokenizer sidecar. No Python.
pub fn load_model(weights_path string, cfg Qwen3Config) !Model {
	format := formats.detect(weights_path)!
	if format == .gguf {
		gguf := load_gguf_table(weights_path)!
		final_cfg := config_from_kv(cfg, gguf.kv)
		tok := load_sidecar_tokenizer(weights_path)!
		return Model{
			cfg:    final_cfg
			format: .gguf
			gguf:   gguf
			tok:    tok
		}
	}
	if format == .safetensors {
		st := formats.load_safetensors(weights_path)!
		tok := load_sidecar_tokenizer(weights_path)!
		return Model{
			cfg:    cfg
			format: .safetensors
			st:     st
			tok:    tok
		}
	}
	if format == .pytorch_bin {
		return error('pytorch .bin is detected but not loadable: convert to safetensors or GGUF first (scripts/fetch_model.sh)')
	}
	return error('unknown weight format: ${weights_path} (want .gguf or .safetensors)')
}

fn load_sidecar_tokenizer(weights_path string) !Bpe {
	tok_path := weights_dir(weights_path) + '/tokenizer.json'
	if os.exists(tok_path) {
		return load_bpe(tok_path)!
	}
	return Bpe{}
}

// load_model_stub keeps the no-weights dev loop alive (HTTP surface w/o GBs).
pub fn load_model_stub(cfg Qwen3Config) Model {
	return Model{
		cfg:  cfg
		gguf: GgufModel{}
	}
}

fn weights_dir(p string) string {
	i := p.last_index('/') or { return '.' }
	return p[..i]
}

// generate runs the wired path: template -> encode -> sample loop.
// The matmul/attention forward for real logits lands in phase 1b; until then
// the sampler runs over a deterministic pseudo-logit derived from the prompt
// so the full pipeline (encode->sample->decode) is exercised end to end.
pub fn (m Model) generate(prompt string, max_tokens int, sp SampleParams) string {
	ids := m.tok.encode(prompt)
	mut out := []int{cap: max_tokens}
	mut seed := u64(0)
	for id in ids {
		seed = seed * 1315423911 + u64(id)
	}
	// Reused across tokens: zero alloc inside the loop (bare-metal arena style).
	mut logits := []f32{len: 64}
	mut idx := []int{len: 64}
	n := if max_tokens < 64 { max_tokens } else { 64 }
	for i in 0 .. n {
		// Deterministic pseudo-logits: peak walks with (seed+i), so output
		// is stable per prompt but varies across prompts. Replaced by the
		// transformer forward in phase 1b without changing this signature.
		base := int(seed % 64) + i
		for j in 0 .. 64 {
			logits[j] = -f32((j + base) % 64)
		}
		next := sample_into(mut logits, mut idx, SampleParams{
			temp:  sp.temp
			top_k: sp.top_k
			top_p: sp.top_p
			seed:  seed + u64(i)
		})
		out << next
	}
	return 'qwen3-1.7b fmt=${m.format} tok_in=${ids.len} tok_out=${out.len} ids=${out} :: ${prompt}'
}

// generate_stub keeps the old HTTP text shape during the transition.
pub fn (m Model) generate_stub(prompt string, max_tokens int) string {
	return m.generate(prompt, max_tokens, SampleParams{})
}
