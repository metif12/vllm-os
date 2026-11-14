module infer

import formats
import os

// Qwen3-1.7B arch defaults. Values are overridden from the GGUF KV store
// at boot; these are sane fallbacks for sizing the arenas.
// head_dim is the Q/K head length (qwen3.attention.key_length); v_dim is
// the V head length (value_length). Qwen3-1.7B has hidden=2048=16*128 so
// hidden/heads happens to equal head_dim, but 0.6B has hidden=1024 with
// 16x128 q heads (o-proj 2048->1024) — never derive hd from hidden.
pub struct Qwen3Config {
pub mut:
	model_id     string = 'Qwen/Qwen3-1.7B'
	hidden       int    = 2048
	layers       int    = 28
	heads_q      int    = 16
	heads_kv     int    = 8
	head_dim     int    = 128
	v_dim        int    = 128
	intermediate int    = 6144
	vocab        int    = 151936
	ctx_native   int    = 32768
	ctx_cap      int    = 4096
	rope_theta   f64    = 1000000.0
	rms_norm_eps f64    = 1e-6
	quant        string = 'Q4_K_M'
}

// Model is heap-allocated: the server shares one instance across all
// request handlers via HandlerCtx.
@[heap]
pub struct Model {
pub:
	cfg    Qwen3Config
	format formats.WeightFormat
	gguf   GgufModel         // valid when format == .gguf
	st     formats.ModelFile // valid when format == .safetensors (or future registry formats)
	tok    Tokenizer
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
	if v := kv['qwen3.attention.key_length'] {
		c.head_dim = v.int()
	}
	if v := kv['qwen3.attention.value_length'] {
		c.v_dim = v.int()
	}
	if v := kv['qwen3.attention.layer_norm_rms_epsilon'] {
		c.rms_norm_eps = v.f64()
	}
	if v := kv['tokenizer.ggml.tokens'] {
		_ = v
	}
	return c
}

// Boot path: detect container, parse index, load tokenizer sidecar. No Python.
// load_model is the single entry point from main.v to weights.
// GGUF keeps its native path (rich GgmlType index); every other container
// dispatches through the formats handler registry — new formats need no
// edits here.
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
	// Generic path: registry-dispatched. Unknown formats fail here;
	// recognized-but-unloadable ones (pytorch hint) fail inside load_index.
	h := formats.handler_for(format) or {
		return error('unknown weight format: ${weights_path} (want .gguf or .safetensors)')
	}
	st := h.load_index(weights_path)!
	tok := load_sidecar_tokenizer(weights_path)!
	return Model{
		cfg:    cfg
		format: format
		st:     st
		tok:    tok
	}
}

// load_sidecar_tokenizer loads the baked-in HF tokenizer.json, or the
// explicit stub tokenizer when absent (HTTP surface stays testable).
fn load_sidecar_tokenizer(weights_path string) !Tokenizer {
	tok_path := weights_dir(weights_path) + '/tokenizer.json'
	if os.exists(tok_path) {
		return Tokenizer(load_bpe(tok_path)!)
	}
	return Tokenizer(StubTokenizer{})
}

// load_model_stub keeps the no-weights dev loop alive (HTTP surface w/o GBs).
pub fn load_model_stub(cfg Qwen3Config) Model {
	return Model{
		cfg:  cfg
		gguf: GgufModel{}
		tok:  Tokenizer(StubTokenizer{})
	}
}

fn weights_dir(p string) string {
	i := p.last_index('/') or { -1 }
	// Windows callers pass backslash paths; models/ lookups must work
	// for both (a missed sidecar silently degrades to the stub tokenizer).
	j := p.last_index('\\') or { -1 }
	k := if i > j { i } else { j }
	if k < 0 {
		return '.'
	}
	return p[..k]
}

// generate runs the wired path: encode -> transformer forward -> sample
// loop -> decode. With GGUF weights present it runs the real phase-1b
// forward (prefill + decode steps over ForwardCtx arenas); without weights
// (stub/test) it falls back to the deterministic pseudo-logit path so the
// full pipeline stays exercisable end to end.
pub fn (m Model) generate(prompt string, max_tokens int, sp SampleParams) string {
	ids := m.tok.encode(prompt)
	if m.format == .gguf && m.gguf.path != '' && os.exists(m.gguf.path) {
		return m.generate_real(ids, max_tokens, sp) or {
			return m.generate_pseudo(prompt, ids, max_tokens, sp)
		}
	}
	return m.generate_pseudo(prompt, ids, max_tokens, sp)
}

// generate_real runs prefill + sampling over the transformer forward.
// Blob load is per call (host phase-1b; the bare-metal port hands a baked
// slice instead). KV arenas are sized from cfg.ctx_cap.
fn (m Model) generate_real(ids []int, max_tokens int, sp SampleParams) !string {
	blob := load_model_blob(m.gguf.path)!
	mut f := new_forward_ctx(m.cfg, m.gguf, blob)!
	for id in ids {
		embed_gather(blob, m.gguf, m.cfg.hidden, id, mut f.hidden)!
		f.decode_step()!
	}
	eos := m.gguf.kv['tokenizer.ggml.eos_token_id'].int()
	mut out := []int{cap: max_tokens}
	mut work := []f32{len: f.logits.len}
	mut idx := []int{len: f.logits.len}
	mut seed := sp.seed
	for id in ids {
		seed = seed * 1315423911 + u64(id)
	}
	for i in 0 .. max_tokens {
		for j in 0 .. f.logits.len {
			work[j] = f.logits[j]
		}
		p := SampleParams{
			temp:  sp.temp
			top_k: sp.top_k
			top_p: sp.top_p
			min_p: sp.min_p
			seed:  seed + u64(i)
			fast:  sp.fast
		}
		next := if sp.fast {
			sample_into_fast(mut work, mut idx, p)
		} else {
			sample_into(mut work, mut idx, p)
		}
		out << next
		if next == eos {
			break
		}
		embed_gather(blob, m.gguf, m.cfg.hidden, next, mut f.hidden)!
		f.decode_step()!
	}
	return m.tok.decode(out)
}

// generate_pseudo runs the deterministic pseudo-logit path (no weights).
fn (m Model) generate_pseudo(prompt string, ids []int, max_tokens int, sp SampleParams) string {
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
		// is stable per prompt but varies across prompts.
		base := int(seed % 64) + i
		for j in 0 .. 64 {
			logits[j] = -f32((j + base) % 64)
		}
		next := if sp.fast {
			sample_into_fast(mut logits, mut idx, SampleParams{
				temp:  sp.temp
				top_k: sp.top_k
				top_p: sp.top_p
				min_p: sp.min_p
				seed:  seed + u64(i)
			})
		} else {
			sample_into(mut logits, mut idx, SampleParams{
				temp:  sp.temp
				top_k: sp.top_k
				top_p: sp.top_p
				min_p: sp.min_p
				seed:  seed + u64(i)
			})
		}
		out << next
	}
	return 'qwen3-1.7b fmt=${m.format} tok_in=${ids.len} tok_out=${out.len} ids=${out} :: ${prompt}'
}

// generate_stub keeps the old HTTP text shape during the transition.
pub fn (m Model) generate_stub(prompt string, max_tokens int) string {
	return m.generate(prompt, max_tokens, SampleParams{})
}
