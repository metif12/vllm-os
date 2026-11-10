module infer

// Qwen3-1.7B arch defaults. Values MUST be overridden from the baked
// config.json / GGUF KV store at boot; these are sane fallbacks for sizing.
pub struct Qwen3Config {
pub:
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
	cfg  Qwen3Config
	gguf GgufModel
}

// Boot path: open weights, no Python, no lazy download.
pub fn load_model(weights_path string, cfg Qwen3Config) !Model {
	gguf := load_gguf_header(weights_path)!
	return Model{
		cfg:  cfg
		gguf: gguf
	}
}

// Phase-1 stub: proves boot -> model -> text path before kernels land.
// Phase 2 replaces this with matmul/dequant/attention + KV-cache.
pub fn (m Model) generate_stub(prompt string, max_tokens int) string {
	toks := encode_stub(prompt)
	budget := if max_tokens < 64 { max_tokens } else { 64 }
	return '[vllm-os qwen3-1.7b stub] prompt_tokens=${toks.len} budget=${budget} gguf_tensors=${m.gguf.meta.n_tensors} version=${m.gguf.meta.version} :: ${prompt}'
}
